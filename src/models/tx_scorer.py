"""
DeepTransactionScorer — Edge / Transaction-Level Deep Learning Anomaly Detector.

Directly addresses transaction-granularity detection ("Phát hiện bất thường trong giao dịch
Ethereum bằng học sâu") by fusing:
  1. Sender & Receiver deep representations z_u, z_v in R^{64} from ModularTGAE
  2. Structural interaction terms |z_u - z_v| and z_u * z_v
  3. 5-D directed edge attributes [tx_count_log, total_value_log, mean_value_log, last_tx_age_norm, avg_gas_log]
  4. 8-D Transaction Behavioral Shock signals (value spike vs. sender historical average,
     sender/receiver TG-AE reconstruction scores, zero-value contract call flag, burstiness, asymmetry).

Trained jointly with Edge Autoencoder Reconstruction + Focal Contrastive Head and calibrated on Validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from src.models.data_utils import extract_labels, extract_static_features
from src.schemas import (
    ALL_FEATURE_COLS,
    GRAPH_TOPO_COLS,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    RANDOM_SEED,
    TABULAR_TX_COLS,
)

TABULAR_DIM = len(TABULAR_TX_COLS)
TOPO_DIM = len(GRAPH_TOPO_COLS)

# Column indices in 25-D static vector for fast behavioral shock extraction
_IDX_AVG_SENT = ALL_FEATURE_COLS.index("avg_eth_sent")
_IDX_SENT_RECV_RATIO = ALL_FEATURE_COLS.index("sent_recv_ratio")
_IDX_ZERO_RATIO = ALL_FEATURE_COLS.index("zero_val_tx_ratio")
_IDX_FAILED_RATIO = ALL_FEATURE_COLS.index("failed_tx_ratio")
_IDX_BURSTINESS = ALL_FEATURE_COLS.index("burstiness_index")


class DeepTxEdgeNet(nn.Module):
    """
    Deep Edge Anomaly Network combining Residual Gated Bottleneck Autoencoder + Focal Anomaly Head.
    Separates three interpretable branches before fusion:
      - Sender Representation Branch (z_u + sender risk)
      - Receiver Representation Branch (z_v + receiver risk)
      - Transaction Dynamics & Behavioral Shock Branch (edge_attr + shock signals + |z_u-z_v|, z_u*z_v)
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.sender_proj = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.receiver_proj = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        # Interaction (2 * latent_dim) + edge_attr (5) + shock_context (6)
        dyn_in = 2 * latent_dim + 5 + 6
        self.dynamics_proj = nn.Sequential(
            nn.Linear(dyn_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        fused_in = 3 * (hidden_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fused_in),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z_u: torch.Tensor,
        z_v: torch.Tensor,
        s_u: torch.Tensor,
        s_v: torch.Tensor,
        edge_attr: torch.Tensor,
        shock_ctx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_s = self.sender_proj(torch.cat([z_u, s_u.unsqueeze(-1)], dim=-1))
        h_r = self.receiver_proj(torch.cat([z_v, s_v.unsqueeze(-1)], dim=-1))
        inter = torch.cat([torch.abs(z_u - z_v), z_u * z_v, edge_attr, shock_ctx], dim=-1)
        h_d = self.dynamics_proj(inter)

        h_cat = torch.cat([h_s, h_r, h_d], dim=-1)
        z_edge = self.encoder(h_cat)
        h_hat = self.decoder(z_edge)

        d = h_s.shape[-1]
        err_sender = (h_hat[:, :d] - h_s).pow(2).mean(dim=-1)
        err_receiver = (h_hat[:, d : 2 * d] - h_r).pow(2).mean(dim=-1)
        err_dynamics = (h_hat[:, 2 * d :] - h_d).pow(2).mean(dim=-1)
        recon_err = (err_sender + err_receiver + err_dynamics) / 3.0

        logits = self.cls_head(torch.cat([z_edge, recon_err.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return {
            "logits": logits,
            "recon_err": recon_err,
            "err_sender": err_sender,
            "err_receiver": err_receiver,
            "err_dynamics": err_dynamics,
        }


def build_shock_context(
    x_u: torch.Tensor,
    x_v: torch.Tensor,
    s_u: torch.Tensor,
    s_v: torch.Tensor,
    edge_attr: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 6-D behavioral shock tensor comparing transaction attributes against sender/receiver profiles:
      0: max(s_u, s_v)
      1: |s_u - s_v|
      2: value_spike_vs_sender (edge mean_value_log - sender scaled avg_eth_sent)
      3: zero_or_tiny_value_flag (edge total_value_log <= 1e-4)
      4: sender_burstiness
      5: sender_out_in_asymmetry
    """
    max_s = torch.maximum(s_u, s_v)
    diff_s = torch.abs(s_u - s_v)
    val_spike = edge_attr[:, 2] - x_u[:, _IDX_AVG_SENT]
    is_zero_val = (edge_attr[:, 1] <= 1e-4).to(edge_attr.dtype)
    burst = x_u[:, _IDX_BURSTINESS]
    asym = x_u[:, _IDX_SENT_RECV_RATIO]
    return torch.stack([max_s, diff_s, val_spike, is_zero_val, burst, asym], dim=-1)


@torch.no_grad()
def extract_fast_tgae_node_latents_and_scores(
    tgae: Any,
    X_np: np.ndarray,
    node_ids: np.ndarray,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast vectorized extraction of 64-D latent vectors z_i and normalized [0, 1] node anomaly scores
    from a fitted ModularTGAE across arbitrary node subsets.
    """
    assert tgae.net_ is not None
    tgae.net_.eval()
    dev = tgae.device_
    tau = float(tgae.threshold_ if tgae.threshold_ is not None else 1.0)

    unique_nodes = np.unique(node_ids)
    n_u = len(unique_nodes)
    z_all = np.zeros((n_u, tgae.latent_dim), dtype=np.float32)
    s_all = np.zeros(n_u, dtype=np.float32)

    for start in range(0, n_u, batch_size):
        end = min(start + batch_size, n_u)
        idx_chunk = unique_nodes[start:end]
        b = len(idx_chunk)
        x_t = torch.tensor(X_np[idx_chunk], dtype=torch.float32, device=dev)
        batch_dict = {
            "x_tab": x_t[:, :TABULAR_DIM],
            "x_topo": x_t[:, TABULAR_DIM : TABULAR_DIM + TOPO_DIM],
            "x_temp_stat": x_t[:, TABULAR_DIM + TOPO_DIM :],
            "temporal_sequence": torch.zeros((b, 10, 8), dtype=torch.float32, device=dev),
            "edge_index": torch.empty((2, 0), dtype=torch.long, device=dev),
            "edge_attr": torch.empty((0, 5), dtype=torch.float32, device=dev),
            "batch_size": b,
        }
        out = tgae.net_(batch_dict, batch_size=b)
        z_all[start:end] = out["z"].cpu().numpy()

        raw_errs = {
            "feature": out["e_feature"].cpu().numpy().astype(np.float64),
            "graph": out["e_graph"].cpu().numpy().astype(np.float64),
            "temporal": out["e_temporal"].cpu().numpy().astype(np.float64),
        }
        cal_errs = tgae._apply_iqr_calibration(raw_errs)
        raw_score = tgae._combine_calibrated_errors(cal_errs)
        s_all[start:end] = (1.0 / (1.0 + np.exp(-0.35 * (raw_score - tau)))).astype(np.float32)

    pos = np.searchsorted(unique_nodes, node_ids)
    return z_all[pos], s_all[pos]


def extract_labeled_edges(
    graph_data: Any,
    max_negatives: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify directed transaction edges with verified ground-truth labels:
      - y_edge = 1 if sender OR receiver is a verified phishing/malicious address (label == 1)
      - y_edge = 0 if sender OR receiver is a verified normal address and neither is anomalous
    """
    y_node = extract_labels(graph_data)
    ei = graph_data.edge_index.cpu().numpy()
    u_lbl = y_node[ei[0]]
    v_lbl = y_node[ei[1]]

    pos_mask = (u_lbl == LABEL_ANOMALY) | (v_lbl == LABEL_ANOMALY)
    neg_mask = (~pos_mask) & ((u_lbl == LABEL_NORMAL) | (v_lbl == LABEL_NORMAL))

    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(neg_mask)

    if max_negatives is not None and len(neg_idx) > max_negatives:
        rng = np.random.default_rng(seed)
        neg_idx = rng.choice(neg_idx, size=max_negatives, replace=False)

    edge_idx = np.concatenate([pos_idx, neg_idx])
    y_edge = np.concatenate(
        [np.ones(len(pos_idx), dtype=np.int64), np.zeros(len(neg_idx), dtype=np.int64)]
    )
    return edge_idx, y_edge


class DeepTransactionScorer:
    """
    Deep Edge-Level Transaction Anomaly Detector coupled with ModularTGAE.
    Supports both bulk graph-edge evaluation and single real-time transaction scoring (`tx_hash`).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        epochs: int = 40,
        batch_size: int = 1024,
        lr: float = 1.5e-3,
        weight_decay: float = 1e-4,
        recon_weight: float = 0.25,
        random_state: int = RANDOM_SEED,
        device: str = "cpu",
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.recon_weight = float(recon_weight)
        self.random_state = int(random_state)
        self.device_ = torch.device(device)

        self.net_: DeepTxEdgeNet | None = None
        self.threshold_: float = 0.5
        self.val_best_f1_: float = 0.0
        self.is_fitted_: bool = False

    def _prepare_edge_tensors(
        self,
        tgae: Any,
        graph_data: Any,
        edge_indices: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_np = extract_static_features(graph_data)
        ei = graph_data.edge_index.cpu().numpy()[:, edge_indices]
        ea = graph_data.edge_attr.cpu().numpy()[edge_indices]

        endpoints = np.unique(np.concatenate([ei[0], ei[1]]))
        z_ep, s_ep = extract_fast_tgae_node_latents_and_scores(tgae, X_np, endpoints)
        u_pos = np.searchsorted(endpoints, ei[0])
        v_pos = np.searchsorted(endpoints, ei[1])

        z_u = torch.tensor(z_ep[u_pos], dtype=torch.float32, device=self.device_)
        z_v = torch.tensor(z_ep[v_pos], dtype=torch.float32, device=self.device_)
        s_u = torch.tensor(s_ep[u_pos], dtype=torch.float32, device=self.device_)
        s_v = torch.tensor(s_ep[v_pos], dtype=torch.float32, device=self.device_)
        x_u = torch.tensor(X_np[ei[0]], dtype=torch.float32, device=self.device_)
        x_v = torch.tensor(X_np[ei[1]], dtype=torch.float32, device=self.device_)
        ea_t = torch.tensor(ea, dtype=torch.float32, device=self.device_)
        shock = build_shock_context(x_u, x_v, s_u, s_v, ea_t)
        return z_u, z_v, s_u, s_v, ea_t, shock

    def fit(self, tgae: Any, train_graph: Any, val_graph: Any | None = None) -> DeepTransactionScorer:
        torch.manual_seed(self.random_state)
        tr_eidx, y_tr = extract_labeled_edges(train_graph, max_negatives=12_000, seed=self.random_state)
        z_u, z_v, s_u, s_v, ea_t, shock = self._prepare_edge_tensors(tgae, train_graph, tr_eidx)
        y_t = torch.tensor(y_tr, dtype=torch.float32, device=self.device_)

        self.net_ = DeepTxEdgeNet(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n_pos = max(1.0, float((y_tr == 1).sum()))
        n_neg = max(1.0, float((y_tr == 0).sum()))
        pos_weight = min(25.0, math.sqrt(n_neg / n_pos)) if "math" in globals() else float(np.sqrt(n_neg / n_pos))

        n = len(y_tr)
        rng = np.random.default_rng(self.random_state)
        self.net_.train()
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                out = self.net_(z_u[idx], z_v[idx], s_u[idx], s_v[idx], ea_t[idx], shock[idx])
                yb = y_t[idx]

                bce = F.binary_cross_entropy_with_logits(
                    out["logits"],
                    yb,
                    pos_weight=torch.tensor(pos_weight, device=self.device_),
                    reduction="none",
                )
                pt = torch.exp(-F.binary_cross_entropy_with_logits(out["logits"], yb, reduction="none"))
                focal_loss = ((1.0 - pt).pow(2.0) * bce).mean()

                # Contrastive edge reconstruction: minimize on normal edges, push above margin=2.0 on attack edges
                normal_recon = out["recon_err"][yb == 0].mean() if (yb == 0).any() else focal_loss.new_tensor(0.0)
                anom_recon = (
                    F.relu(2.0 - out["recon_err"][yb == 1]).mean()
                    if (yb == 1).any()
                    else focal_loss.new_tensor(0.0)
                )
                loss = focal_loss + self.recon_weight * (normal_recon + anom_recon)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
                opt.step()

        self.is_fitted_ = True
        if val_graph is not None:
            self.calibrate_threshold(tgae, val_graph)
        return self

    @torch.no_grad()
    def predict_edges(
        self,
        tgae: Any,
        graph_data: Any,
        edge_indices: np.ndarray,
    ) -> np.ndarray:
        assert self.is_fitted_ and self.net_ is not None
        self.net_.eval()
        scores = np.zeros(len(edge_indices), dtype=np.float64)
        chunk = 16_384
        for start in range(0, len(edge_indices), chunk):
            end = min(start + chunk, len(edge_indices))
            z_u, z_v, s_u, s_v, ea_t, shock = self._prepare_edge_tensors(
                tgae, graph_data, edge_indices[start:end]
            )
            out = self.net_(z_u, z_v, s_u, s_v, ea_t, shock)
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
            scores[start:end] = probs
        return scores

    def calibrate_threshold(self, tgae: Any, val_graph: Any) -> float:
        va_eidx, y_va = extract_labeled_edges(val_graph, max_negatives=25_000, seed=self.random_state)
        scores = self.predict_edges(tgae, val_graph, va_eidx)
        prec, rec, thresh = precision_recall_curve(y_va, scores)
        if len(thresh) > 0:
            f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
            best_i = int(np.argmax(f1s))
            self.threshold_ = float(np.clip(thresh[best_i], 0.05, 0.98))
            self.val_best_f1_ = float(f1s[best_i])
        return self.threshold_

    def evaluate_edges(
        self,
        tgae: Any,
        graph_data: Any,
        edge_indices: np.ndarray,
        y_true: np.ndarray,
    ) -> dict[str, Any]:
        scores = self.predict_edges(tgae, graph_data, edge_indices)
        preds = (scores >= self.threshold_).astype(np.int64)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        order = np.argsort(-scores)
        p_at_50 = float(y_true[order[: min(50, len(y_true))]].mean())
        p_at_100 = float(y_true[order[: min(100, len(y_true))]].mean())
        return {
            "pr_auc": float(average_precision_score(y_true, scores)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "f1_binary": float(f1_score(y_true, preds, zero_division=0)),
            "precision": float(tp / max(1, tp + fp)),
            "recall": float(tp / max(1, tp + fn)),
            "threshold": float(self.threshold_),
            "num_eval_edges": int(len(y_true)),
            "num_positives": int((y_true == 1).sum()),
            "num_negatives": int((y_true == 0).sum()),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision_at_50": p_at_50,
            "precision_at_100": p_at_100,
        }

    @torch.no_grad()
    def score_single_transaction(
        self,
        z_u: np.ndarray | torch.Tensor,
        z_v: np.ndarray | torch.Tensor,
        x_u_scaled: np.ndarray | torch.Tensor,
        x_v_scaled: np.ndarray | torch.Tensor,
        s_u: float,
        s_v: float,
        value_eth: float,
        gas_price_wei: float,
        tx_count_pair: int = 1,
        age_norm: float = 0.0,
    ) -> dict[str, Any]:
        """Score one live or historical transaction and decompose risk into Sender / Receiver / Dynamics %."""
        assert self.is_fitted_ and self.net_ is not None
        self.net_.eval()

        zu_t = torch.as_tensor(z_u, dtype=torch.float32, device=self.device_).view(1, -1)
        zv_t = torch.as_tensor(z_v, dtype=torch.float32, device=self.device_).view(1, -1)
        xu_t = torch.as_tensor(x_u_scaled, dtype=torch.float32, device=self.device_).view(1, -1)
        xv_t = torch.as_tensor(x_v_scaled, dtype=torch.float32, device=self.device_).view(1, -1)
        su_t = torch.tensor([float(s_u)], dtype=torch.float32, device=self.device_)
        sv_t = torch.tensor([float(s_v)], dtype=torch.float32, device=self.device_)

        val_pos = max(0.0, float(value_eth))
        gas_gwei = max(0.0, float(gas_price_wei) / 1e9)
        ea_t = torch.tensor(
            [
                [
                    float(np.log1p(max(1, int(tx_count_pair)))),
                    float(np.log1p(val_pos * max(1, int(tx_count_pair)))),
                    float(np.log1p(val_pos)),
                    float(np.clip(age_norm, 0.0, 1.0)),
                    float(np.log1p(gas_gwei)),
                ]
            ],
            dtype=torch.float32,
            device=self.device_,
        )
        shock = build_shock_context(xu_t, xv_t, su_t, sv_t, ea_t)
        out = self.net_(zu_t, zv_t, su_t, sv_t, ea_t, shock)

        prob = float(torch.sigmoid(out["logits"][0]).item())
        e_s = max(1e-8, float(out["err_sender"][0].item()) * (0.5 + float(s_u)))
        e_r = max(1e-8, float(out["err_receiver"][0].item()) * (0.5 + float(s_v)))
        e_d = max(1e-8, float(out["err_dynamics"][0].item()))
        tot = e_s + e_r + e_d

        return {
            "tx_anomaly_score": round(prob, 4),
            "threshold": round(float(self.threshold_), 4),
            "is_anomaly": bool(prob >= self.threshold_),
            "component_breakdown": {
                "sender_risk_pct": round(100.0 * e_s / tot, 1),
                "receiver_risk_pct": round(100.0 * e_r / tot, 1),
                "tx_dynamics_pct": round(100.0 * e_d / tot, 1),
            },
            "val_spike_delta": round(float(shock[0, 2].item()), 4),
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        assert self.net_ is not None
        torch.save(
            {
                "params": {
                    "latent_dim": self.latent_dim,
                    "hidden_dim": self.hidden_dim,
                    "dropout": self.dropout,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "recon_weight": self.recon_weight,
                    "random_state": self.random_state,
                },
                "state_dict": {k: v.cpu() for k, v in self.net_.state_dict().items()},
                "threshold_": self.threshold_,
                "val_best_f1_": self.val_best_f1_,
            },
            out,
        )
        return out

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> DeepTransactionScorer:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(**ckpt["params"], device=device)
        obj.net_ = DeepTxEdgeNet(
            latent_dim=obj.latent_dim,
            hidden_dim=obj.hidden_dim,
            dropout=obj.dropout,
        ).to(obj.device_)
        obj.net_.load_state_dict(ckpt["state_dict"])
        obj.net_.eval()
        obj.threshold_ = float(ckpt["threshold_"])
        obj.val_best_f1_ = float(ckpt.get("val_best_f1_", 0.0))
        obj.is_fitted_ = True
        return obj
