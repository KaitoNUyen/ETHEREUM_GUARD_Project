"""
Independent Deep Learning & GNN Baseline Detectors for Ethereum Anomaly Detection:
  1. DOMINANTDetector (Ding et al., SDM 2019): Deep Graph Convolutional Autoencoder
     jointly reconstructing node attributes X and graph structure A.
  2. GATDetector (Velickovic et al., ICLR 2018): Multi-Head Graph Attention Network
     trained with class-balanced Focal Loss.

Implemented in pure PyTorch (zero C++ torch-scatter / pyg-lib dependency) so they run
reliably on both CPU and CUDA across million-node Ethereum transaction graphs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseDetector
from src.models.data_utils import extract_labels, extract_static_features, get_node_address
from src.schemas import ALL_FEATURE_COLS, LABEL_ANOMALY, LABEL_NORMAL, RANDOM_SEED


def _sample_local_subgraph(
    edge_index: torch.Tensor,
    seed_nodes: np.ndarray,
    num_nodes: int,
    max_edges_per_batch: int = 40_000,
) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Extract a compact 1-hop induced/incident subgraph around ``seed_nodes`` for scalable mini-batch GNN training."""
    seed_arr = np.unique(np.asarray(seed_nodes, dtype=np.int64))
    if edge_index is None or edge_index.numel() == 0:
        local_pos = np.searchsorted(seed_arr, seed_nodes)
        return seed_arr, torch.empty((2, 0), dtype=torch.long), local_pos

    src_np = edge_index[0].cpu().numpy()
    dst_np = edge_index[1].cpu().numpy()

    mask = np.isin(src_np, seed_arr) | np.isin(dst_np, seed_arr)
    e_idx = np.flatnonzero(mask)
    if len(e_idx) > max_edges_per_batch:
        rng = np.random.default_rng(RANDOM_SEED)
        e_idx = rng.choice(e_idx, size=max_edges_per_batch, replace=False)

    sub_src = src_np[e_idx]
    sub_dst = dst_np[e_idx]
    all_nodes = np.unique(np.concatenate([seed_arr, sub_src, sub_dst]))

    local_src = np.searchsorted(all_nodes, sub_src)
    local_dst = np.searchsorted(all_nodes, sub_dst)
    sub_edge_index = torch.tensor(np.stack([local_src, local_dst], axis=0), dtype=torch.long)
    seed_local_pos = np.searchsorted(all_nodes, seed_nodes)
    return all_nodes, sub_edge_index, seed_local_pos


class PureTorchGCNConv(nn.Module):
    """Symmetric normalized Graph Convolution: D^{-1/2} (A + I) D^{-1/2} X W."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = self.lin(x)
        self_loops = torch.arange(n, device=x.device, dtype=torch.long)
        if edge_index is not None and edge_index.numel() > 0:
            src = torch.cat([edge_index[0].long(), self_loops], dim=0)
            dst = torch.cat([edge_index[1].long(), self_loops], dim=0)
        else:
            src = self_loops
            dst = self_loops

        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        out = torch.zeros_like(h)
        out.index_add_(0, dst, h[src] * norm.unsqueeze(-1))
        return out


class PureTorchGATConv(nn.Module):
    """Multi-Head Graph Attention layer (Velickovic et al., ICLR 2018) in pure PyTorch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.out_channels = int(out_channels)
        self.concat = bool(concat)
        self.dropout = float(dropout)

        self.lin = nn.Linear(in_channels, self.heads * self.out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, self.heads, self.out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, self.heads, self.out_channels))
        out_dim = self.heads * self.out_channels if self.concat else self.out_channels
        self.res = nn.Linear(in_channels, out_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.res.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h, c = self.heads, self.out_channels
        proj = self.lin(x).view(n, h, c)

        self_loops = torch.arange(n, device=x.device, dtype=torch.long)
        if edge_index is not None and edge_index.numel() > 0:
            src = torch.cat([edge_index[0].long(), self_loops], dim=0)
            dst = torch.cat([edge_index[1].long(), self_loops], dim=0)
        else:
            src = self_loops
            dst = self_loops

        alpha_s = (proj * self.att_src).sum(dim=-1)  # (N, H)
        alpha_d = (proj * self.att_dst).sum(dim=-1)  # (N, H)
        e = F.leaky_relu(alpha_s[src] + alpha_d[dst], negative_slope=0.2)

        max_dst = e.new_full((n, h), -1e30)
        max_dst.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, h), e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - max_dst[dst])
        denom = e.new_zeros((n, h))
        denom.index_add_(0, dst, exp_e)
        attn = exp_e / denom[dst].clamp_min(1e-16)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = x.new_zeros((n, h, c))
        out.index_add_(0, dst, proj[src] * attn.unsqueeze(-1))
        out_flat = out.view(n, h * c) if self.concat else out.mean(dim=1)
        return out_flat + self.res(x)


class _DOMINANTNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 96, latent_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.gcn1 = PureTorchGCNConv(in_dim, hidden_dim)
        self.gcn2 = PureTorchGCNConv(hidden_dim, latent_dim)
        self.dropout = float(dropout)
        self.attr_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gcn1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.gcn2(h, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, edge_index)
        x_hat = self.attr_decoder(z)
        return z, x_hat


class DOMINANTDetector(BaseDetector):
    """
    Deep Graph Autoencoder baseline (DOMINANT — Ding et al., SDM 2019).
    Trained strictly on Normal + unlabeled non-anomalous neighborhood to reconstruct X and A.
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        latent_dim: int = 64,
        struct_weight: float = 0.35,
        epochs: int = 35,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        random_state: int = RANDOM_SEED,
        device: str = "cpu",
        feature_names: tuple[str, ...] | list[str] = ALL_FEATURE_COLS,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.struct_weight = float(struct_weight)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.random_state = int(random_state)
        self.device_ = torch.device(device)
        self.net_: _DOMINANTNet | None = None

    def fit(self, train_data: Any, val_data: Any | None = None) -> DOMINANTDetector:
        torch.manual_seed(self.random_state)
        X_np = extract_static_features(train_data)
        y_np = extract_labels(train_data)
        edge_index = getattr(train_data, "edge_index", torch.empty((2, 0), dtype=torch.long))

        normal_idx = np.flatnonzero(y_np == LABEL_NORMAL)
        rng = np.random.default_rng(self.random_state)
        unlabeled_idx = np.flatnonzero(y_np < 0)
        bg_sample = (
            rng.choice(unlabeled_idx, size=min(2048, len(unlabeled_idx)), replace=False)
            if len(unlabeled_idx) > 0
            else np.array([], dtype=np.int64)
        )
        train_seeds = np.unique(np.concatenate([normal_idx, bg_sample]))

        sub_nodes, sub_ei, seed_pos = _sample_local_subgraph(edge_index, train_seeds, len(X_np))
        normal_pos = np.searchsorted(sub_nodes, normal_idx)

        x_sub = torch.tensor(X_np[sub_nodes], dtype=torch.float32, device=self.device_)
        sub_ei = sub_ei.to(self.device_)
        normal_pos_t = torch.tensor(normal_pos, dtype=torch.long, device=self.device_)

        self.net_ = _DOMINANTNet(
            in_dim=X_np.shape[1],
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.net_.train()
        n_sub = x_sub.shape[0]
        for _ in range(self.epochs):
            opt.zero_grad()
            z, x_hat = self.net_(x_sub, sub_ei)
            loss_attr = F.huber_loss(x_hat[normal_pos_t], x_sub[normal_pos_t], delta=1.5)

            if sub_ei.numel() > 0:
                pos_logits = (z[sub_ei[0]] * z[sub_ei[1]]).sum(dim=-1)
                neg_src = torch.randint(0, n_sub, (sub_ei.shape[1],), device=self.device_)
                neg_dst = torch.randint(0, n_sub, (sub_ei.shape[1],), device=self.device_)
                neg_logits = (z[neg_src] * z[neg_dst]).sum(dim=-1)
                loss_struct = 0.5 * (
                    F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                    + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
                )
            else:
                loss_struct = loss_attr.new_tensor(0.0)

            loss = (1.0 - self.struct_weight) * loss_attr + self.struct_weight * loss_struct
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
            opt.step()

        self.is_fitted_ = True
        if val_data is not None:
            self.calibrate_threshold(val_data)
        return self

    @torch.no_grad()
    def predict_score(self, data: Any, node_indices: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted_ or self.net_ is None:
            raise RuntimeError("DOMINANTDetector is not fitted yet.")
        self.net_.eval()
        X_np = extract_static_features(data)
        edge_index = getattr(data, "edge_index", torch.empty((2, 0), dtype=torch.long))
        target_idx = (
            np.asarray(node_indices, dtype=np.int64)
            if node_indices is not None
            else np.arange(len(X_np), dtype=np.int64)
        )

        sub_nodes, sub_ei, local_pos = _sample_local_subgraph(edge_index, target_idx, len(X_np))
        x_sub = torch.tensor(X_np[sub_nodes], dtype=torch.float32, device=self.device_)
        sub_ei = sub_ei.to(self.device_)

        z, x_hat = self.net_(x_sub, sub_ei)
        attr_err = F.smooth_l1_loss(x_hat, x_sub, reduction="none").mean(dim=-1)

        # Degree-normalized structural reconstruction discrepancy per node
        struct_err = torch.zeros(x_sub.shape[0], device=self.device_)
        if sub_ei.numel() > 0:
            edge_prob = torch.sigmoid((z[sub_ei[0]] * z[sub_ei[1]]).sum(dim=-1))
            edge_loss = 1.0 - edge_prob
            deg = torch.zeros(x_sub.shape[0], device=self.device_)
            struct_err.index_add_(0, sub_ei[1], edge_loss)
            deg.index_add_(0, sub_ei[1], torch.ones_like(edge_loss))
            struct_err = struct_err / deg.clamp_min(1.0)

        composite = (1.0 - self.struct_weight) * attr_err + self.struct_weight * struct_err
        return composite[local_pos].cpu().numpy().astype(np.float64)

    def explain(self, data: Any, node_idx: int) -> dict[str, Any]:
        score = float(self.predict_score(data, np.array([node_idx]))[0])
        return {
            "model": "DOMINANTDetector",
            "node_idx": int(node_idx),
            "address": get_node_address(data, node_idx),
            "score": score,
            "threshold": self.threshold_,
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        assert self.net_ is not None
        torch.save(
            {
                "params": {
                    "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim,
                    "struct_weight": self.struct_weight,
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "dropout": self.dropout,
                    "random_state": self.random_state,
                    "feature_names": self.feature_names_,
                },
                "state_dict": {k: v.cpu() for k, v in self.net_.state_dict().items()},
                "threshold_": self.threshold_,
                "val_best_f1_": self.val_best_f1_,
            },
            out,
        )
        return out

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> DOMINANTDetector:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(**ckpt["params"], device=device)
        obj.net_ = _DOMINANTNet(
            in_dim=len(obj.feature_names_),
            hidden_dim=obj.hidden_dim,
            latent_dim=obj.latent_dim,
            dropout=obj.dropout,
        ).to(obj.device_)
        obj.net_.load_state_dict(ckpt["state_dict"])
        obj.net_.eval()
        obj.threshold_ = ckpt["threshold_"]
        obj.val_best_f1_ = ckpt.get("val_best_f1_")
        obj.is_fitted_ = True
        return obj


class _GATClassifierNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, heads: int = 4, dropout: float = 0.25) -> None:
        super().__init__()
        self.gat1 = PureTorchGATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.gat2 = PureTorchGATConv(hidden_dim * heads, hidden_dim * 2, heads=2, concat=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.norm1(self.gat1(x, edge_index)))
        h = F.elu(self.norm2(self.gat2(h, edge_index)))
        return self.head(h).squeeze(-1)


class GATDetector(BaseDetector):
    """
    Supervised Multi-Head Graph Attention Network (GAT — Velickovic et al., ICLR 2018)
    trained with Focal Loss on labeled Ethereum transaction graph nodes.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        heads: int = 4,
        epochs: int = 45,
        lr: float = 2e-3,
        weight_decay: float = 5e-4,
        dropout: float = 0.25,
        gamma_focal: float = 2.0,
        random_state: int = RANDOM_SEED,
        device: str = "cpu",
        feature_names: tuple[str, ...] | list[str] = ALL_FEATURE_COLS,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.gamma_focal = float(gamma_focal)
        self.random_state = int(random_state)
        self.device_ = torch.device(device)
        self.net_: _GATClassifierNet | None = None

    def fit(self, train_data: Any, val_data: Any | None = None) -> GATDetector:
        torch.manual_seed(self.random_state)
        X_np = extract_static_features(train_data)
        y_np = extract_labels(train_data)
        edge_index = getattr(train_data, "edge_index", torch.empty((2, 0), dtype=torch.long))

        labeled_idx = np.flatnonzero((y_np == LABEL_NORMAL) | (y_np == LABEL_ANOMALY))
        sub_nodes, sub_ei, labeled_pos = _sample_local_subgraph(edge_index, labeled_idx, len(X_np))

        x_sub = torch.tensor(X_np[sub_nodes], dtype=torch.float32, device=self.device_)
        sub_ei = sub_ei.to(self.device_)
        pos_t = torch.tensor(labeled_pos, dtype=torch.long, device=self.device_)
        y_t = torch.tensor(y_np[labeled_idx], dtype=torch.float32, device=self.device_)

        self.net_ = _GATClassifierNet(
            in_dim=X_np.shape[1],
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            dropout=self.dropout,
        ).to(self.device_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self.net_(x_sub, sub_ei)[pos_t]
            bce = F.binary_cross_entropy_with_logits(logits, y_t, reduction="none")
            pt = torch.exp(-bce)
            loss = ((1.0 - pt).pow(self.gamma_focal) * bce).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
            opt.step()

        self.is_fitted_ = True
        if val_data is not None:
            self.calibrate_threshold(val_data)
        return self

    @torch.no_grad()
    def predict_score(self, data: Any, node_indices: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted_ or self.net_ is None:
            raise RuntimeError("GATDetector is not fitted yet.")
        self.net_.eval()
        X_np = extract_static_features(data)
        edge_index = getattr(data, "edge_index", torch.empty((2, 0), dtype=torch.long))
        target_idx = (
            np.asarray(node_indices, dtype=np.int64)
            if node_indices is not None
            else np.arange(len(X_np), dtype=np.int64)
        )
        sub_nodes, sub_ei, local_pos = _sample_local_subgraph(edge_index, target_idx, len(X_np))
        x_sub = torch.tensor(X_np[sub_nodes], dtype=torch.float32, device=self.device_)
        probs = torch.sigmoid(self.net_(x_sub, sub_ei.to(self.device_)))
        return probs[local_pos].cpu().numpy().astype(np.float64)

    def explain(self, data: Any, node_idx: int) -> dict[str, Any]:
        score = float(self.predict_score(data, np.array([node_idx]))[0])
        return {
            "model": "GATDetector",
            "node_idx": int(node_idx),
            "address": get_node_address(data, node_idx),
            "score": score,
            "threshold": self.threshold_,
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        assert self.net_ is not None
        torch.save(
            {
                "params": {
                    "hidden_dim": self.hidden_dim,
                    "heads": self.heads,
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "dropout": self.dropout,
                    "gamma_focal": self.gamma_focal,
                    "random_state": self.random_state,
                    "feature_names": self.feature_names_,
                },
                "state_dict": {k: v.cpu() for k, v in self.net_.state_dict().items()},
                "threshold_": self.threshold_,
                "val_best_f1_": self.val_best_f1_,
            },
            out,
        )
        return out

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> GATDetector:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(**ckpt["params"], device=device)
        obj.net_ = _GATClassifierNet(
            in_dim=len(obj.feature_names_),
            hidden_dim=obj.hidden_dim,
            heads=obj.heads,
            dropout=obj.dropout,
        ).to(obj.device_)
        obj.net_.load_state_dict(ckpt["state_dict"])
        obj.net_.eval()
        obj.threshold_ = ckpt["threshold_"]
        obj.val_best_f1_ = ckpt.get("val_best_f1_")
        obj.is_fitted_ = True
        return obj
