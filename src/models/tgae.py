"""Modular Temporal-Graph Autoencoder (TGAE) for Ethereum Guard (Tasks TV2-04, TV2-05, TV2-06)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.sampler import NeighborSampler

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    PyGData = None  # type: ignore[assignment,misc]
    NeighborLoader = None  # type: ignore[assignment,misc]
    NeighborSampler = None  # type: ignore[assignment,misc]

from src.models.base import BaseDetector, find_optimal_f1_threshold
from src.models.data_utils import (
    GraphSnapshotFallback,
    extract_labels,
    get_eval_mask_and_labels,
    get_node_address,
    load_config,
)
from src.models.encoders import (
    TABULAR_DIM,
    TEMPORAL_STAT_DIM,
    TOPO_DIM,
    EdgeTransformerEncoder,
    GraphSAGEEncoder,
    TabularMLP,
    TemporalAttentionGRU,
    _install_neighbor_sampler_fallback,
)
from src.schemas import (
    ALL_FEATURE_COLS,
    EDGE_FEATURE_DIM,
    GRAPH_TOPO_COLS,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    RANDOM_SEED,
    TABULAR_TX_COLS,
    TEMPORAL_FEATURE_DIM,
    TEMPORAL_SEQUENCE_COLS,
    TEMPORAL_STAT_COLS,
    TEMPORAL_WINDOWS,
)

_install_neighbor_sampler_fallback()


def normalize_branch_weights(
    alpha: float,
    beta: float,
    gamma: float,
    use_tabular: bool,
    use_graph: bool,
    use_temporal: bool,
) -> tuple[float, float, float]:
    """Zero out disabled branches and normalize active branch weights to sum to 1.0."""
    if not (use_tabular or use_graph or use_temporal):
        raise ValueError("At least one branch (use_tabular, use_graph, use_temporal) must be True.")

    w_tab = max(0.0, float(alpha)) if use_tabular else 0.0
    w_graph = max(0.0, float(beta)) if use_graph else 0.0
    w_temp = max(0.0, float(gamma)) if use_temporal else 0.0

    total = w_tab + w_graph + w_temp
    if total <= 1e-12:
        n_active = int(use_tabular) + int(use_graph) + int(use_temporal)
        w_tab = (1.0 / n_active) if use_tabular else 0.0
        w_graph = (1.0 / n_active) if use_graph else 0.0
        w_temp = (1.0 / n_active) if use_temporal else 0.0
        return w_tab, w_graph, w_temp

    return w_tab / total, w_graph / total, w_temp / total


def compute_mean_neighbor_topo(
    x_topo: torch.Tensor,
    edge_index: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    """Compute mean neighbor topological features ``(B, 6)`` on the current batch edges via ``index_add_``."""
    mean_neigh = x_topo.new_zeros((batch_size, x_topo.shape[1]))
    if edge_index is None or edge_index.numel() == 0:
        return mean_neigh

    deg = x_topo.new_zeros((batch_size, 1))
    src, dst = edge_index[0], edge_index[1]
    num_nodes = x_topo.shape[0]

    in_mask = (dst >= 0) & (dst < batch_size) & (src >= 0) & (src < num_nodes)
    if in_mask.any():
        dst_in = dst[in_mask]
        mean_neigh.index_add_(0, dst_in, x_topo[src[in_mask]])
        deg.index_add_(0, dst_in, deg.new_ones((dst_in.shape[0], 1)))

    out_mask = (src >= 0) & (src < batch_size) & (dst >= 0) & (dst < num_nodes)
    if out_mask.any():
        src_out = src[out_mask]
        mean_neigh.index_add_(0, src_out, x_topo[dst[out_mask]])
        deg.index_add_(0, src_out, deg.new_ones((src_out.shape[0], 1)))

    return mean_neigh / deg.clamp_min(1.0)


def _clean(x: torch.Tensor) -> torch.Tensor:
    """Cast to float32 and replace NaN/Inf. No log transform: TV1 snapshots are already log10 + RobustScaler."""
    return torch.nan_to_num(x.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _extract_modality_tensors(
    data: Any,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Extract ``(x_tab, x_topo, x_temp_stat, temporal_sequence, edge_index, edge_attr)`` from Data or dict."""
    getter = data.get if isinstance(data, dict) else lambda k, d=None: getattr(data, k, d)

    x_all = getter("x", None)
    # Modalities of disabled branches may be absent (minimal Data); only fall back to ``x`` slices.
    x_tab = getter("x_tab", None)
    if x_tab is None and x_all is not None:
        x_tab = x_all[:, :TABULAR_DIM]

    x_topo = getter("x_topo", None)
    if x_topo is None and x_all is not None:
        x_topo = x_all[:, TABULAR_DIM : TABULAR_DIM + TOPO_DIM]

    x_temp_stat = getter("x_temp_stat", None)
    if x_temp_stat is None and x_all is not None:
        x_temp_stat = x_all[
            :, TABULAR_DIM + TOPO_DIM : TABULAR_DIM + TOPO_DIM + TEMPORAL_STAT_DIM
        ]

    temporal_seq = getter("temporal_sequence", None)
    if temporal_seq is None:
        temporal_seq = getter("temporal_seq", None)

    edge_index = getter("edge_index", None)
    edge_attr = getter("edge_attr", None)

    return x_tab, x_topo, x_temp_stat, temporal_seq, edge_index, edge_attr


def _count_nodes(*tensors: torch.Tensor | None) -> int:
    for t in tensors:
        if t is not None:
            return int(t.shape[0])
    raise ValueError("Data contains none of x_tab / x_topo / x_temp_stat / temporal_sequence.")


class ModularTGAENet(nn.Module):
    """Tri-Modal Masked Gated Autoencoder with Multi-Head Reconstruction and Auxiliary Classifier."""

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.2,
        use_tabular: bool = True,
        use_graph: bool = True,
        use_temporal: bool = True,
        edge_dim: int = EDGE_FEATURE_DIM,
        heads: int = 4,
        seq_use_log1p: bool = False,
    ) -> None:
        super().__init__()
        if not (use_tabular or use_graph or use_temporal):
            raise ValueError(
                "At least one modality (use_tabular, use_graph, use_temporal) must be enabled."
            )

        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.use_tabular = bool(use_tabular)
        self.use_graph = bool(use_graph)
        self.use_temporal = bool(use_temporal)
        self.edge_dim = int(edge_dim)
        self.heads = int(heads)
        self.seq_use_log1p = bool(seq_use_log1p)

        self.register_buffer(
            "modality_mask",
            torch.tensor(
                [float(self.use_tabular), float(self.use_graph), float(self.use_temporal)],
                dtype=torch.float32,
            ),
        )

        # Z-score buffers on the TV1-preprocessed features (fitted on normal TRAIN nodes only, frozen afterwards).
        self.register_buffer("tab_mean", torch.zeros(TABULAR_DIM, dtype=torch.float32))
        self.register_buffer("tab_std", torch.ones(TABULAR_DIM, dtype=torch.float32))
        self.register_buffer("topo_mean", torch.zeros(TOPO_DIM, dtype=torch.float32))
        self.register_buffer("topo_std", torch.ones(TOPO_DIM, dtype=torch.float32))
        self.register_buffer("temp_stat_mean", torch.zeros(TEMPORAL_STAT_DIM, dtype=torch.float32))
        self.register_buffer("temp_stat_std", torch.ones(TEMPORAL_STAT_DIM, dtype=torch.float32))
        self.register_buffer(
            "seq_mean", torch.zeros((1, 1, TEMPORAL_FEATURE_DIM), dtype=torch.float32)
        )
        self.register_buffer(
            "seq_std", torch.ones((1, 1, TEMPORAL_FEATURE_DIM), dtype=torch.float32)
        )

        # 1. Modality Encoders & Learnable Gates
        if self.use_tabular:
            self.tab_encoder: TabularMLP | None = TabularMLP(
                in_dim=TABULAR_DIM,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )
            self.gate_tab: nn.Module | None = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.tab_encoder = None
            self.gate_tab = None

        if self.use_graph:
            self.graph_encoder: GraphSAGEEncoder | None = GraphSAGEEncoder(
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                use_tabular=self.use_tabular,
                use_graph=True,
                edge_dim=self.edge_dim,
                heads=self.heads,
            )
            self.gate_graph: nn.Module | None = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.graph_encoder = None
            self.gate_graph = None

        if self.use_temporal:
            self.temp_encoder: TemporalAttentionGRU | None = TemporalAttentionGRU(
                seq_dim=TEMPORAL_FEATURE_DIM,
                stat_dim=TEMPORAL_STAT_DIM,
                num_windows=TEMPORAL_WINDOWS,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )
            self.gate_temp: nn.Module | None = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.temp_encoder = None
            self.gate_temp = None

        # 2. Masked Gated Fusion -> z (B, latent_dim)
        self.mlp_fuse = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.latent_norm = nn.LayerNorm(self.latent_dim)

        # 3. Multi-Head Decoders (instantiated only for active branches)
        if self.use_tabular:
            self.feature_decoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, TABULAR_DIM),
            )
        else:
            self.feature_decoder = None

        if self.use_graph:
            self.graph_topo_decoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, TOPO_DIM),
            )
            self.graph_neigh_decoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, TOPO_DIM),
            )
        else:
            self.graph_topo_decoder = None
            self.graph_neigh_decoder = None

        if self.use_temporal:
            self.temp_seq_decoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, TEMPORAL_WINDOWS * TEMPORAL_FEATURE_DIM),
            )
            self.temp_stat_decoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, TEMPORAL_STAT_DIM),
            )
        else:
            self.temp_seq_decoder = None
            self.temp_stat_decoder = None

        # 4. Auxiliary Supervised Classifier Head
        self.aux_classifier = nn.Linear(self.latent_dim, 1)

    @torch.no_grad()
    def fit_normalizer(
        self,
        x_tab: torch.Tensor | None = None,
        x_topo: torch.Tensor | None = None,
        x_temp_stat: torch.Tensor | None = None,
        temporal_sequence: torch.Tensor | None = None,
    ) -> None:
        """Fit z-score buffers on Normal (``y == 0``) TRAIN nodes (no extra log transform)."""
        dev = self.modality_mask.device
        if x_tab is not None and x_tab.numel() > 0:
            t = _clean(x_tab)
            self.tab_mean.copy_(t.mean(dim=0).to(dev))
            self.tab_std.copy_(t.std(dim=0, unbiased=False).clamp_min(1e-4).to(dev))
        if x_topo is not None and x_topo.numel() > 0:
            t = _clean(x_topo)
            self.topo_mean.copy_(t.mean(dim=0).to(dev))
            self.topo_std.copy_(t.std(dim=0, unbiased=False).clamp_min(1e-4).to(dev))
        if x_temp_stat is not None and x_temp_stat.numel() > 0:
            t = _clean(x_temp_stat)
            self.temp_stat_mean.copy_(t.mean(dim=0).to(dev))
            self.temp_stat_std.copy_(t.std(dim=0, unbiased=False).clamp_min(1e-4).to(dev))
        if temporal_sequence is not None and temporal_sequence.numel() > 0:
            t = _clean(temporal_sequence)
            if self.seq_use_log1p:
                t = torch.log1p(t.clamp_min(0.0))
            self.seq_mean.copy_(t.mean(dim=(0, 1), keepdim=True).to(dev))
            raw_std = t.std(dim=(0, 1), unbiased=False, keepdim=True)
            # Prevent 10,000x explosion on zero-variance sparse channels (set scale=1.0 when flat)
            safe_std = torch.where(raw_std > 1e-3, raw_std, torch.ones_like(raw_std))
            self.seq_std.copy_(safe_std.to(dev))

    def _norm_tab(self, x_tab: torch.Tensor) -> torch.Tensor:
        return torch.clamp((_clean(x_tab) - self.tab_mean) / self.tab_std, -20.0, 20.0)

    def _norm_topo(self, x_topo: torch.Tensor) -> torch.Tensor:
        return torch.clamp((_clean(x_topo) - self.topo_mean) / self.topo_std, -20.0, 20.0)

    def _norm_temp_stat(self, x_temp_stat: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (_clean(x_temp_stat) - self.temp_stat_mean) / self.temp_stat_std, -20.0, 20.0
        )

    def _norm_seq(self, seq: torch.Tensor) -> torch.Tensor:
        t = _clean(seq)
        if self.seq_use_log1p:
            t = torch.log1p(t.clamp_min(0.0))
        return torch.clamp((t - self.seq_mean) / self.seq_std, -20.0, 20.0)

    def forward(
        self,
        data_or_x_tab: Any = None,
        x_topo: torch.Tensor | None = None,
        x_temp_stat: torch.Tensor | None = None,
        temporal_sequence: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Run Tri-Modal Masked Gated Fusion and Multi-Head Reconstruction."""
        if not isinstance(data_or_x_tab, torch.Tensor) and data_or_x_tab is not None:
            if batch_size is None:
                batch_size = (
                    data_or_x_tab.get("batch_size", None)
                    if isinstance(data_or_x_tab, dict)
                    else getattr(data_or_x_tab, "batch_size", None)
                )
            x_tab_raw, x_topo_raw, x_temp_stat_raw, seq_raw, e_idx, e_attr = (
                _extract_modality_tensors(data_or_x_tab)
            )
            if x_topo is None:
                x_topo = x_topo_raw
            if x_temp_stat is None:
                x_temp_stat = x_temp_stat_raw
            if temporal_sequence is None:
                temporal_sequence = seq_raw
            if edge_index is None:
                edge_index = e_idx
            if edge_attr is None:
                edge_attr = e_attr
            x_tab = x_tab_raw
        else:
            x_tab = data_or_x_tab

        # Determine target batch_size B and device
        ref_tensor: torch.Tensor | None = None
        for candidate in (x_tab, x_topo, x_temp_stat, temporal_sequence):
            if candidate is not None:
                ref_tensor = candidate
                break
        if ref_tensor is None:
            raise ValueError("At least one input feature tensor must be provided to ModularTGAENet.")

        device = ref_tensor.device
        B = int(batch_size) if batch_size is not None else int(ref_tensor.shape[0])

        x_tab_norm: torch.Tensor | None = None
        x_tab_target: torch.Tensor | None = None
        # x_tab is only consumed when the tabular branch is on (Abl-B graph encoder sees x_topo only).
        if self.use_tabular and x_tab is not None:
            x_tab_norm = self._norm_tab(x_tab)
            x_tab_target = x_tab_norm[:B]

        x_topo_norm: torch.Tensor | None = None
        x_topo_target: torch.Tensor | None = None
        if self.use_graph:
            if x_topo is None:
                raise ValueError("use_graph=True requires x_topo tensor.")
            x_topo_norm = self._norm_topo(x_topo)
            x_topo_target = x_topo_norm[:B]

        x_temp_stat_target: torch.Tensor | None = None
        seq_target: torch.Tensor | None = None
        if self.use_temporal:
            if x_temp_stat is None or temporal_sequence is None:
                raise ValueError(
                    "use_temporal=True requires both x_temp_stat and temporal_sequence tensors."
                )
            x_temp_stat_target = self._norm_temp_stat(x_temp_stat[:B])
            seq_target = self._norm_seq(temporal_sequence[:B])

        # --- 1. Encoders & Masked Gated Fusion ---
        m_tab = self.modality_mask[0]
        m_graph = self.modality_mask[1]
        m_temp = self.modality_mask[2]

        if self.use_tabular and self.tab_encoder is not None and self.gate_tab is not None:
            if x_tab_target is None:
                raise ValueError("use_tabular=True requires x_tab tensor.")
            h_tab = self.tab_encoder(x_tab_target)
            g_tab = self.gate_tab(h_tab)
            gated_tab = m_tab * g_tab * h_tab
        else:
            h_tab = torch.zeros((B, self.hidden_dim), device=device, dtype=torch.float32)
            g_tab = torch.zeros((B, self.hidden_dim), device=device, dtype=torch.float32)
            gated_tab = h_tab

        if self.use_graph and self.graph_encoder is not None and self.gate_graph is not None:
            assert x_topo_norm is not None
            edge_attr_clean = (
                torch.nan_to_num(
                    edge_attr.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
                )
                if edge_attr is not None
                else None
            )
            if self.use_tabular:
                if x_tab_norm is None:
                    raise ValueError("GraphSAGEEncoder with use_tabular=True requires x_tab.")
                h_graph = self.graph_encoder(
                    x_topo_norm,
                    edge_index=edge_index,
                    edge_attr=edge_attr_clean,
                    x_tab=x_tab_norm,
                    batch_size=B,
                )
            else:
                h_graph = self.graph_encoder(
                    x_topo_norm,
                    edge_index=edge_index,
                    edge_attr=edge_attr_clean,
                    batch_size=B,
                )
            g_graph = self.gate_graph(h_graph)
            gated_graph = m_graph * g_graph * h_graph
        else:
            h_graph = torch.zeros((B, self.hidden_dim), device=device, dtype=torch.float32)
            g_graph = torch.zeros((B, self.hidden_dim), device=device, dtype=torch.float32)
            gated_graph = h_graph

        if self.use_temporal and self.temp_encoder is not None and self.gate_temp is not None:
            assert seq_target is not None and x_temp_stat_target is not None
            h_temp = self.temp_encoder(seq_target, x_temp_stat=x_temp_stat_target, batch_size=B)
            g_temp = self.gate_temp(h_temp)
            gated_temp = m_temp * g_temp * h_temp
        else:
            h_temp = torch.zeros((B, self.hidden_dim), device=device, dtype=torch.float32)
            g_temp = torch.zeros((B, self.hidden_dim), device=device, dtype=torch.float32)
            gated_temp = h_temp

        fused_input = torch.cat([gated_tab, gated_graph, gated_temp], dim=-1)
        z = self.latent_norm(self.mlp_fuse(fused_input))
        aux_logits = self.aux_classifier(z).squeeze(-1)

        # --- 2. Multi-Head Decoders (active branches only) ---
        x_tab_hat: torch.Tensor | None = None
        err_tab_feat: torch.Tensor | None = None
        if self.use_tabular and self.feature_decoder is not None:
            assert x_tab_target is not None
            x_tab_hat = self.feature_decoder(z)
            err_tab_feat = F.huber_loss(x_tab_hat, x_tab_target, reduction="none")
            e_feature = err_tab_feat.mean(dim=-1)
        else:
            e_feature = z.new_zeros(B)

        x_topo_hat: torch.Tensor | None = None
        mean_neighbor_topo_hat: torch.Tensor | None = None
        mean_neighbor_topo_target: torch.Tensor | None = None
        err_topo_feat: torch.Tensor | None = None
        err_neigh_feat: torch.Tensor | None = None
        if (
            self.use_graph
            and self.graph_topo_decoder is not None
            and self.graph_neigh_decoder is not None
        ):
            assert x_topo_norm is not None and x_topo_target is not None
            x_topo_hat = self.graph_topo_decoder(z)
            mean_neighbor_topo_hat = self.graph_neigh_decoder(z)
            mean_neighbor_topo_target = compute_mean_neighbor_topo(
                x_topo_norm, edge_index=edge_index, batch_size=B
            )
            err_topo_feat = F.mse_loss(x_topo_hat, x_topo_target, reduction="none")
            err_neigh_feat = F.mse_loss(
                mean_neighbor_topo_hat, mean_neighbor_topo_target, reduction="none"
            )
            e_graph = 0.5 * (err_topo_feat.mean(dim=-1) + err_neigh_feat.mean(dim=-1))
        else:
            e_graph = z.new_zeros(B)

        s_hat: torch.Tensor | None = None
        x_temp_stat_hat: torch.Tensor | None = None
        err_temp_stat_feat: torch.Tensor | None = None
        err_seq_feat: torch.Tensor | None = None
        if (
            self.use_temporal
            and self.temp_seq_decoder is not None
            and self.temp_stat_decoder is not None
        ):
            assert seq_target is not None and x_temp_stat_target is not None
            s_hat = self.temp_seq_decoder(z).view(B, TEMPORAL_WINDOWS, TEMPORAL_FEATURE_DIM)
            x_temp_stat_hat = self.temp_stat_decoder(z)
            err_seq_elem = F.mse_loss(s_hat, seq_target, reduction="none")
            err_seq_feat = err_seq_elem.mean(dim=1)
            err_temp_stat_feat = F.huber_loss(
                x_temp_stat_hat, x_temp_stat_target, reduction="none"
            )
            e_temporal = err_seq_elem.mean(dim=(1, 2)) + err_temp_stat_feat.mean(dim=-1)
        else:
            e_temporal = z.new_zeros(B)

        return {
            "z": z,
            "aux_logits": aux_logits,
            "e_feature": e_feature,
            "e_graph": e_graph,
            "e_temporal": e_temporal,
            "x_tab_hat": x_tab_hat,
            "x_tab_target": x_tab_target,
            "x_topo_hat": x_topo_hat,
            "x_topo_target": x_topo_target,
            "mean_neighbor_topo_hat": mean_neighbor_topo_hat,
            "mean_neighbor_topo": mean_neighbor_topo_target,
            "s_hat": s_hat,
            "seq_target": seq_target,
            "x_temp_stat_hat": x_temp_stat_hat,
            "x_temp_stat_target": x_temp_stat_target,
            "err_tab_feat": err_tab_feat,
            "err_topo_feat": err_topo_feat,
            "err_neigh_feat": err_neigh_feat,
            "err_temp_stat_feat": err_temp_stat_feat,
            "err_seq_feat": err_seq_feat,
            "gates": (g_tab, g_graph, g_temp),
        }


class _PureCSCNeighborLoader:
    """Fast pure-PyTorch/NumPy 2-hop CSC mini-batch neighbor loader when torch_geometric is absent."""

    def __init__(
        self,
        pyg_data: GraphSnapshotFallback,
        csc_index: tuple[np.ndarray, np.ndarray, np.ndarray],
        num_neighbors: list[int],
        batch_size: int,
        input_nodes: torch.Tensor | None,
        shuffle: bool,
    ) -> None:
        self.data = pyg_data
        self.colptr, self.perm, self.src_sorted = csc_index
        self.num_neighbors = [int(n) for n in num_neighbors]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        if input_nodes is None:
            self.seeds = np.arange(int(pyg_data.num_nodes), dtype=np.int64)
        else:
            self.seeds = input_nodes.detach().cpu().numpy().astype(np.int64, copy=True)

    def __iter__(self):
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        order = rng.permutation(self.seeds) if self.shuffle else self.seeds
        for start in range(0, len(order), self.batch_size):
            batch_seeds = order[start : start + self.batch_size]
            yield self._sample_batch(batch_seeds, rng)

    def _sample_batch(self, batch_seeds: np.ndarray, rng: np.random.Generator) -> GraphSnapshotFallback:
        nodes_list: list[int] = batch_seeds.tolist()
        g2l: dict[int, int] = {int(u): i for i, u in enumerate(nodes_list)}
        frontier = batch_seeds
        frontier_local = np.arange(len(nodes_list), dtype=np.int64)

        orig_edge_ids: list[np.ndarray] = []
        dst_local_chunks: list[np.ndarray] = []
        src_global_chunks: list[np.ndarray] = []

        for req in self.num_neighbors:
            if req == 0 or frontier.size == 0:
                break
            starts = self.colptr[frontier]
            ends = self.colptr[frontier + 1]
            degs = ends - starts

            hop_pos: list[np.ndarray] = []
            hop_dst: list[np.ndarray] = []
            for i in range(frontier.size):
                d = int(degs[i])
                if d <= 0:
                    continue
                s, e = int(starts[i]), int(ends[i])
                if req < 0 or d <= req:
                    chosen = np.arange(s, e, dtype=np.int64)
                else:
                    chosen = s + rng.choice(d, size=req, replace=False).astype(np.int64)
                hop_pos.append(chosen)
                hop_dst.append(np.full(chosen.size, frontier_local[i], dtype=np.int64))

            if not hop_pos:
                break

            pos_arr = np.concatenate(hop_pos)
            dst_arr = np.concatenate(hop_dst)
            s_glob = self.src_sorted[pos_arr]
            orig_edge_ids.append(self.perm[pos_arr])
            dst_local_chunks.append(dst_arr)
            src_global_chunks.append(s_glob)

            new_front: list[int] = []
            new_front_loc: list[int] = []
            for g in s_glob:
                gi = int(g)
                if gi not in g2l:
                    loc = len(nodes_list)
                    g2l[gi] = loc
                    nodes_list.append(gi)
                    new_front.append(gi)
                    new_front_loc.append(loc)
            frontier = np.asarray(new_front, dtype=np.int64)
            frontier_local = np.asarray(new_front_loc, dtype=np.int64)

        node_idx_t = torch.from_numpy(np.asarray(nodes_list, dtype=np.int64))
        if orig_edge_ids:
            all_orig_e = torch.from_numpy(np.concatenate(orig_edge_ids))
            all_dst = torch.from_numpy(np.concatenate(dst_local_chunks))
            all_src_g = np.concatenate(src_global_chunks)
            all_src = torch.from_numpy(
                np.fromiter((g2l[int(g)] for g in all_src_g), dtype=np.int64, count=all_src_g.size)
            )
            edge_index = torch.stack([all_src, all_dst], dim=0)
            edge_attr = (
                self.data.edge_attr[all_orig_e]
                if getattr(self.data, "edge_attr", None) is not None
                else None
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = (
                torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)
                if getattr(self.data, "edge_attr", None) is not None
                else None
            )

        kw: dict[str, Any] = {
            "batch_size": int(batch_seeds.size),
            "num_nodes": int(node_idx_t.numel()),
            "edge_index": edge_index,
        }
        if edge_attr is not None:
            kw["edge_attr"] = edge_attr
        for attr in ("x_tab", "x_topo", "x_temp_stat", "temporal_sequence", "y"):
            val = getattr(self.data, attr, None)
            if val is not None:
                kw[attr] = val[node_idx_t]
        return GraphSnapshotFallback(**kw)


def _build_csc_arrays(edge_index: torch.Tensor, num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if edge_index is None or edge_index.numel() == 0:
        return (
            np.zeros(num_nodes + 1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    src_np = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst_np = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    perm = np.argsort(dst_np, kind="mergesort")
    dst_sorted = dst_np[perm]
    src_sorted = src_np[perm]
    counts = np.bincount(dst_sorted, minlength=num_nodes)
    colptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=colptr[1:])
    return colptr, perm, src_sorted


class ModularTGAE(BaseDetector):
    """Modular Temporal-Graph Autoencoder Detector supporting all 7 Ablations (A -> G)."""

    def __init__(
        self,
        hidden_dim: int | None = None,
        latent_dim: int | None = None,
        dropout: float | None = None,
        use_tabular: bool | None = None,
        use_graph: bool | None = None,
        use_temporal: bool | None = None,
        graph_neighbor_samples: list[int] | tuple[int, ...] | None = None,
        alpha_feature: float | None = None,
        beta_graph: float | None = None,
        gamma_temporal: float | None = None,
        contrastive_margin: float | None = None,
        supervised_aux_weight: float | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        epochs: int | None = None,
        early_stopping_patience: int | None = None,
        use_amp: bool | None = None,
        tune_weights_on_val: bool = True,
        device: str | torch.device | None = None,
        random_state: int | None = None,
        seq_use_log1p: bool = True,
        strict_unseen_val: bool = False,
        config_path: str | Path | None = "configs/default.yaml",
        feature_names: tuple[str, ...] | list[str] = ALL_FEATURE_COLS,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self.seq_use_log1p = bool(seq_use_log1p)
        self.strict_unseen_val = bool(strict_unseen_val)

        cfg: dict[str, Any] = {}
        if config_path is not None and Path(config_path).exists():
            cfg = load_config(config_path)

        model_cfg = cfg.get("model", {})
        loss_cfg = model_cfg.get("loss_weights", {})
        train_cfg = cfg.get("training", {})
        proj_cfg = cfg.get("project", {})

        self.hidden_dim = int(
            hidden_dim if hidden_dim is not None else model_cfg.get("hidden_dim", 128)
        )
        self.latent_dim = int(
            latent_dim if latent_dim is not None else model_cfg.get("latent_dim", 64)
        )
        self.dropout = float(dropout if dropout is not None else model_cfg.get("dropout", 0.2))

        self.use_tabular = bool(
            use_tabular if use_tabular is not None else model_cfg.get("use_tabular", True)
        )
        self.use_graph = bool(
            use_graph if use_graph is not None else model_cfg.get("use_graph", True)
        )
        self.use_temporal = bool(
            use_temporal if use_temporal is not None else model_cfg.get("use_temporal", True)
        )

        raw_samples = (
            graph_neighbor_samples
            if graph_neighbor_samples is not None
            else model_cfg.get("graph_neighbor_samples", [15, 10])
        )
        self.graph_neighbor_samples: list[int] = [int(s) for s in raw_samples]

        self.alpha_feature = float(
            alpha_feature if alpha_feature is not None else loss_cfg.get("alpha_feature", 0.4)
        )
        self.beta_graph = float(
            beta_graph if beta_graph is not None else loss_cfg.get("beta_graph", 0.3)
        )
        self.gamma_temporal = float(
            gamma_temporal if gamma_temporal is not None else loss_cfg.get("gamma_temporal", 0.3)
        )

        self.alpha_, self.beta_, self.gamma_ = normalize_branch_weights(
            self.alpha_feature,
            self.beta_graph,
            self.gamma_temporal,
            self.use_tabular,
            self.use_graph,
            self.use_temporal,
        )

        self.contrastive_margin = float(
            contrastive_margin
            if contrastive_margin is not None
            else model_cfg.get("contrastive_margin", 5.0)
        )
        self.supervised_aux_weight = float(
            supervised_aux_weight
            if supervised_aux_weight is not None
            else model_cfg.get("supervised_aux_weight", 0.2)
        )

        self.batch_size = int(
            batch_size if batch_size is not None else train_cfg.get("batch_size", 512)
        )
        self.lr = float(lr if lr is not None else train_cfg.get("lr", 1e-3))
        self.weight_decay = float(
            weight_decay if weight_decay is not None else train_cfg.get("weight_decay", 1e-4)
        )
        self.epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 60))
        self.early_stopping_patience = int(
            early_stopping_patience
            if early_stopping_patience is not None
            else train_cfg.get("early_stopping_patience", 10)
        )
        self.use_amp = bool(use_amp if use_amp is not None else train_cfg.get("use_amp", True))
        self.tune_weights_on_val = bool(tune_weights_on_val)
        self.random_state = int(
            random_state
            if random_state is not None
            else proj_cfg.get("random_seed", RANDOM_SEED)
        )

        dev_arg = device if device is not None else proj_cfg.get("device", "auto")
        if str(dev_arg) == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            req_dev = torch.device(dev_arg)
            if req_dev.type == "cuda" and not torch.cuda.is_available():
                self.device_ = torch.device("cpu")
            else:
                self.device_ = req_dev

        self.net_: ModularTGAENet | None = None
        self.iqr_stats_: dict[str, dict[str, float]] = {
            "feature": {"median": 0.0, "iqr": 1.0},
            "graph": {"median": 0.0, "iqr": 1.0},
            "temporal": {"median": 0.0, "iqr": 1.0},
        }
        self.best_val_pr_auc_: float | None = None
        self.pos_weight_: float = 1.0
        self.training_history_: list[dict[str, float]] = []
        # cache value keeps a strong ref to the source data so ``id(data)`` can never be recycled.
        self._sampler_cache_: dict[tuple[Any, ...], tuple[Data, Any, Any]] = {}

    def _build_net(self) -> ModularTGAENet:
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        net = ModularTGAENet(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            use_tabular=self.use_tabular,
            use_graph=self.use_graph,
            use_temporal=self.use_temporal,
            seq_use_log1p=self.seq_use_log1p,
        )
        return net.to(self.device_)

    def _build_minimal_pyg_data(
        self,
        data: Any,
        include_labels: bool = False,
    ) -> Any:
        """Build a lightweight Data object containing only tensors required by enabled branches."""
        x_tab, x_topo, x_temp_stat, temporal_seq, edge_index, edge_attr = (
            _extract_modality_tensors(data)
        )
        num_nodes = _count_nodes(x_tab, x_topo, x_temp_stat, temporal_seq)
        kwargs: dict[str, Any] = {"num_nodes": num_nodes}

        if self.use_tabular:
            kwargs["x_tab"] = x_tab
        if self.use_graph:
            kwargs["x_topo"] = x_topo
            if edge_index is None:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            kwargs["edge_index"] = edge_index
            if edge_attr is not None:
                kwargs["edge_attr"] = edge_attr
        if self.use_temporal:
            if temporal_seq is None:
                raise ValueError(
                    "use_temporal=True requires 'temporal_sequence' on data."
                )
            kwargs["x_temp_stat"] = x_temp_stat
            kwargs["temporal_sequence"] = temporal_seq

        if include_labels:
            y = extract_labels(data)
            kwargs["y"] = torch.as_tensor(y, dtype=torch.long)

        if HAS_PYG and PyGData is not None:
            return PyGData(**kwargs)
        return GraphSnapshotFallback(**kwargs)

    def _get_graph_loader(
        self,
        data: Any,
        input_nodes: torch.Tensor | None,
        shuffle: bool,
        include_labels: bool = False,
    ) -> Any:
        """Build NeighborLoader (or pure-PyTorch CSC fallback) reusing cached CSC index."""
        has_y = (
            ("y" in data and data["y"] is not None)
            if isinstance(data, dict)
            else (hasattr(data, "y") and data.y is not None)
        )
        cache_key = (
            id(data),
            self.use_tabular,
            self.use_graph,
            self.use_temporal,
            tuple(self.graph_neighbor_samples),
            HAS_PYG,
        )
        if cache_key in self._sampler_cache_:
            pyg_data, sampler, _ = self._sampler_cache_[cache_key]
            if HAS_PYG and NeighborLoader is not None:
                return NeighborLoader(
                    pyg_data,
                    num_neighbors=self.graph_neighbor_samples,
                    batch_size=self.batch_size,
                    input_nodes=input_nodes,
                    shuffle=shuffle,
                    neighbor_sampler=sampler,
                )
            return _PureCSCNeighborLoader(
                pyg_data,
                csc_index=sampler,
                num_neighbors=self.graph_neighbor_samples,
                batch_size=self.batch_size,
                input_nodes=input_nodes,
                shuffle=shuffle,
            )

        pyg_data = self._build_minimal_pyg_data(data, include_labels=bool(include_labels or has_y))
        if HAS_PYG and NeighborSampler is not None and NeighborLoader is not None:
            sampler: Any = NeighborSampler(pyg_data, num_neighbors=self.graph_neighbor_samples)
        else:
            sampler = _build_csc_arrays(getattr(pyg_data, "edge_index", None), int(pyg_data.num_nodes))
        if len(self._sampler_cache_) >= 4:
            self._sampler_cache_.clear()
        self._sampler_cache_[cache_key] = (pyg_data, sampler, data)
        if HAS_PYG and NeighborLoader is not None:
            return NeighborLoader(
                pyg_data,
                num_neighbors=self.graph_neighbor_samples,
                batch_size=self.batch_size,
                input_nodes=input_nodes,
                shuffle=shuffle,
                neighbor_sampler=sampler,
            )
        return _PureCSCNeighborLoader(
            pyg_data,
            csc_index=sampler,
            num_neighbors=self.graph_neighbor_samples,
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            shuffle=shuffle,
        )

    def _compute_batch_loss(
        self,
        out: dict[str, Any],
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Normal-Only + Contrastive Margin Reconstruction Loss + Hybrid Focal/BCE Aux Loss."""
        e_total = (
            self.alpha_ * out["e_feature"]
            + self.beta_ * out["e_graph"]
            + self.gamma_ * out["e_temporal"]
        )

        normal_mask = y_batch == LABEL_NORMAL
        anom_mask = y_batch == LABEL_ANOMALY

        loss_terms: list[torch.Tensor] = []
        if normal_mask.any():
            loss_terms.append(e_total[normal_mask].mean())
        if anom_mask.any():
            margin_err = F.relu(self.contrastive_margin - e_total[anom_mask])
            loss_terms.append(margin_err.mean())

        recon_loss = (
            torch.stack(loss_terms).mean() if loss_terms else e_total.mean() * 0.0
        )

        labeled_mask = normal_mask | anom_mask
        if self.supervised_aux_weight > 0.0 and labeled_mask.any():
            logits = out["aux_logits"][labeled_mask]
            targets = (y_batch[labeled_mask] == LABEL_ANOMALY).to(dtype=logits.dtype)
            pos_w = torch.tensor(self.pos_weight_, device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_w, reduction="none"
            )
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            focal_loss = (((1.0 - p_t) ** 2.0) * bce).mean()
            return recon_loss + self.supervised_aux_weight * focal_loss

        return recon_loss

    @torch.no_grad()
    def _predict_raw_branch_errors(
        self,
        data: Data | dict[str, Any],
        node_indices: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute raw ``(e_feature, e_graph, e_temporal)`` in mini-batches without OOM."""
        if self.net_ is None:
            raise RuntimeError("ModularTGAE network is not initialized.")

        self.net_.eval()
        x_tab, x_topo, x_temp_stat, temporal_seq, _, _ = _extract_modality_tensors(data)
        total_nodes = _count_nodes(x_tab, x_topo, x_temp_stat, temporal_seq)

        if node_indices is not None:
            idx_tensor = torch.as_tensor(node_indices, dtype=torch.long).view(-1).cpu()
            n_target = int(idx_tensor.numel())
        else:
            idx_tensor = None
            n_target = total_nodes

        e_feat_arr = np.zeros(n_target, dtype=np.float64)
        e_graph_arr = np.zeros(n_target, dtype=np.float64)
        e_temp_arr = np.zeros(n_target, dtype=np.float64)

        if n_target == 0:
            return {"feature": e_feat_arr, "graph": e_graph_arr, "temporal": e_temp_arr}

        use_amp_actual = bool(self.use_amp and self.device_.type == "cuda")

        if self.use_graph:
            loader = self._get_graph_loader(
                data,
                input_nodes=idx_tensor,
                shuffle=False,
                include_labels=False,
            )
            offset = 0
            # Fixed seed => the random neighbor sampling is reproducible, so val-calibrated
            # threshold/IQR transfer exactly to test and save()/load() round-trips give equal scores.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.random_state)
                for batch in loader:
                    bsz = int(batch.batch_size)
                    batch = batch.to(self.device_)
                    with torch.amp.autocast(
                        device_type=self.device_.type, enabled=use_amp_actual
                    ):
                        out = self.net_(batch, batch_size=bsz)
                    end = offset + bsz
                    if self.use_tabular:
                        e_feat_arr[offset:end] = (
                            out["e_feature"].detach().cpu().numpy().astype(np.float64, copy=False)
                        )
                    if self.use_graph:
                        e_graph_arr[offset:end] = (
                            out["e_graph"].detach().cpu().numpy().astype(np.float64, copy=False)
                        )
                    if self.use_temporal:
                        e_temp_arr[offset:end] = (
                            out["e_temporal"].detach().cpu().numpy().astype(np.float64, copy=False)
                        )
                    offset = end
        else:
            chunk_size = max(self.batch_size, 4096)
            for start in range(0, n_target, chunk_size):
                end = min(start + chunk_size, n_target)
                sel = slice(start, end) if idx_tensor is None else idx_tensor[start:end]
                b_tab = x_tab[sel].to(self.device_) if self.use_tabular else None
                b_temp_stat = x_temp_stat[sel].to(self.device_) if self.use_temporal else None
                b_seq = (
                    temporal_seq[sel].to(self.device_)
                    if (self.use_temporal and temporal_seq is not None)
                    else None
                )
                bsz = end - start
                with torch.amp.autocast(device_type=self.device_.type, enabled=use_amp_actual):
                    out = self.net_(
                        data_or_x_tab=b_tab,
                        x_temp_stat=b_temp_stat,
                        temporal_sequence=b_seq,
                        batch_size=bsz,
                    )
                if self.use_tabular:
                    e_feat_arr[start:end] = (
                        out["e_feature"].detach().cpu().numpy().astype(np.float64, copy=False)
                    )
                if self.use_temporal:
                    e_temp_arr[start:end] = (
                        out["e_temporal"].detach().cpu().numpy().astype(np.float64, copy=False)
                    )

        return {"feature": e_feat_arr, "graph": e_graph_arr, "temporal": e_temp_arr}

    def _fit_iqr_stats_from_errors(
        self,
        raw_errors: dict[str, np.ndarray],
        normal_mask: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Compute median and IQR of ``(E_feature, E_graph, E_temporal)`` on normal samples."""
        stats: dict[str, dict[str, float]] = {}
        branch_flags = {
            "feature": self.use_tabular,
            "graph": self.use_graph,
            "temporal": self.use_temporal,
        }
        for key, enabled in branch_flags.items():
            if not enabled:
                stats[key] = {"median": 0.0, "iqr": 1.0}
                continue
            vals = np.asarray(raw_errors[key][normal_mask], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                stats[key] = {"median": 0.0, "iqr": 1.0}
                continue
            med = float(np.median(vals))
            q75, q25 = np.percentile(vals, [75, 25])
            iqr = float(q75 - q25)
            if iqr <= 1e-6:
                iqr = max(float(np.std(vals)), 1e-6)
            stats[key] = {"median": med, "iqr": iqr}
        return stats

    def _apply_iqr_calibration(
        self,
        raw_errors: dict[str, np.ndarray],
        iqr_stats: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Standardize raw branch errors ``E_k`` into ``E_tilde_k = (E_k - median_k) / IQR_k``."""
        stats = self.iqr_stats_ if iqr_stats is None else iqr_stats
        calibrated: dict[str, np.ndarray] = {}
        branch_flags = {
            "feature": self.use_tabular,
            "graph": self.use_graph,
            "temporal": self.use_temporal,
        }
        for key, enabled in branch_flags.items():
            arr = np.asarray(raw_errors[key], dtype=np.float64)
            if not enabled:
                calibrated[key] = np.zeros_like(arr, dtype=np.float64)
            else:
                med = float(stats[key]["median"])
                iqr = max(float(stats[key]["iqr"]), 1e-6)
                calibrated[key] = np.nan_to_num(
                    (arr - med) / iqr, nan=0.0, posinf=0.0, neginf=0.0
                )
        return calibrated

    def _combine_calibrated_errors(
        self,
        calibrated_errors: dict[str, np.ndarray],
        weights: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        alpha, beta, gamma = (
            (self.alpha_, self.beta_, self.gamma_) if weights is None else weights
        )
        return (
            alpha * calibrated_errors["feature"]
            + beta * calibrated_errors["graph"]
            + gamma * calibrated_errors["temporal"]
        )

    def _tune_weights_on_validation(
        self,
        calibrated_errors: dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> tuple[float, float, float]:
        """Grid-search ``(alpha, beta, gamma)`` on Validation set to maximize PR-AUC."""
        n_active = int(self.use_tabular) + int(self.use_graph) + int(self.use_temporal)
        default_w = normalize_branch_weights(
            self.alpha_feature,
            self.beta_graph,
            self.gamma_temporal,
            self.use_tabular,
            self.use_graph,
            self.use_temporal,
        )
        if not self.tune_weights_on_val or n_active <= 1 or len(np.unique(y_val)) < 2:
            return default_w

        candidates: list[tuple[float, float, float]] = [default_w]
        grid = np.linspace(0.1, 0.8, 8)
        for a in grid if self.use_tabular else [0.0]:
            for b in grid if self.use_graph else [0.0]:
                for g in grid if self.use_temporal else [0.0]:
                    candidates.append(
                        normalize_branch_weights(
                            a, b, g, self.use_tabular, self.use_graph, self.use_temporal
                        )
                    )

        best_w = default_w
        best_score = -1.0
        for w in candidates:
            scores = self._combine_calibrated_errors(calibrated_errors, weights=w)
            pr_auc = float(average_precision_score(y_val, scores))
            if pr_auc > best_score:
                best_score = pr_auc
                best_w = w
        return best_w

    def fit(self, train_data: Any, val_data: Any | None = None) -> ModularTGAE:
        """Fit ModularTGAE on ``train_data`` and calibrate IQR, branch weights, and ``threshold_`` on ``val_data``."""
        y_train = extract_labels(train_data)
        normal_mask_tr = y_train == LABEL_NORMAL
        anom_mask_tr = y_train == LABEL_ANOMALY
        labeled_mask_tr = normal_mask_tr | anom_mask_tr

        if not normal_mask_tr.any():
            raise ValueError(
                "ModularTGAE requires at least one Normal (y == 0) sample in train_data."
            )

        num_norm = int(normal_mask_tr.sum())
        num_anom = int(anom_mask_tr.sum())
        self.pos_weight_ = float(np.clip(num_norm / max(1, num_anom), 0.5, 50.0))

        self.net_ = self._build_net()
        x_tab_tr, x_topo_tr, x_temp_stat_tr, seq_tr, _, _ = _extract_modality_tensors(train_data)

        norm_idx_tr = torch.from_numpy(np.where(normal_mask_tr)[0]).long()
        self.net_.fit_normalizer(
            x_tab=x_tab_tr[norm_idx_tr] if (self.use_tabular or self.use_graph) else None,
            x_topo=x_topo_tr[norm_idx_tr] if self.use_graph else None,
            x_temp_stat=x_temp_stat_tr[norm_idx_tr] if self.use_temporal else None,
            temporal_sequence=(
                seq_tr[norm_idx_tr] if (self.use_temporal and seq_tr is not None) else None
            ),
        )

        train_indices = torch.from_numpy(np.where(labeled_mask_tr)[0]).long()
        if self.use_graph:
            train_loader: Any = self._get_graph_loader(
                train_data,
                input_nodes=train_indices,
                shuffle=True,
                include_labels=True,
            )
        else:
            tensors: list[torch.Tensor] = [
                torch.from_numpy(y_train[labeled_mask_tr]).long(),
                x_tab_tr[train_indices]
                if self.use_tabular
                else torch.empty(( len(train_indices), 0)),
                x_temp_stat_tr[train_indices]
                if self.use_temporal
                else torch.empty((len(train_indices), 0)),
                seq_tr[train_indices]
                if (self.use_temporal and seq_tr is not None)
                else torch.empty((len(train_indices), 0, 0)),
            ]
            dataset = TensorDataset(*tensors)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_eval_indices: np.ndarray | None = None
        y_val_eval: np.ndarray | None = None
        val_normal_in_eval: np.ndarray | None = None
        if val_data is not None:
            val_mask, _ = get_eval_mask_and_labels(val_data)
            if self.strict_unseen_val:
                tr_addrs = (
                    train_data.get("addresses")
                    if isinstance(train_data, dict)
                    else getattr(train_data, "addresses", None)
                )
                if tr_addrs is not None:
                    tr_labeled_set = {
                        str(tr_addrs[i]).strip().lower()
                        for i in np.where(labeled_mask_tr)[0]
                    }
                    val_mask = self._filter_eval_mask(val_data, val_mask, exclude_addresses=tr_labeled_set)
            y_val_eval = extract_labels(val_data)[val_mask]
            val_eval_indices = np.where(val_mask)[0]
            val_normal_in_eval = y_val_eval == LABEL_NORMAL

        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        use_amp_actual = bool(self.use_amp and self.device_.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp_actual)

        best_val_pr_auc = -1.0
        best_state_dict = copy.deepcopy(self.net_.state_dict())
        best_iqr_stats = copy.deepcopy(self.iqr_stats_)
        patience_counter = 0
        self.training_history_ = []

        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                if self.use_graph:
                    bsz = int(batch.batch_size)
                    batch = batch.to(self.device_)
                    y_b = batch.y[:bsz]
                    with torch.amp.autocast(
                        device_type=self.device_.type, enabled=use_amp_actual
                    ):
                        out = self.net_(batch, batch_size=bsz)
                        loss = self._compute_batch_loss(out, y_b)
                else:
                    y_b, b_tab, b_temp_stat, b_seq = batch
                    bsz = int(y_b.shape[0])
                    y_b = y_b.to(self.device_)
                    b_tab_dev = b_tab.to(self.device_) if self.use_tabular else None
                    b_stat_dev = b_temp_stat.to(self.device_) if self.use_temporal else None
                    b_seq_dev = b_seq.to(self.device_) if self.use_temporal else None
                    with torch.amp.autocast(
                        device_type=self.device_.type, enabled=use_amp_actual
                    ):
                        out = self.net_(
                            data_or_x_tab=b_tab_dev,
                            x_temp_stat=b_stat_dev,
                            temporal_sequence=b_seq_dev,
                            batch_size=bsz,
                        )
                        loss = self._compute_batch_loss(out, y_b)

                if use_amp_actual:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=5.0)
                    optimizer.step()

                epoch_loss += float(loss.detach().item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            record: dict[str, float] = {"epoch": float(epoch), "train_loss": avg_loss}

            if (
                val_data is not None
                and val_eval_indices is not None
                and y_val_eval is not None
                and val_normal_in_eval is not None
            ):
                val_raw = self._predict_raw_branch_errors(
                    val_data, node_indices=val_eval_indices
                )
                epoch_iqr = self._fit_iqr_stats_from_errors(val_raw, val_normal_in_eval)
                val_cal = self._apply_iqr_calibration(val_raw, iqr_stats=epoch_iqr)
                val_scores = self._combine_calibrated_errors(val_cal)
                has_both = len(np.unique(y_val_eval)) == 2
                val_pr_auc = (
                    float(average_precision_score(y_val_eval, val_scores))
                    if has_both
                    else 0.0
                )
                record["val_pr_auc"] = val_pr_auc
                self.training_history_.append(record)

                if val_pr_auc > best_val_pr_auc + 1e-6:
                    best_val_pr_auc = val_pr_auc
                    best_state_dict = copy.deepcopy(self.net_.state_dict())
                    best_iqr_stats = copy.deepcopy(epoch_iqr)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if (
                        self.early_stopping_patience > 0
                        and patience_counter >= self.early_stopping_patience
                    ):
                        break
            else:
                self.training_history_.append(record)
                best_state_dict = copy.deepcopy(self.net_.state_dict())

        self.net_.load_state_dict(best_state_dict)
        self.is_fitted_ = True

        if (
            val_data is not None
            and val_eval_indices is not None
            and y_val_eval is not None
            and val_normal_in_eval is not None
        ):
            self.best_val_pr_auc_ = best_val_pr_auc
            val_raw = self._predict_raw_branch_errors(val_data, node_indices=val_eval_indices)
            self.iqr_stats_ = self._fit_iqr_stats_from_errors(val_raw, val_normal_in_eval)
            val_cal = self._apply_iqr_calibration(val_raw, iqr_stats=self.iqr_stats_)
            self.alpha_, self.beta_, self.gamma_ = self._tune_weights_on_validation(
                val_cal, y_val_eval
            )
            val_eval_scores = self._combine_calibrated_errors(val_cal)
            self.best_val_pr_auc_ = (
                float(average_precision_score(y_val_eval, val_eval_scores))
                if len(np.unique(y_val_eval)) == 2
                else best_val_pr_auc
            )
            best_tau, best_f1 = find_optimal_f1_threshold(y_val_eval, val_eval_scores)
            self.threshold_ = best_tau
            self.val_best_f1_ = best_f1
        else:
            train_raw = self._predict_raw_branch_errors(
                train_data, node_indices=train_indices
            )
            y_tr_labeled = y_train[labeled_mask_tr]
            self.iqr_stats_ = self._fit_iqr_stats_from_errors(
                train_raw, y_tr_labeled == LABEL_NORMAL
            )
            if (y_tr_labeled == LABEL_ANOMALY).any():
                tr_cal = self._apply_iqr_calibration(train_raw, iqr_stats=self.iqr_stats_)
                tr_scores = self._combine_calibrated_errors(tr_cal)
                best_tau, best_f1 = find_optimal_f1_threshold(y_tr_labeled, tr_scores)
                self.threshold_ = best_tau
                self.val_best_f1_ = best_f1

        return self

    def calibrate_threshold(
        self,
        val_data: Any,
        scores: np.ndarray | None = None,
        exclude_addresses: set[str] | None = None,
        inductive_only: bool = False,
    ) -> float:
        """Calibrate IQR statistics, branch weights, and F1 optimal ``threshold_`` on Validation data."""
        mask, _ = get_eval_mask_and_labels(val_data)
        mask = self._filter_eval_mask(
            val_data, mask, exclude_addresses=exclude_addresses, inductive_only=inductive_only
        )
        y_val = extract_labels(val_data)[mask]
        if scores is None:
            eval_indices = np.where(mask)[0]
            val_raw = self._predict_raw_branch_errors(val_data, node_indices=eval_indices)
            self.iqr_stats_ = self._fit_iqr_stats_from_errors(
                val_raw, y_val == LABEL_NORMAL
            )
            val_cal = self._apply_iqr_calibration(val_raw, iqr_stats=self.iqr_stats_)
            self.alpha_, self.beta_, self.gamma_ = self._tune_weights_on_validation(
                val_cal, y_val
            )
            eval_scores = self._combine_calibrated_errors(val_cal)
            best_tau, best_f1 = find_optimal_f1_threshold(y_val, eval_scores)
            self.threshold_ = best_tau
            self.val_best_f1_ = best_f1
            return best_tau
        return super().calibrate_threshold(
            val_data,
            scores=scores,
            exclude_addresses=exclude_addresses,
            inductive_only=inductive_only,
        )

    def predict_score(
        self,
        data: Any,
        node_indices: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Compute 1D IQR-calibrated anomaly scores in mini-batches (safe for 1.79M+ nodes)."""
        if not self.is_fitted_ or self.net_ is None:
            raise RuntimeError("ModularTGAE is not fitted yet.")

        raw_errors = self._predict_raw_branch_errors(data, node_indices=node_indices)
        calibrated = self._apply_iqr_calibration(raw_errors)
        return self._combine_calibrated_errors(calibrated)

    def evaluate(
        self,
        data: Any,
        threshold: float | None = None,
        exclude_addresses: set[str] | None = None,
        inductive_only: bool = False,
    ) -> dict[str, float]:
        """Compute evaluation metrics strictly on ``eval_mask & (y in {0, 1})`` using mini-batches."""
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        mask, _ = get_eval_mask_and_labels(data)
        mask = self._filter_eval_mask(
            data, mask, exclude_addresses=exclude_addresses, inductive_only=inductive_only
        )
        y_true = extract_labels(data)[mask]
        eval_indices = np.where(mask)[0]
        scores = np.asarray(
            self.predict_score(data, node_indices=eval_indices), dtype=np.float64
        )

        tau = self.threshold_ if threshold is None else float(threshold)
        if tau is None:
            raise RuntimeError(
                "Decision threshold is not set. Calibrate on validation data before evaluating."
            )
        preds = (scores >= tau).astype(np.int64)

        has_both_classes = len(np.unique(y_true)) == 2
        pr_auc = float(average_precision_score(y_true, scores)) if has_both_classes else 0.0
        roc_auc = float(roc_auc_score(y_true, scores)) if has_both_classes else 0.0

        return {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "threshold": float(tau),
            "num_eval_nodes": int(mask.sum()),
            "num_eval_positives": int((y_true == 1).sum()),
            "num_eval_negatives": int((y_true == 0).sum()),
        }

    def _extract_local_2hop_subgraph(
        self,
        data: Data | dict[str, Any],
        node_idx: int,
    ) -> dict[str, Any]:
        """Extract local 2-hop subgraph around ``node_idx`` (placed at index 0) without full-graph CSC."""
        x_tab, x_topo, x_temp_stat, temporal_seq, edge_index, edge_attr = (
            _extract_modality_tensors(data)
        )
        num_nodes = _count_nodes(x_tab, x_topo, x_temp_stat, temporal_seq)
        if node_idx < 0 or node_idx >= num_nodes:
            raise IndexError(f"node_idx {node_idx} out of range [0, {num_nodes})")

        if not self.use_graph or edge_index is None or edge_index.numel() == 0:
            sub_idx = torch.tensor([node_idx], dtype=torch.long)
            return {
                "x_tab": x_tab[sub_idx].to(self.device_),
                "x_topo": x_topo[sub_idx].to(self.device_),
                "x_temp_stat": x_temp_stat[sub_idx].to(self.device_),
                "temporal_sequence": (
                    temporal_seq[sub_idx].to(self.device_) if temporal_seq is not None else None
                ),
                "edge_index": torch.empty((2, 0), dtype=torch.long, device=self.device_),
                "edge_attr": (
                    torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=self.device_)
                    if edge_attr is not None
                    else None
                ),
                "batch_size": 1,
            }

        src, dst = edge_index[0], edge_index[1]
        max_h1 = self.graph_neighbor_samples[0] if self.graph_neighbor_samples else 15
        max_h2 = (
            self.graph_neighbor_samples[1]
            if len(self.graph_neighbor_samples) > 1
            else 10
        )

        # Only IN-edges (src -> dst), matching NeighborSampler(direction="forward") used in fit/predict.
        hop1_edge_idx = torch.where(dst == node_idx)[0]
        if hop1_edge_idx.numel() > max_h1:
            hop1_edge_idx = hop1_edge_idx[:max_h1]

        hop1_nodes = torch.unique(
            torch.cat([src[hop1_edge_idx], dst[hop1_edge_idx]])
        )
        hop1_frontier = hop1_nodes[hop1_nodes != node_idx]
        if hop1_frontier.numel() > max_h1:
            hop1_frontier = hop1_frontier[:max_h1]

        if hop1_frontier.numel() > 0 and max_h2 > 0:
            hop2_edge_idx = torch.where(torch.isin(dst, hop1_frontier))[0]
            max_edges_h2 = int(hop1_frontier.numel()) * max_h2
            if hop2_edge_idx.numel() > max_edges_h2:
                hop2_edge_idx = hop2_edge_idx[:max_edges_h2]
            chosen_edges = torch.unique(torch.cat([hop1_edge_idx, hop2_edge_idx]))
        else:
            chosen_edges = hop1_edge_idx

        if chosen_edges.numel() > 0:
            sub_src = src[chosen_edges]
            sub_dst = dst[chosen_edges]
            other_nodes = torch.unique(torch.cat([sub_src, sub_dst]))
            other_nodes = other_nodes[other_nodes != node_idx]
            sub_nodes = torch.cat(
                [torch.tensor([node_idx], dtype=other_nodes.dtype, device=other_nodes.device), other_nodes]
            )
            node_list = sub_nodes.detach().cpu().tolist()
            g2l = {int(g): loc for loc, g in enumerate(node_list)}
            local_src = torch.tensor(
                [g2l[int(g)] for g in sub_src.detach().cpu().tolist()],
                dtype=torch.long,
                device=self.device_,
            )
            local_dst = torch.tensor(
                [g2l[int(g)] for g in sub_dst.detach().cpu().tolist()],
                dtype=torch.long,
                device=self.device_,
            )
            local_edge_index = torch.stack([local_src, local_dst], dim=0)
            local_edge_attr = (
                edge_attr[chosen_edges].to(self.device_) if edge_attr is not None else None
            )
        else:
            sub_nodes = torch.tensor([node_idx], dtype=torch.long)
            local_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device_)
            local_edge_attr = (
                torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=self.device_)
                if edge_attr is not None
                else None
            )

        sub_nodes_cpu = sub_nodes.detach().cpu()
        return {
            "x_tab": x_tab[sub_nodes_cpu].to(self.device_),
            "x_topo": x_topo[sub_nodes_cpu].to(self.device_),
            "x_temp_stat": x_temp_stat[sub_nodes_cpu].to(self.device_),
            "temporal_sequence": (
                temporal_seq[sub_nodes_cpu].to(self.device_)
                if temporal_seq is not None
                else None
            ),
            "edge_index": local_edge_index,
            "edge_attr": local_edge_attr,
            "batch_size": 1,
        }

    @torch.no_grad()
    def explain(self, data: Any, node_idx: int) -> dict[str, Any]:
        """Run local 2-hop subgraph inference for ``node_idx`` and return branch % + Top-5 feature errors."""
        if not self.is_fitted_ or self.net_ is None:
            raise RuntimeError("ModularTGAE is not fitted yet.")

        self.net_.eval()
        local_sub = self._extract_local_2hop_subgraph(data, int(node_idx))
        use_amp_actual = bool(self.use_amp and self.device_.type == "cuda")
        with torch.amp.autocast(device_type=self.device_.type, enabled=use_amp_actual):
            out = self.net_(local_sub, batch_size=1)

        raw_single = {
            "feature": np.array([float(out["e_feature"][0].item())], dtype=np.float64),
            "graph": np.array([float(out["e_graph"][0].item())], dtype=np.float64),
            "temporal": np.array([float(out["e_temporal"][0].item())], dtype=np.float64),
        }
        cal_single = self._apply_iqr_calibration(raw_single)
        score = float(self._combine_calibrated_errors(cal_single)[0])

        # Compute non-negative branch contributions (feature_pct, graph_pct, temporal_pct)
        w_feat_pos = self.alpha_ * max(float(cal_single["feature"][0]), 0.0) if self.use_tabular else 0.0
        w_graph_pos = self.beta_ * max(float(cal_single["graph"][0]), 0.0) if self.use_graph else 0.0
        w_temp_pos = self.gamma_ * max(float(cal_single["temporal"][0]), 0.0) if self.use_temporal else 0.0
        pos_sum = w_feat_pos + w_graph_pos + w_temp_pos

        if pos_sum <= 1e-12:
            w_feat_pos = self.alpha_ * float(raw_single["feature"][0]) if self.use_tabular else 0.0
            w_graph_pos = self.beta_ * float(raw_single["graph"][0]) if self.use_graph else 0.0
            w_temp_pos = self.gamma_ * float(raw_single["temporal"][0]) if self.use_temporal else 0.0
            pos_sum = w_feat_pos + w_graph_pos + w_temp_pos

        if pos_sum > 1e-12:
            feature_pct = 100.0 * w_feat_pos / pos_sum
            graph_pct = 100.0 * w_graph_pos / pos_sum
            temporal_pct = 100.0 * w_temp_pos / pos_sum
        else:
            feature_pct = 100.0 * self.alpha_
            graph_pct = 100.0 * self.beta_
            temporal_pct = 100.0 * self.gamma_

        # Collect per-feature reconstruction errors at node_idx (local index 0)
        x_tab_raw = local_sub["x_tab"][0].detach().cpu().numpy()
        x_topo_raw = local_sub["x_topo"][0].detach().cpu().numpy()
        x_temp_stat_raw = local_sub["x_temp_stat"][0].detach().cpu().numpy()

        feature_candidates: list[dict[str, Any]] = []
        feature_errors_map: dict[str, float] = {}

        if self.use_tabular and out["err_tab_feat"] is not None and out["x_tab_hat"] is not None:
            errs = out["err_tab_feat"][0].detach().cpu().numpy()
            recons = out["x_tab_hat"][0].detach().cpu().numpy()
            for i, col_name in enumerate(TABULAR_TX_COLS):
                err_val = float(errs[i])
                feature_errors_map[col_name] = err_val
                feature_candidates.append(
                    {
                        "feature": col_name,
                        "modality": "tabular",
                        "value": float(x_tab_raw[i]),
                        "reconstructed_norm": float(recons[i]),
                        "reconstruction_error": err_val,
                    }
                )

        if self.use_graph and out["err_topo_feat"] is not None and out["x_topo_hat"] is not None:
            errs = out["err_topo_feat"][0].detach().cpu().numpy()
            recons = out["x_topo_hat"][0].detach().cpu().numpy()
            for i, col_name in enumerate(GRAPH_TOPO_COLS):
                err_val = float(errs[i])
                feature_errors_map[col_name] = err_val
                feature_candidates.append(
                    {
                        "feature": col_name,
                        "modality": "graph",
                        "value": float(x_topo_raw[i]),
                        "reconstructed_norm": float(recons[i]),
                        "reconstruction_error": err_val,
                    }
                )

        if (
            self.use_temporal
            and out["err_temp_stat_feat"] is not None
            and out["x_temp_stat_hat"] is not None
        ):
            errs = out["err_temp_stat_feat"][0].detach().cpu().numpy()
            recons = out["x_temp_stat_hat"][0].detach().cpu().numpy()
            for i, col_name in enumerate(TEMPORAL_STAT_COLS):
                err_val = float(errs[i])
                feature_errors_map[col_name] = err_val
                feature_candidates.append(
                    {
                        "feature": col_name,
                        "modality": "temporal",
                        "value": float(x_temp_stat_raw[i]),
                        "reconstructed_norm": float(recons[i]),
                        "reconstruction_error": err_val,
                    }
                )

        feature_candidates.sort(key=lambda item: item["reconstruction_error"], reverse=True)
        top_5_features = feature_candidates[:5]

        temporal_seq_errors: dict[str, float] = {}
        if self.use_temporal and out["err_seq_feat"] is not None:
            seq_errs = out["err_seq_feat"][0].detach().cpu().numpy()
            for i, col_name in enumerate(TEMPORAL_SEQUENCE_COLS):
                temporal_seq_errors[col_name] = float(seq_errs[i])

        predicted_label = (
            int(score >= self.threshold_) if self.threshold_ is not None else None
        )

        return {
            "model": "ModularTGAE",
            "node_idx": int(node_idx),
            "address": get_node_address(data, int(node_idx)),
            "score": score,
            "threshold": self.threshold_,
            "predicted_label": predicted_label,
            "feature_pct": float(feature_pct),
            "graph_pct": float(graph_pct),
            "temporal_pct": float(temporal_pct),
            "branch_contributions": {
                "feature_pct": float(feature_pct),
                "graph_pct": float(graph_pct),
                "temporal_pct": float(temporal_pct),
            },
            "branch_errors_raw": {
                "feature": float(raw_single["feature"][0]),
                "graph": float(raw_single["graph"][0]),
                "temporal": float(raw_single["temporal"][0]),
            },
            "branch_errors_calibrated": {
                "feature": float(cal_single["feature"][0]),
                "graph": float(cal_single["graph"][0]),
                "temporal": float(cal_single["temporal"][0]),
            },
            "top_features": top_5_features,
            "feature_errors": feature_errors_map,
            "temporal_sequence_errors": temporal_seq_errors,
        }

    @torch.no_grad()
    def predict_ego_subgraph(
        self,
        x_vec_25d: np.ndarray | torch.Tensor,
        temporal_seq_10x8: np.ndarray | torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Lightweight single-wallet inference without loading the 1.79M-node graph.
        Accepts the target wallet's 25-D scaled feature vector (shape ``(1, 25)`` or ``(N_ego, 25)``)
        and optional ``(1, 10, 8)`` temporal sequence + local ego edges, returning calibrated
        TGAE score, normalized [0, 1] display score, branch %, and per-feature reconstruction errors.
        """
        if not self.is_fitted_ or self.net_ is None:
            raise RuntimeError("ModularTGAE is not fitted yet.")

        x_t = torch.as_tensor(x_vec_25d, dtype=torch.float32, device=self.device_)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        n_ego = int(x_t.shape[0])

        if temporal_seq_10x8 is None:
            seq_t = torch.zeros(
                (1, TEMPORAL_WINDOWS, TEMPORAL_FEATURE_DIM),
                dtype=torch.float32,
                device=self.device_,
            )
        else:
            seq_t = torch.as_tensor(temporal_seq_10x8, dtype=torch.float32, device=self.device_)
            if seq_t.ndim == 2:
                seq_t = seq_t.unsqueeze(0)

        e_idx = (
            edge_index.to(self.device_)
            if edge_index is not None
            else torch.empty((2, 0), dtype=torch.long, device=self.device_)
        )
        e_attr = (
            edge_attr.to(self.device_)
            if edge_attr is not None
            else torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=self.device_)
        )

        ego_dict: dict[str, Any] = {
            "x_tab": x_t[:, :TABULAR_DIM],
            "x_topo": x_t[:, TABULAR_DIM : TABULAR_DIM + TOPO_DIM],
            "x_temp_stat": x_t[:, TABULAR_DIM + TOPO_DIM :],
            "temporal_sequence": seq_t[:1] if n_ego > 1 and seq_t.shape[0] == 1 else seq_t,
            "edge_index": e_idx,
            "edge_attr": e_attr,
            "batch_size": 1,
        }
        res = self.explain_ego_dict(ego_dict)
        return res

    @torch.no_grad()
    def explain_ego_dict(self, local_sub: dict[str, Any]) -> dict[str, Any]:
        """Run forward pass & branch attribution on an already-assembled 1-wallet ego dict."""
        assert self.net_ is not None
        self.net_.eval()
        use_amp_actual = bool(self.use_amp and self.device_.type == "cuda")
        with torch.amp.autocast(device_type=self.device_.type, enabled=use_amp_actual):
            out = self.net_(local_sub, batch_size=1)

        raw_single = {
            "feature": np.array([float(out["e_feature"][0].item())], dtype=np.float64),
            "graph": np.array([float(out["e_graph"][0].item())], dtype=np.float64),
            "temporal": np.array([float(out["e_temporal"][0].item())], dtype=np.float64),
        }
        cal_single = self._apply_iqr_calibration(raw_single)
        raw_score = float(self._combine_calibrated_errors(cal_single)[0])
        aux_prob = float(torch.sigmoid(out["aux_logits"][0]).item())

        w_feat_pos = self.alpha_ * max(float(cal_single["feature"][0]), 0.0) if self.use_tabular else 0.0
        w_graph_pos = self.beta_ * max(float(cal_single["graph"][0]), 0.0) if self.use_graph else 0.0
        w_temp_pos = self.gamma_ * max(float(cal_single["temporal"][0]), 0.0) if self.use_temporal else 0.0
        pos_sum = w_feat_pos + w_graph_pos + w_temp_pos
        if pos_sum <= 1e-12:
            w_feat_pos = self.alpha_ * float(raw_single["feature"][0]) if self.use_tabular else 0.0
            w_graph_pos = self.beta_ * float(raw_single["graph"][0]) if self.use_graph else 0.0
            w_temp_pos = self.gamma_ * float(raw_single["temporal"][0]) if self.use_temporal else 0.0
            pos_sum = w_feat_pos + w_graph_pos + w_temp_pos

        if pos_sum > 1e-12:
            feature_pct = 100.0 * w_feat_pos / pos_sum
            graph_pct = 100.0 * w_graph_pos / pos_sum
            temporal_pct = 100.0 * w_temp_pos / pos_sum
        else:
            feature_pct = 100.0 * self.alpha_
            graph_pct = 100.0 * self.beta_
            temporal_pct = 100.0 * self.gamma_

        tau = float(self.threshold_ if self.threshold_ is not None else 1.0)
        # Smooth [0, 1] display normalization centered so score == tau maps to 0.50
        norm_display_score = float(1.0 / (1.0 + np.exp(-0.35 * (raw_score - tau))))

        feature_errors_map: dict[str, float] = {}
        if self.use_tabular and out["err_tab_feat"] is not None:
            errs = out["err_tab_feat"][0].detach().cpu().numpy()
            for i, c in enumerate(TABULAR_TX_COLS):
                feature_errors_map[c] = float(errs[i])
        if self.use_graph and out["err_topo_feat"] is not None:
            errs = out["err_topo_feat"][0].detach().cpu().numpy()
            for i, c in enumerate(GRAPH_TOPO_COLS):
                feature_errors_map[c] = float(errs[i])
        if self.use_temporal and out["err_temp_stat_feat"] is not None:
            errs = out["err_temp_stat_feat"][0].detach().cpu().numpy()
            for i, c in enumerate(TEMPORAL_STAT_COLS):
                feature_errors_map[c] = float(errs[i])

        return {
            "raw_tgae_score": raw_score,
            "threshold": tau,
            "normalized_score": norm_display_score,
            "aux_probability": aux_prob,
            "is_anomaly": bool(raw_score >= tau),
            "latent_z": out["z"][0].detach().cpu().numpy(),
            "branch_contributions": {
                "feature_pct": round(float(feature_pct), 2),
                "graph_pct": round(float(graph_pct), 2),
                "temporal_pct": round(float(temporal_pct), 2),
            },
            "branch_errors_calibrated": {
                "feature": round(float(cal_single["feature"][0]), 4),
                "graph": round(float(cal_single["graph"][0]), 4),
                "temporal": round(float(cal_single["temporal"][0]), 4),
            },
            "feature_errors": feature_errors_map,
        }

    def save(self, path: str | Path) -> Path:
        """Persist trained ModularTGAE checkpoint, IQR calibration stats, and ``threshold_``."""
        if not self.is_fitted_ or self.net_ is None:
            raise RuntimeError("Cannot save an unfitted ModularTGAE detector.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class": "ModularTGAE",
            "params": {
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "dropout": self.dropout,
                "use_tabular": self.use_tabular,
                "use_graph": self.use_graph,
                "use_temporal": self.use_temporal,
                "graph_neighbor_samples": self.graph_neighbor_samples,
                "alpha_feature": self.alpha_feature,
                "beta_graph": self.beta_graph,
                "gamma_temporal": self.gamma_temporal,
                "contrastive_margin": self.contrastive_margin,
                "supervised_aux_weight": self.supervised_aux_weight,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "use_amp": self.use_amp,
                "tune_weights_on_val": self.tune_weights_on_val,
                "random_state": self.random_state,
                "seq_use_log1p": self.seq_use_log1p,
                "strict_unseen_val": self.strict_unseen_val,
                "config_path": None,
                "feature_names": self.feature_names_,
            },
            "state": {
                "net_state_dict": {k: v.cpu() for k, v in self.net_.state_dict().items()},
                "iqr_stats_": copy.deepcopy(self.iqr_stats_),
                "alpha_": self.alpha_,
                "beta_": self.beta_,
                "gamma_": self.gamma_,
                "threshold_": self.threshold_,
                "val_best_f1_": self.val_best_f1_,
                "best_val_pr_auc_": self.best_val_pr_auc_,
                "pos_weight_": self.pos_weight_,
                "is_fitted_": self.is_fitted_,
                "training_history_": self.training_history_,
            },
        }
        torch.save(payload, out_path)
        return out_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device | None = None,
    ) -> ModularTGAE:
        """Load a persisted ModularTGAE checkpoint from ``path``."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        params = dict(payload["params"])
        # Legacy checkpoints saved before seq_use_log1p was introduced used un-logged sequences
        params.setdefault("seq_use_log1p", False)
        if device is not None:
            params["device"] = device
        obj = cls(**params)
        obj.net_ = obj._build_net()
        obj.net_.load_state_dict(payload["state"]["net_state_dict"])
        obj.net_.to(obj.device_)
        obj.net_.eval()

        for k, v in payload["state"].items():
            if k != "net_state_dict":
                setattr(obj, k, v)
        return obj
