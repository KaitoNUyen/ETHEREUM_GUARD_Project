"""Strict Tri-Modal Encoders for Ethereum Guard (Task TV2-03).

Provides both native PyTorch Geometric execution (when ``torch_geometric`` is installed)
and an exact pure-PyTorch ``TransformerConv`` + CSC ``NeighborSampler`` fallback so checkpoints
like ``best_tgae.pt`` load, train, and infer identically on CPU-only environments.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.schemas import (
    EDGE_FEATURE_DIM,
    GRAPH_TOPO_COLS,
    TABULAR_TX_COLS,
    TEMPORAL_FEATURE_DIM,
    TEMPORAL_STAT_COLS,
    TEMPORAL_WINDOWS,
)

try:
    import torch_geometric.loader.utils as pyg_loader_utils
    import torch_geometric.typing
    from torch_geometric.data import Data as PyGData
    from torch_geometric.nn import TransformerConv as PyGTransformerConv
    from torch_geometric.sampler import NeighborSampler
    from torch_geometric.sampler.base import SamplerOutput

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    PyGData = None  # type: ignore[assignment,misc]
    PyGTransformerConv = None  # type: ignore[assignment,misc]

TABULAR_DIM: int = len(TABULAR_TX_COLS)
TOPO_DIM: int = len(GRAPH_TOPO_COLS)
TEMPORAL_STAT_DIM: int = len(TEMPORAL_STAT_COLS)


class PureTorchTransformerConv(nn.Module):
    """
    Exact parameter- and math-compatible pure-PyTorch implementation of
    ``torch_geometric.nn.TransformerConv(..., concat=False, beta=False, root_weight=True)``.

    Matches exact state_dict keys:
      - ``lin_key.{weight,bias}``
      - ``lin_query.{weight,bias}``
      - ``lin_value.{weight,bias}``
      - ``lin_edge.weight``
      - ``lin_skip.{weight,bias}``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.beta = bool(beta)
        self.dropout = float(dropout)
        self.edge_dim = edge_dim

        total_out = self.heads * self.out_channels
        self.lin_key = nn.Linear(self.in_channels, total_out, bias=True)
        self.lin_query = nn.Linear(self.in_channels, total_out, bias=True)
        self.lin_value = nn.Linear(self.in_channels, total_out, bias=True)
        if edge_dim is not None:
            self.lin_edge: nn.Linear | None = nn.Linear(int(edge_dim), total_out, bias=False)
        else:
            self.lin_edge = None

        skip_out = total_out if self.concat else self.out_channels
        self.lin_skip = nn.Linear(self.in_channels, skip_out, bias=True)
        if self.beta:
            self.lin_beta: nn.Linear | None = nn.Linear(3 * skip_out, 1, bias=False)
        else:
            self.lin_beta = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.lin_beta is not None:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = x.shape[0]
        H, C = self.heads, self.out_channels

        query = self.lin_query(x).view(N, H, C)
        key = self.lin_key(x).view(N, H, C)
        value = self.lin_value(x).view(N, H, C)

        out = x.new_zeros((N, H, C))
        if edge_index is not None and edge_index.numel() > 0:
            src = edge_index[0].long()
            dst = edge_index[1].long()

            key_j = key[src]      # (E, H, C)
            value_j = value[src]  # (E, H, C)
            query_i = query[dst]  # (E, H, C)

            if self.lin_edge is not None and edge_attr is not None:
                e_emb = self.lin_edge(edge_attr).view(-1, H, C)
                key_j = key_j + e_emb
                value_j = value_j + e_emb

            # Scaled dot-product attention per head: (E, H)
            alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(float(C))

            # Numerically stable segment softmax over incoming edges grouped by dst
            max_per_dst = alpha.new_full((N, H), -1e30)
            max_per_dst.scatter_reduce_(
                0,
                dst.unsqueeze(-1).expand(-1, H),
                alpha,
                reduce="amax",
                include_self=True,
            )
            alpha_exp = torch.exp(alpha - max_per_dst[dst])
            denom_per_dst = alpha.new_zeros((N, H))
            denom_per_dst.index_add_(0, dst, alpha_exp)
            attn = alpha_exp / denom_per_dst[dst].clamp_min(1e-16)
            attn = F.dropout(attn, p=self.dropout, training=self.training)

            msg = value_j * attn.unsqueeze(-1)  # (E, H, C)
            out.index_add_(0, dst, msg)

        if self.concat:
            out_flat = out.view(N, H * C)
        else:
            out_flat = out.mean(dim=1)

        x_r = self.lin_skip(x)
        if self.beta and self.lin_beta is not None:
            b = torch.sigmoid(self.lin_beta(torch.cat([out_flat, x_r, out_flat - x_r], dim=-1)))
            return b * x_r + (1.0 - b) * out_flat
        return out_flat + x_r


TransformerConvLayer = PyGTransformerConv if HAS_PYG else PureTorchTransformerConv


def _is_graph_container(obj: object) -> bool:
    if HAS_PYG and PyGData is not None and isinstance(obj, PyGData):
        return True
    return hasattr(obj, "x_topo") or hasattr(obj, "x_tab") or hasattr(obj, "x")


def _install_neighbor_sampler_fallback() -> None:
    """Provide pure-PyTorch/NumPy CSC neighbor sampling when pyg-lib/torch-sparse are absent."""
    if not HAS_PYG:
        return
    if not getattr(pyg_loader_utils, "_list_index_select_installed", False):
        orig_index_select = pyg_loader_utils.index_select

        def _index_select_with_list(
            value: object,
            index: torch.Tensor,
            dim: int = 0,
        ) -> object:
            if isinstance(value, list):
                idx_list = index.detach().cpu().tolist()
                return [value[i] for i in idx_list]
            if isinstance(value, tuple):
                idx_list = index.detach().cpu().tolist()
                return tuple(value[i] for i in idx_list)
            return orig_index_select(value, index, dim=dim)

        pyg_loader_utils.index_select = _index_select_with_list  # type: ignore[assignment]
        pyg_loader_utils._list_index_select_installed = True  # type: ignore[attr-defined]

    if torch_geometric.typing.WITH_PYG_LIB or torch_geometric.typing.WITH_TORCH_SPARSE:
        return
    if getattr(NeighborSampler, "_pure_torch_fallback_installed", False):
        return

    orig_sample = NeighborSampler._sample

    def _fallback_sample(
        self: Any,
        seed: torch.Tensor | dict,
        seed_time: torch.Tensor | dict | None = None,
        **kwargs: object,
    ) -> Any:
        if isinstance(seed, dict):
            return orig_sample(self, seed, seed_time=seed_time, **kwargs)

        colptr_np = self.colptr.detach().cpu().numpy()
        row_np = self.row.detach().cpu().numpy()
        seed_np = seed.detach().cpu().numpy().astype(np.int64, copy=False)

        nodes_list: list[int] = seed_np.tolist()
        node_to_local: dict[int, int] = {int(u): idx for idx, u in enumerate(nodes_list)}

        frontier = seed_np
        frontier_local = np.arange(len(nodes_list), dtype=np.int64)

        edge_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        src_global_chunks: list[np.ndarray] = []
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))

        for num_neigh in self.num_neighbors.get_mapped_values():
            req = int(num_neigh)
            if req == 0 or frontier.size == 0:
                break

            starts = colptr_np[frontier]
            ends = colptr_np[frontier + 1]
            degs = ends - starts

            hop_edges: list[np.ndarray] = []
            hop_cols: list[np.ndarray] = []

            for i in range(frontier.size):
                d = int(degs[i])
                if d <= 0:
                    continue
                s = int(starts[i])
                e = int(ends[i])
                if req < 0 or (not self.replace and d <= req):
                    chosen = np.arange(s, e, dtype=np.int64)
                elif self.replace:
                    chosen = s + rng.integers(0, d, size=req, dtype=np.int64)
                else:
                    chosen = s + rng.choice(d, size=req, replace=False).astype(np.int64)
                hop_edges.append(chosen)
                hop_cols.append(np.full(chosen.size, frontier_local[i], dtype=np.int64))

            if not hop_edges:
                break

            e_arr = np.concatenate(hop_edges)
            c_arr = np.concatenate(hop_cols)
            s_global = row_np[e_arr]

            edge_chunks.append(e_arr)
            col_chunks.append(c_arr)
            src_global_chunks.append(s_global)

            new_frontier: list[int] = []
            new_frontier_local: list[int] = []
            for g in s_global:
                gi = int(g)
                if gi not in node_to_local:
                    loc = len(nodes_list)
                    node_to_local[gi] = loc
                    nodes_list.append(gi)
                    new_frontier.append(gi)
                    new_frontier_local.append(loc)

            frontier = np.asarray(new_frontier, dtype=np.int64)
            frontier_local = np.asarray(new_frontier_local, dtype=np.int64)

        if edge_chunks:
            all_edges = np.concatenate(edge_chunks)
            all_cols = np.concatenate(col_chunks)
            all_src_global = np.concatenate(src_global_chunks)
            all_rows = np.fromiter(
                (node_to_local[int(g)] for g in all_src_global),
                dtype=np.int64,
                count=all_src_global.size,
            )
        else:
            all_edges = np.empty(0, dtype=np.int64)
            all_cols = np.empty(0, dtype=np.int64)
            all_rows = np.empty(0, dtype=np.int64)

        if self.sample_direction == "backward":
            all_rows, all_cols = all_cols, all_rows

        return SamplerOutput(
            node=torch.from_numpy(np.asarray(nodes_list, dtype=np.int64)),
            row=torch.from_numpy(all_rows),
            col=torch.from_numpy(all_cols),
            edge=torch.from_numpy(all_edges),
            batch=None,
            num_sampled_nodes=None,
            num_sampled_edges=None,
        )

    NeighborSampler._sample = _fallback_sample  # type: ignore[method-assign]
    NeighborSampler._pure_torch_fallback_installed = True  # type: ignore[attr-defined]


_install_neighbor_sampler_fallback()


class TabularMLP(nn.Module):
    """Tabular encoder operating strictly on ``x_tab`` of shape ``(B, 12)``."""

    def __init__(
        self,
        in_dim: int = TABULAR_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x_tab: Any,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(x_tab, torch.Tensor):
            if batch_size is None:
                batch_size = getattr(x_tab, "batch_size", None)
            x_tab = x_tab.x_tab
        if batch_size is not None:
            x_tab = x_tab[:batch_size]
        if x_tab.ndim != 2 or x_tab.shape[1] != self.in_dim:
            raise ValueError(
                f"TabularMLP expected input shape (B, {self.in_dim}), got {tuple(x_tab.shape)}"
            )
        return self.net(x_tab)


class EdgeTransformerEncoder(nn.Module):
    """2-layer Multi-Head Edge-Aware Graph Transformer encoder (``TransformerConv``)
    paired with GraphSAGE-style 2-hop ``NeighborLoader`` sampling.

    Strict Ablation contract:
      - ``use_tabular=False, use_graph=True`` (Only-Graph Abl-B): consumes ONLY ``x_topo`` ``(N, 6)``.
      - ``use_tabular=True, use_graph=True``: consumes ``[x_tab || x_topo]`` ``(N, 18)``.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_tabular: bool = True,
        use_graph: bool = True,
        edge_dim: int = EDGE_FEATURE_DIM,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if not use_graph:
            raise ValueError("EdgeTransformerEncoder requires use_graph=True.")

        self.use_tabular = bool(use_tabular)
        self.use_graph = bool(use_graph)
        self.in_dim = (TABULAR_DIM + TOPO_DIM) if self.use_tabular else TOPO_DIM
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.dropout = float(dropout)
        self.heads = int(heads)

        self.conv1 = TransformerConvLayer(
            in_channels=self.in_dim,
            out_channels=self.hidden_dim,
            heads=self.heads,
            concat=False,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(self.dropout)

        self.conv2 = TransformerConvLayer(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.heads,
            concat=False,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
        )
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.act2 = nn.GELU()

    def _assemble_node_features(
        self,
        x_or_data: Any,
        x_tab: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(x_or_data, torch.Tensor):
            x_topo = x_or_data.x_topo
            if not self.use_tabular:
                return x_topo
            return torch.cat([x_or_data.x_tab, x_topo], dim=-1)

        if not self.use_tabular:
            if x_tab is not None:
                raise ValueError("use_tabular=False prohibits passing x_tab to EdgeTransformerEncoder.")
            return x_or_data

        if x_tab is not None:
            return torch.cat([x_tab, x_or_data], dim=-1)
        return x_or_data

    def forward(
        self,
        x_or_data: Any,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        x_tab: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(x_or_data, torch.Tensor):
            if edge_index is None:
                edge_index = x_or_data.edge_index
            if edge_attr is None:
                edge_attr = getattr(x_or_data, "edge_attr", None)
            if batch_size is None:
                batch_size = getattr(x_or_data, "batch_size", None)

        if edge_index is None:
            raise ValueError("EdgeTransformerEncoder requires edge_index.")

        x = self._assemble_node_features(x_or_data, x_tab=x_tab)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(
                f"EdgeTransformerEncoder (use_tabular={self.use_tabular}) expected node features "
                f"of shape (N, {self.in_dim}), got {tuple(x.shape)}"
            )

        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.shape[1], self.edge_dim))
        elif edge_attr.ndim != 2 or edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                f"EdgeTransformerEncoder expected edge_attr shape (E, {self.edge_dim}), got {tuple(edge_attr.shape)}"
            )

        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.drop1(self.act1(self.norm1(h)))
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.act2(self.norm2(h))

        if batch_size is not None:
            h = h[:batch_size]
        return h


# Backwards-compatible alias preserving Master Plan & checkpoint class import contract
GraphSAGEEncoder = EdgeTransformerEncoder


class TemporalAttentionGRU(nn.Module):
    """2-layer GRU with Additive Temporal Attention over 10 windows fused with ``x_temp_stat``."""

    def __init__(
        self,
        seq_dim: int = TEMPORAL_FEATURE_DIM,
        stat_dim: int = TEMPORAL_STAT_DIM,
        num_windows: int = TEMPORAL_WINDOWS,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_dim = int(seq_dim)
        self.stat_dim = int(stat_dim)
        self.num_windows = int(num_windows)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.gru = nn.GRU(
            input_size=self.seq_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        self.attn_score = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )

        self.stat_mlp = nn.Sequential(
            nn.Linear(self.stat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.last_attn_weights_: torch.Tensor | None = None

    def forward(
        self,
        temporal_sequence: Any,
        x_temp_stat: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(temporal_sequence, torch.Tensor):
            data = temporal_sequence
            if batch_size is None:
                batch_size = getattr(data, "batch_size", None)
            x_temp_stat = data.x_temp_stat
            temporal_sequence = data.temporal_sequence

        if x_temp_stat is None:
            raise ValueError("TemporalAttentionGRU requires x_temp_stat of shape (B, 7).")

        if batch_size is not None:
            temporal_sequence = temporal_sequence[:batch_size]
            x_temp_stat = x_temp_stat[:batch_size]

        if (
            temporal_sequence.ndim != 3
            or temporal_sequence.shape[1] != self.num_windows
            or temporal_sequence.shape[2] != self.seq_dim
        ):
            raise ValueError(
                f"Expected temporal_sequence shape (B, {self.num_windows}, {self.seq_dim}), "
                f"got {tuple(temporal_sequence.shape)}"
            )
        if x_temp_stat.ndim != 2 or x_temp_stat.shape[1] != self.stat_dim:
            raise ValueError(
                f"Expected x_temp_stat shape (B, {self.stat_dim}), got {tuple(x_temp_stat.shape)}"
            )
        if temporal_sequence.shape[0] != x_temp_stat.shape[0]:
            raise ValueError(
                f"Batch size mismatch between temporal_sequence ({temporal_sequence.shape[0]}) "
                f"and x_temp_stat ({x_temp_stat.shape[0]})."
            )

        gru_out, _ = self.gru(temporal_sequence)
        attn_logits = self.attn_score(gru_out)
        attn_weights = torch.softmax(attn_logits, dim=1)
        self.last_attn_weights_ = attn_weights.squeeze(-1).detach()

        context = torch.sum(attn_weights * gru_out, dim=1)
        stat_emb = self.stat_mlp(x_temp_stat)

        fused = torch.cat([context, stat_emb], dim=-1)
        return self.out_proj(fused)
