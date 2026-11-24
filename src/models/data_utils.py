"""Utilities for loading and validating processed Ethereum Guard datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

if TYPE_CHECKING:
    from torch_geometric.data import Data

from src.schemas import (
    ALL_FEATURE_COLS,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    SPLITS,
    TEMPORAL_FEATURE_DIM,
    TEMPORAL_WINDOWS,
)


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_graph_path(processed_dir: Path, split: str) -> Path:
    candidates = [
        processed_dir / "graphs" / split / f"snapshot_{split}.pt",
        processed_dir / "graphs" / split / "graph.pt",
        processed_dir / f"snapshot_{split}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Graph snapshot for split '{split}' not found. Tried: {[str(c) for c in candidates]}"
    )


def _resolve_temporal_path(processed_dir: Path, split: str) -> Path | None:
    candidates = [
        processed_dir / "features" / split / "temporal.pt",
        processed_dir / "features" / split / f"temporal_{split}.pt",
        processed_dir / "temporal" / split / "temporal.pt",
        processed_dir / "temporal" / split / f"temporal_{split}.pt",
        processed_dir / "temporal" / f"temporal_{split}.pt",
        processed_dir / f"temporal_{split}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_nodes_parquet_path(processed_dir: Path, split: str) -> Path | None:
    # Prefer graphs/{split}/nodes.parquet because it has all 25 features (including topo) populated.
    candidates = [
        processed_dir / "graphs" / split / "nodes.parquet",
        processed_dir / "features" / split / "nodes.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_temporal_data(
    temporal_path: str | Path,
    expected_num_nodes: int | None = None,
    expected_node_ids: torch.Tensor | None = None,
    expected_addresses: list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load temporal tensor from dict (``temporal_data['sequence']``) or direct Tensor.

    Returns:
        Tuple of ``(sequence_tensor, metadata_dict)`` where ``sequence_tensor`` is
        ``torch.float32`` with shape ``(N, TEMPORAL_WINDOWS, TEMPORAL_FEATURE_DIM)``.
    """
    path = Path(temporal_path)
    if not path.exists():
        raise FileNotFoundError(f"Temporal file not found: {path}")

    raw = torch.load(path, map_location="cpu", weights_only=False)

    metadata: dict[str, Any] = {}
    if isinstance(raw, dict):
        if "sequence" not in raw:
            raise KeyError(
                f"Temporal dict at {path} missing 'sequence' key; keys={list(raw.keys())}"
            )
        seq = raw["sequence"]
        metadata = {k: v for k, v in raw.items() if k != "sequence"}
    elif isinstance(raw, torch.Tensor):
        seq = raw
    else:
        raise TypeError(f"Unsupported temporal data type in {path}: {type(raw)}")

    if not isinstance(seq, torch.Tensor):
        seq = torch.as_tensor(seq)
    seq = seq.to(dtype=torch.float32)

    if seq.ndim != 3 or seq.shape[1:] != (TEMPORAL_WINDOWS, TEMPORAL_FEATURE_DIM):
        raise ValueError(
            f"Invalid temporal sequence shape {tuple(seq.shape)} in {path}; "
            f"expected (N, {TEMPORAL_WINDOWS}, {TEMPORAL_FEATURE_DIM})"
        )

    num_nodes = seq.shape[0]
    if expected_num_nodes is not None and num_nodes != expected_num_nodes:
        raise ValueError(
            f"Node count mismatch between graph ({expected_num_nodes}) "
            f"and temporal sequence ({num_nodes}) in {path}"
        )

    if expected_node_ids is not None and "node_ids" in metadata:
        temp_ids = torch.as_tensor(metadata["node_ids"], dtype=expected_node_ids.dtype)
        if temp_ids.shape != expected_node_ids.shape or not torch.equal(temp_ids, expected_node_ids):
            raise ValueError(f"node_ids order/values mismatch between graph and {path}")

    if expected_addresses is not None and "addresses" in metadata:
        temp_addrs = list(metadata["addresses"])
        if len(temp_addrs) != len(expected_addresses) or temp_addrs != list(expected_addresses):
            raise ValueError(f"addresses order/values mismatch between graph and {path}")

    return seq, metadata


def _verify_nodes_parquet_alignment(
    parquet_path: Path,
    expected_num_nodes: int,
    expected_node_ids: torch.Tensor | None,
    expected_addresses: list[str] | None,
) -> None:
    """Verify node_id and address alignment against nodes.parquet without loading all columns."""
    schema_cols = set(pq.read_schema(parquet_path).names)
    cols_to_read = [c for c in ("node_id", "address") if c in schema_cols]
    if not cols_to_read:
        return

    table = pq.read_table(parquet_path, columns=cols_to_read)
    if table.num_rows != expected_num_nodes:
        raise ValueError(
            f"Node count mismatch between graph ({expected_num_nodes}) "
            f"and {parquet_path} ({table.num_rows})"
        )

    if expected_node_ids is not None and "node_id" in cols_to_read:
        pq_ids = torch.from_numpy(table["node_id"].to_numpy(zero_copy_only=False)).to(
            dtype=expected_node_ids.dtype
        )
        if not torch.equal(pq_ids, expected_node_ids):
            raise ValueError(f"node_id alignment mismatch between graph and {parquet_path}")

    if expected_addresses is not None and "address" in cols_to_read:
        pq_addrs = table["address"].to_pylist()
        if pq_addrs != list(expected_addresses):
            raise ValueError(f"address alignment mismatch between graph and {parquet_path}")


def _validate_feature_columns(data: Data) -> None:
    if hasattr(data, "feature_columns") and data.feature_columns is not None:
        cols = tuple(data.feature_columns)
        if cols != ALL_FEATURE_COLS:
            raise ValueError(
                f"Feature columns mismatch: expected {ALL_FEATURE_COLS}, got {cols}"
            )
    if data.x.ndim != 2 or data.x.shape[1] != len(ALL_FEATURE_COLS):
        raise ValueError(
            f"Expected data.x shape (N, {len(ALL_FEATURE_COLS)}), got {tuple(data.x.shape)}"
        )


class GraphSnapshotFallback:
    """Lightweight attribute container unpacking PyG Data `.pt` files when torch_geometric is absent."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setstate__(self, state: dict[str, Any]) -> None:
        store = state.get("_store")
        if store is not None and hasattr(store, "__dict__"):
            mapping = store.__dict__.get("_mapping", store.__dict__)
            if isinstance(mapping, dict):
                for k, v in mapping.items():
                    setattr(self, k, v)
                return
        self.__dict__.update(state)

    def to(self, device: str | torch.device) -> GraphSnapshotFallback:
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


def _load_pyg_pt_file(graph_path: Path) -> Any:
    """Load a PyG `.pt` snapshot natively if torch_geometric is installed, else via safe fallback unpickler."""
    try:
        from torch_geometric.data import Data as PyGData

        data = torch.load(graph_path, map_location="cpu", weights_only=False)
        if not isinstance(data, PyGData):
            raise TypeError(f"Expected torch_geometric.data.Data in {graph_path}, got {type(data)}")
        return data
    except ImportError:
        import pickle

        class _StubUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if module.startswith("torch_geometric"):
                    return GraphSnapshotFallback
                return super().find_class(module, name)

        fake_pickle = type(
            "_FakePickle",
            (),
            {"Unpickler": _StubUnpickler, "__name__": "pickle", "loads": pickle.loads},
        )
        return torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False,
            pickle_module=fake_pickle,
        )


def load_split_data(
    split: str,
    processed_dir: str | Path = "data/processed",
    load_temporal: bool = True,
    verify_alignment: bool = True,
    verify_parquet: bool = True,
    device: str | torch.device | None = None,
) -> Any:
    """Load a processed split (``train``, ``val``, or ``test``) as a PyG ``Data`` (or fallback container).

    Validates alignment across the graph snapshot, static 25-dim features,
    temporal sequence ``(N, 10, 8)``, and ``nodes.parquet``.
    """
    if split not in SPLITS:
        raise ValueError(f"Invalid split '{split}'. Expected one of {SPLITS}.")

    base_dir = Path(processed_dir)
    graph_path = _resolve_graph_path(base_dir, split)
    data = _load_pyg_pt_file(graph_path)

    num_nodes = int(data.num_nodes if data.num_nodes is not None else data.x.shape[0])
    if data.x.shape[0] != num_nodes:
        raise ValueError(
            f"Inconsistent num_nodes ({num_nodes}) and data.x rows ({data.x.shape[0]}) in {graph_path}"
        )
    if data.y is not None and data.y.shape[0] != num_nodes:
        raise ValueError(
            f"Inconsistent num_nodes ({num_nodes}) and data.y rows ({data.y.shape[0]}) in {graph_path}"
        )

    _validate_feature_columns(data)

    expected_node_ids = getattr(data, "node_id", None)
    expected_addresses = list(data.addresses) if getattr(data, "addresses", None) is not None else None

    if expected_node_ids is not None and expected_node_ids.shape[0] != num_nodes:
        raise ValueError("data.node_id length does not match num_nodes")
    if expected_addresses is not None and len(expected_addresses) != num_nodes:
        raise ValueError("data.addresses length does not match num_nodes")

    if load_temporal:
        temporal_path = _resolve_temporal_path(base_dir, split)
        if temporal_path is not None:
            seq, _ = load_temporal_data(
                temporal_path,
                expected_num_nodes=num_nodes,
                expected_node_ids=expected_node_ids if verify_alignment else None,
                expected_addresses=expected_addresses if verify_alignment else None,
            )
            if verify_alignment and hasattr(data, "temporal_sequence") and data.temporal_sequence is not None:
                if data.temporal_sequence.shape != seq.shape:
                    raise ValueError(
                        f"Snapshot temporal_sequence shape {tuple(data.temporal_sequence.shape)} "
                        f"differs from {temporal_path} shape {tuple(seq.shape)}"
                    )
            data.temporal_sequence = seq
        elif not hasattr(data, "temporal_sequence") or data.temporal_sequence is None:
            if hasattr(data, "temporal_seq") and data.temporal_seq is not None:
                data.temporal_sequence = data.temporal_seq.to(dtype=torch.float32)
            else:
                raise FileNotFoundError(
                    f"No temporal file found for split '{split}' and snapshot has no temporal_sequence."
                )

    if verify_alignment and verify_parquet:
        parquet_path = _resolve_nodes_parquet_path(base_dir, split)
        if parquet_path is not None:
            _verify_nodes_parquet_alignment(
                parquet_path,
                expected_num_nodes=num_nodes,
                expected_node_ids=expected_node_ids,
                expected_addresses=expected_addresses,
            )

    if device is not None:
        data = data.to(device)

    return data


def load_all_splits(
    processed_dir: str | Path = "data/processed",
    load_temporal: bool = True,
    verify_alignment: bool = True,
    verify_parquet: bool = True,
    device: str | torch.device | None = None,
) -> dict[str, Data]:
    """Load ``train``, ``val``, and ``test`` splits into a dictionary."""
    return {
        split: load_split_data(
            split=split,
            processed_dir=processed_dir,
            load_temporal=load_temporal,
            verify_alignment=verify_alignment,
            verify_parquet=verify_parquet,
            device=device,
        )
        for split in SPLITS
    }


def extract_static_features(data: Any) -> np.ndarray:
    """Extract the 25 static node features as a contiguous ``float32`` numpy array."""
    if isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
    elif isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy().astype(np.float32, copy=False)
    elif isinstance(data, dict) and "x" in data:
        x = data["x"]
        arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        arr = arr.astype(np.float32, copy=False)
    elif hasattr(data, "x") and data.x is not None:
        x = data.x
        arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        arr = arr.astype(np.float32, copy=False)
    else:
        raise TypeError(f"Cannot extract static features from type {type(data)}")

    if arr.ndim != 2 or arr.shape[1] != len(ALL_FEATURE_COLS):
        raise ValueError(
            f"Expected static features shape (N, {len(ALL_FEATURE_COLS)}), got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def extract_labels(data: Any) -> np.ndarray:
    """Extract 1D integer node labels ``y``."""
    if isinstance(data, dict) and "y" in data:
        y = data["y"]
    elif hasattr(data, "y") and data.y is not None:
        y = data.y
    else:
        raise ValueError("Data object does not contain labels 'y'.")
    arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    return arr.astype(np.int64, copy=False).reshape(-1)


def get_eval_mask_and_labels(data: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean mask (``eval_mask & (y in {0, 1})``) and binary labels on that mask."""
    y = extract_labels(data)
    if isinstance(data, dict) and "eval_mask" in data and data["eval_mask"] is not None:
        raw_mask = data["eval_mask"]
        eval_mask = (
            raw_mask.detach().cpu().numpy()
            if isinstance(raw_mask, torch.Tensor)
            else np.asarray(raw_mask)
        ).astype(bool)
    elif hasattr(data, "eval_mask") and data.eval_mask is not None:
        raw_mask = data.eval_mask
        eval_mask = (
            raw_mask.detach().cpu().numpy()
            if isinstance(raw_mask, torch.Tensor)
            else np.asarray(raw_mask)
        ).astype(bool)
    else:
        eval_mask = np.ones_like(y, dtype=bool)

    labeled_mask = (y == LABEL_NORMAL) | (y == LABEL_ANOMALY)
    mask = eval_mask & labeled_mask
    if not mask.any():
        raise ValueError("No valid labeled evaluation samples (eval_mask & y in {0, 1}).")
    return mask, y[mask]


def get_node_address(data: Any, node_idx: int) -> str | None:
    """Return address string for ``node_idx`` if available on ``data``."""
    addrs = None
    if isinstance(data, dict):
        addrs = data.get("addresses")
    elif hasattr(data, "addresses"):
        addrs = data.addresses
    if addrs is None or node_idx < 0 or node_idx >= len(addrs):
        return None
    return str(addrs[node_idx])
