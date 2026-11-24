"""Model definitions and data loading utilities for Ethereum Guard."""

from __future__ import annotations

from typing import Any

from src.models.base import BaseDetector, find_optimal_f1_threshold
from src.models.baselines import IsolationForestDetector, XGBoostDetector
from src.models.data_utils import (
    extract_labels,
    extract_static_features,
    get_eval_mask_and_labels,
    load_all_splits,
    load_config,
    load_split_data,
    load_temporal_data,
)

_PYG_EXPORTS = {
    "DOMINANTDetector": ("src.models.deep_baselines", "DOMINANTDetector"),
    "DeepTransactionScorer": ("src.models.tx_scorer", "DeepTransactionScorer"),
    "EdgeTransformerEncoder": ("src.models.encoders", "EdgeTransformerEncoder"),
    "GATDetector": ("src.models.deep_baselines", "GATDetector"),
    "GraphSAGEEncoder": ("src.models.encoders", "GraphSAGEEncoder"),
    "TabularMLP": ("src.models.encoders", "TabularMLP"),
    "TemporalAttentionGRU": ("src.models.encoders", "TemporalAttentionGRU"),
    "ModularTGAE": ("src.models.tgae", "ModularTGAE"),
    "ModularTGAENet": ("src.models.tgae", "ModularTGAENet"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load PyTorch Geometric model classes so CPU/tabular callers work without torch_geometric."""
    if name in _PYG_EXPORTS:
        module_name, attr_name = _PYG_EXPORTS[name]
        import importlib

        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseDetector",
    "GraphSAGEEncoder",
    "IsolationForestDetector",
    "ModularTGAE",
    "ModularTGAENet",
    "TabularMLP",
    "TemporalAttentionGRU",
    "XGBoostDetector",
    "extract_labels",
    "extract_static_features",
    "find_optimal_f1_threshold",
    "get_eval_mask_and_labels",
    "load_all_splits",
    "load_config",
    "load_split_data",
    "load_temporal_data",
]
