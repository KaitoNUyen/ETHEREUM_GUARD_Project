"""Abstract base class and threshold calibration utilities for anomaly detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.data_utils import get_eval_mask_and_labels
from src.schemas import ALL_FEATURE_COLS


def find_optimal_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """Find threshold ``tau*`` maximizing binary F1-score on validation predictions.

    Returns:
        Tuple of ``(best_threshold, best_f1)``.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Length mismatch between y_true ({y_true.shape[0]}) and y_score ({y_score.shape[0]})."
        )
    if y_true.size == 0:
        raise ValueError("Cannot calibrate threshold on empty validation labels.")
    if not np.isfinite(y_score).all():
        raise ValueError("Validation scores contain NaN or Inf.")

    positives = int((y_true == 1).sum())
    if positives == 0 or positives == y_true.size or np.all(y_score == y_score[0]):
        fallback = float(np.median(y_score))
        preds = (y_score >= fallback).astype(np.int64)
        return fallback, float(f1_score(y_true, preds, zero_division=0))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        fallback = float(np.median(y_score))
        preds = (y_score >= fallback).astype(np.int64)
        return fallback, float(f1_score(y_true, preds, zero_division=0))

    p = precisions[:-1]
    r = recalls[:-1]
    denom = p + r
    f1_scores = np.divide(2.0 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


class BaseDetector(ABC):
    """Abstract base interface for all Ethereum Guard anomaly detectors."""

    def __init__(self, feature_names: tuple[str, ...] | list[str] = ALL_FEATURE_COLS) -> None:
        self.feature_names_: list[str] = list(feature_names)
        self.threshold_: float | None = None
        self.val_best_f1_: float | None = None
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(self, train_data: Any, val_data: Any | None = None) -> BaseDetector:
        """Fit detector on ``train_data`` and calibrate ``threshold_`` on ``val_data``."""

    @abstractmethod
    def predict_score(self, data: Any, node_indices: np.ndarray | None = None) -> np.ndarray:
        """Return 1D anomaly scores (higher = more anomalous).

        Shape is ``(N,)`` for all nodes, or ``(len(node_indices),)`` when ``node_indices`` is given.
        """

    def _filter_eval_mask(
        self,
        data: Any,
        mask: np.ndarray,
        exclude_addresses: set[str] | None = None,
        inductive_only: bool = False,
    ) -> np.ndarray:
        """Optionally restrict evaluation mask to unseen/inductive nodes (prevents cumulative leakage)."""
        filtered = mask.copy()
        if inductive_only:
            holdout = (
                data.get("is_inductive_holdout")
                if isinstance(data, dict)
                else getattr(data, "is_inductive_holdout", None)
            )
            if holdout is not None:
                holdout_np = (
                    holdout.detach().cpu().numpy()
                    if hasattr(holdout, "detach")
                    else np.asarray(holdout)
                ).astype(bool)
                filtered &= holdout_np

        if exclude_addresses:
            addrs = (
                data.get("addresses")
                if isinstance(data, dict)
                else getattr(data, "addresses", None)
            )
            if addrs is not None:
                lower_excl = {str(a).strip().lower() for a in exclude_addresses}
                keep = np.fromiter(
                    (str(a).strip().lower() not in lower_excl for a in addrs),
                    dtype=bool,
                    count=len(addrs),
                )
                filtered &= keep

        if not filtered.any():
            raise ValueError("No labeled evaluation nodes remain after applying leakage filters.")
        return filtered

    def calibrate_threshold(
        self,
        val_data: Any,
        scores: np.ndarray | None = None,
        exclude_addresses: set[str] | None = None,
        inductive_only: bool = False,
    ) -> float:
        """Tune ``threshold_`` to maximize F1 on Validation nodes (``eval_mask & y in {0, 1}``).

        Must ONLY be called with Validation data, never Test data.
        Pass ``exclude_addresses`` (e.g. train labeled addresses) or ``inductive_only=True``
        to prevent cumulative-snapshot evaluation leakage.
        """
        mask, _ = get_eval_mask_and_labels(val_data)
        mask = self._filter_eval_mask(
            val_data, mask, exclude_addresses=exclude_addresses, inductive_only=inductive_only
        )
        from src.models.data_utils import extract_labels

        y_val = extract_labels(val_data)[mask]
        if scores is None:
            # Score ONLY the labeled eval nodes (not the whole graph).
            val_scores = np.asarray(self.predict_score(val_data, node_indices=np.where(mask)[0]))
        else:
            val_scores = np.asarray(scores)
            if val_scores.shape[0] != mask.shape[0]:
                raise ValueError(
                    f"Score length ({val_scores.shape[0]}) does not match node count ({mask.shape[0]})."
                )
            val_scores = val_scores[mask]

        best_tau, best_f1 = find_optimal_f1_threshold(y_val, val_scores)
        self.threshold_ = best_tau
        self.val_best_f1_ = best_f1
        return best_tau

    def predict_label(
        self,
        data: Any,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Predict binary anomaly labels (0=Normal, 1=Anomaly) using ``threshold`` or ``threshold_``."""
        tau = self.threshold_ if threshold is None else float(threshold)
        if tau is None:
            raise RuntimeError(
                "Decision threshold is not set. Pass `threshold` or calibrate on validation data first."
            )
        scores = np.asarray(self.predict_score(data), dtype=np.float64)
        return (scores >= tau).astype(np.int64)

    def evaluate(
        self,
        data: Any,
        threshold: float | None = None,
        exclude_addresses: set[str] | None = None,
        inductive_only: bool = False,
    ) -> dict[str, float]:
        """Compute evaluation metrics strictly on ``eval_mask & (y in {0, 1})``.

        Pass ``exclude_addresses`` or ``inductive_only=True`` to report strictly unseen metrics.
        """
        mask, _ = get_eval_mask_and_labels(data)
        mask = self._filter_eval_mask(
            data, mask, exclude_addresses=exclude_addresses, inductive_only=inductive_only
        )
        from src.models.data_utils import extract_labels

        y_true = extract_labels(data)[mask]
        scores = np.asarray(
            self.predict_score(data, node_indices=np.where(mask)[0]), dtype=np.float64
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

    @abstractmethod
    def explain(self, data: Any, node_idx: int) -> dict[str, Any]:
        """Return per-feature explanation for a single node at ``node_idx``."""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist trained detector and calibrated ``threshold_`` to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseDetector:
        """Load a persisted detector from ``path``."""
