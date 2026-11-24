"""
ETHEREUM GUARD - TV3-02: Unified Scientific Evaluator & Error Case Analyzer.

Computes the full Master Plan metric suite (Section 7.2):
  - PR-AUC (Primary metric for imbalanced fraud detection)
  - ROC-AUC
  - F1-Score (Binary & Macro)
  - Precision, Recall
  - Precision@K (adaptive K = 10, 50, 100)
  - FPR@90%TPR
  - Inference Latency (ms/wallet)
  - Peak GPU VRAM (MB) & Peak Python RAM (MB)

Supports BOTH:
  - Cumulative evaluation (`eval_mask & y in {0, 1}`)
  - Strict Inductive / Unseen evaluation (`exclude_addresses=train_labeled_addresses` or `inductive_only=True`)
    to prevent cumulative-snapshot evaluation leakage.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models.base import BaseDetector
from src.models.data_utils import extract_labels, get_eval_mask_and_labels
from src.schemas import ALL_FEATURE_COLS

logger = logging.getLogger("ethereum_guard.evaluator")


def compute_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute Precision@K over the top-K highest anomaly scores."""
    n = int(y_true.size)
    if n == 0 or k <= 0:
        return 0.0
    eff_k = min(int(k), n)
    top_idx = np.argsort(-y_score)[:eff_k]
    return float(np.mean(y_true[top_idx] == 1))


def compute_fpr_at_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_tpr: float = 0.90,
) -> float:
    """Compute False Positive Rate at a target True Positive Rate (default: FPR@90%TPR)."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= target_tpr)[0]
    if idx.size == 0:
        return 1.0
    return float(fpr[idx[0]])


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    ks: tuple[int, ...] = (10, 50, 100),
) -> dict[str, float]:
    """Compute complete imbalanced-classification metric dictionary."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    preds = (y_score >= float(threshold)).astype(np.int64)

    has_both = len(np.unique(y_true)) == 2
    metrics: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_both else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both else 0.0,
        "f1_binary": float(f1_score(y_true, preds, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "fpr_at_90_tpr": compute_fpr_at_tpr(y_true, y_score, target_tpr=0.90),
        "threshold": float(threshold),
        "num_eval_nodes": int(y_true.size),
        "num_positives": int((y_true == 1).sum()),
        "num_negatives": int((y_true == 0).sum()),
        "tp": int(((preds == 1) & (y_true == 1)).sum()),
        "fp": int(((preds == 1) & (y_true == 0)).sum()),
        "tn": int(((preds == 0) & (y_true == 0)).sum()),
        "fn": int(((preds == 0) & (y_true == 1)).sum()),
    }
    for k in ks:
        metrics[f"precision_at_{k}"] = compute_precision_at_k(y_true, y_score, k)
    return metrics


def analyze_error_cases(
    model_name: str,
    split: str,
    addresses: list[str] | np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    features_raw: np.ndarray | None = None,
    feature_names: tuple[str, ...] | list[str] = ALL_FEATURE_COLS,
    is_inductive_holdout: np.ndarray | None = None,
    unseen_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Extract all False Positive (FP) and False Negative (FN) cases for Error Analysis (TV3-05).
    """
    preds = (y_score >= float(threshold)).astype(np.int64)
    rows: list[dict[str, Any]] = []

    for i in range(len(y_true)):
        yt = int(y_true[i])
        yp = int(preds[i])
        if yt == yp:
            continue
        error_type = "FALSE_POSITIVE" if (yp == 1 and yt == 0) else "FALSE_NEGATIVE"
        row: dict[str, Any] = {
            "model": model_name,
            "split": split,
            "address": str(addresses[i]),
            "error_type": error_type,
            "true_label": yt,
            "predicted_label": yp,
            "anomaly_score": round(float(y_score[i]), 6),
            "threshold": round(float(threshold), 6),
            "score_margin": round(float(y_score[i] - threshold), 6),
            "is_inductive_holdout": (
                bool(is_inductive_holdout[i]) if is_inductive_holdout is not None else False
            ),
            "is_strictly_unseen_from_train": (
                bool(unseen_mask[i]) if unseen_mask is not None else False
            ),
        }
        if features_raw is not None:
            key_cols = (
                "total_eth_sent",
                "total_eth_received",
                "tx_sent_count",
                "tx_recv_count",
                "in_degree",
                "out_degree",
                "local_ego_density",
                "burstiness_index",
            )
            name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            for col in key_cols:
                if col in name_to_idx:
                    row[col] = round(float(features_raw[i, name_to_idx[col]]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["error_type", "score_margin"], ascending=[True, False]
        ).reset_index(drop=True)
    return df


class UnifiedEvaluator:
    """
    Standardized evaluation harness enforcing Master Plan metrics, latency measurement,
    VRAM/RAM tracking, and strict anti-leakage reporting.
    """

    def __init__(self, ks: tuple[int, ...] = (10, 50, 100)) -> None:
        self.ks = ks

    def evaluate_detector(
        self,
        detector: BaseDetector,
        data: Any,
        model_name: str,
        split: str = "test",
        threshold: float | None = None,
        exclude_addresses: set[str] | None = None,
        inductive_only: bool = False,
        eval_protocol: str = "cumulative",
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate a BaseDetector with latency and memory profiling.

        Returns:
            (metrics_row_dict, eval_indices, y_true, scores)
        """
        mask, _ = get_eval_mask_and_labels(data)
        mask = detector._filter_eval_mask(
            data,
            mask,
            exclude_addresses=exclude_addresses,
            inductive_only=inductive_only,
        )
        eval_indices = np.where(mask)[0]
        y_true = extract_labels(data)[mask]

        tau = detector.threshold_ if threshold is None else float(threshold)
        if tau is None:
            raise RuntimeError(
                f"Detector {model_name} has no calibrated threshold_. Call calibrate_threshold(val_data) first."
            )

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        tracemalloc.start()
        t0 = time.perf_counter()
        scores = np.asarray(
            detector.predict_score(data, node_indices=eval_indices),
            dtype=np.float64,
        )
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_ram_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            if use_cuda
            else 0.0
        )
        peak_ram_mb = float(peak_ram_bytes) / (1024.0 * 1024.0)

        metrics = compute_binary_metrics(y_true, scores, threshold=tau, ks=self.ks)
        n_eval = max(1, int(eval_indices.size))
        row: dict[str, Any] = {
            "model": model_name,
            "split": split,
            "protocol": eval_protocol,
            **metrics,
            "total_inference_ms": round(elapsed_ms, 3),
            "inference_latency_ms_per_wallet": round(elapsed_ms / n_eval, 4),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "peak_ram_mb": round(peak_ram_mb, 2),
        }
        return row, eval_indices, y_true, scores
