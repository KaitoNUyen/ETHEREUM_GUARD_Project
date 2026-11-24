"""Scientific Evaluation & Error Analysis Package (Task TV3-02)."""

from src.evaluation.evaluator import (
    UnifiedEvaluator,
    analyze_error_cases,
    compute_binary_metrics,
    compute_fpr_at_tpr,
    compute_precision_at_k,
)

__all__ = [
    "UnifiedEvaluator",
    "analyze_error_cases",
    "compute_binary_metrics",
    "compute_fpr_at_tpr",
    "compute_precision_at_k",
]
