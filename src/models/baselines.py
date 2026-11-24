"""Tabular baseline detectors: IsolationForest and XGBoost."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

from src.models.base import BaseDetector
from src.models.data_utils import (
    extract_labels,
    extract_static_features,
    get_eval_mask_and_labels,
    get_node_address,
)
from src.schemas import ALL_FEATURE_COLS, LABEL_ANOMALY, LABEL_NORMAL, RANDOM_SEED


class IsolationForestDetector(BaseDetector):
    """Unsupervised Isolation Forest trained strictly on Normal (``y == 0``) accounts."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: str | int | float = "auto",
        contamination: str | float = "auto",
        max_features: float = 1.0,
        random_state: int = RANDOM_SEED,
        n_jobs: int = -1,
        batch_size: int = 250_000,
        feature_names: tuple[str, ...] | list[str] = ALL_FEATURE_COLS,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.batch_size = int(batch_size)

        self.model_: IsolationForest | None = None
        self.normal_mean_: np.ndarray | None = None
        self.normal_std_: np.ndarray | None = None
        self.normal_median_: np.ndarray | None = None
        self.normal_iqr_: np.ndarray | None = None
        self.n_train_normal_: int = 0

    def fit(self, train_data: Any, val_data: Any | None = None) -> IsolationForestDetector:
        """Fit IsolationForest on ``y == 0`` train nodes and calibrate ``threshold_`` on ``val_data``."""
        X_train = extract_static_features(train_data)
        y_train = extract_labels(train_data)

        normal_mask = y_train == LABEL_NORMAL
        if not normal_mask.any():
            raise ValueError("IsolationForestDetector requires at least one Normal (y == 0) sample in train_data.")

        X_normal = X_train[normal_mask]
        self.n_train_normal_ = int(X_normal.shape[0])

        # Store Normal baseline distribution statistics for explain().
        self.normal_mean_ = np.mean(X_normal, axis=0, dtype=np.float64)
        self.normal_std_ = np.std(X_normal, axis=0, dtype=np.float64)
        self.normal_median_ = np.median(X_normal, axis=0).astype(np.float64)
        q75 = np.percentile(X_normal, 75, axis=0).astype(np.float64)
        q25 = np.percentile(X_normal, 25, axis=0).astype(np.float64)
        self.normal_iqr_ = q75 - q25

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_normal)
        self.is_fitted_ = True

        if val_data is not None:
            self.calibrate_threshold(val_data)

        return self

    def predict_score(self, data: Any, node_indices: np.ndarray | None = None) -> np.ndarray:
        """Return normalized anomaly scores in ``[0, 1]`` where higher means more anomalous."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("IsolationForestDetector is not fitted yet.")

        X = extract_static_features(data)
        if node_indices is not None:
            X = X[np.asarray(node_indices, dtype=np.int64)]
        n = X.shape[0]
        scores = np.empty(n, dtype=np.float64)

        # Chunk scoring to bound peak memory on million-node snapshots.
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            # sklearn score_samples returns -2^{-E(h)/c(n)} in [-1, 0]; negating gives [0, 1].
            scores[start:end] = np.clip(-self.model_.score_samples(X[start:end]), 0.0, 1.0)

        return scores

    def explain(self, data: Any, node_idx: int) -> dict[str, Any]:
        """Explain anomaly score at ``node_idx`` via per-feature Z-score and IQR deviation."""
        if not self.is_fitted_ or self.normal_mean_ is None:
            raise RuntimeError("IsolationForestDetector is not fitted yet.")

        X = extract_static_features(data)
        if node_idx < 0 or node_idx >= X.shape[0]:
            raise IndexError(f"node_idx {node_idx} out of range [0, {X.shape[0]})")

        x_row = X[node_idx].astype(np.float64)
        score = float(self.predict_score(X[node_idx : node_idx + 1])[0])

        z_scores = np.divide(
            x_row - self.normal_mean_,
            self.normal_std_,
            out=np.zeros_like(x_row),
            where=self.normal_std_ > 1e-12,
        )

        # Use IQR when non-zero; fall back to std when IQR collapses on sparse features.
        iqr_scale = np.where(self.normal_iqr_ > 1e-12, self.normal_iqr_, self.normal_std_)
        iqr_devs = np.divide(
            x_row - self.normal_median_,
            iqr_scale,
            out=np.zeros_like(x_row),
            where=iqr_scale > 1e-12,
        )

        feature_values = {
            name: float(x_row[i]) for i, name in enumerate(self.feature_names_)
        }
        z_score_map = {
            name: float(z_scores[i]) for i, name in enumerate(self.feature_names_)
        }
        iqr_dev_map = {
            name: float(iqr_devs[i]) for i, name in enumerate(self.feature_names_)
        }

        order = np.argsort(-np.maximum(np.abs(z_scores), np.abs(iqr_devs)))
        top_features = [
            {
                "feature": self.feature_names_[i],
                "value": float(x_row[i]),
                "z_score": float(z_scores[i]),
                "iqr_deviation": float(iqr_devs[i]),
                "normal_mean": float(self.normal_mean_[i]),
                "normal_median": float(self.normal_median_[i]),
            }
            for i in order
        ]

        predicted_label = (
            int(score >= self.threshold_) if self.threshold_ is not None else None
        )

        return {
            "model": "IsolationForestDetector",
            "node_idx": int(node_idx),
            "address": get_node_address(data, node_idx),
            "score": score,
            "threshold": self.threshold_,
            "predicted_label": predicted_label,
            "feature_values": feature_values,
            "z_scores": z_score_map,
            "iqr_deviations": iqr_dev_map,
            "top_features": top_features,
        }

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class": "IsolationForestDetector",
            "params": {
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "contamination": self.contamination,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "batch_size": self.batch_size,
                "feature_names": self.feature_names_,
            },
            "state": {
                "model_": self.model_,
                "threshold_": self.threshold_,
                "val_best_f1_": self.val_best_f1_,
                "is_fitted_": self.is_fitted_,
                "normal_mean_": self.normal_mean_,
                "normal_std_": self.normal_std_,
                "normal_median_": self.normal_median_,
                "normal_iqr_": self.normal_iqr_,
                "n_train_normal_": self.n_train_normal_,
            },
        }
        joblib.dump(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> IsolationForestDetector:
        payload = joblib.load(Path(path))
        obj = cls(**payload["params"])
        for k, v in payload["state"].items():
            setattr(obj, k, v)
        return obj


class XGBoostDetector(BaseDetector):
    """Supervised XGBoost binary classifier on the 25 static node features."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        early_stopping_rounds: int = 30,
        eval_metric: str = "aucpr",
        random_state: int = RANDOM_SEED,
        n_jobs: int = -1,
        batch_size: int = 500_000,
        feature_names: tuple[str, ...] | list[str] = ALL_FEATURE_COLS,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.eval_metric = str(eval_metric)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.batch_size = int(batch_size)

        self.model_: xgb.XGBClassifier | None = None
        self.scale_pos_weight_: float = 1.0
        self.n_train_labeled_: int = 0

    def fit(self, train_data: Any, val_data: Any | None = None) -> XGBoostDetector:
        """Fit XGBoost on labeled (``y in {0, 1}``) train nodes with early stopping on ``val_data``."""
        if xgb is None:
            raise ImportError("xgboost is required for XGBoostDetector. Install with: pip install xgboost")
        X_train = extract_static_features(train_data)
        y_train = extract_labels(train_data)

        labeled_mask = (y_train == LABEL_NORMAL) | (y_train == LABEL_ANOMALY)
        if not labeled_mask.any():
            raise ValueError("XGBoostDetector requires labeled (y in {0, 1}) samples in train_data.")

        X_tr = X_train[labeled_mask]
        y_tr = y_train[labeled_mask]
        self.n_train_labeled_ = int(y_tr.shape[0])

        num_pos = int((y_tr == LABEL_ANOMALY).sum())
        num_neg = int((y_tr == LABEL_NORMAL).sum())
        self.scale_pos_weight_ = float(num_neg / max(1, num_pos))

        eval_set = None
        if val_data is not None:
            X_val_all = extract_static_features(val_data)
            val_mask, y_val = get_eval_mask_and_labels(val_data)
            eval_set = [(X_val_all[val_mask], y_val)]

        model_kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "scale_pos_weight": self.scale_pos_weight_,
            "eval_metric": self.eval_metric,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tree_method": "hist",
        }
        if eval_set is not None and self.early_stopping_rounds > 0:
            model_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self.model_ = xgb.XGBClassifier(**model_kwargs)
        if eval_set is not None:
            self.model_.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        else:
            self.model_.fit(X_tr, y_tr, verbose=False)

        self.is_fitted_ = True

        if val_data is not None:
            self.calibrate_threshold(val_data)

        return self

    def predict_score(self, data: Any, node_indices: np.ndarray | None = None) -> np.ndarray:
        """Return class-1 probability scores in ``[0, 1]``."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("XGBoostDetector is not fitted yet.")

        X = extract_static_features(data)
        if node_indices is not None:
            X = X[np.asarray(node_indices, dtype=np.int64)]
        n = X.shape[0]
        scores = np.empty(n, dtype=np.float64)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            scores[start:end] = self.model_.predict_proba(X[start:end])[:, 1]

        return scores

    def explain(self, data: Any, node_idx: int) -> dict[str, Any]:
        """Explain prediction at ``node_idx`` using exact TreeSHAP feature contributions."""
        if xgb is None:
            raise ImportError("xgboost is required for XGBoostDetector.explain(). Install with: pip install xgboost")
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("XGBoostDetector is not fitted yet.")

        X = extract_static_features(data)
        if node_idx < 0 or node_idx >= X.shape[0]:
            raise IndexError(f"node_idx {node_idx} out of range [0, {X.shape[0]})")

        x_row = X[node_idx : node_idx + 1]
        score = float(self.predict_score(x_row)[0])

        dmatrix = xgb.DMatrix(x_row, feature_names=self.feature_names_)
        contribs = self.model_.get_booster().predict(dmatrix, pred_contribs=True)[0]
        feat_contribs = contribs[:-1]
        base_value = float(contribs[-1])

        feature_values = {
            name: float(x_row[0, i]) for i, name in enumerate(self.feature_names_)
        }
        contrib_map = {
            name: float(feat_contribs[i]) for i, name in enumerate(self.feature_names_)
        }

        order = np.argsort(-np.abs(feat_contribs))
        top_features = [
            {
                "feature": self.feature_names_[i],
                "value": float(x_row[0, i]),
                "contribution": float(feat_contribs[i]),
            }
            for i in order
        ]

        predicted_label = (
            int(score >= self.threshold_) if self.threshold_ is not None else None
        )

        return {
            "model": "XGBoostDetector",
            "node_idx": int(node_idx),
            "address": get_node_address(data, node_idx),
            "score": score,
            "threshold": self.threshold_,
            "predicted_label": predicted_label,
            "base_value": base_value,
            "feature_values": feature_values,
            "feature_contributions": contrib_map,
            "top_features": top_features,
        }

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class": "XGBoostDetector",
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "reg_lambda": self.reg_lambda,
                "reg_alpha": self.reg_alpha,
                "early_stopping_rounds": self.early_stopping_rounds,
                "eval_metric": self.eval_metric,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "batch_size": self.batch_size,
                "feature_names": self.feature_names_,
            },
            "state": {
                "model_": self.model_,
                "threshold_": self.threshold_,
                "val_best_f1_": self.val_best_f1_,
                "is_fitted_": self.is_fitted_,
                "scale_pos_weight_": self.scale_pos_weight_,
                "n_train_labeled_": self.n_train_labeled_,
            },
        }
        joblib.dump(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> XGBoostDetector:
        payload = joblib.load(Path(path))
        obj = cls(**payload["params"])
        for k, v in payload["state"].items():
            setattr(obj, k, v)
        return obj
