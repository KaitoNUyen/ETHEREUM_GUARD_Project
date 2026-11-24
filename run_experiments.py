"""
ETHEREUM GUARD — Automated Dual-Level Benchmark (Transaction & Wallet), 7-Config Ablation & Error Analysis CLI.

Usage:
    python run_experiments.py --config configs/default.yaml --suite all
    python run_experiments.py --config configs/default.yaml --suite baselines
    python run_experiments.py --config configs/default.yaml --suite ablation
    python run_experiments.py --config configs/default.yaml --suite transactions

Outputs:
    artifacts/results/benchmark_comparison.csv
    artifacts/results/transaction_benchmark.csv
    artifacts/results/ablation_study.csv
    artifacts/results/error_cases.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import UnifiedEvaluator, analyze_error_cases
from src.models.base import BaseDetector
from src.models.baselines import IsolationForestDetector, XGBoostDetector
from src.models.data_utils import extract_static_features, load_config, load_split_data
from src.models.deep_baselines import DOMINANTDetector, GATDetector
from src.models.tgae import ModularTGAE
from src.models.tx_scorer import DeepTransactionScorer, extract_labeled_edges
from src.pipeline.live_feature_extractor import LiveFeatureExtractor
from src.schemas import ALL_FEATURE_COLS, LABEL_ANOMALY, LABEL_NORMAL, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ethereum_guard.experiments")

ABLATION_CONFIGS: list[dict[str, Any]] = [
    {"code": "Abl-A", "name": "Tabular Autoencoder", "use_tabular": True, "use_graph": False, "use_temporal": False},
    {"code": "Abl-B", "name": "Graph Transformer AE", "use_tabular": False, "use_graph": True, "use_temporal": False},
    {"code": "Abl-C", "name": "Temporal GRU-Attn AE", "use_tabular": False, "use_graph": False, "use_temporal": True},
    {"code": "Abl-D", "name": "Tabular + Graph AE", "use_tabular": True, "use_graph": True, "use_temporal": False},
    {"code": "Abl-E", "name": "Tabular + Temporal AE", "use_tabular": True, "use_graph": False, "use_temporal": True},
    {"code": "Abl-F", "name": "Graph + Temporal AE", "use_tabular": False, "use_graph": True, "use_temporal": True},
    {"code": "Abl-G", "name": "Full Proposed TG-AE", "use_tabular": True, "use_graph": True, "use_temporal": True},
]


class _HistGBDTFallbackDetector(BaseDetector):
    """Histogram Gradient Boosted Decision Trees fallback when C++ xgboost wheel is absent."""

    def __init__(self) -> None:
        super().__init__(feature_names=ALL_FEATURE_COLS)
        self.clf_ = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_SEED
        )

    def fit(self, train_data: Any, val_data: Any | None = None) -> _HistGBDTFallbackDetector:
        mask = train_data["eval_mask"]
        self.clf_.fit(train_data["x"][mask], train_data["y"][mask])
        self.is_fitted_ = True
        if val_data is not None:
            self.calibrate_threshold(val_data)
        return self

    def predict_score(self, data: Any, node_indices: np.ndarray | None = None) -> np.ndarray:
        X = extract_static_features(data)
        if node_indices is not None:
            X = X[np.asarray(node_indices, dtype=np.int64)]
        return self.clf_.predict_proba(X)[:, 1].astype(np.float64)

    def explain(self, data: Any, node_idx: int) -> dict[str, Any]:
        return {"node_idx": int(node_idx), "score": float(self.predict_score(data, np.array([node_idx]))[0])}


def _load_tabular_split_from_parquet(split: str, processed_dir: Path) -> dict[str, Any]:
    graph_nodes = processed_dir / "graphs" / split / "nodes.parquet"
    feat_nodes = processed_dir / "features" / split / "nodes.parquet"
    path = graph_nodes if graph_nodes.exists() else feat_nodes
    if not path.exists():
        raise FileNotFoundError(f"Could not locate nodes.parquet for split '{split}'")

    cols_available = set(pq.read_schema(path).names)
    read_cols = ["address", "label"] + [c for c in ALL_FEATURE_COLS if c in cols_available]
    if "is_inductive_holdout" in cols_available:
        read_cols.append("is_inductive_holdout")

    df = pq.read_table(path, columns=read_cols).to_pandas()
    for col in ALL_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = (
        df[list(ALL_FEATURE_COLS)]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    y = df["label"].to_numpy(dtype=np.int64)
    eval_mask = (y == LABEL_NORMAL) | (y == LABEL_ANOMALY)
    holdout = (
        df["is_inductive_holdout"].fillna(False).to_numpy(dtype=bool)
        if "is_inductive_holdout" in df.columns
        else np.zeros(len(df), dtype=bool)
    )

    return {
        "x": X,
        "y": y,
        "eval_mask": eval_mask,
        "addresses": df["address"].astype(str).str.lower().tolist(),
        "is_inductive_holdout": holdout,
        "source_path": str(path.relative_to(ROOT)),
    }


def _score_edge_baseline_metrics(
    model_name: str,
    protocol: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    val_y: np.ndarray,
    val_scores: np.ndarray,
    elapsed_ms: float,
) -> dict[str, Any]:
    prec, rec, thresh = precision_recall_curve(val_y, val_scores)
    if len(thresh) > 0:
        f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        tau = float(thresh[int(np.argmax(f1s))])
    else:
        tau = 0.5

    preds = (scores >= tau).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    order = np.argsort(-scores)
    return {
        "model": model_name,
        "granularity": "transaction_edge",
        "protocol": protocol,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "f1_binary": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "threshold": tau,
        "num_eval_edges": int(len(y_true)),
        "num_positives": int((y_true == 1).sum()),
        "num_negatives": int((y_true == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_at_50": float(y_true[order[: min(50, len(y_true))]].mean()),
        "precision_at_100": float(y_true[order[: min(100, len(y_true))]].mean()),
        "latency_ms_per_1k_txs": round(elapsed_ms / max(1e-3, len(y_true) / 1000.0), 4),
    }


def run_experiments(
    config_path: Path,
    suite: str = "all",
    retrain_ablation: bool = False,
) -> dict[str, Path]:
    cfg = load_config(config_path)
    processed_dir = ROOT / cfg.get("data", {}).get("processed_dir", "data/processed")
    results_dir = ROOT / cfg.get("output", {}).get("results_dir", "artifacts/results")
    checkpoints_dir = ROOT / cfg.get("output", {}).get("checkpoints_dir", "artifacts/checkpoints")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    evaluator = UnifiedEvaluator(ks=(10, 50, 100))
    extractor = LiveFeatureExtractor(ROOT)

    logger.info("Loading train, val & test Parquet tables...")
    train_tab = _load_tabular_split_from_parquet("train", processed_dir)
    val_tab = _load_tabular_split_from_parquet("val", processed_dir)
    test_tab = _load_tabular_split_from_parquet("test", processed_dir)

    train_labeled_addrs = {
        train_tab["addresses"][i] for i in np.where(train_tab["eval_mask"])[0]
    }

    logger.info("Loading multi-modal graph snapshots (train, val, test)...")
    train_pyg = load_split_data("train", processed_dir=processed_dir, verify_parquet=False)
    val_pyg = load_split_data("val", processed_dir=processed_dir, verify_parquet=False)
    test_pyg = load_split_data("test", processed_dir=processed_dir, verify_parquet=False)

    benchmark_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    tx_benchmark_rows: list[dict[str, Any]] = []
    error_frames: list[pd.DataFrame] = []

    has_xgb = importlib.util.find_spec("xgboost") is not None

    # ------------------------------------------------------------------
    # 1. TRADITIONAL ML & INDEPENDENT DEEP GNN BASELINES (Account-Level)
    # ------------------------------------------------------------------
    iso_ckpt = checkpoints_dir / "isolation_forest.pkl"
    iso_det = IsolationForestDetector.load(iso_ckpt) if iso_ckpt.exists() else IsolationForestDetector().fit(train_tab, val_tab)

    if suite in ("all", "baselines"):
        for proto, excl, ind_only in (
            ("cumulative_test", None, False),
            ("strict_unseen_from_train", train_labeled_addrs, False),
            ("inductive_holdout_only", None, True),
        ):
            row, eval_idx, y_true, scores = evaluator.evaluate_detector(
                iso_det,
                test_tab,
                model_name="BL-1: Isolation Forest",
                split="test",
                exclude_addresses=excl,
                inductive_only=ind_only,
                eval_protocol=proto,
            )
            benchmark_rows.append(row)
            if proto == "cumulative_test":
                eval_addrs = [test_tab["addresses"][i] for i in eval_idx]
                raw_mat = np.zeros((len(eval_idx), len(ALL_FEATURE_COLS)), dtype=np.float64)
                for r_i, node_i in enumerate(eval_idx):
                    scaled_map = {col: float(test_tab["x"][node_i, c_i]) for c_i, col in enumerate(ALL_FEATURE_COLS)}
                    raw_map = extractor.inverse_scale_features(scaled_map)
                    raw_mat[r_i] = [raw_map[col] for col in ALL_FEATURE_COLS]

                error_frames.append(
                    analyze_error_cases(
                        model_name="BL-1: Isolation Forest",
                        split="test",
                        addresses=eval_addrs,
                        y_true=y_true,
                        y_score=scores,
                        threshold=iso_det.threshold_,
                        features_raw=raw_mat,
                        is_inductive_holdout=test_tab["is_inductive_holdout"][eval_idx],
                        unseen_mask=np.array([a not in train_labeled_addrs for a in eval_addrs], dtype=bool),
                    )
                )

        # BL-2: XGBoost (or HistGBDT exact tree-boosting equivalent when xgboost wheel is absent)
        xgb_ckpt = checkpoints_dir / "xgboost.pkl"
        if has_xgb and xgb_ckpt.exists():
            gb_det: BaseDetector = XGBoostDetector.load(xgb_ckpt)
            gb_name = "BL-2: XGBoost"
        elif has_xgb:
            gb_det = XGBoostDetector().fit(train_tab, val_tab)
            gb_det.save(xgb_ckpt)
            gb_name = "BL-2: XGBoost"
        else:
            gb_det = _HistGBDTFallbackDetector().fit(train_tab, val_tab)
            gb_name = "BL-2: Gradient Boosted Trees (HistGBDT)"

        for proto, excl, ind_only in (
            ("cumulative_test", None, False),
            ("strict_unseen_from_train", train_labeled_addrs, False),
            ("inductive_holdout_only", None, True),
        ):
            row, eval_idx, y_true, scores = evaluator.evaluate_detector(
                gb_det,
                test_tab,
                model_name=gb_name,
                split="test",
                exclude_addresses=excl,
                inductive_only=ind_only,
                eval_protocol=proto,
            )
            benchmark_rows.append(row)
            if proto == "cumulative_test":
                eval_addrs = [test_tab["addresses"][i] for i in eval_idx]
                error_frames.append(
                    analyze_error_cases(
                        model_name=gb_name,
                        split="test",
                        addresses=eval_addrs,
                        y_true=y_true,
                        y_score=scores,
                        threshold=gb_det.threshold_,
                        is_inductive_holdout=test_tab["is_inductive_holdout"][eval_idx],
                        unseen_mask=np.array([a not in train_labeled_addrs for a in eval_addrs], dtype=bool),
                    )
                )

        # DL-BL1: DOMINANT (Deep Graph Convolutional Autoencoder)
        dom_ckpt = checkpoints_dir / "dominant_gae.pt"
        logger.info("Training & evaluating DL-BL1: DOMINANT Deep Graph Autoencoder...")
        dom_det = DOMINANTDetector(epochs=40).fit(train_pyg, val_pyg)
        dom_det.save(dom_ckpt)
        for proto, excl in (("cumulative_test", None), ("strict_unseen_from_train", train_labeled_addrs)):
            row, eval_idx, y_true, scores = evaluator.evaluate_detector(
                dom_det,
                test_pyg,
                model_name="DL-BL1: DOMINANT (Deep Graph AE)",
                split="test",
                exclude_addresses=excl,
                eval_protocol=proto,
            )
            benchmark_rows.append(row)
            if proto == "cumulative_test":
                eval_addrs = [test_tab["addresses"][i] for i in eval_idx]
                error_frames.append(
                    analyze_error_cases(
                        model_name="DL-BL1: DOMINANT (Deep Graph AE)",
                        split="test",
                        addresses=eval_addrs,
                        y_true=y_true,
                        y_score=scores,
                        threshold=dom_det.threshold_,
                        is_inductive_holdout=test_tab["is_inductive_holdout"][eval_idx],
                        unseen_mask=np.array([a not in train_labeled_addrs for a in eval_addrs], dtype=bool),
                    )
                )

        # DL-BL2: GAT (Multi-Head Graph Attention Network + Focal Loss)
        gat_ckpt = checkpoints_dir / "gat_detector.pt"
        logger.info("Training & evaluating DL-BL2: GAT (Graph Attention Network)...")
        gat_det = GATDetector(epochs=45).fit(train_pyg, val_pyg)
        gat_det.save(gat_ckpt)
        for proto, excl in (("cumulative_test", None), ("strict_unseen_from_train", train_labeled_addrs)):
            row, eval_idx, y_true, scores = evaluator.evaluate_detector(
                gat_det,
                test_pyg,
                model_name="DL-BL2: GAT (Graph Attention Net)",
                split="test",
                exclude_addresses=excl,
                eval_protocol=proto,
            )
            benchmark_rows.append(row)
            if proto == "cumulative_test":
                eval_addrs = [test_tab["addresses"][i] for i in eval_idx]
                error_frames.append(
                    analyze_error_cases(
                        model_name="DL-BL2: GAT (Graph Attention Net)",
                        split="test",
                        addresses=eval_addrs,
                        y_true=y_true,
                        y_score=scores,
                        threshold=gat_det.threshold_,
                        is_inductive_holdout=test_tab["is_inductive_holdout"][eval_idx],
                        unseen_mask=np.array([a not in train_labeled_addrs for a in eval_addrs], dtype=bool),
                    )
                )

    # ------------------------------------------------------------------
    # 2. PROPOSED MODULAR TG-AE & 7-CONFIG ABLATION
    # ------------------------------------------------------------------
    tgae_ckpt = checkpoints_dir / "best_tgae.pt"
    tgae_det = ModularTGAE.load(tgae_ckpt)

    if suite in ("all", "ablation"):
        for proto, excl, ind_only in (
            ("cumulative_test", None, False),
            ("strict_unseen_from_train", train_labeled_addrs, False),
            ("inductive_holdout_only", None, True),
        ):
            row, eval_idx, y_true, scores = evaluator.evaluate_detector(
                tgae_det,
                test_pyg,
                model_name="Proposed: Modular TG-AE (best_tgae.pt)",
                split="test",
                exclude_addresses=excl,
                inductive_only=ind_only,
                eval_protocol=proto,
            )
            benchmark_rows.append(row)
            if proto == "cumulative_test":
                eval_addrs = [test_tab["addresses"][i] for i in eval_idx]
                error_frames.append(
                    analyze_error_cases(
                        model_name="Proposed: Modular TG-AE",
                        split="test",
                        addresses=eval_addrs,
                        y_true=y_true,
                        y_score=scores,
                        threshold=tgae_det.threshold_,
                        is_inductive_holdout=test_tab["is_inductive_holdout"][eval_idx],
                        unseen_mask=np.array([a not in train_labeled_addrs for a in eval_addrs], dtype=bool),
                    )
                )

        ablation_csv_existing = results_dir / "ablation_study.csv"
        if retrain_ablation or not ablation_csv_existing.exists():
            model_cfg = cfg.get("model", {})
            train_cfg = cfg.get("training", {})
            for abl in ABLATION_CONFIGS:
                logger.info("Training %s (%s) with seq_use_log1p=True...", abl["code"], abl["name"])
                det = ModularTGAE(
                    hidden_dim=int(model_cfg.get("hidden_dim", 128)),
                    latent_dim=int(model_cfg.get("latent_dim", 64)),
                    dropout=float(model_cfg.get("dropout", 0.2)),
                    use_tabular=abl["use_tabular"],
                    use_graph=abl["use_graph"],
                    use_temporal=abl["use_temporal"],
                    epochs=int(train_cfg.get("epochs", 60)),
                    batch_size=int(train_cfg.get("batch_size", 512)),
                    lr=float(train_cfg.get("lr", 0.001)),
                    seq_use_log1p=True,
                )
                det.fit(train_pyg, val_pyg)
                for proto, excl in (
                    ("cumulative_test", None),
                    ("strict_unseen_from_train", train_labeled_addrs),
                ):
                    row, _, _, _ = evaluator.evaluate_detector(
                        det,
                        test_pyg,
                        model_name=f"{abl['code']}: {abl['name']}",
                        split="test",
                        exclude_addresses=excl,
                        eval_protocol=proto,
                    )
                    ablation_rows.append(
                        {
                            "ablation_code": abl["code"],
                            "use_tabular": abl["use_tabular"],
                            "use_graph": abl["use_graph"],
                            "use_temporal": abl["use_temporal"],
                            **row,
                        }
                    )
                    if abl["code"] == "Abl-G":
                        benchmark_rows.append({**row, "model": "Proposed: Modular TG-AE (Log1p-Seq Fixed)"})
        else:
            existing_abl_df = pd.read_csv(ablation_csv_existing)
            ablation_rows = existing_abl_df.to_dict(orient="records")
            for r in ablation_rows:
                if r.get("ablation_code") == "Abl-G":
                    row_copy = {k: v for k, v in r.items() if k not in {"ablation_code", "use_tabular", "use_graph", "use_temporal"}}
                    row_copy["model"] = "Proposed: Modular TG-AE (Log1p-Seq Fixed)"
                    benchmark_rows.append(row_copy)

    # ------------------------------------------------------------------
    # 3. TRANSACTION-LEVEL (EDGE-LEVEL) DEEP LEARNING BENCHMARK
    # ------------------------------------------------------------------
    if suite in ("all", "transactions"):
        logger.info("Training & evaluating DeepTransactionScorer on Ethereum transaction flows...")
        tx_scorer = DeepTransactionScorer(epochs=40).fit(tgae_det, train_pyg, val_pyg)
        tx_ckpt = checkpoints_dir / "deep_tx_scorer.pt"
        tx_scorer.save(tx_ckpt)

        va_eidx, y_va_edge = extract_labeled_edges(val_pyg, max_negatives=25_000, seed=RANDOM_SEED)
        te_eidx, y_te_edge = extract_labeled_edges(test_pyg, max_negatives=25_000, seed=RANDOM_SEED)

        # Build strict unseen transaction edge mask using global node_id pairs (zero train overlap)
        tr_nid = train_pyg.node_id.cpu().numpy()
        tr_ei = train_pyg.edge_index.cpu().numpy()
        train_global_edges = set(zip(tr_nid[tr_ei[0]].tolist(), tr_nid[tr_ei[1]].tolist()))

        te_nid = test_pyg.node_id.cpu().numpy()
        te_ei_eval = test_pyg.edge_index.cpu().numpy()[:, te_eidx]
        unseen_edge_mask = np.array(
            [(int(te_nid[u]), int(te_nid[v])) not in train_global_edges for u, v in zip(te_ei_eval[0], te_ei_eval[1])],
            dtype=bool,
        )

        # 3a. Tx-BL1: Isolation Forest Edge Baseline (max endpoint IF score + log tx value)
        def _if_edge_scores(g: Any, e_idx: np.ndarray) -> np.ndarray:
            ei = g.edge_index.cpu().numpy()[:, e_idx]
            ea = g.edge_attr.cpu().numpy()[e_idx]
            ep = np.unique(np.concatenate([ei[0], ei[1]]))
            ep_scores = iso_det.predict_score(g, ep)
            su = ep_scores[np.searchsorted(ep, ei[0])]
            sv = ep_scores[np.searchsorted(ep, ei[1])]
            return np.clip(0.85 * np.maximum(su, sv) + 0.15 * (1.0 / (1.0 + np.exp(-ea[:, 1]))), 0.0, 1.0)

        t0 = time.perf_counter()
        va_if_s = _if_edge_scores(val_pyg, va_eidx)
        te_if_s = _if_edge_scores(test_pyg, te_eidx)
        el_if = (time.perf_counter() - t0) * 1000.0
        tx_benchmark_rows.append(
            _score_edge_baseline_metrics(
                "Tx-BL1: Isolation Forest Edge Scorer", "cumulative_test_txs", y_te_edge, te_if_s, y_va_edge, va_if_s, el_if
            )
        )
        if unseen_edge_mask.any() and len(np.unique(y_te_edge[unseen_edge_mask])) == 2:
            tx_benchmark_rows.append(
                _score_edge_baseline_metrics(
                    "Tx-BL1: Isolation Forest Edge Scorer",
                    "strict_unseen_txs",
                    y_te_edge[unseen_edge_mask],
                    te_if_s[unseen_edge_mask],
                    y_va_edge,
                    va_if_s,
                    el_if,
                )
            )

        # 3b. Tx-DL1: DOMINANT Deep Graph AE Edge Scorer
        dom_det_tx = DOMINANTDetector.load(checkpoints_dir / "dominant_gae.pt") if (checkpoints_dir / "dominant_gae.pt").exists() else DOMINANTDetector(epochs=35).fit(train_pyg, val_pyg)

        def _dom_edge_scores(g: Any, e_idx: np.ndarray) -> np.ndarray:
            ei = g.edge_index.cpu().numpy()[:, e_idx]
            ep = np.unique(np.concatenate([ei[0], ei[1]]))
            ep_s = dom_det_tx.predict_score(g, ep)
            su = ep_s[np.searchsorted(ep, ei[0])]
            sv = ep_s[np.searchsorted(ep, ei[1])]
            return np.maximum(su, sv)

        t0 = time.perf_counter()
        va_dom_s = _dom_edge_scores(val_pyg, va_eidx)
        te_dom_s = _dom_edge_scores(test_pyg, te_eidx)
        el_dom = (time.perf_counter() - t0) * 1000.0
        tx_benchmark_rows.append(
            _score_edge_baseline_metrics(
                "Tx-DL1: DOMINANT Graph-AE Edge Scorer", "cumulative_test_txs", y_te_edge, te_dom_s, y_va_edge, va_dom_s, el_dom
            )
        )
        if unseen_edge_mask.any() and len(np.unique(y_te_edge[unseen_edge_mask])) == 2:
            tx_benchmark_rows.append(
                _score_edge_baseline_metrics(
                    "Tx-DL1: DOMINANT Graph-AE Edge Scorer",
                    "strict_unseen_txs",
                    y_te_edge[unseen_edge_mask],
                    te_dom_s[unseen_edge_mask],
                    y_va_edge,
                    va_dom_s,
                    el_dom,
                )
            )

        # 3c. Proposed DeepTransactionScorer (Modular TG-AE Embeddings + Behavioral Shock)
        t0 = time.perf_counter()
        res_cum = tx_scorer.evaluate_edges(tgae_det, test_pyg, te_eidx, y_te_edge)
        el_prop = (time.perf_counter() - t0) * 1000.0
        tx_benchmark_rows.append(
            {
                "model": "Proposed: DeepTransactionScorer (TG-AE + Shock)",
                "granularity": "transaction_edge",
                "protocol": "cumulative_test_txs",
                **res_cum,
                "latency_ms_per_1k_txs": round(el_prop / max(1e-3, len(y_te_edge) / 1000.0), 4),
            }
        )
        if unseen_edge_mask.any() and len(np.unique(y_te_edge[unseen_edge_mask])) == 2:
            t0_u = time.perf_counter()
            res_uns = tx_scorer.evaluate_edges(
                tgae_det, test_pyg, te_eidx[unseen_edge_mask], y_te_edge[unseen_edge_mask]
            )
            el_uns = (time.perf_counter() - t0_u) * 1000.0
            tx_benchmark_rows.append(
                {
                    "model": "Proposed: DeepTransactionScorer (TG-AE + Shock)",
                    "granularity": "transaction_edge",
                    "protocol": "strict_unseen_txs",
                    **res_uns,
                    "latency_ms_per_1k_txs": round(el_uns / max(1e-3, int(unseen_edge_mask.sum()) / 1000.0), 4),
                }
            )

    # Write standardized CSV deliverables
    bench_path = results_dir / "benchmark_comparison.csv"
    tx_bench_path = results_dir / "transaction_benchmark.csv"
    ablation_path = results_dir / "ablation_study.csv"
    error_path = results_dir / "error_cases.csv"

    if benchmark_rows:
        pd.DataFrame(benchmark_rows).to_csv(bench_path, index=False)
    if tx_benchmark_rows:
        pd.DataFrame(tx_benchmark_rows).to_csv(tx_bench_path, index=False)
    if ablation_rows:
        pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)
    if error_frames:
        pd.concat(error_frames, ignore_index=True).to_csv(error_path, index=False)

    logger.info("Saved account benchmark     -> %s (%d rows)", bench_path, len(benchmark_rows))
    logger.info("Saved transaction benchmark -> %s (%d rows)", tx_bench_path, len(tx_benchmark_rows))
    logger.info("Saved ablation table        -> %s (%d rows)", ablation_path, len(ablation_rows))
    logger.info("Saved error case table      -> %s", error_path)

    return {
        "benchmark_comparison": bench_path,
        "transaction_benchmark": tx_bench_path,
        "ablation_study": ablation_path,
        "error_cases": error_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ethereum Guard Dual-Level Benchmark & Ablation Runner")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--suite", choices=("all", "baselines", "ablation", "transactions"), default="all")
    parser.add_argument("--retrain-ablation", action="store_true")
    args = parser.parse_args()
    outputs = run_experiments(
        config_path=args.config.resolve(),
        suite=args.suite,
        retrain_ablation=args.retrain_ablation,
    )
    print("\n=== EXPERIMENT ARTIFACTS GENERATED ===")
    for k, v in outputs.items():
        print(f"  {k:24s}: {v.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
