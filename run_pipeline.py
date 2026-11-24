"""
ETHEREUM GUARD — Zero-to-One Pipeline CLI (Master Plan Section 3).

Usage:
    python run_pipeline.py --config configs/default.yaml --mode verify
    python run_pipeline.py --config configs/default.yaml --mode sample
    python run_pipeline.py --config configs/default.yaml --mode full
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schemas import ALL_FEATURE_COLS


def _run_step(script_rel: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / script_rel), *(extra_args or [])]
    print(f"\n>>> Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(ROOT), check=False)
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Pipeline step failed ({script_rel}) with exit code {res.returncode}")
    print(f"<<< Completed {script_rel} in {elapsed:.2f}s")


def run_verify_mode() -> dict[str, Any]:
    """Audit all expected processed datasets, scalers, and checkpoints on disk."""
    expected_artifacts = {
        "sample_transactions": ROOT / "data/sample/transactions.parquet",
        "sample_labels": ROOT / "data/sample/address_labels.parquet",
        "features_train_nodes": ROOT / "data/processed/features/train/nodes.parquet",
        "features_val_nodes": ROOT / "data/processed/features/val/nodes.parquet",
        "features_test_nodes": ROOT / "data/processed/features/test/nodes.parquet",
        "graphs_train_nodes": ROOT / "data/processed/graphs/train/nodes.parquet",
        "graphs_val_nodes": ROOT / "data/processed/graphs/val/nodes.parquet",
        "graphs_test_nodes": ROOT / "data/processed/graphs/test/nodes.parquet",
        "scaler_tabular": ROOT / "artifacts/scalers/feature_robust_scaler.joblib",
        "scaler_graph": ROOT / "artifacts/scalers/graph_feature_scaler.joblib",
        "ckpt_isolation_forest": ROOT / "artifacts/checkpoints/isolation_forest.pkl",
        "ckpt_xgboost": ROOT / "artifacts/checkpoints/xgboost.pkl",
        "ckpt_tgae": ROOT / "artifacts/checkpoints/best_tgae.pt",
    }

    report: dict[str, Any] = {"mode": "verify", "artifacts": {}}
    print("=== ETHEREUM GUARD PIPELINE ARTIFACT AUDIT ===")
    for name, path in expected_artifacts.items():
        exists = path.exists()
        size_mb = round(path.stat().st_size / (1024 * 1024), 2) if exists else 0.0
        status = "OK" if exists else "MISSING"
        report["artifacts"][name] = {"exists": exists, "size_mb": size_mb, "path": str(path.relative_to(ROOT))}
        print(f"  [{status:7s}] {name:24s} | {size_mb:8.2f} MB | {path.relative_to(ROOT)}")

    return report


def run_sample_mode() -> dict[str, Any]:
    """
    Fast <10s end-to-end verification on the compact 1,000-address sample fixture.
    Verifies schema contract, LiveFeatureExtractor, and inference engine without requiring data/raw/.
    """
    from src.app.engine import EthereumGuardEngine
    from src.pipeline.live_feature_extractor import LiveFeatureExtractor

    t0 = time.perf_counter()
    sample_tx_path = ROOT / "data/sample/transactions.parquet"
    sample_lbl_path = ROOT / "data/sample/address_labels.parquet"
    if not sample_tx_path.exists() or not sample_lbl_path.exists():
        raise FileNotFoundError(
            "Sample fixture missing under data/sample/. Run src/pipeline/build_sample.py after ingestion."
        )

    tx_df = pq.read_table(sample_tx_path).to_pandas()
    lbl_df = pq.read_table(sample_lbl_path).to_pandas()

    extractor = LiveFeatureExtractor(ROOT)
    engine = EthereumGuardEngine(ROOT)

    sample_addrs = lbl_df["address"].astype(str).str.lower().head(10).tolist()
    checked = 0
    for addr in sample_addrs:
        raw, vec = extractor.extract_features(addr, tx_df)
        assert len(raw) == len(ALL_FEATURE_COLS) == 25
        assert vec.shape == (1, 25)
        pred = engine.predict(addr)
        assert 0.0 <= pred.anomaly_score <= 1.0
        checked += 1

    elapsed = time.perf_counter() - t0
    summary = {
        "mode": "sample",
        "status": "PASS",
        "sample_transactions": int(len(tx_df)),
        "sample_addresses": int(len(lbl_df)),
        "verified_e2e_wallets": checked,
        "elapsed_seconds": round(elapsed, 2),
    }
    print("=== SAMPLE MODE PIPELINE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


def run_full_mode(config_path: Path) -> None:
    """Execute the full TV1-02 -> TV1-05 ingestion, split, feature, graph, and audit pipeline."""
    steps = [
        ("src/pipeline/ingest.py", []),
        ("src/pipeline/build_label_registry.py", []),
        ("src/pipeline/create_splits.py", ["--config", str(config_path)]),
        ("src/pipeline/features.py", ["--config", str(config_path)]),
        ("src/pipeline/graph_builder.py", ["--config", str(config_path)]),
        ("src/pipeline/audit_splits.py", []),
    ]
    for script, args in steps:
        _run_step(script, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ethereum Guard Unified Data & Graph Pipeline CLI")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument(
        "--mode",
        choices=("sample", "full", "verify"),
        default="verify",
        help="sample: fast 1000-wallet E2E verification; full: rebuild all splits/features/graphs; verify: audit artifacts",
    )
    args = parser.parse_args()

    if args.mode == "verify":
        run_verify_mode()
    elif args.mode == "sample":
        run_sample_mode()
    else:
        run_full_mode(args.config.resolve())


if __name__ == "__main__":
    main()
