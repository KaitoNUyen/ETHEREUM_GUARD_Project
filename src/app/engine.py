"""
Ethereum Guard - Backend Serving Engine & Dual-Granularity Deep Predictor.
Enterprise-grade modular inference service for BOTH:
  1. Transaction-Level Anomaly Detection (`tx_hash` -> DeepTransactionScorer + Modular TG-AE)
  2. Account-Level Anomaly Detection (`address` -> Tri-Modal Modular TG-AE primary + IsolationForest baseline)
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from src.pipeline.live_feature_extractor import (
    LiveFeatureExtractor,
    _clean_address_value,
)
from src.schemas import (
    ALL_FEATURE_COLS,
    GRAPH_TOPO_COLS,
    TABULAR_TX_COLS,
    TEMPORAL_STAT_COLS,
)

logger = logging.getLogger("ethereum_guard.engine")
ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")

EVM_METHOD_LABELS: dict[str, str] = {
    "0x": "Native ETH Transfer",
    "0xa9059cbb": "ERC-20 transfer(address,uint256)",
    "0x23b872dd": "ERC-20 transferFrom(address,address,uint256)",
    "0x095ea7b3": "ERC-20 approve(address,uint256) [Approval Grant]",
    "0xa22cb465": "NFT setApprovalForAll(address,bool) [Operator Grant]",
    "0xd505accf": "EIP-2612 permit(...) [Gasless Signature Approval]",
    "0x3593564c": "Uniswap UniversalRouter.execute(...)",
}


def validate_ethereum_address(address: str) -> str:
    """Validate and normalize a 40-hex-character Ethereum address (prevents injection/path traversal)."""
    if not isinstance(address, str):
        raise ValueError(
            "Invalid Ethereum address: must be a string starting with 0x followed by 40 hex characters."
        )
    clean_addr = address.strip().lower()
    if not ADDRESS_RE.fullmatch(clean_addr):
        raise ValueError(
            f"Invalid Ethereum address: {address}. Must be 0x followed by 40 hex characters."
        )
    return clean_addr


def validate_tx_hash(tx_hash: str) -> str:
    """Validate and normalize a 64-hex-character Ethereum transaction hash."""
    if not isinstance(tx_hash, str):
        raise ValueError(
            "Invalid Ethereum transaction hash: must be a string starting with 0x followed by 64 hex characters."
        )
    clean_hash = tx_hash.strip().lower()
    if not TX_HASH_RE.fullmatch(clean_hash):
        raise ValueError(
            f"Invalid Ethereum transaction hash: {tx_hash}. Must be 0x followed by 64 hex characters."
        )
    return clean_hash


def is_transaction_hash(query: str) -> bool:
    """Return True if ``query`` matches a 66-char 0x transaction hash."""
    return isinstance(query, str) and bool(TX_HASH_RE.fullmatch(query.strip()))


def _parse_single_timestamp(value: Any) -> float:
    """Convert heterogeneous timestamp value into Unix seconds for deterministic sorting."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        fval = float(value)
        if not np.isfinite(fval):
            return 0.0
        return fval / 1000.0 if abs(fval) >= 1e11 else fval
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return 0.0
        dt = value.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat"}:
        return 0.0
    try:
        fval = float(text)
        return fval / 1000.0 if abs(fval) >= 1e11 else fval
    except ValueError:
        pass
    try:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(ts):
            return 0.0
        return ts.to_pydatetime().timestamp()
    except Exception:
        return 0.0


class _TokenBucketRateLimiter:
    """Thread-safe Token Bucket rate limiter for outbound on-chain API calls."""

    def __init__(self, rate_per_sec: float = 4.5, capacity: int = 5) -> None:
        self.rate = max(0.1, float(rate_per_sec))
        self.capacity = float(max(1, int(capacity)))
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._last_refill)
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last_refill = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait_seconds = (tokens - self._tokens) / self.rate
            time.sleep(min(wait_seconds, 1.0))


class _BoundedTTLCache:
    """Thread-safe bounded LRU cache with per-entry TTL (prevents unbounded memory growth)."""

    def __init__(self, maxsize: int = 2048, ttl_seconds: float | None = 900.0) -> None:
        self.maxsize = max(1, int(maxsize))
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if self.ttl_seconds is not None and (now - ts) >= self.ttl_seconds:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        now = time.monotonic()
        with self._lock:
            self._store[key] = (now, value)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


@dataclass
class AnomalyPrediction:
    address: str
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    risk_level: str
    verified_label: int  # 1 = phishing, 0 = normal, -1 = unlabeled
    label_text: str
    feature_contributions: list[dict[str, Any]]
    raw_features: dict[str, float]
    normalized_features: dict[str, float]
    inference_time_ms: float
    source: str = "OFFLINE_SNAPSHOT"  # "OFFLINE_SNAPSHOT" | "LIVE_ONCHAIN" | "CACHE_HIT"
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class TransactionAnomalyPrediction:
    tx_hash: str
    from_address: str
    to_address: str | None
    value_eth: float
    gas_used: int
    gas_price_gwei: float
    timestamp: str
    block_number: int
    success: bool
    input_method_id: str
    method_label: str
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    risk_level: str
    sender_score: float
    receiver_score: float
    sender_label_text: str
    receiver_label_text: str
    component_breakdown: dict[str, float]
    forensic_signals: list[dict[str, Any]]
    inference_time_ms: float
    source: str = "OFFLINE_SNAPSHOT"
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class EgoTransaction:
    tx_hash: str
    block_number: int
    timestamp: str
    from_address: str
    to_address: str | None
    value_eth: float
    gas_used: int
    gas_price_gwei: float
    success: bool
    direction: str  # "IN", "OUT", "SELF"


@dataclass
class EgoNetworkGraph:
    center_address: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    total_volume_eth: float
    in_degree: int
    out_degree: int
    unique_counterparties: int


class EthereumGuardEngine:
    """
    Dual-Granularity Deep Learning Anomaly Detection & Forensics Engine.
    Primary Models:
      - Account-Level: Modular TG-AE (`best_tgae.pt` — Tabular MLP + TransformerConv + GRU-Attention)
      - Transaction-Level: DeepTransactionScorer (`deep_tx_scorer.pt` — TG-AE Embeddings + Behavioral Shock)
      - Fast Baseline: Isolation Forest (`isolation_forest.pkl`)
    """

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[2]
        self.checkpoint_path = (
            self.root_dir / "artifacts" / "checkpoints" / "isolation_forest.pkl"
        )
        self.tgae_checkpoint_path = (
            self.root_dir / "artifacts" / "checkpoints" / "best_tgae.pt"
        )
        self.tx_scorer_checkpoint_path = (
            self.root_dir / "artifacts" / "checkpoints" / "deep_tx_scorer.pt"
        )
        self.nodes_path = (
            self.root_dir / "data" / "processed" / "graphs" / "test" / "nodes.parquet"
        )
        self.temporal_test_path = (
            self.root_dir / "data" / "processed" / "features" / "test" / "temporal.pt"
        )
        self.tx_path = self.root_dir / "data" / "sample" / "transactions.parquet"
        self.live_cache_dir = (self.root_dir / "data" / "cache" / "live_wallets").resolve()
        self.live_cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tgae_model = None
        self.tx_scorer = None
        self.if_threshold = 0.37477
        self.threshold = 0.50
        self.normal_mean: np.ndarray | None = None
        self.normal_std: np.ndarray | None = None
        self.normal_median: np.ndarray | None = None
        self.normal_iqr: np.ndarray | None = None
        self.is_loaded = False
        self._mmap_temporal_seq: torch.Tensor | None = None
        self._mmap_lock = threading.Lock()

        self._cache_ttl_seconds = 900.0
        self._prediction_cache = _BoundedTTLCache(maxsize=2048, ttl_seconds=self._cache_ttl_seconds)
        self._tx_prediction_cache = _BoundedTTLCache(maxsize=2048, ttl_seconds=self._cache_ttl_seconds)
        self._live_tx_cache = _BoundedTTLCache(maxsize=512, ttl_seconds=self._cache_ttl_seconds)
        self._offline_miss_cache = _BoundedTTLCache(maxsize=10_000, ttl_seconds=None)
        self._disk_write_lock = threading.Lock()
        self._etherscan_limiter = _TokenBucketRateLimiter(rate_per_sec=4.5, capacity=5)

        self.live_extractor = LiveFeatureExtractor(self.root_dir)
        self._load_models()

    def _load_models(self) -> None:
        if self.checkpoint_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                    iso_dict = joblib.load(self.checkpoint_path)
                state = iso_dict.get("state", {})
                self.model = state.get("model_")
                self.if_threshold = float(state.get("threshold_", 0.37477))
                self.normal_mean = state.get("normal_mean_")
                self.normal_std = state.get("normal_std_")
                self.normal_median = state.get("normal_median_")
                self.normal_iqr = state.get("normal_iqr_")
                self.is_loaded = True
                logger.info("Loaded IsolationForest baseline checkpoint.")
            except Exception as exc:
                logger.error("Failed to load IsolationForest state: %s", exc)
                raise

        if self.tgae_checkpoint_path.exists():
            try:
                from src.models.tgae import ModularTGAE

                self.tgae_model = ModularTGAE.load(self.tgae_checkpoint_path, device="cpu")
                self.threshold = 0.50
                self.is_loaded = True
                logger.info("Loaded primary Deep Learning model: ModularTGAE (best_tgae.pt).")
            except Exception as exc:
                logger.warning("ModularTGAE checkpoint could not be loaded: %s", exc)
                self.threshold = self.if_threshold
        else:
            self.threshold = self.if_threshold

        if self.tx_scorer_checkpoint_path.exists():
            try:
                from src.models.tx_scorer import DeepTransactionScorer

                self.tx_scorer = DeepTransactionScorer.load(self.tx_scorer_checkpoint_path, device="cpu")
                logger.info("Loaded primary Transaction Deep Scorer (deep_tx_scorer.pt).")
            except Exception as exc:
                logger.warning("DeepTransactionScorer checkpoint could not be loaded: %s", exc)

    def _get_offline_temporal_seq(self, node_id: int | None) -> np.ndarray | None:
        """Memory-map temporal.pt once so offline snapshot wallets get real (1, 10, 8) sequences in <1ms."""
        if node_id is None or node_id < 0 or not self.temporal_test_path.exists():
            return None
        if self._mmap_temporal_seq is None:
            with self._mmap_lock:
                if self._mmap_temporal_seq is None:
                    try:
                        raw = torch.load(
                            self.temporal_test_path,
                            map_location="cpu",
                            weights_only=False,
                            mmap=True,
                        )
                        if isinstance(raw, dict) and isinstance(raw.get("sequence"), torch.Tensor):
                            self._mmap_temporal_seq = raw["sequence"]
                        elif isinstance(raw, torch.Tensor):
                            self._mmap_temporal_seq = raw
                    except Exception as exc:
                        logger.warning("Could not memory-map temporal.pt: %s", exc)
                        return None
        if self._mmap_temporal_seq is not None and 0 <= int(node_id) < self._mmap_temporal_seq.shape[0]:
            return self._mmap_temporal_seq[int(node_id)].unsqueeze(0).numpy().astype(np.float32)
        return None

    def _get_local_wallet_transactions(self, clean_addr: str) -> pd.DataFrame:
        """Retrieve cached or sample transactions for a wallet without triggering external network calls."""
        cached_df = self._live_tx_cache.get(clean_addr)
        if cached_df is not None and not cached_df.empty:
            return cached_df

        cache_file = self._safe_cache_path(clean_addr)
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self._live_tx_cache.set(clean_addr, df)
                if not df.empty:
                    return df
            except Exception:
                pass

        if self.tx_path.exists():
            try:
                dataset = ds.dataset(str(self.tx_path), format="parquet")
                expr = (ds.field("from_address") == clean_addr) | (ds.field("to_address") == clean_addr)
                tbl = dataset.to_table(filter=expr)
                if tbl.num_rows > 0:
                    df = tbl.to_pandas()
                    self._live_tx_cache.set(clean_addr, df)
                    return df
            except Exception:
                pass

        return pd.DataFrame()

    def _safe_cache_path(self, clean_addr: str) -> Path:
        validated = validate_ethereum_address(clean_addr)
        candidate = (self.live_cache_dir / f"{validated}.parquet").resolve()
        if candidate.parent != self.live_cache_dir:
            raise ValueError(f"Unsafe cache path rejected for address: {clean_addr}")
        return candidate

    def _atomic_write_parquet(self, df: pd.DataFrame, target_path: Path) -> None:
        with self._disk_write_lock:
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=".tmp_wallet_", suffix=".parquet", dir=str(target_path.parent)
            )
            os.close(tmp_fd)
            tmp_path = Path(tmp_name)
            try:
                df.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, target_path)
            except Exception:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise

    def get_address_record(self, address: str) -> dict[str, Any] | None:
        clean_addr = validate_ethereum_address(address)
        if self._offline_miss_cache.contains(clean_addr):
            return None
        if not self.nodes_path.exists():
            return None

        dataset = ds.dataset(str(self.nodes_path), format="parquet")
        table = dataset.to_table(filter=ds.field("address") == clean_addr)
        if table.num_rows == 0:
            self._offline_miss_cache.set(clean_addr, True)
            return None
        return table.to_pylist()[0]

    def _fetch_live_transactions(self, clean_addr: str) -> tuple[pd.DataFrame, str]:
        clean_addr = validate_ethereum_address(clean_addr)

        cached_df = self._live_tx_cache.get(clean_addr)
        if cached_df is not None:
            return cached_df, "MEM_CACHE"

        cache_file = self._safe_cache_path(clean_addr)
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self._live_tx_cache.set(clean_addr, df)
                return df, "DISK_CACHE"
            except Exception as exc:
                logger.warning("Failed to read disk cache for %s: %s", clean_addr, exc)

        if self.tx_path.exists():
            try:
                dataset = ds.dataset(str(self.tx_path), format="parquet")
                filter_expr = (ds.field("from_address") == clean_addr) | (
                    ds.field("to_address") == clean_addr
                )
                tbl = dataset.to_table(filter=filter_expr)
                if tbl.num_rows > 0:
                    df = tbl.to_pandas()
                    self._live_tx_cache.set(clean_addr, df)
                    self._atomic_write_parquet(df, cache_file)
                    return df, "SAMPLE_POOL"
            except Exception as exc:
                logger.warning("Failed to query sample transactions for %s: %s", clean_addr, exc)

        api_key = os.getenv("ETHERSCAN_API_KEY", "").strip()
        if api_key:
            try:
                from src.pipeline.etherscan_adapter import fetch_normal_transactions

                self._etherscan_limiter.acquire()
                df = fetch_normal_transactions(
                    clean_addr,
                    api_key=api_key,
                    offset=1000,
                    max_pages=2,
                    request_delay=0.22,
                )
                self._live_tx_cache.set(clean_addr, df)
                if not df.empty:
                    self._atomic_write_parquet(df, cache_file)
                return df, "ETHERSCAN_LIVE"
            except Exception as exc:
                logger.error("Etherscan live fetch failed for %s: %s", clean_addr, exc)
                empty_df = pd.DataFrame(
                    columns=["timestamp", "from_address", "to_address", "value_eth", "gas_price", "success"]
                )
                return empty_df, "PROVIDER_ERROR"

        empty_df = pd.DataFrame(
            columns=["timestamp", "from_address", "to_address", "value_eth", "gas_price", "success"]
        )
        return empty_df, "NO_API_KEY"

    def predict(
        self,
        address: str,
        allow_live: bool = True,
        model_type: str = "tgae",
    ) -> AnomalyPrediction:
        """
        Produce account-level risk assessment and XAI attribution report.
        Defaults to ``model_type="tgae"`` (Primary Tri-Modal Deep Learning GNN),
        feeding real 1-hop ego-subgraph edges + (1, 10, 8) temporal sequence tensors!
        """
        model_key = str(model_type or "tgae").strip().lower()
        if model_key not in {"tgae", "isolation_forest"}:
            raise ValueError(f"Unsupported model_type: {model_type}. Expected 'tgae' or 'isolation_forest'.")

        if model_key == "tgae" and self.tgae_model is None:
            if self.model is not None:
                model_key = "isolation_forest"
            else:
                raise RuntimeError("ModularTGAE checkpoint is not loaded.")
        elif model_key == "isolation_forest" and (not self.is_loaded or self.model is None):
            raise RuntimeError("IsolationForestDetector model state is not loaded.")

        start_time = time.perf_counter()
        clean_addr = validate_ethereum_address(address)
        cache_key = clean_addr if model_key == "tgae" else f"{clean_addr}:{model_key}"

        cached_pred = self._prediction_cache.get(cache_key)
        if cached_pred is not None:
            return AnomalyPrediction(
                address=cached_pred.address,
                anomaly_score=cached_pred.anomaly_score,
                is_anomaly=cached_pred.is_anomaly,
                threshold=cached_pred.threshold,
                risk_level=cached_pred.risk_level,
                verified_label=cached_pred.verified_label,
                label_text=cached_pred.label_text,
                feature_contributions=cached_pred.feature_contributions,
                raw_features=cached_pred.raw_features,
                normalized_features=cached_pred.normalized_features,
                inference_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                source="CACHE_HIT",
                metadata={
                    **cached_pred.metadata,
                    "cached_at": cached_pred.evaluated_at,
                },
            )

        record = self.get_address_record(clean_addr)
        source = "OFFLINE_SNAPSHOT"
        meta: dict[str, Any] = {}
        should_cache = True
        tx_df_for_ego: pd.DataFrame | None = None
        offline_node_id: int | None = None

        if record is not None:
            scaled_feat_map = {col: float(record.get(col, 0.0) or 0.0) for col in ALL_FEATURE_COLS}
            raw_feat_map = self.live_extractor.inverse_scale_features(scaled_feat_map)
            verified_label = int(record.get("label", -1))
            offline_node_id = int(record["node_id"]) if record.get("node_id") is not None else None
            X_vec = np.array(
                [[scaled_feat_map[col] for col in ALL_FEATURE_COLS]], dtype=np.float32
            )
            meta["storage"] = "data/processed/graphs/test/nodes.parquet"
            meta["feature_space"] = "raw_inverted_from_snapshot"
            if model_key == "tgae":
                tx_df_for_ego = self._get_local_wallet_transactions(clean_addr)
        elif allow_live:
            source = "LIVE_ONCHAIN"
            tx_df, fetch_status = self._fetch_live_transactions(clean_addr)
            tx_df_for_ego = tx_df
            raw_feat_map, X_vec = self.live_extractor.extract_features(clean_addr, tx_df)
            verified_label = -1
            meta["transactions_analyzed"] = int(len(tx_df))
            meta["fetch_status"] = fetch_status
            meta["feature_space"] = "raw_onchain"

            if fetch_status == "ETHERSCAN_LIVE":
                meta["live_provider"] = "Etherscan API V2"
                meta["notice"] = "Live on-chain analysis completed via Etherscan API V2."
            elif fetch_status in {"MEM_CACHE", "DISK_CACHE", "SAMPLE_POOL"}:
                meta["live_provider"] = "Local On-Chain Cache"
                meta["notice"] = f"Live on-chain features computed from {fetch_status.lower()}."
            elif fetch_status == "PROVIDER_ERROR":
                should_cache = False
                meta["live_provider"] = "Etherscan API V2 (Degraded)"
                meta["notice"] = (
                    "Etherscan request failed temporarily; result not cached. Retry shortly."
                )
            else:
                meta["live_provider"] = "Local On-Chain Adapter"
                meta["notice"] = (
                    "Evaluated with local on-chain engine. Set ETHERSCAN_API_KEY for continuous live block sync."
                )
        else:
            raise KeyError(
                f"Address {clean_addr} was not observed in current processed dataset snapshots."
            )

        X_vec = np.nan_to_num(X_vec, nan=0.0, posinf=0.0, neginf=0.0)
        meta["model_type"] = model_key

        x_row = X_vec[0].astype(np.float64)
        z_scores = np.zeros(len(ALL_FEATURE_COLS), dtype=np.float64)
        iqr_devs = np.zeros(len(ALL_FEATURE_COLS), dtype=np.float64)

        if self.normal_mean is not None and self.normal_std is not None:
            z_scores = np.divide(
                x_row - self.normal_mean,
                self.normal_std,
                out=np.zeros_like(x_row),
                where=self.normal_std > 1e-12,
            )

        if (
            self.normal_iqr is not None
            and self.normal_median is not None
            and self.normal_std is not None
        ):
            iqr_scale = np.where(
                self.normal_iqr > 1e-12, self.normal_iqr, self.normal_std
            )
            iqr_devs = np.divide(
                x_row - self.normal_median,
                iqr_scale,
                out=np.zeros_like(x_row),
                where=iqr_scale > 1e-12,
            )

        if model_key == "tgae" and self.tgae_model is not None:
            # Assemble real 1-hop ego subgraph + (1, 10, 8) temporal sequence for true Tri-Modal GNN inference
            if tx_df_for_ego is not None and not tx_df_for_ego.empty:
                mm = self.live_extractor.extract_multimodal_ego(clean_addr, tx_df_for_ego)
                x_ego = mm["x_ego"].copy()
                x_ego[0] = X_vec[0]
                seq_10x8 = mm["temporal_seq_10x8"]
                ei_t = torch.from_numpy(mm["edge_index_np"]).long()
                ea_t = torch.from_numpy(mm["edge_attr_np"]).float()
            else:
                x_ego = X_vec
                mmap_seq = self._get_offline_temporal_seq(offline_node_id)
                seq_10x8 = (
                    mmap_seq
                    if mmap_seq is not None
                    else np.zeros((1, 10, 8), dtype=np.float32)
                )
                ei_t = torch.empty((2, 0), dtype=torch.long)
                ea_t = torch.empty((0, 5), dtype=torch.float32)

            tgae_res = self.tgae_model.predict_ego_subgraph(
                x_ego,
                temporal_seq_10x8=seq_10x8,
                edge_index=ei_t,
                edge_attr=ea_t,
            )
            norm_score = float(np.clip(tgae_res["normalized_score"], 0.0, 1.0))
            active_threshold = 0.50
            is_anom = bool(tgae_res["is_anomaly"])
            meta["branch_contributions"] = tgae_res["branch_contributions"]
            meta["branch_errors_calibrated"] = tgae_res["branch_errors_calibrated"]
            meta["raw_tgae_score"] = round(float(tgae_res["raw_tgae_score"]), 4)
            meta["raw_tgae_threshold"] = round(float(tgae_res["threshold"]), 4)
            meta["aux_probability"] = round(float(tgae_res["aux_probability"]), 4)
            meta["active_modalities"] = {
                "tabular_dim": 12,
                "ego_graph_nodes": int(x_ego.shape[0]),
                "ego_graph_edges": int(ei_t.shape[1]),
                "nonempty_temporal_windows": int((np.abs(seq_10x8[0]).sum(axis=-1) > 0).sum()),
            }
            meta["_latent_z"] = tgae_res["latent_z"].tolist()
            feat_err_map = tgae_res["feature_errors"]
            impact_magnitudes = np.array(
                [float(feat_err_map.get(col, 0.0)) for col in ALL_FEATURE_COLS],
                dtype=np.float64,
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                norm_score = float(np.clip(-self.model.score_samples(X_vec)[0], 0.0, 1.0))
            active_threshold = float(self.if_threshold)
            is_anom = norm_score >= active_threshold
            impact_magnitudes = np.maximum(np.abs(z_scores), np.abs(iqr_devs))

        ranked_indices = np.argsort(-impact_magnitudes)

        contributions = []
        for idx in ranked_indices[:8]:
            col_name = ALL_FEATURE_COLS[idx]
            category = (
                "Transaction Flow"
                if col_name in TABULAR_TX_COLS
                else (
                    "Graph Topology"
                    if col_name in GRAPH_TOPO_COLS
                    else "Temporal Dynamics"
                )
            )
            base_ref = (
                float(self.normal_mean[idx])
                if self.normal_mean is not None
                else 0.0
            )
            contributions.append(
                {
                    "feature": col_name,
                    "category": category,
                    "raw_value": raw_feat_map[col_name],
                    "transformed_value": float(x_row[idx]),
                    "baseline_mean": base_ref,
                    "z_score": float(z_scores[idx]),
                    "iqr_deviation": float(iqr_devs[idx]),
                    "direction": "HIGH" if x_row[idx] >= base_ref else "LOW",
                }
            )

        if norm_score >= max(0.65, active_threshold * 1.35):
            risk_level = "CRITICAL"
        elif norm_score >= active_threshold:
            risk_level = "HIGH"
        elif norm_score >= active_threshold * 0.85:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if source == "LIVE_ONCHAIN" and meta.get("transactions_analyzed", 0) == 0:
            meta["insufficient_history"] = True
            if meta.get("fetch_status") == "ETHERSCAN_LIVE":
                meta["warning"] = (
                    "Address has 0 observed normal transactions on-chain; anomaly score reflects "
                    "distance from active normal training accounts, not confirmed malicious activity."
                )

        label_text_map = {
            1: "Known Phishing (Verified Malicious)",
            0: "Verified Normal (Whitelisted / Protocol)",
            -1: (
                "Live On-Chain (Dynamic Assessment)"
                if source == "LIVE_ONCHAIN"
                else "Unlabeled (Under Assessment)"
            ),
        }

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        # Keep public metadata JSON-clean without bulky internal arrays unless needed
        latent_z_list = meta.pop("_latent_z", None)

        pred = AnomalyPrediction(
            address=clean_addr,
            anomaly_score=round(norm_score, 4),
            is_anomaly=is_anom,
            threshold=round(active_threshold, 4),
            risk_level=risk_level,
            verified_label=verified_label,
            label_text=label_text_map.get(verified_label, "Unknown"),
            feature_contributions=contributions,
            raw_features=raw_feat_map,
            normalized_features={
                col: round(float(x_row[i]), 4)
                for i, col in enumerate(ALL_FEATURE_COLS)
            },
            inference_time_ms=round(latency_ms, 2),
            source=source,
            metadata=meta,
        )
        if latent_z_list is not None:
            pred.metadata["latent_dim"] = len(latent_z_list)

        if should_cache:
            self._prediction_cache.set(cache_key, pred)
        return pred

    def _get_node_embedding_and_profile(self, addr: str | None, allow_live: bool = True) -> tuple[np.ndarray, np.ndarray, float, str]:
        """Extract 64-D ModularTGAE latent vector, 25-D scaled features, [0, 1] score, and label text for a wallet."""
        latent_dim = self.tgae_model.latent_dim if self.tgae_model is not None else 64
        if not addr or not ADDRESS_RE.fullmatch(str(addr).strip().lower()):
            return (
                np.zeros(latent_dim, dtype=np.float32),
                np.zeros(len(ALL_FEATURE_COLS), dtype=np.float32),
                0.0,
                "Contract Creation / Null Destination",
            )
        pred = self.predict(addr, allow_live=allow_live, model_type="tgae")
        x_scaled = np.array([pred.normalized_features[c] for c in ALL_FEATURE_COLS], dtype=np.float32)
        if self.tgae_model is not None:
            res = self.tgae_model.predict_ego_subgraph(x_scaled)
            z = res["latent_z"].astype(np.float32)
        else:
            z = np.zeros(latent_dim, dtype=np.float32)
        return z, x_scaled, float(pred.anomaly_score), pred.label_text

    def _lookup_transaction_record(self, clean_hash: str, allow_live: bool = True) -> tuple[dict[str, Any], str]:
        """Locate a transaction by 66-char hash across local sample Parquet, wallet caches, or live Etherscan."""
        if self.tx_path.exists():
            dataset = ds.dataset(str(self.tx_path), format="parquet")
            tbl = dataset.to_table(filter=ds.field("transaction_hash") == clean_hash)
            if tbl.num_rows > 0:
                return tbl.to_pylist()[0], "OFFLINE_SNAPSHOT"

        for p_file in self.live_cache_dir.glob("0x*.parquet"):
            try:
                tbl = ds.dataset(str(p_file), format="parquet").to_table(
                    filter=ds.field("transaction_hash") == clean_hash
                )
                if tbl.num_rows > 0:
                    return tbl.to_pylist()[0], "DISK_CACHE"
            except Exception:
                continue

        if allow_live:
            api_key = os.getenv("ETHERSCAN_API_KEY", "").strip()
            if api_key:
                from src.pipeline.etherscan_adapter import fetch_transaction_by_hash

                self._etherscan_limiter.acquire()
                tx_rec = fetch_transaction_by_hash(clean_hash, api_key=api_key)
                return tx_rec, "LIVE_ONCHAIN"

        raise KeyError(
            f"Transaction {clean_hash} not found in local dataset. Set ETHERSCAN_API_KEY to query Ethereum Mainnet live."
        )

    def _score_tx_dict(
        self,
        tx_row: dict[str, Any],
        source: str = "OFFLINE_SNAPSHOT",
        allow_live_endpoints: bool = True,
    ) -> TransactionAnomalyPrediction:
        t0 = time.perf_counter()
        tx_hash = str(tx_row.get("transaction_hash") or tx_row.get("hash") or "").strip().lower()
        src_addr = _clean_address_value(tx_row.get("from_address") or tx_row.get("from")) or ""
        dst_addr = _clean_address_value(tx_row.get("to_address") or tx_row.get("to"))

        val_eth = float(tx_row.get("value_eth") or 0.0)
        if not np.isfinite(val_eth):
            val_eth = 0.0
        gas_used = int(tx_row.get("gas_used") or tx_row.get("gasUsed") or 0)
        gas_price_wei = float(tx_row.get("gas_price") or tx_row.get("gasPrice") or 0.0)
        gas_price_gwei = round(gas_price_wei / 1e9, 4)
        block_num = int(tx_row.get("block_number") or tx_row.get("blockNumber") or 0)
        ts_str = str(tx_row.get("timestamp") or tx_row.get("timeStamp") or "")
        success = bool(tx_row.get("success", True))
        method_id = str(tx_row.get("input_method_id") or "0x").lower()
        method_label = EVM_METHOD_LABELS.get(method_id, f"Smart Contract Call ({method_id})")

        z_u, x_u, s_u, lbl_u = self._get_node_embedding_and_profile(src_addr, allow_live=allow_live_endpoints)
        z_v, x_v, s_v, lbl_v = self._get_node_embedding_and_profile(dst_addr, allow_live=allow_live_endpoints)

        if self.tx_scorer is not None and self.tx_scorer.is_fitted_:
            scored = self.tx_scorer.score_single_transaction(
                z_u=z_u,
                z_v=z_v,
                x_u_scaled=x_u,
                x_v_scaled=x_v,
                s_u=s_u,
                s_v=s_v,
                value_eth=val_eth,
                gas_price_wei=gas_price_wei,
            )
            tx_score = float(scored["tx_anomaly_score"])
            tau = float(scored["threshold"])
            is_anom = bool(scored["is_anomaly"])
            comp = scored["component_breakdown"]
            val_spike = float(scored["val_spike_delta"])
        else:
            tx_score = round(max(s_u, s_v), 4)
            tau = 0.50
            is_anom = tx_score >= tau
            comp = {"sender_risk_pct": 45.0, "receiver_risk_pct": 45.0, "tx_dynamics_pct": 10.0}
            val_spike = 0.0

        if tx_score >= max(0.75, tau * 1.15):
            risk_level = "CRITICAL"
        elif is_anom:
            risk_level = "HIGH"
        elif tx_score >= tau * 0.70:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        signals: list[dict[str, Any]] = [
            {
                "signal": "Sender Deep Embedding Risk (z_u)",
                "value": f"{s_u:.4f}",
                "status": "ELEVATED" if s_u >= 0.5 else "NORMAL",
                "detail": f"{src_addr[:10]}... ({lbl_u})",
            },
            {
                "signal": "Receiver Deep Embedding Risk (z_v)",
                "value": f"{s_v:.4f}",
                "status": "ELEVATED" if s_v >= 0.5 else "NORMAL",
                "detail": f"{(dst_addr or 'Contract Creation')[:10]}... ({lbl_v})",
            },
            {
                "signal": "Value Shock vs. Sender Baseline",
                "value": f"{val_eth:.4f} ETH (Δ={val_spike:+.2f})",
                "status": "SPIKE" if abs(val_spike) >= 1.5 else "TYPICAL",
                "detail": "Compares log transaction value against sender historical average",
            },
            {
                "signal": "EVM Execution Semantics",
                "value": method_label,
                "status": "WATCH" if (val_eth == 0.0 or not success or method_id in {"0x095ea7b3", "0xa22cb465", "0xd505accf"}) else "STANDARD",
                "detail": f"Gas used: {gas_used:,} @ {gas_price_gwei:.2f} Gwei | Success={success}",
            },
        ]

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return TransactionAnomalyPrediction(
            tx_hash=tx_hash,
            from_address=src_addr,
            to_address=dst_addr,
            value_eth=round(val_eth, 6),
            gas_used=gas_used,
            gas_price_gwei=gas_price_gwei,
            timestamp=ts_str,
            block_number=block_num,
            success=success,
            input_method_id=method_id,
            method_label=method_label,
            anomaly_score=round(tx_score, 4),
            is_anomaly=is_anom,
            threshold=round(tau, 4),
            risk_level=risk_level,
            sender_score=round(s_u, 4),
            receiver_score=round(s_v, 4),
            sender_label_text=lbl_u,
            receiver_label_text=lbl_v,
            component_breakdown=comp,
            forensic_signals=signals,
            inference_time_ms=elapsed_ms,
            source=source,
            metadata={"model_type": "deep_tx_scorer_tgae"},
        )

    def predict_transaction(self, tx_hash: str, allow_live: bool = True) -> TransactionAnomalyPrediction:
        """Evaluate anomaly risk of a single Ethereum transaction (`0x` + 64 hex chars) using Deep Learning."""
        clean_hash = validate_tx_hash(tx_hash)
        cached = self._tx_prediction_cache.get(clean_hash)
        if cached is not None:
            return cached

        tx_row, src = self._lookup_transaction_record(clean_hash, allow_live=allow_live)
        pred = self._score_tx_dict(tx_row, source=src, allow_live_endpoints=allow_live)
        self._tx_prediction_cache.set(clean_hash, pred)
        return pred

    def score_wallet_transactions(self, address: str, max_txs: int = 20) -> list[dict[str, Any]]:
        """Score up to ``max_txs`` recent transactions of ``address`` and rank from highest risk to lowest."""
        clean_addr = validate_ethereum_address(address)
        tx_df = self._get_local_wallet_transactions(clean_addr)
        if tx_df.empty and os.getenv("ETHERSCAN_API_KEY", "").strip():
            tx_df, _ = self._fetch_live_transactions(clean_addr)
        if tx_df.empty:
            return []

        rows = tx_df.to_dict(orient="records")
        rows.sort(
            key=lambda r: _parse_single_timestamp(r.get("timestamp") or r.get("timeStamp")),
            reverse=True,
        )

        scored_list: list[dict[str, Any]] = []
        for row in rows[: max(1, int(max_txs))]:
            tx_pred = self._score_tx_dict(row, source="WALLET_HISTORY", allow_live_endpoints=False)
            direction = "OUT" if tx_pred.from_address == clean_addr else "IN"
            counterparty = tx_pred.to_address if direction == "OUT" else tx_pred.from_address
            scored_list.append(
                {
                    "tx_hash": tx_pred.tx_hash,
                    "direction": direction,
                    "counterparty": counterparty or "Contract Creation",
                    "value_eth": tx_pred.value_eth,
                    "tx_anomaly_score": tx_pred.anomaly_score,
                    "risk_level": tx_pred.risk_level,
                    "is_anomaly": tx_pred.is_anomaly,
                    "timestamp": tx_pred.timestamp,
                }
            )

        scored_list.sort(key=lambda item: float(item["tx_anomaly_score"]), reverse=True)
        return scored_list

    def get_ego_network(self, address: str, max_edges: int = 50) -> EgoNetworkGraph:
        clean_addr = validate_ethereum_address(address)

        tx_rows: list[dict[str, Any]] = []
        df_local = self._get_local_wallet_transactions(clean_addr)
        if not df_local.empty:
            tx_rows = df_local.to_dict(orient="records")
        elif os.getenv("ETHERSCAN_API_KEY", "").strip():
            live_df, _ = self._fetch_live_transactions(clean_addr)
            if not live_df.empty:
                tx_rows = live_df.to_dict(orient="records")

        center_node = {
            "id": clean_addr,
            "label": f"CENTER\n{clean_addr[:6]}...{clean_addr[-4:]}",
            "type": "center",
            "color": "#ef4444",
            "size": 25,
        }

        if not tx_rows:
            return EgoNetworkGraph(
                center_address=clean_addr,
                nodes=[center_node],
                edges=[],
                total_volume_eth=0.0,
                in_degree=0,
                out_degree=0,
                unique_counterparties=0,
            )

        nodes_map: dict[str, dict[str, Any]] = {clean_addr: center_node}
        edges: list[dict[str, Any]] = []
        total_vol = 0.0
        in_deg = 0
        out_deg = 0
        counterparties: set[str] = set()

        tx_rows.sort(
            key=lambda row: _parse_single_timestamp(row.get("timestamp") or row.get("timeStamp")),
            reverse=True,
        )

        for tx in tx_rows[:max_edges]:
            src = _clean_address_value(tx.get("from_address") if "from_address" in tx else tx.get("from"))
            dst = _clean_address_value(tx.get("to_address") if "to_address" in tx else tx.get("to"))

            if not src or not dst:
                continue

            raw_val = tx.get("value_eth", 0.0)
            try:
                val_eth = float(raw_val) if raw_val is not None and not pd.isna(raw_val) else 0.0
            except (TypeError, ValueError):
                val_eth = 0.0
            if not np.isfinite(val_eth):
                val_eth = 0.0

            if src == clean_addr and dst == clean_addr:
                direction = "SELF"
                out_deg += 1
                in_deg += 1
                target_peer = None
            elif src == clean_addr:
                out_deg += 1
                direction = "OUT"
                target_peer = dst
            elif dst == clean_addr:
                in_deg += 1
                direction = "IN"
                target_peer = src
            else:
                continue

            total_vol += val_eth

            if target_peer is not None:
                counterparties.add(target_peer)
                if target_peer not in nodes_map:
                    nodes_map[target_peer] = {
                        "id": target_peer,
                        "label": f"{target_peer[:6]}...{target_peer[-4:]}",
                        "type": "peer",
                        "color": "#3b82f6" if direction == "IN" else "#10b981",
                        "size": 15,
                    }

            raw_gas = tx.get("gas_used") if "gas_used" in tx else tx.get("gasUsed")
            try:
                gas_used = int(raw_gas) if raw_gas is not None and not pd.isna(raw_gas) else 0
            except (TypeError, ValueError):
                gas_used = 0

            edges.append(
                {
                    "source": src,
                    "target": dst,
                    "value_eth": round(val_eth, 4),
                    "timestamp": str(tx.get("timestamp") or tx.get("timeStamp") or ""),
                    "tx_hash": str(tx.get("transaction_hash") or tx.get("hash") or ""),
                    "gas_used": gas_used,
                    "success": bool(tx.get("success", True)),
                    "direction": direction,
                }
            )

        return EgoNetworkGraph(
            center_address=clean_addr,
            nodes=list(nodes_map.values()),
            edges=edges,
            total_volume_eth=round(total_vol, 4),
            in_degree=in_deg,
            out_degree=out_deg,
            unique_counterparties=len(counterparties),
        )


_engine_instance: EthereumGuardEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> EthereumGuardEngine:
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = EthereumGuardEngine()
    return _engine_instance
