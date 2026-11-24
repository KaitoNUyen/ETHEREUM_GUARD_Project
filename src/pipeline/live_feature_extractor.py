"""
ETHEREUM GUARD - Real-time Live Wallet Feature Extractor.
Extracts 25 canonical features and ego-graph topology from on-chain transactions
with 100% data contract compatibility with the pre-trained model and scalers.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.pipeline.feature_transform import (
    LOG1P_COLS,
    LOG_COLS,
    SIGNED_LOG_COLS,
    transform_feature_frame,
)
from src.schemas import (
    ALL_FEATURE_COLS,
    GRAPH_TOPO_COLS,
    TABULAR_TX_COLS,
    TEMPORAL_STAT_COLS,
)

logger = logging.getLogger("ethereum_guard.live_extractor")

WEI_PER_ETH = Decimal(10) ** 18
ADDRESS_RE = re.compile(r"^0x[a-f0-9]{40}$")
_NULL_ADDRESS_TOKENS = {"", "none", "nan", "null", "nat", "<na>"}


def _clean_address_value(value: Any) -> str | None:
    """Normalize an Ethereum address value, mapping null/NaN tokens cleanly to None."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in _NULL_ADDRESS_TOKENS:
        return None
    return text


def _clean_address_series(series: pd.Series) -> pd.Series:
    """Vectorized null-safe address normalization (never turns None/NaN into 'none'/'nan')."""
    out = series.astype("string").str.strip().str.lower()
    invalid_mask = out.isna() | out.isin(_NULL_ADDRESS_TOKENS)
    return out.mask(invalid_mask, pd.NA)


def _parse_timestamp_to_unix_seconds(series: pd.Series) -> pd.Series:
    """
    Convert Unix int/float/string (seconds or milliseconds) or datetime64[s/ms/us/ns]
    into exact Unix seconds (int64-compatible float64 with NaN for invalid rows).
    Matches src/pipeline/features.py:timestamp_to_unix_seconds resolution handling.
    """
    if series.empty:
        return pd.Series(dtype=np.float64, index=series.index)

    if pd.api.types.is_numeric_dtype(series):
        nums = pd.to_numeric(series, errors="coerce").astype(np.float64)
        # Normalize millisecond Unix timestamps (e.g., 1.7e12) down to seconds
        nums = nums.where(nums.abs() < 1e11, nums / 1000.0)
        return nums

    # Handle string-encoded numeric Unix timestamps (e.g., Etherscan timeStamp="1700000000")
    stripped = series.astype("string").str.strip()
    is_numeric_str = stripped.str.fullmatch(r"\d+(\.\d+)?").fillna(False)
    if is_numeric_str.any() and is_numeric_str.all():
        nums = pd.to_numeric(stripped, errors="coerce").astype(np.float64)
        nums = nums.where(nums.abs() < 1e11, nums / 1000.0)
        return nums

    ts = pd.to_datetime(series, utc=True, errors="coerce")
    unit = getattr(ts.dtype, "unit", "ns")
    divisors = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
    if unit not in divisors:
        raise ValueError(f"Unsupported timestamp resolution: {unit}")
    raw = ts.astype("int64").to_numpy(dtype=np.int64)
    valid = ts.notna().to_numpy()
    seconds = raw // divisors[unit]
    out = pd.Series(seconds.astype(np.float64), index=series.index)
    out.loc[~valid] = np.nan
    return out


def _wei_to_eth_safe(value: Any) -> float:
    """Convert raw Wei string/int to ETH float using Decimal to avoid precision loss."""
    if value is None:
        return 0.0
    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        return float(Decimal(str(value).strip()) / WEI_PER_ETH)
    except (InvalidOperation, ValueError):
        return 0.0


def _bundle_uses_log_transform(bundle: dict[str, Any] | None) -> bool:
    """
    Detect whether a persisted scaler bundle was fitted after transform_feature_frame().
    Prevents train/serve skew when legacy un-logged scalers coexist with log10 scalers.
    """
    if not isinstance(bundle, dict):
        return False
    transform_tag = str(bundle.get("transform") or "").strip().lower()
    return "log10" in transform_tag or "log" in transform_tag


class LiveFeatureExtractor:
    """
    Computes exact 25 canonical features for an arbitrary Ethereum address from its transaction list.
    Preserves exact mathematical definitions and scaling contract of the loaded scaler artifacts.
    """

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[2]
        self.tabular_scaler_path = (
            self.root_dir / "artifacts" / "scalers" / "feature_robust_scaler.joblib"
        )
        self.graph_scaler_path = (
            self.root_dir / "artifacts" / "scalers" / "graph_feature_scaler.joblib"
        )

        self.tab_scaler = None
        self.tab_cols: list[str] = []
        self.tab_uses_log_transform: bool = False

        self.graph_scaler = None
        self.graph_cols: list[str] = []
        self.graph_uses_log_transform: bool = False

        self._load_scalers()

    def _load_scalers(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            if self.tabular_scaler_path.exists():
                tab_pack = joblib.load(self.tabular_scaler_path)
                self.tab_scaler = tab_pack["scaler"]
                self.tab_cols = list(tab_pack["columns"])
                self.tab_uses_log_transform = _bundle_uses_log_transform(tab_pack)
                logger.info(
                    "Loaded tabular RobustScaler (%d cols, log_transform=%s)",
                    len(self.tab_cols),
                    self.tab_uses_log_transform,
                )
            else:
                logger.warning("Tabular RobustScaler not found at %s", self.tabular_scaler_path)

            if self.graph_scaler_path.exists():
                graph_pack = joblib.load(self.graph_scaler_path)
                self.graph_scaler = graph_pack["scaler"]
                self.graph_cols = list(graph_pack["columns"])
                self.graph_uses_log_transform = _bundle_uses_log_transform(graph_pack)
                logger.info(
                    "Loaded graph RobustScaler (%d cols, log_transform=%s)",
                    len(self.graph_cols),
                    self.graph_uses_log_transform,
                )
            else:
                logger.warning("Graph RobustScaler not found at %s", self.graph_scaler_path)

    def extract_features(
        self,
        target_address: str,
        transactions_df: pd.DataFrame,
        window_hours: int = 24,
        num_windows: int = 10,
        max_transactions: int = 10_000,
    ) -> tuple[dict[str, float], np.ndarray]:
        """
        Extract canonical 25 raw features and return (raw_feature_dict, scaled_vector_25d).
        """
        clean_addr = _clean_address_value(target_address)
        if not clean_addr:
            raise ValueError(f"Invalid target address: {target_address!r}")

        # Handle edge case: Wallet with zero transactions
        if transactions_df is None or transactions_df.empty:
            return self._build_empty_wallet_features(clean_addr)

        df = transactions_df.copy()

        # Standardize address columns (null-safe; never convert None/NaN to "none"/"nan")
        if "from_address" not in df.columns and "from" in df.columns:
            df["from_address"] = df["from"]
        if "to_address" not in df.columns and "to" in df.columns:
            df["to_address"] = df["to"]

        if "from_address" not in df.columns:
            return self._build_empty_wallet_features(clean_addr)
        if "to_address" not in df.columns:
            df["to_address"] = pd.NA

        df["from_address"] = _clean_address_series(df["from_address"])
        df["to_address"] = _clean_address_series(df["to_address"])

        # Filter transactions involving the target wallet early (reduces work on multi-wallet frames)
        is_sender = df["from_address"].eq(clean_addr).fillna(False)
        is_receiver = df["to_address"].eq(clean_addr).fillna(False)
        tx = df.loc[is_sender | is_receiver].copy()

        if tx.empty:
            return self._build_empty_wallet_features(clean_addr)

        # Standardize timestamps using resolution-aware converter (fixes ms/us/s corruption)
        if "timestamp" in tx.columns:
            tx["timestamp"] = _parse_timestamp_to_unix_seconds(tx["timestamp"])
        elif "timeStamp" in tx.columns:
            tx["timestamp"] = _parse_timestamp_to_unix_seconds(tx["timeStamp"])
        else:
            tx["timestamp"] = np.nan

        tx = tx.dropna(subset=["timestamp", "from_address"])
        if tx.empty:
            return self._build_empty_wallet_features(clean_addr)
        tx["timestamp"] = tx["timestamp"].astype(np.int64)

        # Deduplicate by transaction_hash if available
        if "transaction_hash" in tx.columns:
            tx = tx.drop_duplicates(subset=["transaction_hash"], keep="last")
        elif "hash" in tx.columns:
            tx = tx.drop_duplicates(subset=["hash"], keep="last")

        # Bound maximum transactions analyzed (keep most recent window for deterministic latency/RAM)
        if max_transactions > 0 and len(tx) > max_transactions:
            tx = tx.nlargest(max_transactions, "timestamp")

        # Standardize ETH values
        if "value_eth" in tx.columns:
            tx["value_eth"] = (
                pd.to_numeric(tx["value_eth"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                .astype(np.float64)
            )
        elif "value_wei" in tx.columns:
            tx["value_eth"] = tx["value_wei"].map(_wei_to_eth_safe).astype(np.float64)
        elif "value" in tx.columns:
            tx["value_eth"] = tx["value"].map(_wei_to_eth_safe).astype(np.float64)
        else:
            tx["value_eth"] = 0.0

        # Standardize gas price
        if "gas_price" not in tx.columns and "gasPrice" in tx.columns:
            tx["gas_price"] = pd.to_numeric(tx["gasPrice"], errors="coerce").fillna(0.0)
        elif "gas_price" not in tx.columns:
            tx["gas_price"] = 0.0
        else:
            tx["gas_price"] = pd.to_numeric(tx["gas_price"], errors="coerce").fillna(0.0)

        # Standardize success flag
        if "success" in tx.columns:
            tx["success"] = tx["success"].fillna(False).astype(bool)
        elif "txreceipt_status" in tx.columns:
            rs = tx["txreceipt_status"].astype("string").str.strip()
            is_err = (
                tx["isError"].astype("string").str.strip()
                if "isError" in tx.columns
                else pd.Series("0", index=tx.index, dtype="string")
            )
            tx["success"] = np.where(rs.isin(["0", "1"]), rs == "1", is_err == "0")
        elif "isError" in tx.columns:
            tx["success"] = tx["isError"].astype("string").str.strip() == "0"
        else:
            tx["success"] = True

        # -------------------------------------------------------------
        # 1. Tabular Transaction Metrics (12 features)
        #    Exact wallet-side event parity with src/pipeline/features.py
        # -------------------------------------------------------------
        out_txs = tx.loc[tx["from_address"].eq(clean_addr).fillna(False)]
        in_txs = tx.loc[tx["to_address"].eq(clean_addr).fillna(False)]

        tx_sent_count = float(len(out_txs))
        tx_recv_count = float(len(in_txs))
        event_total = tx_sent_count + tx_recv_count

        total_eth_sent = float(out_txs["value_eth"].sum()) if tx_sent_count > 0 else 0.0
        total_eth_received = float(in_txs["value_eth"].sum()) if tx_recv_count > 0 else 0.0
        net_balance_diff = total_eth_received - total_eth_sent

        avg_eth_sent = float(out_txs["value_eth"].mean()) if tx_sent_count > 0 else 0.0
        max_eth_sent = float(out_txs["value_eth"].max()) if tx_sent_count > 0 else 0.0
        avg_eth_received = float(in_txs["value_eth"].mean()) if tx_recv_count > 0 else 0.0
        max_eth_received = float(in_txs["value_eth"].max()) if tx_recv_count > 0 else 0.0

        # Exact formula from features.py:235-237: (sent + 1) / (recv + 1)
        sent_recv_ratio = (tx_sent_count + 1.0) / (tx_recv_count + 1.0)

        # Wallet-side event sums (matches features.py:240-247 on self-loops too)
        zero_total = float((out_txs["value_eth"] == 0.0).sum() + (in_txs["value_eth"] == 0.0).sum())
        failed_total = float((~out_txs["success"]).sum() + (~in_txs["success"]).sum())
        zero_val_tx_ratio = zero_total / max(1.0, event_total)
        failed_tx_ratio = failed_total / max(1.0, event_total)

        # -------------------------------------------------------------
        # 2. Graph Topology Metrics (6 features)
        #    Exact non-self valid directed edge policy matching graph_builder.py:157-160
        # -------------------------------------------------------------
        valid_directed_out = out_txs.loc[
            out_txs["to_address"].notna() & out_txs["to_address"].ne(clean_addr)
        ]
        valid_directed_in = in_txs.loc[
            in_txs["from_address"].notna() & in_txs["from_address"].ne(clean_addr)
        ]

        in_degree = float(len(valid_directed_in))
        out_degree = float(len(valid_directed_out))

        unique_in_counterparties = set(valid_directed_in["from_address"].dropna().tolist())
        unique_out_counterparties = set(valid_directed_out["to_address"].dropna().tolist())

        unique_in_neighbors = float(len(unique_in_counterparties))
        unique_out_neighbors = float(len(unique_out_counterparties))

        # Formula from graph_builder.py:795-797
        repeat_denom = in_degree + out_degree + 1.0
        neighbor_repeat_ratio = float(
            np.clip(
                1.0 - (unique_in_neighbors + unique_out_neighbors) / repeat_denom,
                0.0,
                1.0,
            )
        )

        # Directed 1-hop ego density lower bound (center <-> neighbors; inter-neighbor edges unavailable in 1-wallet txlist)
        all_neighbors = unique_in_counterparties | unique_out_counterparties
        k_nodes = 1 + len(all_neighbors)
        if k_nodes <= 1:
            local_ego_density = 0.0
        else:
            observed_directed_edges = len(unique_in_counterparties) + len(unique_out_counterparties)
            local_ego_density = float(
                min(1.0, observed_directed_edges / float(k_nodes * (k_nodes - 1)))
            )

        # -------------------------------------------------------------
        # 3. Temporal Dynamic Metrics (7 features)
        #    Exact wallet-side event ordering & ddof=1 std matching features.py:250-285
        # -------------------------------------------------------------
        event_timestamps = np.concatenate([
            out_txs["timestamp"].to_numpy(dtype=np.int64),
            in_txs["timestamp"].to_numpy(dtype=np.int64),
        ])
        event_timestamps.sort(kind="mergesort")

        first_seen = float(event_timestamps[0])
        last_seen = float(event_timestamps[-1])

        active_duration_hours = max(0.0, (last_seen - first_seen) / 3600.0)
        active_days = max(1.0 / 24.0, (last_seen - first_seen) / 86400.0)
        tx_frequency_per_day = event_total / active_days

        if len(event_timestamps) > 1:
            inter_arrivals = np.diff(event_timestamps).astype(np.float64)
            mean_inter_tx_time = float(np.mean(inter_arrivals))
            # Match pandas Series.std(ddof=1) in features.py:258 (returns 0.0 when only 1 delta)
            std_inter_tx_time = (
                float(np.std(inter_arrivals, ddof=1)) if len(inter_arrivals) > 1 else 0.0
            )
            denom = std_inter_tx_time + mean_inter_tx_time
            burstiness_index = float(
                (std_inter_tx_time - mean_inter_tx_time) / denom if denom > 1e-9 else 0.0
            )
        else:
            mean_inter_tx_time = 0.0
            std_inter_tx_time = 0.0
            burstiness_index = 0.0

        # Recent spike calculation relative to wallet's last observed activity
        window_seconds = window_hours * 3600
        cutoff = int(last_seen)
        history_start = cutoff - num_windows * window_seconds
        recent_start = cutoff - window_seconds

        recent_txs = out_txs.loc[out_txs["timestamp"] > recent_start]
        history_txs = out_txs.loc[
            (out_txs["timestamp"] > history_start) & (out_txs["timestamp"] <= recent_start)
        ]

        recent_value = float(recent_txs["value_eth"].sum()) if not recent_txs.empty else 0.0
        recent_count = float(len(recent_txs))

        n_hist_windows = max(num_windows - 1, 1)
        history_value = float(history_txs["value_eth"].sum()) if not history_txs.empty else 0.0
        history_count = float(len(history_txs))

        base_vol = history_value / n_hist_windows
        base_cnt = history_count / n_hist_windows

        recent_volume_spike_ratio = float(recent_value / base_vol if base_vol > 0.0 else 0.0)
        recent_tx_spike_ratio = float(recent_count / base_cnt if base_cnt > 0.0 else 0.0)

        raw_features: dict[str, float] = {
            "total_eth_sent": total_eth_sent,
            "total_eth_received": total_eth_received,
            "net_balance_diff": net_balance_diff,
            "avg_eth_sent": avg_eth_sent,
            "max_eth_sent": max_eth_sent,
            "avg_eth_received": avg_eth_received,
            "max_eth_received": max_eth_received,
            "tx_sent_count": tx_sent_count,
            "tx_recv_count": tx_recv_count,
            "sent_recv_ratio": sent_recv_ratio,
            "zero_val_tx_ratio": zero_val_tx_ratio,
            "failed_tx_ratio": failed_tx_ratio,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "unique_in_neighbors": unique_in_neighbors,
            "unique_out_neighbors": unique_out_neighbors,
            "neighbor_repeat_ratio": neighbor_repeat_ratio,
            "local_ego_density": local_ego_density,
            "active_duration_hours": active_duration_hours,
            "tx_frequency_per_day": tx_frequency_per_day,
            "mean_inter_tx_time": mean_inter_tx_time,
            "std_inter_tx_time": std_inter_tx_time,
            "burstiness_index": burstiness_index,
            "recent_volume_spike_ratio": recent_volume_spike_ratio,
            "recent_tx_spike_ratio": recent_tx_spike_ratio,
        }

        # Sanitize any non-finite numbers before scaling
        raw_features = {
            k: (0.0 if not np.isfinite(v) else float(v))
            for k, v in raw_features.items()
        }

        scaled_vec = self._scale_features(raw_features)
        return raw_features, scaled_vec

    def _scale_features(self, raw_features: dict[str, float]) -> np.ndarray:
        """
        Applies log transforms ONLY when the loaded scaler bundle was fitted with them,
        then applies the train-fitted RobustScalers in exact ALL_FEATURE_COLS order.
        """
        df_row = pd.DataFrame([raw_features])

        # 1. Scale 19 tabular + temporal features
        if self.tab_scaler is not None and self.tab_cols:
            tab_df = df_row[self.tab_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if self.tab_uses_log_transform:
                if "sent_recv_ratio" in tab_df.columns:
                    tab_df["sent_recv_ratio"] = tab_df["sent_recv_ratio"].clip(lower=1e-6)
                tab_df = transform_feature_frame(tab_df, self.tab_cols)
            tab_scaled = self.tab_scaler.transform(tab_df)
            tab_res = dict(zip(self.tab_cols, tab_scaled[0]))
        else:
            tab_res = {col: raw_features[col] for col in (TABULAR_TX_COLS + TEMPORAL_STAT_COLS)}

        # 2. Scale 6 graph topology features
        if self.graph_scaler is not None and self.graph_cols:
            graph_df = df_row[self.graph_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if self.graph_uses_log_transform:
                graph_df = transform_feature_frame(graph_df, self.graph_cols)
            graph_scaled = self.graph_scaler.transform(graph_df)
            graph_res = dict(zip(self.graph_cols, graph_scaled[0]))
        else:
            graph_res = {col: raw_features[col] for col in GRAPH_TOPO_COLS}

        combined_scaled = {**tab_res, **graph_res}
        vector_25d = np.array(
            [[combined_scaled[col] for col in ALL_FEATURE_COLS]],
            dtype=np.float32,
        )
        vector_25d = np.nan_to_num(vector_25d, nan=0.0, posinf=0.0, neginf=0.0)
        return vector_25d

    def inverse_scale_features(self, scaled_features: dict[str, float]) -> dict[str, float]:
        """
        Invert RobustScaler (and log10 transforms when the scaler bundle used them)
        so offline snapshot records expose true human-readable raw feature units.
        """
        out: dict[str, float] = {col: float(scaled_features.get(col, 0.0) or 0.0) for col in ALL_FEATURE_COLS}

        def _invert_log(col: str, val: float) -> float:
            if not np.isfinite(val):
                return 0.0
            clipped = float(np.clip(val, -30.0, 30.0))
            if col in LOG1P_COLS:
                return max(0.0, float(10.0**clipped - 1.0))
            if col in SIGNED_LOG_COLS:
                return float(np.sign(clipped) * (10.0 ** abs(clipped) - 1.0))
            if col in LOG_COLS:
                return float(10.0**clipped)
            return float(val)

        if self.tab_scaler is not None and self.tab_cols:
            tab_arr = np.array([[out[col] for col in self.tab_cols]], dtype=np.float64)
            tab_unscaled = self.tab_scaler.inverse_transform(tab_arr)[0]
            for col, val in zip(self.tab_cols, tab_unscaled):
                raw_val = _invert_log(col, float(val)) if self.tab_uses_log_transform else float(val)
                out[col] = 0.0 if not np.isfinite(raw_val) else round(raw_val, 6)

        if self.graph_scaler is not None and self.graph_cols:
            grp_arr = np.array([[out[col] for col in self.graph_cols]], dtype=np.float64)
            grp_unscaled = self.graph_scaler.inverse_transform(grp_arr)[0]
            for col, val in zip(self.graph_cols, grp_unscaled):
                raw_val = _invert_log(col, float(val)) if self.graph_uses_log_transform else float(val)
                out[col] = 0.0 if not np.isfinite(raw_val) else round(raw_val, 6)

        return out

    def _build_empty_wallet_features(self, address: str) -> tuple[dict[str, float], np.ndarray]:
        """Provides normalized representation for virgin address with 0 transactions."""
        raw = {col: 0.0 for col in ALL_FEATURE_COLS}
        raw["sent_recv_ratio"] = 1.0  # (0+1)/(0+1)
        scaled_vec = self._scale_features(raw)
        return raw, scaled_vec

    def extract_multimodal_ego(
        self,
        target_address: str,
        transactions_df: pd.DataFrame | None,
        window_hours: int = 24,
        num_windows: int = 10,
        max_neighbors: int = 32,
    ) -> dict[str, Any]:
        """
        Extract full Tri-Modal inputs for ModularTGAE live/ego inference:
          - raw_features: 25-feature dict
          - x_ego: (N_ego, 25) float32 array (row 0 = target wallet, rows 1..K = 1-hop counterparties)
          - temporal_seq_10x8: (1, num_windows, 8) float32 trailing temporal sequence
          - edge_index_np: (2, E_ego) int64 directed 1-hop edge index
          - edge_attr_np: (E_ego, 5) float32 edge attributes matching graph_builder.py
        """
        raw_features, target_vec = self.extract_features(
            target_address,
            transactions_df if transactions_df is not None else pd.DataFrame(),
            window_hours=window_hours,
            num_windows=num_windows,
        )
        empty_result = {
            "raw_features": raw_features,
            "x_ego": target_vec,
            "temporal_seq_10x8": np.zeros((1, num_windows, 8), dtype=np.float32),
            "edge_index_np": np.empty((2, 0), dtype=np.int64),
            "edge_attr_np": np.empty((0, 5), dtype=np.float32),
        }
        clean_addr = _clean_address_value(target_address)
        if not clean_addr or transactions_df is None or transactions_df.empty:
            return empty_result

        df = transactions_df.copy()
        if "from_address" not in df.columns and "from" in df.columns:
            df["from_address"] = df["from"]
        if "to_address" not in df.columns and "to" in df.columns:
            df["to_address"] = df["to"]
        if "from_address" not in df.columns:
            return empty_result
        if "to_address" not in df.columns:
            df["to_address"] = pd.NA

        df["from_address"] = _clean_address_series(df["from_address"])
        df["to_address"] = _clean_address_series(df["to_address"])
        tx = df.loc[
            df["from_address"].eq(clean_addr).fillna(False)
            | df["to_address"].eq(clean_addr).fillna(False)
        ].copy()
        if tx.empty:
            return empty_result

        ts_col = "timestamp" if "timestamp" in tx.columns else ("timeStamp" if "timeStamp" in tx.columns else None)
        if ts_col is None:
            return empty_result
        tx["timestamp"] = _parse_timestamp_to_unix_seconds(tx[ts_col])
        tx = tx.dropna(subset=["timestamp", "from_address"])
        if tx.empty:
            return empty_result
        tx["timestamp"] = tx["timestamp"].astype(np.int64)

        if "value_eth" in tx.columns:
            tx["value_eth"] = pd.to_numeric(tx["value_eth"], errors="coerce").fillna(0.0).clip(lower=0.0)
        elif "value_wei" in tx.columns:
            tx["value_eth"] = tx["value_wei"].map(_wei_to_eth_safe)
        else:
            tx["value_eth"] = 0.0

        gp_col = "gas_price" if "gas_price" in tx.columns else ("gasPrice" if "gasPrice" in tx.columns else None)
        tx["gas_gwei"] = (
            pd.to_numeric(tx[gp_col], errors="coerce").fillna(0.0).clip(lower=0.0) / 1e9
            if gp_col is not None
            else 0.0
        )

        # 1. Build (1, num_windows, 8) trailing temporal sequence
        cutoff = int(tx["timestamp"].max())
        min_ts = int(tx["timestamp"].min())
        span_sec = max(1.0, float(cutoff - min_ts))
        win_sec = max(1, int(window_hours * 3600))
        start_ts = cutoff - num_windows * win_sec

        seq = np.zeros((1, num_windows, 8), dtype=np.float32)
        seq_tx = tx.loc[(tx["timestamp"] > start_ts) & (tx["timestamp"] <= cutoff)].copy()
        if not seq_tx.empty:
            seq_tx["w"] = np.clip((seq_tx["timestamp"] - start_ts) // win_sec, 0, num_windows - 1).astype(int)
            for w_idx, grp in seq_tx.groupby("w", sort=False):
                out_g = grp.loc[grp["from_address"].eq(clean_addr).fillna(False)]
                in_g = grp.loc[grp["to_address"].eq(clean_addr).fillna(False)]
                peers_w = set(out_g["to_address"].dropna().tolist()) | set(in_g["from_address"].dropna().tolist())
                peers_w.discard(clean_addr)
                ts_sorted = np.sort(grp["timestamp"].to_numpy(dtype=np.int64))
                mean_iat = float(np.mean(np.diff(ts_sorted))) if len(ts_sorted) > 1 else 0.0

                seq[0, int(w_idx), 0] = float(len(in_g))
                seq[0, int(w_idx), 1] = float(len(out_g))
                seq[0, int(w_idx), 2] = float(in_g["value_eth"].sum())
                seq[0, int(w_idx), 3] = float(out_g["value_eth"].sum())
                seq[0, int(w_idx), 4] = float(grp["value_eth"].max())
                seq[0, int(w_idx), 5] = float(len(peers_w))
                seq[0, int(w_idx), 6] = float(grp["gas_gwei"].mean())
                seq[0, int(w_idx), 7] = mean_iat

        # 2. Build 1-hop directed ego graph (target = node 0, top counterparties = 1..K)
        valid_edges = tx.loc[
            tx["to_address"].notna() & (tx["from_address"] != tx["to_address"])
        ].copy()
        if valid_edges.empty:
            return {
                "raw_features": raw_features,
                "x_ego": target_vec,
                "temporal_seq_10x8": seq,
                "edge_index_np": np.empty((2, 0), dtype=np.int64),
                "edge_attr_np": np.empty((0, 5), dtype=np.float32),
            }

        valid_edges["peer"] = np.where(
            valid_edges["from_address"] == clean_addr,
            valid_edges["to_address"],
            valid_edges["from_address"],
        )
        top_peers = (
            valid_edges["peer"].dropna().value_counts().head(max_neighbors).index.tolist()
        )
        peer_to_idx = {p: i + 1 for i, p in enumerate(top_peers)}

        x_ego = np.repeat(self._build_empty_wallet_features(clean_addr)[1], 1 + len(top_peers), axis=0)
        x_ego[0] = target_vec[0]

        # Populate lightweight local scaled profile for each neighbor from observed ego interactions
        for p, p_idx in peer_to_idx.items():
            p_out = valid_edges.loc[valid_edges["from_address"] == p]
            p_in = valid_edges.loc[valid_edges["to_address"] == p]
            p_raw = {col: 0.0 for col in ALL_FEATURE_COLS}
            p_raw["total_eth_sent"] = float(p_out["value_eth"].sum())
            p_raw["total_eth_received"] = float(p_in["value_eth"].sum())
            p_raw["net_balance_diff"] = p_raw["total_eth_received"] - p_raw["total_eth_sent"]
            p_raw["tx_sent_count"] = float(len(p_out))
            p_raw["tx_recv_count"] = float(len(p_in))
            p_raw["sent_recv_ratio"] = (p_raw["tx_sent_count"] + 1.0) / (p_raw["tx_recv_count"] + 1.0)
            p_raw["in_degree"] = p_raw["tx_recv_count"]
            p_raw["out_degree"] = p_raw["tx_sent_count"]
            p_raw["unique_in_neighbors"] = 1.0 if len(p_in) > 0 else 0.0
            p_raw["unique_out_neighbors"] = 1.0 if len(p_out) > 0 else 0.0
            x_ego[p_idx] = self._scale_features(p_raw)[0]

        sub_edges = valid_edges.loc[valid_edges["peer"].isin(peer_to_idx)]
        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_attrs: list[list[float]] = []

        for (u_addr, v_addr), e_grp in sub_edges.groupby(["from_address", "to_address"], sort=False):
            u_id = 0 if u_addr == clean_addr else peer_to_idx.get(str(u_addr), -1)
            v_id = 0 if v_addr == clean_addr else peer_to_idx.get(str(v_addr), -1)
            if u_id < 0 or v_id < 0:
                continue
            cnt = len(e_grp)
            tot_v = float(e_grp["value_eth"].sum())
            mean_v = float(e_grp["value_eth"].mean())
            last_age = float(np.clip((cutoff - int(e_grp["timestamp"].max())) / span_sec, 0.0, 1.0))
            avg_g = float(e_grp["gas_gwei"].mean())

            edge_src.append(u_id)
            edge_dst.append(v_id)
            edge_attrs.append(
                [
                    float(np.log1p(cnt)),
                    float(np.log1p(max(0.0, tot_v))),
                    float(np.log1p(max(0.0, mean_v))),
                    last_age,
                    float(np.log1p(max(0.0, avg_g))),
                ]
            )

        if not edge_src:
            ei_np = np.empty((2, 0), dtype=np.int64)
            ea_np = np.empty((0, 5), dtype=np.float32)
        else:
            ei_np = np.array([edge_src, edge_dst], dtype=np.int64)
            ea_np = np.array(edge_attrs, dtype=np.float32)

        return {
            "raw_features": raw_features,
            "x_ego": x_ego,
            "temporal_seq_10x8": seq,
            "edge_index_np": ei_np,
            "edge_attr_np": ea_np,
        }
