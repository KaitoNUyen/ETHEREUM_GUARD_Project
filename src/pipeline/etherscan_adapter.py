"""
ETHEREUM GUARD - TV1-06
Etherscan API V2 adapter for one Ethereum wallet.

Usage (PowerShell):
    $env:ETHERSCAN_API_KEY="YOUR_KEY"
    python -m src.pipeline.etherscan_adapter 0xYourWalletAddress

The API key is read from the environment and is never written to disk.
Only normal/external transactions (account/txlist) are fetched here.
Internal transactions and ERC-20 transfers are intentionally separate data
types and are not silently mixed into the current canonical transaction schema.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 1
WEI_PER_ETH = Decimal(10) ** 18
ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")


class EtherscanError(RuntimeError):
    pass


def validate_address(address: str) -> str:
    address = address.strip()
    if not ADDRESS_RE.fullmatch(address):
        raise ValueError("Ethereum address must be 0x followed by 40 hexadecimal characters.")
    return address.lower()


def validate_tx_hash(tx_hash: str) -> str:
    tx_hash = tx_hash.strip()
    if not TX_HASH_RE.fullmatch(tx_hash):
        raise ValueError("Ethereum transaction hash must be 0x followed by 64 hexadecimal characters.")
    return tx_hash.lower()


def _rpc_proxy_get(api_key: str, action: str, extra_params: dict[str, str], timeout: int = 20) -> Any:
    params = {
        "chainid": str(CHAIN_ID),
        "module": "proxy",
        "action": action,
        **extra_params,
        "apikey": api_key,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "ethereum-guard/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if isinstance(payload, dict) and payload.get("error"):
        raise EtherscanError(f"Etherscan proxy error ({action}): {payload['error']}")
    return payload.get("result") if isinstance(payload, dict) else None


def fetch_transaction_by_hash(tx_hash: str, api_key: str, timeout: int = 20) -> dict[str, Any]:
    """Fetch a single Ethereum transaction + receipt + block timestamp via Etherscan V2 JSON-RPC proxy."""
    clean_hash = validate_tx_hash(tx_hash)
    if not api_key:
        raise ValueError("Missing ETHERSCAN_API_KEY environment variable.")

    tx_obj = _rpc_proxy_get(api_key, "eth_getTransactionByHash", {"txhash": clean_hash}, timeout=timeout)
    if not isinstance(tx_obj, dict) or not tx_obj.get("hash"):
        raise KeyError(f"Transaction {clean_hash} not found on Ethereum Mainnet.")

    receipt = _rpc_proxy_get(api_key, "eth_getTransactionReceipt", {"txhash": clean_hash}, timeout=timeout)
    block_hex = str(tx_obj.get("blockNumber") or "0x0")
    block_num = int(block_hex, 16) if block_hex.startswith("0x") else int(block_hex or 0)

    ts_unix = int(time.time())
    if block_hex and block_hex != "0x0":
        blk = _rpc_proxy_get(
            api_key,
            "eth_getBlockByNumber",
            {"tag": block_hex, "boolean": "false"},
            timeout=timeout,
        )
        if isinstance(blk, dict) and blk.get("timestamp"):
            ts_hex = str(blk["timestamp"])
            ts_unix = int(ts_hex, 16) if ts_hex.startswith("0x") else int(ts_hex)

    val_hex = str(tx_obj.get("value") or "0x0")
    val_wei_int = int(val_hex, 16) if val_hex.startswith("0x") else int(val_hex)
    val_eth = float(Decimal(val_wei_int) / WEI_PER_ETH)

    gp_hex = str(tx_obj.get("gasPrice") or "0x0")
    gas_price = int(gp_hex, 16) if gp_hex.startswith("0x") else int(gp_hex)

    gas_used_hex = str((receipt or {}).get("gasUsed") or tx_obj.get("gas") or "0x0")
    gas_used = int(gas_used_hex, 16) if gas_used_hex.startswith("0x") else int(gas_used_hex)

    status_hex = str((receipt or {}).get("status") or "0x1")
    success = (int(status_hex, 16) == 1) if status_hex.startswith("0x") else (status_hex == "1")

    raw_input = str(tx_obj.get("input") or "0x").lower()
    method_id = raw_input[:10] if len(raw_input) >= 10 else "0x"

    return {
        "transaction_hash": clean_hash,
        "block_number": block_num,
        "timestamp": datetime.fromtimestamp(ts_unix, tz=timezone.utc),
        "from_address": (tx_obj.get("from") or "").lower() or None,
        "to_address": (tx_obj.get("to") or "").lower() or None,
        "value_wei": str(val_wei_int),
        "value_eth": val_eth,
        "gas_used": gas_used,
        "gas_price": gas_price,
        "success": success,
        "input_method_id": method_id,
        "is_contract_call": raw_input not in {"", "0x"},
        "chain_id": CHAIN_ID,
    }


def request_page(
    address: str,
    api_key: str,
    page: int,
    offset: int,
    timeout: int = 30,
    retries: int = 4,
    action: str = "txlist",
) -> list[dict[str, Any]]:
    params = {
        "chainid": str(CHAIN_ID),
        "module": "account",
        "action": action,
        "address": address,
        "startblock": "0",
        "endblock": "999999999",
        "page": str(page),
        "offset": str(offset),
        "sort": "desc",
        "apikey": api_key,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    last_error = None

    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ethereum-guard/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            status = str(payload.get("status", ""))
            result = payload.get("result", [])

            if status == "1" and isinstance(result, list):
                return result
            if status == "0" and isinstance(result, list) and not result:
                # Etherscan returns status=0 for an address with no txs.
                if "no transactions" in str(payload.get("message", "")).lower():
                    return []
            message = str(payload.get("result") or payload.get("message") or payload)
            if any(token in message.lower() for token in ("rate limit", "max rate", "temporarily")):
                raise EtherscanError(f"Etherscan rate/API limit: {message}")
            raise EtherscanError(f"Etherscan rejected request: {message}")

        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, EtherscanError) as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(min(2 ** attempt, 8))

    raise EtherscanError(f"Request failed after {retries} attempts: {last_error}")


def fetch_normal_transactions(
    address: str,
    api_key: str,
    offset: int = 1000,
    max_pages: int = 10,
    request_delay: float = 0.25,
) -> pd.DataFrame:
    """Fetch paginated normal transactions; default cap is 10,000 rows."""
    address = validate_address(address)
    if not api_key:
        raise ValueError("Missing ETHERSCAN_API_KEY environment variable.")
    if not 1 <= offset <= 1000:
        raise ValueError("offset must be between 1 and 1000.")
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1.")

    records: list[dict[str, Any]] = []
    pages_fetched = 0
    last_page_size = 0
    for page in range(1, max_pages + 1):
        rows = request_page(address, api_key, page, offset)
        pages_fetched += 1
        last_page_size = len(rows)
        if not rows:
            break
        records.extend(rows)
        if len(rows) < offset:
            break
        if page < max_pages:
            time.sleep(max(request_delay, 0.0))

    columns = [
        "transaction_hash", "block_number", "transaction_index", "timestamp",
        "from_address", "to_address", "value_wei", "value_eth", "gas_used",
        "gas_price", "success", "chain_id",
    ]
    normalized = []
    for row in records:
        value_wei = str(row.get("value", "0"))
        value_eth = float(Decimal(value_wei) / WEI_PER_ETH)
        receipt_status = str(row.get("txreceipt_status", "")).strip()
        is_error = str(row.get("isError", "0")).strip()
        success = receipt_status == "1" if receipt_status in {"0", "1"} else is_error == "0"

        normalized.append({
            "transaction_hash": str(row.get("hash", "")).lower(),
            "block_number": int(row.get("blockNumber", 0)),
            "transaction_index": int(row.get("transactionIndex", 0)),
            "timestamp": datetime.fromtimestamp(
                int(row.get("timeStamp", 0)), tz=timezone.utc
            ),
            "from_address": (row.get("from") or "").lower() or None,
            "to_address": (row.get("to") or "").lower() or None,
            "value_wei": value_wei,
            "value_eth": value_eth,
            "gas_used": int(row.get("gasUsed") or 0),
            "gas_price": int(row.get("gasPrice") or 0),
            "success": success,
            "chain_id": CHAIN_ID,
        })

    frame = pd.DataFrame(normalized, columns=columns)
    raw_count = len(frame)
    if not frame.empty:
        frame = frame.drop_duplicates("transaction_hash", keep="last")
        frame = frame.sort_values(
            ["timestamp", "block_number", "transaction_index"],
            kind="mergesort",
        ).reset_index(drop=True)

    frame.attrs["fetch_metadata"] = {
        "endpoint": "account/txlist",
        "pages_fetched": pages_fetched,
        "page_size": offset,
        "max_pages": max_pages,
        "api_rows_received": raw_count,
        "unique_transactions": len(frame),
        "deduplicated_rows": raw_count - len(frame),
        "history_cap_reached": (
            pages_fetched == max_pages and last_page_size == offset
        ),
    }
    return frame


def fetch_erc20_transfers(
    address: str,
    api_key: str,
    offset: int = 1000,
    max_pages: int = 2,
    request_delay: float = 0.25,
) -> pd.DataFrame:
    """Fetch paginated ERC-20 token transfer events (account/tokentx) via Etherscan V2."""
    address = validate_address(address)
    if not api_key:
        raise ValueError("Missing ETHERSCAN_API_KEY environment variable.")
    records: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        rows = request_page(address, api_key, page, offset, action="tokentx")
        if not rows:
            break
        records.extend(rows)
        if len(rows) < offset:
            break
        if page < max_pages:
            time.sleep(max(request_delay, 0.0))

    cols = [
        "transaction_hash", "block_number", "timestamp", "from_address",
        "to_address", "contract_address", "token_symbol", "token_decimal", "value_token",
    ]
    normalized = []
    for row in records:
        decimals = max(0, int(row.get("tokenDecimal") or 18))
        raw_val = Decimal(str(row.get("value") or "0"))
        val_tok = float(raw_val / (Decimal(10) ** decimals)) if decimals <= 36 else 0.0
        normalized.append({
            "transaction_hash": str(row.get("hash", "")).lower(),
            "block_number": int(row.get("blockNumber", 0)),
            "timestamp": datetime.fromtimestamp(int(row.get("timeStamp", 0)), tz=timezone.utc),
            "from_address": (row.get("from") or "").lower() or None,
            "to_address": (row.get("to") or "").lower() or None,
            "contract_address": (row.get("contractAddress") or "").lower() or None,
            "token_symbol": str(row.get("tokenSymbol") or ""),
            "token_decimal": decimals,
            "value_token": val_tok,
        })
    return pd.DataFrame(normalized, columns=cols)


def fetch_internal_transactions(
    address: str,
    api_key: str,
    offset: int = 1000,
    max_pages: int = 2,
    request_delay: float = 0.25,
) -> pd.DataFrame:
    """Fetch paginated internal contract calls (account/txlistinternal) via Etherscan V2."""
    address = validate_address(address)
    if not api_key:
        raise ValueError("Missing ETHERSCAN_API_KEY environment variable.")
    records: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        rows = request_page(address, api_key, page, offset, action="txlistinternal")
        if not rows:
            break
        records.extend(rows)
        if len(rows) < offset:
            break
        if page < max_pages:
            time.sleep(max(request_delay, 0.0))

    cols = [
        "transaction_hash", "block_number", "timestamp", "from_address",
        "to_address", "value_wei", "value_eth", "success",
    ]
    normalized = []
    for row in records:
        value_wei = str(row.get("value", "0"))
        normalized.append({
            "transaction_hash": str(row.get("hash", "")).lower(),
            "block_number": int(row.get("blockNumber", 0)),
            "timestamp": datetime.fromtimestamp(int(row.get("timeStamp", 0)), tz=timezone.utc),
            "from_address": (row.get("from") or "").lower() or None,
            "to_address": (row.get("to") or "").lower() or None,
            "value_wei": value_wei,
            "value_eth": float(Decimal(value_wei) / WEI_PER_ETH),
            "success": str(row.get("isError", "0")).strip() == "0",
        })
    return pd.DataFrame(normalized, columns=cols)


def compute_wallet_features(
    address: str,
    transactions: pd.DataFrame,
    num_windows: int = 10,
    window_hours: int = 24,
    apply_scaler: bool = True,
) -> dict[str, Any]:
    """Compute the project's 19 non-graph features and dense W x 8 sequence."""
    address = validate_address(address)
    if transactions.empty:
        return {
            "address": address,
            "transaction_count": 0,
            "history_start_utc": None,
            "history_cutoff_utc": None,
            "history_span_hours": 0.0,
            "fetch_metadata": transactions.attrs.get("fetch_metadata", {}),
            "sequence_metadata": {
                "window_count": num_windows,
                "window_hours": window_hours,
                "sequence_window_hours": num_windows * window_hours,
                "transactions_in_sequence": 0,
                "nonempty_windows": 0,
                "window_order": "same order as compute_tabular_and_temporal output",
            },
            "features_raw": None,
            "features_scaled": None,
            "temporal_sequence": [
                {"window_index": i, **{k: 0.0 for k in (
                    "tx_in_count", "tx_out_count", "val_in_sum", "val_out_sum",
                    "val_max", "unique_counterparties", "avg_gas_price",
                    "mean_inter_arrival_time",
                )}}
                for i in range(num_windows)
            ],
            "warning": "No normal transactions returned by Etherscan.",
        }

    from src.pipeline.features import (
        compute_tabular_and_temporal,
        normalize_transactions,
        SEQUENCE_COLUMNS,
        TABULAR,
        TEMPORAL_STATS,
    )
    from src.pipeline.feature_transform import transform_feature_frame
    import joblib
    tx = normalize_transactions(transactions)
    cutoff = int(tx["timestamp"].max())
    features, sparse_sequence, _ = compute_tabular_and_temporal(
        tx=tx,
        cutoff=cutoff,
        num_windows=num_windows,
        window_hours=window_hours,
    )

    raw_record = None
    scaled_record = None
    dense_sequence = [
        {"window_index": i, **{key: 0.0 for key in SEQUENCE_COLUMNS}}
        for i in range(num_windows)
    ]

    if address in features.index:
        raw = features.loc[address]
        raw_record = {
            key: (None if pd.isna(value) else float(value))
            for key, value in raw.to_dict().items()
        }

        if isinstance(sparse_sequence.index, pd.MultiIndex):
            try:
                wallet_sequence = sparse_sequence.xs(address, level="address")
                for window_index, row in wallet_sequence.iterrows():
                    wi = int(window_index)
                    if 0 <= wi < num_windows:
                        dense_sequence[wi] = {
                            "window_index": wi,
                            **{
                                key: (
                                    0.0 if pd.isna(row.get(key))
                                    else float(row.get(key))
                                )
                                for key in SEQUENCE_COLUMNS
                            },
                        }
            except KeyError:
                pass

        if apply_scaler:
            scaler_path = ROOT / "artifacts" / "scalers" / "feature_robust_scaler.joblib"
            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"Train-fitted scaler not found: {scaler_path}. "
                    "Run the feature pipeline first or pass apply_scaler=False."
                )
            bundle = joblib.load(scaler_path)
            scaler = bundle["scaler"]
            columns = bundle["columns"]
            values = pd.DataFrame(
                [[0.0 if pd.isna(raw.get(col)) else float(raw.get(col)) for col in columns]],
                columns=columns,
            )
            transform_tag = str(bundle.get("transform") or "").strip().lower()
            if "log" in transform_tag:
                values = transform_feature_frame(values, columns)
            transformed = scaler.transform(values)[0]
            scaled_record = {
                col: float(value) for col, value in zip(columns, transformed)
            }
            # Graph features remain unavailable without graph context.
            for col in features.columns:
                if col not in scaled_record:
                    scaled_record[col] = None
        else:
            scaled_record = None

    history_start = tx["timestamp"].min()
    history_end = tx["timestamp"].max()

    # normalize_transactions currently keeps timestamps as Unix seconds
    # (often numpy.int64), not datetime objects.
    def timestamp_to_utc(value: Any) -> datetime:
        if isinstance(value, (pd.Timestamp, datetime)):
            result = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
            return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    history_start_dt = timestamp_to_utc(history_start)
    history_end_dt = timestamp_to_utc(history_end)
    sequence_tx_count = int(sum(
        float(row.get("tx_in_count", 0.0) or 0.0)
        + float(row.get("tx_out_count", 0.0) or 0.0)
        for row in dense_sequence
    ))
    nonempty_windows = sum(
        1 for row in dense_sequence
        if (
            float(row.get("tx_in_count", 0.0) or 0.0)
            + float(row.get("tx_out_count", 0.0) or 0.0)
        ) > 0
    )
    sequence_window_seconds = num_windows * window_hours * 3600
    sequence_start = datetime.fromtimestamp(
        cutoff - sequence_window_seconds, timezone.utc
    )
    sequence_end = datetime.fromtimestamp(cutoff, timezone.utc)

    return {
        "address": address,
        "transaction_count": int(len(tx)),
        "fetch_metadata": transactions.attrs.get("fetch_metadata", {}),
        "history_start_utc": history_start_dt.isoformat(),
        "history_cutoff_utc": history_end_dt.isoformat(),
        "history_span_hours": (
            history_end_dt - history_start_dt
        ).total_seconds() / 3600.0,
        "sequence_metadata": {
            "window_count": num_windows,
            "window_hours": window_hours,
            "sequence_window_hours": num_windows * window_hours,
            "sequence_window_start_utc": sequence_start.isoformat(),
            "sequence_window_end_utc": sequence_end.isoformat(),
            "transactions_in_sequence": sequence_tx_count,
            "nonempty_windows": nonempty_windows,
            "window_order": "same order as compute_tabular_and_temporal output",
            "note": (
                "Sequence covers only the configured trailing time window; "
                "transaction_count covers all fetched normal transactions."
            ),
        },
        "features_raw": raw_record,
        "features_scaled": scaled_record,
        "temporal_sequence": dense_sequence,
        "temporal_sequence_shape": [num_windows, len(SEQUENCE_COLUMNS)],
        "feature_scope": "Normal/external transactions returned by Etherscan txlist only.",
        "limitations": [
            "The adapter requests the latest available normal/external transactions, up to max_pages * offset.",
            "Etherscan txlist does not include internal transactions or ERC-20 transfer events.",
            "Graph topology features require graph context and remain null.",
            "The 10-window sequence is dense and ordered; empty observed windows are represented by zeros.",
            "Scaled features use the train-fitted RobustScaler when available.",
            "The API history cap means this is not guaranteed to be the wallet's complete lifetime history.",
            "transaction_count covers all fetched normal transactions; temporal_sequence covers only the configured trailing windows.",
            "ERC-20 token transfers and internal transactions are not included in normal transaction features.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and featurize one Ethereum wallet via Etherscan V2")
    parser.add_argument("address")
    parser.add_argument("--offset", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api_key = os.getenv("ETHERSCAN_API_KEY", "").strip()
    address = validate_address(args.address)
    tx = fetch_normal_transactions(
        address, api_key, offset=args.offset, max_pages=args.max_pages
    )
    result = compute_wallet_features(
        address, tx, num_windows=args.windows, window_hours=args.window_hours,
        apply_scaler=True,
    )
    output = json.dumps(result, indent=2, ensure_ascii=False, default=str)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
