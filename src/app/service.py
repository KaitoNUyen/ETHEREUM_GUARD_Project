"""
Ethereum Guard - High Performance Dual-Level Deep Learning FastAPI Service.
Production endpoints for BOTH:
  - Transaction-level anomaly scoring (`GET /predict/tx/{tx_hash}`, `GET /transactions/{address}`)
  - Account-level anomaly scoring (`GET /predict/{address}`, `POST /predict/batch`, `GET /graph/{address}`)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.app.engine import (
    AnomalyPrediction,
    EgoNetworkGraph,
    TransactionAnomalyPrediction,
    get_engine,
)
from src.schemas import ALL_FEATURE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ethereum_guard.api")

app = FastAPI(
    title="Ethereum Guard — Deep Learning Transaction & Account Anomaly API",
    description="Dual-granularity Deep Learning REST API (Modular TG-AE + DeepTransactionScorer) for Ethereum transaction & wallet anomaly detection.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_raw_origins = os.getenv("ETHEREUM_GUARD_CORS_ORIGINS", "*").strip()
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]
_allow_credentials = "*" not in _allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    primary_model: str
    tx_scorer_loaded: bool
    version: str
    feature_count: int
    threshold: float


class FeatureAttribution(BaseModel):
    feature: str
    category: str
    raw_value: float
    transformed_value: float
    baseline_mean: float
    z_score: float
    iqr_deviation: float
    direction: str


class PredictResponse(BaseModel):
    address: str
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    threshold: float
    risk_level: str
    verified_label: int
    label_text: str
    inference_time_ms: float
    evaluated_at: str
    source: str = "OFFLINE_SNAPSHOT"
    metadata: dict[str, Any] = Field(default_factory=dict)
    feature_contributions: list[FeatureAttribution]
    raw_features: dict[str, float]


class TransactionPredictResponse(BaseModel):
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
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
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
    source: str
    evaluated_at: str


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    color: str
    size: int


class GraphEdge(BaseModel):
    source: str
    target: str
    value_eth: float
    timestamp: str
    tx_hash: str
    gas_used: int
    success: bool
    direction: str


class EgoNetworkResponse(BaseModel):
    center_address: str
    total_volume_eth: float
    in_degree: int
    out_degree: int
    unique_counterparties: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class BatchPredictRequest(BaseModel):
    addresses: list[str] = Field(..., min_length=1, max_length=50)
    model: str = Field(default="tgae", pattern="^(tgae|isolation_forest)$")


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_processed: int
    total_anomalies: int
    failed_addresses: list[dict[str, str]] = Field(default_factory=list)


class StatsResponse(BaseModel):
    dataset_summary: dict[str, Any]
    benchmark_comparison: list[dict[str, Any]]
    transaction_benchmark: list[dict[str, Any]] = Field(default_factory=list)
    ablation_study: list[dict[str, Any]]
    error_case_summary: dict[str, Any]


def _to_predict_response(pred: AnomalyPrediction) -> PredictResponse:
    return PredictResponse(
        address=pred.address,
        anomaly_score=pred.anomaly_score,
        is_anomaly=pred.is_anomaly,
        threshold=pred.threshold,
        risk_level=pred.risk_level,
        verified_label=pred.verified_label,
        label_text=pred.label_text,
        inference_time_ms=pred.inference_time_ms,
        evaluated_at=pred.evaluated_at,
        source=pred.source,
        metadata=pred.metadata,
        feature_contributions=[FeatureAttribution(**c) for c in pred.feature_contributions],
        raw_features=pred.raw_features,
    )


def _to_tx_response(tp: TransactionAnomalyPrediction) -> TransactionPredictResponse:
    return TransactionPredictResponse(**tp.__dict__)


@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    engine = get_engine()
    return HealthResponse(
        status="ONLINE" if engine.is_loaded else "DEGRADED",
        model_loaded=engine.is_loaded,
        primary_model="ModularTGAE" if engine.tgae_model is not None else "IsolationForest",
        tx_scorer_loaded=engine.tx_scorer is not None,
        version="2.0.0",
        feature_count=len(ALL_FEATURE_COLS),
        threshold=engine.threshold,
    )


def _load_csv_records_safe(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


@app.get("/stats", response_model=StatsResponse, tags=["System"])
@app.get("/api/v1/stats", response_model=StatsResponse, tags=["System"])
def get_system_stats() -> StatsResponse:
    results_dir = ROOT_DIR / "artifacts" / "results"
    split_report_path = results_dir / "split_report.json"
    dataset_summary: dict[str, Any] = {}
    if split_report_path.exists():
        raw_split = json.loads(split_report_path.read_text(encoding="utf-8"))
        dataset_summary = {
            "total_transactions": raw_split.get("input", {}).get("total_transactions", 0),
            "unique_addresses": raw_split.get("address_split_summary", {}).get("unique_addresses", 0),
            "labels": raw_split.get("address_split_summary", {}).get("labels", {}),
            "cutoffs": raw_split.get("cutoffs", {}),
        }

    benchmark_records = _load_csv_records_safe(results_dir / "benchmark_comparison.csv")
    tx_benchmark_records = _load_csv_records_safe(results_dir / "transaction_benchmark.csv")
    ablation_records = _load_csv_records_safe(results_dir / "ablation_study.csv")
    error_records = _load_csv_records_safe(results_dir / "error_cases.csv")

    fp_cnt = sum(1 for r in error_records if r.get("error_type") == "FALSE_POSITIVE")
    fn_cnt = sum(1 for r in error_records if r.get("error_type") == "FALSE_NEGATIVE")

    return StatsResponse(
        dataset_summary=dataset_summary,
        benchmark_comparison=benchmark_records,
        transaction_benchmark=tx_benchmark_records,
        ablation_study=ablation_records,
        error_case_summary={
            "total_errors": len(error_records),
            "false_positives": fp_cnt,
            "false_negatives": fn_cnt,
            "top_cases": error_records[:10],
        },
    )


_inbound_lock = threading.Lock()
_inbound_buckets: dict[str, tuple[float, float]] = {}


def _verify_api_security(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    required_key = os.getenv("ETHEREUM_GUARD_API_KEY", "").strip()
    if required_key and x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")

    client_ip = request.client.host if request.client else "local"
    now = time.monotonic()
    rate_per_sec = 0.5
    capacity = 10.0
    with _inbound_lock:
        tokens, last_ts = _inbound_buckets.get(client_ip, (capacity, now))
        tokens = min(capacity, tokens + max(0.0, now - last_ts) * rate_per_sec)
        if tokens < 1.0:
            _inbound_buckets[client_ip] = (tokens, now)
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Retry shortly.")
        _inbound_buckets[client_ip] = (tokens - 1.0, now)
        if len(_inbound_buckets) > 4096:
            _inbound_buckets.clear()


@app.get("/predict/tx/{tx_hash}", response_model=TransactionPredictResponse, tags=["Transaction Inference"])
def predict_transaction_endpoint(tx_hash: str) -> TransactionPredictResponse:
    """
    Evaluate anomaly risk of an individual Ethereum transaction (`0x` + 64 hex chars)
    using DeepTransactionScorer + Modular TG-AE endpoint embeddings.
    """
    engine = get_engine()
    try:
        tp = engine.predict_transaction(tx_hash)
        return _to_tx_response(tp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error("Error scoring transaction %s: %s", tx_hash, exc)
        raise HTTPException(status_code=500, detail="Transaction inference failed.")


@app.get("/transactions/{address}", tags=["Transaction Inference"])
def score_wallet_transactions_endpoint(
    address: str,
    max_txs: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """Return per-transaction Deep Learning anomaly scores for a wallet, ranked from highest risk to lowest."""
    engine = get_engine()
    try:
        ranked = engine.score_wallet_transactions(address, max_txs=max_txs)
        return {"address": address.strip().lower(), "count": len(ranked), "transactions": ranked}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Error scoring wallet transactions for %s: %s", address, exc)
        raise HTTPException(status_code=500, detail="Wallet transaction scoring failed.")


@app.get("/predict/{address}", response_model=PredictResponse, tags=["Account Inference"])
def predict_address(
    address: str,
    model: str = Query("tgae", pattern="^(tgae|isolation_forest)$"),
) -> PredictResponse:
    """
    Evaluate account-level anomaly risk using Modular TG-AE (default primary Deep GNN) or Isolation Forest.
    """
    engine = get_engine()
    try:
        pred: AnomalyPrediction = engine.predict(address, model_type=model)
        return _to_predict_response(pred)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error("Error predicting address %s: %s", address, exc)
        raise HTTPException(status_code=500, detail="Inference failure due to an internal server error.")


@app.get("/graph/{address}", response_model=EgoNetworkResponse, tags=["Forensics"])
def get_graph(
    address: str,
    max_edges: int = Query(50, ge=5, le=200, description="Max 1-hop edges to return"),
) -> EgoNetworkResponse:
    engine = get_engine()
    try:
        network: EgoNetworkGraph = engine.get_ego_network(address, max_edges=max_edges)
        return EgoNetworkResponse(
            center_address=network.center_address,
            total_volume_eth=network.total_volume_eth,
            in_degree=network.in_degree,
            out_degree=network.out_degree,
            unique_counterparties=network.unique_counterparties,
            nodes=[GraphNode(**n) for n in network.nodes],
            edges=[GraphEdge(**e) for e in network.edges],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Error building ego network for %s: %s", address, exc)
        raise HTTPException(status_code=500, detail="Graph retrieval failed due to an internal server error.")


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Account Inference"],
    dependencies=[Depends(_verify_api_security)],
)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    engine = get_engine()
    results: list[PredictResponse] = []
    failed: list[dict[str, str]] = []
    anom_count = 0

    for addr in payload.addresses:
        try:
            pred = engine.predict(addr, model_type=payload.model)
            if pred.is_anomaly:
                anom_count += 1
            results.append(_to_predict_response(pred))
        except (ValueError, KeyError) as exc:
            failed.append({"address": str(addr), "error": str(exc)})
        except Exception as exc:
            logger.error("Batch item error for %s: %s", addr, exc)
            failed.append({"address": str(addr), "error": "Internal inference error"})

    return BatchPredictResponse(
        results=results,
        total_processed=len(results),
        total_anomalies=anom_count,
        failed_addresses=failed,
    )
