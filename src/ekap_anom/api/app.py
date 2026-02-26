"""FastAPI application: /score, /health, /metrics endpoints."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from ekap_anom import __version__
from ekap_anom.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResult,
)
from ekap_anom.config import load_settings, Settings
from ekap_anom.logging import get_logger, setup_logging
from ekap_anom.models.layer2_iforest import SegmentIForest
from ekap_anom.monitoring.metrics import (
    INFERENCE_LATENCY,
    ALERTS_GENERATED,
    get_metrics_text,
    get_content_type,
)

logger = get_logger(__name__)

# Global state
_state: dict[str, Any] = {
    "settings": None,
    "layer2_model": None,
    "thresholds": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup/shutdown lifecycle."""
    settings = load_settings()
    setup_logging(settings.general.log_level, settings.general.log_format)
    _state["settings"] = settings

    # Try to load models
    model_root = Path(settings.general.model_root)
    l2_path = model_root / "layer2_iforest.pkl"
    if l2_path.exists():
        _state["layer2_model"] = SegmentIForest.load(l2_path)
        logger.info("layer2_model_loaded")

    logger.info("api_started", version=__version__)
    yield
    logger.info("api_shutdown")


app = FastAPI(
    title="EKAP Anomaly Detection API",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded=_state.get("layer2_model") is not None,
    )


@app.post("/score", response_model=BatchScoreResponse)
async def score(request: BatchScoreRequest) -> BatchScoreResponse:
    """Score a batch of log records."""
    start = time.monotonic()
    results: list[ScoreResult] = []

    # Convert to DataFrame
    records = [r.model_dump() for r in request.records]
    if not records:
        return BatchScoreResponse(results=[], total=0, anomalies=0)

    df = pd.DataFrame(records)

    # Enrich
    from ekap_anom.features.layer1_agg import enrich_for_layer1
    from ekap_anom.utils.url import endpoint_group as compute_eg

    # Ensure settings are available (may not be set in test context)
    settings: Settings = _state.get("settings") or load_settings()
    df = enrich_for_layer1(df, settings.url, settings.segmentation)

    # Score with Layer 2 if model available
    model = _state.get("layer2_model")
    if model is not None and hasattr(model, "feature_columns") and model.feature_columns:
        from ekap_anom.features.layer2_req import build_layer2_features
        df = build_layer2_features(df, settings.hashing)

        available_cols = [c for c in model.feature_columns if c in df.columns]
        if available_cols:
            scores = model.predict(df, segment_col="segment_id")
            df["layer2_score"] = scores
        else:
            df["layer2_score"] = 0.0
    else:
        df["layer2_score"] = 0.0

    df["layer1_score"] = 0.0
    df["layer3_score"] = 0.0
    df["final_score"] = df["layer2_score"]
    df["is_anomaly"] = df["final_score"] > 0.5

    for _, row in df.iterrows():
        results.append(ScoreResult(
            endpoint_group=str(row.get("endpoint_group", "")),
            segment_id=str(row.get("segment_id", "")),
            layer1_score=float(row.get("layer1_score", 0)),
            layer2_score=float(row.get("layer2_score", 0)),
            layer3_score=float(row.get("layer3_score", 0)),
            final_score=float(row.get("final_score", 0)),
            is_anomaly=bool(row.get("is_anomaly", False)),
            reason_codes=[],
        ))

    elapsed = time.monotonic() - start
    INFERENCE_LATENCY.labels(layer="api").observe(elapsed)

    n_anomalies = sum(1 for r in results if r.is_anomaly)
    ALERTS_GENERATED.labels(severity="api").inc(n_anomalies)

    return BatchScoreResponse(
        results=results,
        total=len(results),
        anomalies=n_anomalies,
    )


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=get_metrics_text(),
        media_type=get_content_type(),
    )
