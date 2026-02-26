"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = ""
    models_loaded: bool = False


class ScoreRequest(BaseModel):
    """Single log line scoring request."""
    uri_stem: str
    method: str = "GET"
    status: int = 200
    substatus: int = 0
    win32_status: int = 0
    time_taken: int = 0
    sc_bytes: int = 0
    cs_bytes: int = 0
    client_ip: str = "0.0.0.0"
    user_agent: str = ""
    uri_query: str | None = None
    host: str | None = None
    referer: str | None = None
    timestamp: str | None = None


class BatchScoreRequest(BaseModel):
    """Batch scoring request."""
    records: list[ScoreRequest]


class ScoreResult(BaseModel):
    """Single record scoring result."""
    endpoint_group: str = ""
    segment_id: str = ""
    layer1_score: float = 0.0
    layer2_score: float = 0.0
    layer3_score: float = 0.0
    final_score: float = 0.0
    is_anomaly: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class BatchScoreResponse(BaseModel):
    """Batch scoring response."""
    results: list[ScoreResult]
    total: int = 0
    anomalies: int = 0
