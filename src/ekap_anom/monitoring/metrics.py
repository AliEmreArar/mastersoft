"""Prometheus metrics for observability."""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ── Counters ────────────────────────────────────────────────────────────────

LOGS_PARSED = Counter(
    "ekap_logs_parsed_total",
    "Total number of log lines parsed",
    ["source"],
)

REQUESTS_FILTERED = Counter(
    "ekap_requests_filtered_total",
    "Requests filtered (static vs dynamic)",
    ["filter_result"],
)

ALERTS_GENERATED = Counter(
    "ekap_alerts_generated_total",
    "Total alerts generated",
    ["severity"],
)

# ── Histograms ──────────────────────────────────────────────────────────────

INFERENCE_LATENCY = Histogram(
    "ekap_inference_latency_seconds",
    "Inference latency in seconds",
    ["layer"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

SCORE_DISTRIBUTION = Histogram(
    "ekap_score_distribution",
    "Distribution of final anomaly scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Gauges ──────────────────────────────────────────────────────────────────

HOT_WINDOWS = Gauge(
    "ekap_hot_windows_current",
    "Current number of hot windows",
)

DRIFT_PSI = Gauge(
    "ekap_drift_psi",
    "Latest PSI drift metric",
    ["feature"],
)

MODEL_VERSION = Info(
    "ekap_model",
    "Current model information",
)


def get_metrics_text() -> bytes:
    """Return Prometheus metrics in text exposition format."""
    return generate_latest()


def get_content_type() -> str:
    """Return the Prometheus content type header value."""
    return CONTENT_TYPE_LATEST
