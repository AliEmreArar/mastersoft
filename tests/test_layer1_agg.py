"""Tests for Layer 1 window aggregation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ekap_anom.features.layer1_agg import (
    compute_window_features,
    add_burstiness,
    enrich_for_layer1,
)


@pytest.fixture
def sample_enriched_df() -> pd.DataFrame:
    """Create sample data with required columns."""
    n = 100
    base_time = dt.datetime(2024, 4, 22, 10, 0, 0, tzinfo=dt.timezone.utc)
    return pd.DataFrame({
        "timestamp": [base_time + dt.timedelta(seconds=i * 3) for i in range(n)],
        "segment_id": ["seg_00000001"] * n,
        "endpoint_group": ["/ekap/default.aspx"] * n,
        "client_ip": [f"10.0.0.{i % 10}" for i in range(n)],
        "user_agent": [f"Mozilla/{i % 5}" for i in range(n)],
        "status_class": ["2xx"] * 80 + ["4xx"] * 15 + ["5xx"] * 5,
        "time_taken": np.random.randint(10, 5000, size=n),
        "sc_bytes": np.random.randint(100, 50000, size=n),
    })


class TestComputeWindowFeatures:
    def test_produces_required_columns(self, sample_enriched_df: pd.DataFrame) -> None:
        result = compute_window_features(sample_enriched_df, window_minutes=5)
        required = [
            "request_count", "unique_ip_count", "unique_ua_count",
            "4xx_rate", "5xx_rate", "p50_time_taken", "p95_time_taken",
            "p99_time_taken", "sum_sc_bytes", "p95_sc_bytes",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_request_count_positive(self, sample_enriched_df: pd.DataFrame) -> None:
        result = compute_window_features(sample_enriched_df, window_minutes=5)
        assert (result["request_count"] > 0).all()

    def test_rates_between_0_and_1(self, sample_enriched_df: pd.DataFrame) -> None:
        result = compute_window_features(sample_enriched_df, window_minutes=5)
        assert (result["4xx_rate"] >= 0).all() and (result["4xx_rate"] <= 1).all()
        assert (result["5xx_rate"] >= 0).all() and (result["5xx_rate"] <= 1).all()


class TestAddBurstiness:
    def test_burstiness_columns(self, sample_enriched_df: pd.DataFrame) -> None:
        windows = compute_window_features(sample_enriched_df, window_minutes=5)
        result = add_burstiness(windows, rolling_hours=1, window_minutes=5)
        assert "burstiness" in result.columns
        assert "error_burstiness" in result.columns

    def test_burstiness_positive(self, sample_enriched_df: pd.DataFrame) -> None:
        windows = compute_window_features(sample_enriched_df, window_minutes=5)
        result = add_burstiness(windows, rolling_hours=1, window_minutes=5)
        assert (result["burstiness"] >= 0).all()


class TestEnrichForLayer1:
    def test_adds_required_columns(self) -> None:
        df = pd.DataFrame({
            "uri_stem": ["/EKAP/Default.aspx", "/EKAP/Data.ashx"],
            "method": ["GET", "POST"],
            "status": [200, 500],
            "user_agent": ["Mozilla/5.0", "curl/7.68"],
        })
        result = enrich_for_layer1(df)
        assert "endpoint_group" in result.columns
        assert "status_class" in result.columns
        assert "bot_flag" in result.columns
        assert "segment_id" in result.columns

    def test_bot_detection(self) -> None:
        df = pd.DataFrame({
            "uri_stem": ["/test.aspx", "/test.aspx"],
            "method": ["GET", "GET"],
            "status": [200, 200],
            "user_agent": ["Mozilla/5.0", "Googlebot/2.1"],
        })
        result = enrich_for_layer1(df)
        assert result.iloc[0]["bot_flag"] == 0
        assert result.iloc[1]["bot_flag"] == 1
