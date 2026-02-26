"""Tests for Layer 2 request-level features."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ekap_anom.features.layer2_req import (
    compute_time_features,
    compute_performance_features,
    compute_path_features,
    compute_query_features,
    get_layer2_feature_columns,
)
from ekap_anom.config import HashingConfig


@pytest.fixture
def sample_request_df() -> pd.DataFrame:
    """Sample request DataFrame with required columns."""
    return pd.DataFrame({
        "timestamp": [
            dt.datetime(2024, 4, 22, 14, 30, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 4, 23, 9, 15, 0, tzinfo=dt.timezone.utc),
        ],
        "uri_stem": ["/EKAP/Default.aspx", "/EKAP/Ortak/Data.ashx"],
        "uri_query": ["id=123&type=test", None],
        "time_taken": [500, 1200],
        "sc_bytes": [2048, 8192],
        "cs_bytes": [128, 256],
        "status": [200, 500],
        "substatus": [0, 0],
        "win32_status": [0, 64],
        "client_ip": ["10.0.0.1", "10.0.0.2"],
        "user_agent": ["Mozilla/5.0", "curl/7.68"],
        "endpoint_group": ["/ekap/default.aspx", "/ekap/ortak/data.ashx"],
    })


class TestTimeFeatures:
    def test_hour_extraction(self, sample_request_df: pd.DataFrame) -> None:
        result = compute_time_features(sample_request_df)
        assert result.iloc[0]["hour"] == 14
        assert result.iloc[1]["hour"] == 9

    def test_day_of_week(self, sample_request_df: pd.DataFrame) -> None:
        result = compute_time_features(sample_request_df)
        assert result.iloc[0]["day_of_week"] == 0  # Monday


class TestPerformanceFeatures:
    def test_log_transforms(self, sample_request_df: pd.DataFrame) -> None:
        result = compute_performance_features(sample_request_df)
        assert "log_time_taken" in result.columns
        assert "log_sc_bytes" in result.columns
        assert "log_cs_bytes" in result.columns
        assert "bytes_per_ms" in result.columns
        assert (result["log_time_taken"] >= 0).all()

    def test_bytes_per_ms(self, sample_request_df: pd.DataFrame) -> None:
        result = compute_performance_features(sample_request_df)
        assert result.iloc[0]["bytes_per_ms"] == 2048 / 500


class TestPathFeatures:
    def test_path_depth(self, sample_request_df: pd.DataFrame) -> None:
        config = HashingConfig(path_hash_dim=16, query_hash_dim=8)
        result = compute_path_features(sample_request_df, config)
        assert "path_depth" in result.columns
        assert result.iloc[0]["path_depth"] == 2

    def test_hash_vector_dim(self, sample_request_df: pd.DataFrame) -> None:
        config = HashingConfig(path_hash_dim=16, query_hash_dim=8)
        result = compute_path_features(sample_request_df, config)
        path_cols = [c for c in result.columns if c.startswith("path_hash_")]
        assert len(path_cols) == 16


class TestQueryFeatures:
    def test_query_stats(self, sample_request_df: pd.DataFrame) -> None:
        config = HashingConfig(path_hash_dim=16, query_hash_dim=8)
        result = compute_query_features(sample_request_df, config)
        assert "query_key_count" in result.columns
        assert result.iloc[0]["query_key_count"] == 2  # id, type
        assert result.iloc[1]["query_key_count"] == 0  # None

    def test_hash_vector_dim(self, sample_request_df: pd.DataFrame) -> None:
        config = HashingConfig(path_hash_dim=16, query_hash_dim=8)
        result = compute_query_features(sample_request_df, config)
        query_cols = [c for c in result.columns if c.startswith("query_hash_")]
        assert len(query_cols) == 8


class TestFeatureColumns:
    def test_default_dim(self) -> None:
        cols = get_layer2_feature_columns()
        assert len(cols) > 0
        assert "hour" in cols
        assert "path_depth" in cols

    def test_custom_dim(self) -> None:
        config = HashingConfig(path_hash_dim=32, query_hash_dim=16)
        cols = get_layer2_feature_columns(config)
        path_hash_cols = [c for c in cols if c.startswith("path_hash_")]
        query_hash_cols = [c for c in cols if c.startswith("query_hash_")]
        assert len(path_hash_cols) == 32
        assert len(query_hash_cols) == 16
