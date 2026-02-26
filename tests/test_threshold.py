"""Tests for segment-based thresholding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ekap_anom.scoring.threshold import (
    compute_segment_thresholds,
    apply_thresholds,
)


class TestComputeSegmentThresholds:
    def test_per_segment(self) -> None:
        df = pd.DataFrame({
            "segment_id": ["seg_a"] * 100 + ["seg_b"] * 100,
            "final_score": list(np.linspace(0, 0.5, 100)) + list(np.linspace(0, 0.9, 100)),
        })
        thresholds = compute_segment_thresholds(df, quantile=0.95)
        assert "seg_a" in thresholds
        assert "seg_b" in thresholds
        # seg_b has higher scores, so higher threshold
        assert thresholds["seg_b"] > thresholds["seg_a"]

    def test_no_global_threshold(self) -> None:
        """Ensure each segment gets its own threshold, not a single global one."""
        df = pd.DataFrame({
            "segment_id": ["low"] * 1000 + ["high"] * 1000,
            "final_score": list(np.random.uniform(0, 0.2, 1000)) +
                          list(np.random.uniform(0.5, 1.0, 1000)),
        })
        thresholds = compute_segment_thresholds(df, quantile=0.99)
        assert len(thresholds) == 2
        assert abs(thresholds["low"] - thresholds["high"]) > 0.2

    def test_empty_segment(self) -> None:
        df = pd.DataFrame({"segment_id": [], "final_score": []})
        thresholds = compute_segment_thresholds(df)
        assert len(thresholds) == 0


class TestApplyThresholds:
    def test_flags_anomalies(self) -> None:
        df = pd.DataFrame({
            "segment_id": ["seg_a", "seg_a", "seg_a"],
            "final_score": [0.1, 0.5, 0.9],
        })
        thresholds = {"seg_a": 0.8}
        result = apply_thresholds(df, thresholds)
        assert not result.iloc[0]["is_anomaly"]
        assert not result.iloc[1]["is_anomaly"]
        assert result.iloc[2]["is_anomaly"]

    def test_missing_segment_no_anomaly(self) -> None:
        df = pd.DataFrame({
            "segment_id": ["unknown_seg"],
            "final_score": [0.99],
        })
        thresholds = {"seg_a": 0.5}
        result = apply_thresholds(df, thresholds)
        # Unknown segment gets threshold=1.0 (never trigger)
        assert not result.iloc[0]["is_anomaly"]
