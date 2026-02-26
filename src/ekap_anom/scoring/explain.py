"""Reason code generation: explain which feature groups drive anomaly scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ekap_anom.logging import get_logger

logger = get_logger(__name__)

# Feature-to-reason-code mapping
FEATURE_GROUPS: dict[str, list[str]] = {
    "burst_rate": ["burstiness", "error_burstiness", "request_count"],
    "latency_spike": ["p95_time_taken", "p99_time_taken", "log_time_taken"],
    "bytes_spike": ["sum_sc_bytes", "p95_sc_bytes", "log_sc_bytes", "log_cs_bytes", "bytes_per_ms"],
    "error_spike": ["4xx_rate", "5xx_rate"],
    "url_risk": [
        "path_depth", "query_special_char_ratio", "query_decoded_len",
        "query_suspicious_flag", "query_max_value_len",
    ],
    "ua_novelty": ["ua_novelty", "ua_bot_flag"],
    "ip_anomaly": ["ip_rps_last_5m", "ip_unique_endpoints_last_30m", "first_seen_ip_in_7d"],
}


def compute_reason_codes(
    row: pd.Series,
    baseline_means: dict[str, float] | None = None,
    top_n: int = 3,
) -> list[str]:
    """Identify the top contributing feature groups for a single request.

    Uses deviation from baseline means to rank feature groups.

    Args:
        row: Single request row.
        baseline_means: Training-set feature means for comparison.
        top_n: Number of top reason codes to return.

    Returns:
        Sorted list of reason code strings (most impactful first).
    """
    if baseline_means is None:
        baseline_means = {}

    group_scores: dict[str, float] = {}

    for reason, features in FEATURE_GROUPS.items():
        deviations: list[float] = []
        for feat in features:
            if feat in row.index and pd.notna(row[feat]):
                val = float(row[feat])
                mean = baseline_means.get(feat, 0.0)
                std = max(abs(mean) * 0.5, 1e-6)  # rough std estimate
                deviations.append(abs(val - mean) / std)
        if deviations:
            group_scores[reason] = float(np.mean(deviations))

    # Sort by score descending
    sorted_reasons = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
    return [reason for reason, _ in sorted_reasons[:top_n]]


def add_reason_codes(
    df: pd.DataFrame,
    baseline_means: dict[str, float] | None = None,
    top_n: int = 3,
) -> pd.DataFrame:
    """Add ``reason_codes`` column to a DataFrame.

    Only computed for anomalous rows (``is_anomaly == True``).
    """
    df = df.copy()

    if "is_anomaly" not in df.columns:
        df["reason_codes"] = None
        return df

    anomaly_mask = df["is_anomaly"].fillna(False)

    df["reason_codes"] = None
    if anomaly_mask.any():
        df.loc[anomaly_mask, "reason_codes"] = df[anomaly_mask].apply(
            lambda r: compute_reason_codes(r, baseline_means, top_n),
            axis=1,
        )

    return df
