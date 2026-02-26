"""Layer 1: Robust statistical baseline (rolling median + MAD z-score)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ekap_anom.config import Layer1Config
from ekap_anom.logging import get_logger

logger = get_logger(__name__)

# Feature columns used for anomaly scoring
SCORE_FEATURES = [
    "request_count",
    "unique_ip_count",
    "unique_ua_count",
    "4xx_rate",
    "5xx_rate",
    "p95_time_taken",
    "p99_time_taken",
    "sum_sc_bytes",
    "burstiness",
    "error_burstiness",
]


def robust_zscore(series: pd.Series, window: int = 36) -> pd.Series:
    """Compute robust z-score using rolling median and MAD.

    ``z = |x - median| / (1.4826 * MAD)``

    Args:
        series: Input numeric series.
        window: Rolling window size (number of observations).

    Returns:
        Robust z-score series.
    """
    rolling_median = series.rolling(window=window, min_periods=1).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window, min_periods=1).median()
    # Scale MAD to approximate std for normal distributions
    scaled_mad = rolling_mad * 1.4826
    z = (series - rolling_median).abs() / scaled_mad.clip(lower=1e-8)
    return z


def ewma_score(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    """Compute EWMA-based deviation score.

    ``score = |x - ewma| / rolling_std``
    """
    ewma = series.ewm(alpha=alpha, min_periods=1).mean()
    rolling_std = series.rolling(window=36, min_periods=1).std().clip(lower=1e-8)
    return ((series - ewma).abs() / rolling_std)


def score_layer1(
    windows_df: pd.DataFrame,
    config: Layer1Config | None = None,
) -> pd.DataFrame:
    """Score each window using robust z-score aggregation.

    For each segment, computes per-feature robust z-scores, then
    takes the max z-score across features as the raw anomaly signal.
    Final layer1_score is sigmoid-normalized to [0, 1].

    Args:
        windows_df: Layer 1 feature DataFrame (one row per segment×window).
        config: Layer 1 configuration.

    Returns:
        Input DataFrame with ``layer1_score`` and ``window_is_hot`` columns.
    """
    if config is None:
        config = Layer1Config()

    df = windows_df.sort_values(["segment_id", "window_start"]).copy()
    rolling_n = (config.rolling_hours * 60) // config.window_size_minutes

    available_features = [f for f in SCORE_FEATURES if f in df.columns]
    if not available_features:
        logger.warning("no_score_features_available")
        df["layer1_score"] = 0.0
        df["window_is_hot"] = False
        return df

    def _score_segment(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        z_scores = []
        for feat in available_features:
            if config.use_ewma:
                z = ewma_score(group[feat].fillna(0).astype(float), config.ewma_alpha)
            else:
                z = robust_zscore(group[feat].fillna(0).astype(float), window=rolling_n)
            z_scores.append(z)

        # Max z-score across features
        z_matrix = pd.concat(z_scores, axis=1)
        raw_score = z_matrix.max(axis=1)

        # Sigmoid normalization to [0, 1]
        group["layer1_score"] = 1.0 / (1.0 + np.exp(-0.5 * (raw_score - 3)))
        group["window_is_hot"] = group["layer1_score"] >= config.hot_threshold
        return group

    result = df.groupby("segment_id", group_keys=False).apply(_score_segment)

    hot_count = int(result["window_is_hot"].sum())
    logger.info(
        "layer1_scoring_done",
        total_windows=len(result),
        hot_windows=hot_count,
        hot_ratio=round(hot_count / max(len(result), 1), 4),
    )
    return result.reset_index(drop=True)
