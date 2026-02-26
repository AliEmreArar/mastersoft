"""Segment-based thresholding (no global threshold allowed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ekap_anom.config import ScoringConfig
from ekap_anom.logging import get_logger

logger = get_logger(__name__)


def compute_segment_thresholds(
    df: pd.DataFrame,
    score_col: str = "final_score",
    segment_col: str = "segment_id",
    quantile: float = 0.995,
) -> dict[str, float]:
    """Compute per-segment anomaly thresholds at the given quantile.

    No global threshold is allowed.

    Args:
        df: DataFrame with scores and segment IDs.
        score_col: Score column name.
        segment_col: Segment ID column name.
        quantile: Quantile for threshold computation.

    Returns:
        Dict mapping segment_id → threshold value.
    """
    thresholds: dict[str, float] = {}

    for seg_id, group in df.groupby(segment_col):
        scores = group[score_col].dropna()
        if len(scores) == 0:
            thresholds[str(seg_id)] = 1.0  # no data → never trigger
            continue
        thresholds[str(seg_id)] = float(np.quantile(scores, quantile))

    logger.info(
        "segment_thresholds_computed",
        n_segments=len(thresholds),
        quantile=quantile,
        mean_threshold=round(np.mean(list(thresholds.values())), 4),
    )
    return thresholds


def compute_evt_thresholds(
    df: pd.DataFrame,
    score_col: str = "final_score",
    segment_col: str = "segment_id",
    tail_fraction: float = 0.05,
) -> dict[str, float]:
    """Compute per-segment thresholds using EVT/POT tail fitting.

    Uses Generalized Pareto Distribution on exceedances over a high quantile.

    Args:
        df: DataFrame with scores and segment IDs.
        score_col: Score column name.
        segment_col: Segment ID column name.
        tail_fraction: Fraction of data to use as tail.

    Returns:
        Dict mapping segment_id → EVT threshold.
    """
    thresholds: dict[str, float] = {}

    for seg_id, group in df.groupby(segment_col):
        scores = group[score_col].dropna().values
        if len(scores) < 100:
            # Not enough data for EVT, fall back to quantile
            thresholds[str(seg_id)] = float(np.quantile(scores, 0.995)) if len(scores) > 0 else 1.0
            continue

        # POT threshold
        u = np.quantile(scores, 1 - tail_fraction)
        exceedances = scores[scores > u] - u

        if len(exceedances) < 10:
            thresholds[str(seg_id)] = float(u)
            continue

        try:
            # Fit GPD
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
            # 99.5% return level
            n = len(scores)
            n_u = len(exceedances)
            p = 0.005
            if shape == 0:
                threshold = u + scale * np.log(n_u / (n * p))
            else:
                threshold = u + (scale / shape) * ((n_u / (n * p)) ** shape - 1)
            thresholds[str(seg_id)] = float(np.clip(threshold, 0, 1))
        except Exception:
            thresholds[str(seg_id)] = float(np.quantile(scores, 0.995))

    logger.info(
        "evt_thresholds_computed",
        n_segments=len(thresholds),
    )
    return thresholds


def apply_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    score_col: str = "final_score",
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """Flag rows exceeding their segment threshold.

    Args:
        df: DataFrame with scores.
        thresholds: Per-segment thresholds.
        score_col: Score column name.
        segment_col: Segment column name.

    Returns:
        DataFrame with ``is_anomaly`` boolean column.
    """
    df = df.copy()
    df["_seg_threshold"] = df[segment_col].map(thresholds).fillna(1.0)
    df["is_anomaly"] = df[score_col] >= df["_seg_threshold"]
    df.drop(columns=["_seg_threshold"], inplace=True)

    n_anomalies = int(df["is_anomaly"].sum())
    logger.info(
        "thresholds_applied",
        total=len(df),
        anomalies=n_anomalies,
        anomaly_rate=round(n_anomalies / max(len(df), 1), 6),
    )
    return df
