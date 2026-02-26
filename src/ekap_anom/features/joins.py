"""Cross-layer feature joins."""

from __future__ import annotations

import pandas as pd

from ekap_anom.logging import get_logger

logger = get_logger(__name__)


def join_layer1_scores_to_requests(
    requests_df: pd.DataFrame,
    layer1_df: pd.DataFrame,
    window_minutes: int = 5,
) -> pd.DataFrame:
    """Attach Layer 1 window scores to individual requests.

    Each request gets the ``layer1_score`` and ``window_is_hot`` from
    its containing window.

    Args:
        requests_df: Request-level DataFrame with ``segment_id``, ``timestamp``.
        layer1_df: Window-level Layer 1 results with ``segment_id``,
            ``window_start``, ``layer1_score``, ``window_is_hot``.
        window_minutes: Window size for bucketing.

    Returns:
        Requests DataFrame with ``layer1_score`` and ``window_is_hot`` joined.
    """
    df = requests_df.copy()

    # Ensure datetime types
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["_window_start"] = ts.dt.floor(f"{window_minutes}min")

    # Prepare join columns from layer1
    l1 = layer1_df[["segment_id", "window_start", "layer1_score", "window_is_hot"]].copy()
    l1 = l1.rename(columns={"window_start": "_window_start"})

    # Left join
    df = df.merge(l1, on=["segment_id", "_window_start"], how="left")

    # Fill missing scores
    df["layer1_score"] = df["layer1_score"].fillna(0.0)
    df["window_is_hot"] = df["window_is_hot"].fillna(False)

    df.drop(columns=["_window_start"], inplace=True)

    logger.info(
        "layer1_scores_joined",
        total_requests=len(df),
        hot_requests=int(df["window_is_hot"].sum()),
    )
    return df


def filter_for_layer2(
    df: pd.DataFrame,
    critical_endpoints: list[str] | None = None,
) -> pd.DataFrame:
    """Select requests that should be scored by Layer 2.

    Rules:
      - All requests in hot windows (``window_is_hot == True``)
      - All requests to critical endpoints (always scored)

    Args:
        df: Requests with ``window_is_hot`` and ``endpoint_group`` columns.
        critical_endpoints: Always-score endpoint list.

    Returns:
        Filtered DataFrame for Layer 2 scoring.
    """
    mask = df["window_is_hot"].fillna(False).astype(bool)

    if critical_endpoints:
        critical_lower = {ep.lower() for ep in critical_endpoints}
        mask = mask | df["endpoint_group"].str.lower().isin(critical_lower)

    result = df[mask].copy()
    logger.info(
        "layer2_filter_applied",
        total=len(df),
        selected=len(result),
    )
    return result
