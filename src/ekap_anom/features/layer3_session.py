"""Layer 3: Session/sequence anomaly features (optional)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from ekap_anom.config import Layer3Config
from ekap_anom.logging import get_logger
from ekap_anom.utils.hashing import hash_single

logger = get_logger(__name__)


def build_session_id(
    client_ip: str,
    user_agent: str,
    cookie_prefix: str = "",
) -> str:
    """Create session identifier from IP + UA + cookie prefix."""
    raw = f"{client_ip}|{user_agent}|{cookie_prefix}"
    return f"sess_{hash_single(raw):08x}"


def sessionize(
    df: pd.DataFrame,
    timeout_minutes: int = 30,
) -> pd.DataFrame:
    """Assign session IDs and session sequence numbers.

    Args:
        df: DataFrame sorted by timestamp with ``client_ip``, ``user_agent`` columns.
        timeout_minutes: Gap threshold to start new session.

    Returns:
        DataFrame with ``session_id`` and ``session_seq`` columns.
    """
    df = df.sort_values("timestamp").copy()

    # Session ID from IP + UA
    df["session_id"] = df.apply(
        lambda r: build_session_id(
            str(r.get("client_ip", "")),
            str(r.get("user_agent", "")),
            str(r.get("cookie", ""))[:20] if pd.notna(r.get("cookie")) else "",
        ),
        axis=1,
    )

    # Session break detection (timeout)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["_time_diff"] = ts.diff().dt.total_seconds().fillna(0)
    df["_session_break"] = (
        (df["_time_diff"] > timeout_minutes * 60) |
        (df["session_id"] != df["session_id"].shift(1))
    ).astype(int)
    df["session_num"] = df.groupby("session_id")["_session_break"].cumsum()
    df["session_id"] = df["session_id"] + "_" + df["session_num"].astype(str)

    # Session sequence number
    df["session_seq"] = df.groupby("session_id").cumcount()

    df.drop(columns=["_time_diff", "_session_break", "session_num"], inplace=True)
    return df


def extract_session_sequences(
    df: pd.DataFrame,
    critical_endpoints: list[str] | None = None,
) -> dict[str, list[str]]:
    """Extract endpoint sequences per session for critical endpoints.

    Args:
        df: Sessionized DataFrame with ``session_id``, ``endpoint_group``.
        critical_endpoints: Only include sessions touching these endpoints.

    Returns:
        Dict mapping session_id → list of endpoint_group in order.
    """
    if critical_endpoints:
        # Normalize for matching
        critical_lower = {ep.lower() for ep in critical_endpoints}
        sessions_with_critical = df[
            df["endpoint_group"].str.lower().isin(critical_lower)
        ]["session_id"].unique()
        df = df[df["session_id"].isin(sessions_with_critical)]

    sequences: dict[str, list[str]] = {}
    for session_id, group in df.sort_values("session_seq").groupby("session_id"):
        sequences[str(session_id)] = group["endpoint_group"].tolist()

    return sequences


def build_layer3_features(
    df: pd.DataFrame,
    config: Layer3Config | None = None,
) -> pd.DataFrame:
    """End-to-end Layer 3 feature pipeline.

    Args:
        df: Enriched DataFrame (with endpoint_group, etc.).
        config: Layer 3 configuration.

    Returns:
        Sessionized DataFrame with sequence features.
    """
    if config is None:
        config = Layer3Config()

    if not config.enabled:
        logger.info("layer3_disabled")
        return df

    df = sessionize(df, config.session_timeout_minutes)

    logger.info(
        "layer3_features_computed",
        n_sessions=df["session_id"].nunique(),
        n_rows=len(df),
    )
    return df
