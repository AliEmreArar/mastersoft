"""Layer 1: 5-minute window endpoint-based aggregation features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ekap_anom.config import Layer1Config, URLConfig, SegmentationConfig
from ekap_anom.logging import get_logger
from ekap_anom.utils.segment import make_segment_id, status_class
from ekap_anom.utils.url import endpoint_group as compute_endpoint_group
from ekap_anom.utils.ua import is_bot

logger = get_logger(__name__)


def enrich_for_layer1(
    df: pd.DataFrame,
    url_config: URLConfig | None = None,
    seg_config: SegmentationConfig | None = None,
) -> pd.DataFrame:
    """Add endpoint_group, status_class, bot_flag, segment_id, and window columns."""
    if url_config is None:
        url_config = URLConfig()
    if seg_config is None:
        seg_config = SegmentationConfig()

    df = df.copy()

    # Endpoint group
    df["endpoint_group"] = df["uri_stem"].apply(
        lambda u: compute_endpoint_group(
            u,
            lowercase=url_config.lowercase,
            strip_trailing_slash=url_config.strip_trailing_slash,
            numeric_template=url_config.numeric_template,
            aggressive_template=url_config.aggressive_template,
        ) if pd.notna(u) else "unknown"
    )

    # Status class
    df["status_class"] = df["status"].apply(status_class)

    # Bot flag
    df["bot_flag"] = df["user_agent"].apply(
        lambda ua: 1 if (pd.notna(ua) and is_bot(ua)) else 0
    )

    # Method uppercase
    df["method"] = df["method"].str.upper().fillna("UNKNOWN")

    # Segment ID
    df["segment_id"] = df.apply(
        lambda r: make_segment_id(
            r["endpoint_group"],
            r["method"],
            r["status_class"],
            r["bot_flag"],
            host=r.get("host"),
            enable_host=seg_config.enable_host,
        ),
        axis=1,
    )

    return df


def compute_window_features(
    df: pd.DataFrame,
    window_minutes: int = 5,
) -> pd.DataFrame:
    """Compute 5-minute window aggregation features per segment.

    Input must have: timestamp, segment_id, endpoint_group, client_ip,
    user_agent, status_class, time_taken, sc_bytes.

    Returns one row per (segment_id, window_start) with aggregated features.
    """
    df = df.copy()

    # Ensure timestamp
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Window assignment
    df["window_start"] = df["timestamp"].dt.floor(f"{window_minutes}min")

    # Aggregation
    agg = df.groupby(["segment_id", "endpoint_group", "window_start"]).agg(
        request_count=("timestamp", "size"),
        unique_ip_count=("client_ip", "nunique"),
        unique_ua_count=("user_agent", "nunique"),
        _4xx_count=("status_class", lambda s: (s == "4xx").sum()),
        _5xx_count=("status_class", lambda s: (s == "5xx").sum()),
        p50_time_taken=("time_taken", lambda s: s.quantile(0.50) if len(s) > 0 else 0),
        p95_time_taken=("time_taken", lambda s: s.quantile(0.95) if len(s) > 0 else 0),
        p99_time_taken=("time_taken", lambda s: s.quantile(0.99) if len(s) > 0 else 0),
        sum_sc_bytes=("sc_bytes", "sum"),
        p95_sc_bytes=("sc_bytes", lambda s: s.quantile(0.95) if len(s) > 0 else 0),
    ).reset_index()

    # Rates
    agg["4xx_rate"] = agg["_4xx_count"] / agg["request_count"].clip(lower=1)
    agg["5xx_rate"] = agg["_5xx_count"] / agg["request_count"].clip(lower=1)
    agg.drop(columns=["_4xx_count", "_5xx_count"], inplace=True)

    # Fill NaN
    numeric_cols = agg.select_dtypes(include=[np.number]).columns
    agg[numeric_cols] = agg[numeric_cols].fillna(0)

    return agg


def add_burstiness(
    windows_df: pd.DataFrame,
    rolling_hours: int = 3,
    window_minutes: int = 5,
) -> pd.DataFrame:
    """Add burstiness features: current / rolling_mean.

    Must be called on sorted windows per segment.
    """
    df = windows_df.sort_values(["segment_id", "window_start"]).copy()
    n_windows = (rolling_hours * 60) // window_minutes

    def _add_burst(group: pd.DataFrame) -> pd.DataFrame:
        rolling_mean_count = (
            group["request_count"]
            .rolling(window=n_windows, min_periods=1)
            .mean()
        )
        rolling_mean_5xx = (
            group["5xx_rate"]
            .rolling(window=n_windows, min_periods=1)
            .mean()
        )
        group = group.copy()
        group["burstiness"] = group["request_count"] / rolling_mean_count.clip(lower=1)
        group["error_burstiness"] = group["5xx_rate"] / rolling_mean_5xx.clip(lower=0.001)
        return group

    return df.groupby("segment_id", group_keys=False).apply(_add_burst).reset_index(drop=True)


def build_layer1_features(
    df: pd.DataFrame,
    layer1_config: Layer1Config | None = None,
    url_config: URLConfig | None = None,
    seg_config: SegmentationConfig | None = None,
) -> pd.DataFrame:
    """End-to-end Layer 1 feature pipeline.

    1. Enrich raw data (endpoint group, segment ID, etc.)
    2. Compute window aggregation
    3. Add burstiness

    Args:
        df: Filtered dynamic-only DataFrame.
        layer1_config: Layer 1 configuration.
        url_config: URL normalization config.
        seg_config: Segmentation config.

    Returns:
        Window-level feature DataFrame.
    """
    if layer1_config is None:
        layer1_config = Layer1Config()

    enriched = enrich_for_layer1(df, url_config, seg_config)
    windows = compute_window_features(enriched, layer1_config.window_size_minutes)
    windows = add_burstiness(
        windows,
        rolling_hours=layer1_config.rolling_hours,
        window_minutes=layer1_config.window_size_minutes,
    )
    return windows


def save_layer1_features(
    features_df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save Layer 1 features as date-partitioned Parquet."""
    out = Path(output_dir)
    written: list[Path] = []

    features_df["_date"] = pd.to_datetime(features_df["window_start"]).dt.strftime("%Y-%m-%d")

    for date_val, group in features_df.groupby("_date"):
        part_dir = out / f"date={date_val}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "layer1_features.parquet"
        group.drop(columns=["_date"]).to_parquet(out_path, index=False)
        written.append(out_path)

    return written
