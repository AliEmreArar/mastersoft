"""Layer 2: Request-level feature engineering for outlier scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ekap_anom.config import HashingConfig
from ekap_anom.logging import get_logger
from ekap_anom.utils.hashing import hash_tokens_to_vector
from ekap_anom.utils.url import path_tokens, path_depth, parse_query_string
from ekap_anom.utils.ua import parse_ua

logger = get_logger(__name__)


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour and day_of_week from timestamp."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    return df


def compute_ip_stats(df: pd.DataFrame, window_minutes: int = 5) -> pd.DataFrame:
    """Compute per-IP statistics within recent time windows.

    Adds:
      - ``ip_rps_last_5m``: requests per second from this IP in last 5 min
      - ``ip_unique_endpoints_last_30m``: unique endpoints from this IP
      - ``first_seen_ip_in_7d``: 1 if IP not seen in training data (stub)
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)

    # Simple per-IP-per-window counts
    df["_window"] = ts.dt.floor(f"{window_minutes}min")
    ip_window_counts = df.groupby(["client_ip", "_window"]).agg(
        _ip_req_count=("timestamp", "size"),
        _ip_unique_ep=("endpoint_group", "nunique"),
    ).reset_index()

    df = df.merge(ip_window_counts, on=["client_ip", "_window"], how="left")
    df["ip_rps_last_5m"] = df["_ip_req_count"] / (window_minutes * 60)
    df["ip_unique_endpoints_last_30m"] = df["_ip_unique_ep"]  # simplified
    df["first_seen_ip_in_7d"] = 0  # will be set during training context
    df.drop(columns=["_window", "_ip_req_count", "_ip_unique_ep"], inplace=True)

    return df


def compute_ua_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse User-Agent and add encoded features."""
    df = df.copy()

    ua_data = df["user_agent"].fillna("").apply(parse_ua)
    df["ua_browser_family"] = ua_data.apply(lambda u: u.browser_family)
    df["ua_os_family"] = ua_data.apply(lambda u: u.os_family)
    df["ua_device_family"] = ua_data.apply(lambda u: u.device_family)
    df["ua_bot_flag"] = ua_data.apply(lambda u: u.bot_flag)
    df["ua_novelty"] = 0  # batch override later

    return df


def compute_path_features(
    df: pd.DataFrame,
    hash_config: HashingConfig | None = None,
) -> pd.DataFrame:
    """Compute path-based features: depth, token hash vector."""
    if hash_config is None:
        hash_config = HashingConfig()

    df = df.copy()
    df["path_depth"] = df["uri_stem"].fillna("").apply(path_depth)

    # Path token hash vector
    path_vecs = df["uri_stem"].fillna("").apply(
        lambda u: hash_tokens_to_vector(path_tokens(u), dim=hash_config.path_hash_dim)
    )
    path_cols = [f"path_hash_{i}" for i in range(hash_config.path_hash_dim)]
    path_df = pd.DataFrame(path_vecs.tolist(), columns=path_cols, index=df.index)
    df = pd.concat([df, path_df], axis=1)

    return df


def compute_query_features(
    df: pd.DataFrame,
    hash_config: HashingConfig | None = None,
) -> pd.DataFrame:
    """Compute query string features: key hash vector, special char ratio, suspicious flag."""
    if hash_config is None:
        hash_config = HashingConfig()

    df = df.copy()

    query_stats = df["uri_query"].fillna("").apply(parse_query_string)
    df["query_key_count"] = query_stats.apply(lambda q: q["key_count"])
    df["query_special_char_ratio"] = query_stats.apply(lambda q: q["special_char_ratio"])
    df["query_decoded_len"] = query_stats.apply(lambda q: q["decoded_len"])
    df["query_max_value_len"] = query_stats.apply(lambda q: q["max_value_len"])

    # Suspicious flag: very long values or high special char ratio
    df["query_suspicious_flag"] = (
        (df["query_special_char_ratio"] > 0.3) | (df["query_max_value_len"] > 500)
    ).astype(int)

    # Query key hash vector
    query_vecs = query_stats.apply(
        lambda q: hash_tokens_to_vector(q["key_set"], dim=hash_config.query_hash_dim)
    )
    query_cols = [f"query_hash_{i}" for i in range(hash_config.query_hash_dim)]
    query_df = pd.DataFrame(query_vecs.tolist(), columns=query_cols, index=df.index)
    df = pd.concat([df, query_df], axis=1)

    return df


def compute_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance features: log1p transforms, bytes_per_ms."""
    df = df.copy()
    df["log_time_taken"] = np.log1p(df["time_taken"].fillna(0).astype(float))
    df["log_sc_bytes"] = np.log1p(df["sc_bytes"].fillna(0).astype(float))
    df["log_cs_bytes"] = np.log1p(df["cs_bytes"].fillna(0).astype(float))
    df["bytes_per_ms"] = df["sc_bytes"].fillna(0) / df["time_taken"].fillna(1).clip(lower=1)
    return df


def compute_status_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode status triple (status, substatus, win32_status) as token hash."""
    df = df.copy()
    df["status_triple"] = (
        df["status"].fillna(0).astype(int).astype(str) + "_" +
        df["substatus"].fillna(0).astype(int).astype(str) + "_" +
        df["win32_status"].fillna(0).astype(int).astype(str)
    )
    # Simple hash encoding
    df["status_triple_hash"] = df["status_triple"].apply(
        lambda s: hash(s) % 1000
    )
    return df


def build_layer2_features(
    df: pd.DataFrame,
    hash_config: HashingConfig | None = None,
) -> pd.DataFrame:
    """End-to-end Layer 2 request-level feature pipeline.

    Requires enriched DataFrame (endpoint_group, segment_id already present).
    """
    df = compute_time_features(df)
    df = compute_ip_stats(df)
    df = compute_ua_features(df)
    df = compute_path_features(df, hash_config)
    df = compute_query_features(df, hash_config)
    df = compute_performance_features(df)
    df = compute_status_features(df)

    logger.info(
        "layer2_features_computed",
        n_rows=len(df),
        n_features=len(df.columns),
    )
    return df


def get_layer2_feature_columns(hash_config: HashingConfig | None = None) -> list[str]:
    """Return the list of numeric feature column names for Layer 2 model input."""
    if hash_config is None:
        hash_config = HashingConfig()

    cols = [
        "hour", "day_of_week",
        "ip_rps_last_5m", "ip_unique_endpoints_last_30m", "first_seen_ip_in_7d",
        "ua_bot_flag", "ua_novelty",
        "path_depth",
    ]
    cols += [f"path_hash_{i}" for i in range(hash_config.path_hash_dim)]
    cols += [
        "query_key_count", "query_special_char_ratio", "query_decoded_len",
        "query_max_value_len", "query_suspicious_flag",
    ]
    cols += [f"query_hash_{i}" for i in range(hash_config.query_hash_dim)]
    cols += [
        "status_triple_hash",
        "log_time_taken", "log_sc_bytes", "log_cs_bytes", "bytes_per_ms",
    ]
    return cols


def save_layer2_features(
    features_df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save Layer 2 features as date-partitioned Parquet."""
    out = Path(output_dir)
    written: list[Path] = []

    features_df = features_df.copy()
    ts = pd.to_datetime(features_df["timestamp"], utc=True)
    features_df["_date"] = ts.dt.strftime("%Y-%m-%d")

    for date_val, group in features_df.groupby("_date"):
        part_dir = out / f"date={date_val}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "layer2_features.parquet"
        group.drop(columns=["_date"]).to_parquet(out_path, index=False)
        written.append(out_path)

    return written
