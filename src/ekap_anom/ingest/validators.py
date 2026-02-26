"""Data quality validators for parsed IIS log data."""

from __future__ import annotations

import pandas as pd

from ekap_anom.logging import get_logger

logger = get_logger(__name__)


def validate_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing critical columns and log warnings.

    Required columns: ``timestamp``, ``method``, ``uri_stem``, ``status``.
    """
    required = ["timestamp", "method", "uri_stem", "status"]
    before = len(df)

    for col in required:
        if col not in df.columns:
            logger.warning("missing_required_column", column=col)
            return df.head(0)  # empty frame with same schema

    mask = df[required].notna().all(axis=1)
    result = df[mask].copy()
    dropped = before - len(result)
    if dropped > 0:
        logger.info("dropped_invalid_rows", count=dropped, reason="missing_required_fields")
    return result


def validate_status_range(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``status`` is within valid HTTP range [100, 599]."""
    if "status" not in df.columns:
        return df
    mask = df["status"].between(100, 599)
    dropped = (~mask).sum()
    if dropped > 0:
        logger.info("dropped_invalid_status", count=int(dropped))
    return df[mask].copy()


def validate_time_taken(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp negative ``time_taken`` to 0."""
    if "time_taken" not in df.columns:
        return df
    df = df.copy()
    neg_mask = df["time_taken"].fillna(0) < 0
    if neg_mask.any():
        logger.info("clamped_negative_time_taken", count=int(neg_mask.sum()))
        df.loc[neg_mask, "time_taken"] = 0
    return df


def run_all_validations(df: pd.DataFrame) -> pd.DataFrame:
    """Execute all validators in sequence."""
    df = validate_required_columns(df)
    df = validate_status_range(df)
    df = validate_time_taken(df)
    return df
