"""Alert production: Top-K daily alerts with critical endpoint quotas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ekap_anom.config import ScoringConfig
from ekap_anom.logging import get_logger

logger = get_logger(__name__)


def build_alert_payload(row: pd.Series) -> dict[str, Any]:
    """Build a single alert payload from an anomalous row.

    Includes layer scores, reason codes, and context metrics.
    PII fields are excluded (no raw cookies/query values).
    """
    payload: dict[str, Any] = {
        "timestamp": str(row.get("timestamp", "")),
        "endpoint_group": row.get("endpoint_group", ""),
        "method": row.get("method", ""),
        "status": int(row["status"]) if pd.notna(row.get("status")) else None,
        "status_class": row.get("status_class", ""),
    }

    # Scores
    for col in ["layer1_score", "layer2_score", "layer3_score", "final_score"]:
        payload[col] = round(float(row[col]), 4) if col in row.index and pd.notna(row[col]) else None

    # Reason codes
    payload["reason_codes"] = row.get("reason_codes", [])

    # Context (window-level stats if available)
    context: dict[str, Any] = {}
    for ctx_col in ["request_count", "5xx_rate", "p99_time_taken", "burstiness"]:
        if ctx_col in row.index and pd.notna(row[ctx_col]):
            context[ctx_col] = round(float(row[ctx_col]), 4)
    payload["context"] = context

    # UA family (non-PII)
    payload["ua_family"] = row.get("ua_browser_family", "")
    payload["bot_flag"] = int(row.get("bot_flag", 0)) if pd.notna(row.get("bot_flag")) else 0

    # Performance summary
    payload["time_taken"] = int(row["time_taken"]) if pd.notna(row.get("time_taken")) else None
    payload["sc_bytes"] = int(row["sc_bytes"]) if pd.notna(row.get("sc_bytes")) else None

    return payload


def produce_alerts(
    df: pd.DataFrame,
    config: ScoringConfig | None = None,
    critical_endpoints: list[str] | None = None,
) -> pd.DataFrame:
    """Select Top-K alerts per day with critical endpoint quotas.

    Args:
        df: Scored DataFrame with ``is_anomaly``, ``final_score``.
        config: Scoring configuration.
        critical_endpoints: Endpoints with dedicated alert quota.

    Returns:
        Alert DataFrame (subset of input with highest scores).
    """
    if config is None:
        config = ScoringConfig()

    anomalies = df[df.get("is_anomaly", pd.Series(dtype=bool)).fillna(False)].copy()
    if anomalies.empty:
        logger.info("no_anomalies_for_alerts")
        return anomalies

    alerts: list[pd.DataFrame] = []

    # Critical endpoint alerts (separate quota)
    if critical_endpoints:
        critical_lower = {ep.lower() for ep in critical_endpoints}
        crit_mask = anomalies["endpoint_group"].str.lower().isin(critical_lower)
        crit_df = anomalies[crit_mask].nlargest(config.topk_critical, "final_score")
        alerts.append(crit_df)
        # Remove critical from general pool
        anomalies = anomalies[~crit_mask]

    # General Top-K
    remaining_k = config.topk_daily - sum(len(a) for a in alerts)
    if remaining_k > 0 and not anomalies.empty:
        general = anomalies.nlargest(min(remaining_k, len(anomalies)), "final_score")
        alerts.append(general)

    result = pd.concat(alerts, ignore_index=True) if alerts else pd.DataFrame()

    logger.info(
        "alerts_produced",
        total_anomalies=len(df[df.get("is_anomaly", pd.Series(dtype=bool)).fillna(False)]),
        alerts_selected=len(result),
    )
    return result


def save_alerts(
    alerts_df: pd.DataFrame,
    output_dir: str | Path,
    date: str | None = None,
) -> list[Path]:
    """Save alerts as JSONL and Parquet.

    Args:
        alerts_df: Alert DataFrame.
        output_dir: Output root (``data/alerts/``).
        date: Date partition (YYYY-MM-DD).

    Returns:
        List of written file paths.
    """
    out = Path(output_dir)
    written: list[Path] = []

    if alerts_df.empty:
        return written

    if date is None:
        date = "unknown"

    part_dir = out / f"date={date}"
    part_dir.mkdir(parents=True, exist_ok=True)

    # JSONL
    jsonl_path = part_dir / "alerts.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for _, row in alerts_df.iterrows():
            payload = build_alert_payload(row)
            fh.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")
    written.append(jsonl_path)

    # Parquet
    parquet_path = part_dir / "alerts.parquet"
    alerts_df.to_parquet(parquet_path, index=False)
    written.append(parquet_path)

    logger.info("alerts_saved", date=date, files=len(written))
    return written
