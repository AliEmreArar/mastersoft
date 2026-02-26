"""Drift detection: PSI, JSD, and score distribution shift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from ekap_anom.config import DriftConfig
from ekap_anom.logging import get_logger

logger = get_logger(__name__)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Population Stability Index for numeric distributions.

    Args:
        reference: Training/reference distribution values.
        current: Current/production distribution values.
        n_bins: Number of histogram bins.

    Returns:
        PSI value (0 = no shift, >0.2 = significant shift).
    """
    eps = 1e-8
    # Common bin edges from reference
    bins = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    ref_pct = ref_hist / max(ref_hist.sum(), 1) + eps
    cur_pct = cur_hist / max(cur_hist.sum(), 1) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_jsd(
    reference_counts: dict[str, int],
    current_counts: dict[str, int],
) -> float:
    """Jensen-Shannon Divergence for categorical distributions.

    Args:
        reference_counts: Category → count mapping for reference.
        current_counts: Category → count mapping for current.

    Returns:
        JSD value in [0, 1].
    """
    all_keys = set(reference_counts.keys()) | set(current_counts.keys())
    ref_vals = np.array([reference_counts.get(k, 0) for k in sorted(all_keys)], dtype=float)
    cur_vals = np.array([current_counts.get(k, 0) for k in sorted(all_keys)], dtype=float)

    ref_vals = ref_vals / max(ref_vals.sum(), 1)
    cur_vals = cur_vals / max(cur_vals.sum(), 1)

    return float(jensenshannon(ref_vals, cur_vals))


def detect_score_shift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> dict[str, float]:
    """Detect shift in anomaly score distribution.

    Returns:
        Dict with mean_shift, std_shift, and ks_statistic.
    """
    from scipy.stats import ks_2samp

    ks_stat, ks_pval = ks_2samp(reference_scores, current_scores)

    return {
        "mean_shift": float(abs(current_scores.mean() - reference_scores.mean())),
        "std_shift": float(abs(current_scores.std() - reference_scores.std())),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
    }


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    score_col: str = "final_score",
    config: DriftConfig | None = None,
) -> dict[str, Any]:
    """Generate comprehensive drift report.

    Args:
        reference_df: Training/reference data.
        current_df: Current/production data.
        numeric_features: List of numeric feature columns for PSI.
        categorical_features: List of categorical columns for JSD.
        score_col: Anomaly score column.
        config: Drift detection configuration.

    Returns:
        Full drift report dict.
    """
    if config is None:
        config = DriftConfig()

    report: dict[str, Any] = {
        "reference_size": len(reference_df),
        "current_size": len(current_df),
        "features": {},
        "alerts": [],
    }

    # Numeric PSI
    for feat in numeric_features:
        if feat in reference_df.columns and feat in current_df.columns:
            ref_vals = reference_df[feat].dropna().values
            cur_vals = current_df[feat].dropna().values
            if len(ref_vals) > 0 and len(cur_vals) > 0:
                psi = compute_psi(ref_vals, cur_vals)
                report["features"][feat] = {"type": "numeric", "psi": round(psi, 4)}
                if psi > config.psi_threshold:
                    report["alerts"].append({
                        "feature": feat, "metric": "psi", "value": round(psi, 4),
                        "threshold": config.psi_threshold, "action": "retrain_recommended",
                    })

    # Categorical JSD
    for feat in categorical_features:
        if feat in reference_df.columns and feat in current_df.columns:
            ref_counts = reference_df[feat].value_counts().to_dict()
            cur_counts = current_df[feat].value_counts().to_dict()
            jsd = compute_jsd(ref_counts, cur_counts)
            report["features"][feat] = {"type": "categorical", "jsd": round(jsd, 4)}
            if jsd > config.jsd_threshold:
                report["alerts"].append({
                    "feature": feat, "metric": "jsd", "value": round(jsd, 4),
                    "threshold": config.jsd_threshold, "action": "retrain_recommended",
                })

    # Score distribution shift
    if score_col in reference_df.columns and score_col in current_df.columns:
        ref_scores = reference_df[score_col].dropna().values
        cur_scores = current_df[score_col].dropna().values
        if len(ref_scores) > 0 and len(cur_scores) > 0:
            shift = detect_score_shift(ref_scores, cur_scores)
            report["score_shift"] = shift
            if shift["mean_shift"] > config.score_shift_threshold:
                report["alerts"].append({
                    "metric": "score_distribution_shift",
                    "value": round(shift["mean_shift"], 4),
                    "threshold": config.score_shift_threshold,
                    "action": "retrain_recommended",
                })

    report["needs_retrain"] = len(report["alerts"]) > 0

    logger.info(
        "drift_report_generated",
        n_features_checked=len(report["features"]),
        n_alerts=len(report["alerts"]),
        needs_retrain=report["needs_retrain"],
    )
    return report


def save_drift_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """Save drift report as JSON + Markdown summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / "drift_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    # Markdown summary
    md_path = out / "drift_report.md"
    lines = ["# Drift Report\n"]
    lines.append(f"- Reference size: {report['reference_size']}")
    lines.append(f"- Current size: {report['current_size']}")
    lines.append(f"- Needs retrain: **{report['needs_retrain']}**\n")

    if report.get("alerts"):
        lines.append("## Alerts\n")
        for alert in report["alerts"]:
            lines.append(f"- ⚠️ {alert.get('feature', alert.get('metric', 'unknown'))}: "
                        f"{alert['metric']}={alert['value']} > {alert['threshold']}")

    lines.append("\n## Feature Details\n")
    for feat, info in report.get("features", {}).items():
        if info["type"] == "numeric":
            lines.append(f"- `{feat}`: PSI={info['psi']}")
        else:
            lines.append(f"- `{feat}`: JSD={info['jsd']}")

    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))

    return json_path
