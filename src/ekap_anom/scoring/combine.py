"""Score combination: weighted fusion of layer scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ekap_anom.config import ScoringWeights
from ekap_anom.logging import get_logger

logger = get_logger(__name__)


def combine_scores(
    df: pd.DataFrame,
    weights: ScoringWeights | None = None,
    layer1_col: str = "layer1_score",
    layer2_col: str = "layer2_score",
    layer3_col: str = "layer3_score",
) -> pd.DataFrame:
    """Compute weighted final anomaly score.

    ``final_score = w1*layer1 + w2*layer2 + w3*layer3``

    Missing layer scores are treated as 0 and weights are re-normalized.

    Args:
        df: DataFrame with layer score columns.
        weights: Layer weights.
        layer1_col: Column name for Layer 1 score.
        layer2_col: Column name for Layer 2 score.
        layer3_col: Column name for Layer 3 score.

    Returns:
        DataFrame with ``final_score`` column added.
    """
    if weights is None:
        weights = ScoringWeights()

    df = df.copy()

    # Get available scores
    scores: list[tuple[str, float]] = []
    for col, w in [
        (layer1_col, weights.layer1),
        (layer2_col, weights.layer2),
        (layer3_col, weights.layer3),
    ]:
        if col in df.columns:
            scores.append((col, w))

    if not scores:
        df["final_score"] = 0.0
        return df

    # Re-normalize weights for available layers
    total_w = sum(w for _, w in scores)
    if total_w <= 0:
        total_w = 1.0

    df["final_score"] = sum(
        df[col].fillna(0.0).astype(float) * (w / total_w)
        for col, w in scores
    )

    # Clip to [0, 1]
    df["final_score"] = df["final_score"].clip(0.0, 1.0)

    logger.info(
        "scores_combined",
        layers_used=[col for col, _ in scores],
        weight_sum=round(total_w, 3),
    )
    return df
