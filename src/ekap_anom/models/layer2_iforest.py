"""Layer 2 – Phase 1: Per-segment Isolation Forest with fallback hierarchy."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ekap_anom.config import IForestConfig, SegmentationConfig
from ekap_anom.logging import get_logger

logger = get_logger(__name__)


class SegmentIForest:
    """Per-segment Isolation Forest ensemble with parent/global fallback.

    Hierarchy:
      1. Segment-level IF  (if segment has ≥ min_segment_count samples)
      2. Parent-segment IF (endpoint_group + method only)
      3. Global IF         (all data)
    """

    def __init__(
        self,
        if_config: IForestConfig | None = None,
        min_segment_count: int = 5000,
    ) -> None:
        self.if_config = if_config or IForestConfig()
        self.min_segment_count = min_segment_count

        self.segment_models: dict[str, tuple[IsolationForest, StandardScaler]] = {}
        self.parent_models: dict[str, tuple[IsolationForest, StandardScaler]] = {}
        self.global_model: tuple[IsolationForest, StandardScaler] | None = None
        self.feature_columns: list[str] = []

    def _make_iforest(self) -> IsolationForest:
        """Create a fresh Isolation Forest with current config."""
        return IsolationForest(
            n_estimators=self.if_config.n_estimators,
            max_samples=self.if_config.max_samples,
            contamination=self.if_config.contamination,
            random_state=self.if_config.random_state,
            n_jobs=-1,
        )

    def _fit_one(
        self,
        X: np.ndarray,
        label: str = "",
    ) -> tuple[IsolationForest, StandardScaler]:
        """Fit scaler + IF on a numeric matrix."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = self._make_iforest()
        model.fit(X_scaled)
        logger.info("iforest_trained", segment=label, n_samples=X_scaled.shape[0])
        return model, scaler

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        segment_col: str = "segment_id",
        parent_segment_col: str | None = None,
    ) -> None:
        """Train per-segment + parent + global Isolation Forests.

        Args:
            df: Feature DataFrame.
            feature_columns: Numeric feature column names.
            segment_col: Column name for segment ID.
            parent_segment_col: Column for parent segment (optional fallback).
        """
        self.feature_columns = feature_columns
        X_all = df[feature_columns].fillna(0).values.astype(np.float32)

        # 1) Global model
        if len(X_all) > 0:
            self.global_model = self._fit_one(X_all, "GLOBAL")

        # 2) Per-segment models
        for seg_id, group in df.groupby(segment_col):
            if len(group) >= self.min_segment_count:
                X_seg = group[feature_columns].fillna(0).values.astype(np.float32)
                self.segment_models[str(seg_id)] = self._fit_one(X_seg, str(seg_id))

        # 3) Parent segment models (optional)
        if parent_segment_col and parent_segment_col in df.columns:
            for pseg_id, group in df.groupby(parent_segment_col):
                if (
                    len(group) >= self.min_segment_count
                    and str(pseg_id) not in self.parent_models
                ):
                    X_pseg = group[feature_columns].fillna(0).values.astype(np.float32)
                    self.parent_models[str(pseg_id)] = self._fit_one(X_pseg, f"parent_{pseg_id}")

        logger.info(
            "iforest_ensemble_fitted",
            segment_models=len(self.segment_models),
            parent_models=len(self.parent_models),
            has_global=self.global_model is not None,
        )

    def predict(
        self,
        df: pd.DataFrame,
        segment_col: str = "segment_id",
        parent_segment_col: str | None = None,
    ) -> np.ndarray:
        """Score requests. Returns anomaly scores in [0, 1].

        Higher score = more anomalous.
        """
        scores = np.zeros(len(df), dtype=np.float32)
        X_all = df[self.feature_columns].fillna(0).values.astype(np.float32)

        for i, (idx, row) in enumerate(df.iterrows()):
            seg_id = str(row[segment_col])
            x = X_all[i : i + 1]

            if seg_id in self.segment_models:
                model, scaler = self.segment_models[seg_id]
            elif (
                parent_segment_col
                and parent_segment_col in row.index
                and str(row[parent_segment_col]) in self.parent_models
            ):
                model, scaler = self.parent_models[str(row[parent_segment_col])]
            elif self.global_model is not None:
                model, scaler = self.global_model
            else:
                scores[i] = 0.5  # no model available
                continue

            x_scaled = scaler.transform(x)
            raw = model.decision_function(x_scaled)[0]
            # Normalize: IF decision_function is negative for anomalies
            # Map to [0, 1] where 1 is most anomalous
            scores[i] = 1.0 / (1.0 + np.exp(raw))

        return scores

    def save(self, path: str | Path) -> None:
        """Persist the ensemble to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "if_config": self.if_config,
            "min_segment_count": self.min_segment_count,
            "feature_columns": self.feature_columns,
            "segment_models": self.segment_models,
            "parent_models": self.parent_models,
            "global_model": self.global_model,
        }
        with open(p, "wb") as fh:
            pickle.dump(state, fh)
        logger.info("iforest_ensemble_saved", path=str(p))

    @classmethod
    def load(cls, path: str | Path) -> "SegmentIForest":
        """Load a persisted ensemble."""
        with open(Path(path), "rb") as fh:
            state = pickle.load(fh)
        obj = cls(
            if_config=state["if_config"],
            min_segment_count=state["min_segment_count"],
        )
        obj.feature_columns = state["feature_columns"]
        obj.segment_models = state["segment_models"]
        obj.parent_models = state["parent_models"]
        obj.global_model = state["global_model"]
        logger.info("iforest_ensemble_loaded", path=str(path))
        return obj
