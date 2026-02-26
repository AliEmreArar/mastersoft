"""Layer 3 – Sequence Autoencoder (placeholder for future implementation)."""

from __future__ import annotations

from ekap_anom.logging import get_logger

logger = get_logger(__name__)


class SequenceAutoencoder:
    """Placeholder for Seq2Seq Autoencoder on session sequences.

    This module is reserved for Phase 3 implementation.
    Current Layer 3 uses ``MarkovTransitionScorer`` instead.
    """

    def __init__(self) -> None:
        logger.info("seq_ae_placeholder_init", status="not_implemented")

    def fit(self, sequences: dict[str, list[str]]) -> None:
        """Placeholder fit method."""
        raise NotImplementedError("SequenceAutoencoder is a future feature (Phase 3).")

    def predict(self, sequences: dict[str, list[str]]) -> dict[str, float]:
        """Placeholder predict method."""
        raise NotImplementedError("SequenceAutoencoder is a future feature (Phase 3).")
