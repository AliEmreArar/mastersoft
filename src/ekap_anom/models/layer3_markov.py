"""Layer 3: Markov chain transition scoring for session sequences."""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from ekap_anom.logging import get_logger

logger = get_logger(__name__)


class MarkovTransitionScorer:
    """Score session sequences using rare-transition probabilities.

    Builds a first-order Markov transition matrix from training sequences,
    then scores new sequences based on how unlikely their transitions are.
    """

    def __init__(self, smoothing: float = 1e-6) -> None:
        self.smoothing = smoothing
        self.transition_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.transition_probs: dict[str, dict[str, float]] = {}
        self.states: set[str] = set()

    def fit(self, sequences: dict[str, list[str]]) -> None:
        """Learn transition probabilities from training sequences.

        Args:
            sequences: Dict of session_id → list of endpoint states.
        """
        for session_id, seq in sequences.items():
            for i in range(len(seq) - 1):
                from_state = seq[i]
                to_state = seq[i + 1]
                self.transition_counts[from_state][to_state] += 1
                self.states.add(from_state)
                self.states.add(to_state)

        # Compute probabilities with smoothing
        n_states = len(self.states)
        for from_state, targets in self.transition_counts.items():
            total = sum(targets.values()) + self.smoothing * n_states
            self.transition_probs[from_state] = {
                to_state: (count + self.smoothing) / total
                for to_state, count in targets.items()
            }

        logger.info(
            "markov_trained",
            n_states=n_states,
            n_sequences=len(sequences),
        )

    def score_sequence(self, sequence: list[str]) -> float:
        """Score a sequence. Returns anomaly score in [0, 1].

        Higher score = more anomalous (rare transitions).
        """
        if len(sequence) < 2:
            return 0.0

        log_probs: list[float] = []
        n_states = max(len(self.states), 1)

        for i in range(len(sequence) - 1):
            from_state = sequence[i]
            to_state = sequence[i + 1]

            if from_state in self.transition_probs:
                prob = self.transition_probs[from_state].get(
                    to_state, self.smoothing / n_states
                )
            else:
                prob = self.smoothing / n_states

            log_probs.append(np.log(prob))

        # Average -log_prob as anomaly signal
        avg_neg_log_prob = -np.mean(log_probs)

        # Sigmoid normalize to [0, 1]
        score = 1.0 / (1.0 + np.exp(-avg_neg_log_prob + 3))
        return float(np.clip(score, 0, 1))

    def score_sessions(self, sequences: dict[str, list[str]]) -> dict[str, float]:
        """Score multiple sessions."""
        return {
            session_id: self.score_sequence(seq)
            for session_id, seq in sequences.items()
        }

    def save(self, path: str | Path) -> None:
        """Persist the model."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "smoothing": self.smoothing,
            "transition_counts": dict(self.transition_counts),
            "transition_probs": self.transition_probs,
            "states": self.states,
        }
        with open(p, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load(cls, path: str | Path) -> "MarkovTransitionScorer":
        """Load a persisted model."""
        with open(Path(path), "rb") as fh:
            state = pickle.load(fh)
        obj = cls(smoothing=state["smoothing"])
        obj.transition_counts = defaultdict(lambda: defaultdict(int), state["transition_counts"])
        obj.transition_probs = state["transition_probs"]
        obj.states = state["states"]
        return obj
