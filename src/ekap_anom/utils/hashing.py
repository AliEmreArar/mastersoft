"""MurmurHash-based feature hashing for fixed-dimension vector representations."""

from __future__ import annotations

import numpy as np
import mmh3


def hash_tokens_to_vector(tokens: list[str], dim: int, seed: int = 42) -> np.ndarray:
    """Hash a list of string tokens into a fixed-dimension vector using the hashing trick.

    Each token is hashed to a bucket (MurmurHash3) and the corresponding
    element is incremented (+1 / -1 based on a secondary hash for sign).

    Args:
        tokens: List of string tokens.
        dim: Output vector dimensionality.
        seed: Random seed for hashing.

    Returns:
        1-D float32 array of shape ``(dim,)``.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        h = mmh3.hash(token, seed=seed, signed=False)
        bucket = h % dim
        sign = 1 if (mmh3.hash(token, seed=seed + 1, signed=True) >= 0) else -1
        vec[bucket] += sign
    return vec


def hash_single(value: str, seed: int = 42) -> int:
    """Return unsigned 32-bit MurmurHash3 for a single string."""
    return mmh3.hash(value, seed=seed, signed=False)


def segment_id(
    endpoint_group: str,
    method: str,
    status_class: str,
    bot_flag: int,
    host: str | None = None,
) -> str:
    """Compute deterministic segment identifier.

    SegmentID = hash(endpoint_group|method|status_class|bot_flag|host?)
    """
    parts = [endpoint_group, method, status_class, str(bot_flag)]
    if host is not None:
        parts.append(host)
    raw = "|".join(parts)
    return f"seg_{mmh3.hash(raw, seed=42, signed=False):08x}"
