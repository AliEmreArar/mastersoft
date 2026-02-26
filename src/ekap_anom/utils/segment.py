"""Segment key generation for multi-dimensional grouping."""

from __future__ import annotations

import mmh3


def status_class(status_code: int | str | None) -> str:
    """Map HTTP status code to class string (2xx, 3xx, 4xx, 5xx, unknown)."""
    if status_code is None:
        return "unknown"
    try:
        code = int(status_code)
    except (ValueError, TypeError):
        return "unknown"
    if 200 <= code < 300:
        return "2xx"
    if 300 <= code < 400:
        return "3xx"
    if 400 <= code < 500:
        return "4xx"
    if 500 <= code < 600:
        return "5xx"
    return "unknown"


def make_segment_id(
    endpoint_group: str,
    method: str,
    sc_status_class: str,
    bot_flag: int,
    host: str | None = None,
    *,
    enable_host: bool = False,
) -> str:
    """Build deterministic segment identifier.

    ``SegmentID = hash(endpoint_group|method|status_class|bot_flag|host?)``
    """
    parts = [endpoint_group, method.upper(), sc_status_class, str(bot_flag)]
    if enable_host and host:
        parts.append(host.lower())
    raw = "|".join(parts)
    h = mmh3.hash(raw, seed=42, signed=False)
    return f"seg_{h:08x}"


def make_parent_segment_id(
    endpoint_group: str,
    method: str,
) -> str:
    """Fallback parent segment (endpoint_group + method only)."""
    raw = f"{endpoint_group}|{method.upper()}"
    h = mmh3.hash(raw, seed=42, signed=False)
    return f"pseg_{h:08x}"
