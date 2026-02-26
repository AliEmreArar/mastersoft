"""Time-related utilities: UTC parsing, window bucketing, rolling helpers."""

from __future__ import annotations

import datetime as dt
from typing import Iterator


def parse_iis_datetime(date_str: str, time_str: str) -> dt.datetime:
    """Merge IIS ``date`` and ``time`` columns into a UTC datetime.

    Args:
        date_str: e.g. ``"2024-04-22"``
        time_str: e.g. ``"13:05:42"``

    Returns:
        Timezone-aware UTC datetime.
    """
    combined = f"{date_str} {time_str}"
    return dt.datetime.strptime(combined, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.timezone.utc,
    )


def floor_to_window(
    timestamp: dt.datetime,
    window_minutes: int = 5,
) -> dt.datetime:
    """Floor a timestamp to the start of its containing window.

    Example:
        ``2024-04-22 13:07:22`` with window=5 → ``2024-04-22 13:05:00``
    """
    total_minutes = timestamp.hour * 60 + timestamp.minute
    floored_minutes = (total_minutes // window_minutes) * window_minutes
    return timestamp.replace(
        hour=floored_minutes // 60,
        minute=floored_minutes % 60,
        second=0,
        microsecond=0,
    )


def generate_windows(
    start: dt.datetime,
    end: dt.datetime,
    window_minutes: int = 5,
    step_minutes: int = 1,
) -> Iterator[tuple[dt.datetime, dt.datetime]]:
    """Yield ``(window_start, window_end)`` between *start* and *end*.

    Args:
        start: Inclusive start timestamp.
        end: Exclusive end timestamp.
        window_minutes: Size of each window.
        step_minutes: Step between consecutive window starts.

    Yields:
        Tuples of ``(window_start, window_end)``.
    """
    step_delta = dt.timedelta(minutes=step_minutes)
    window_delta = dt.timedelta(minutes=window_minutes)
    current = floor_to_window(start, window_minutes)
    while current + window_delta <= end + step_delta:
        yield (current, current + window_delta)
        current += step_delta


def to_date_partition(timestamp: dt.datetime) -> str:
    """Return ``YYYY-MM-DD`` partition string."""
    return timestamp.strftime("%Y-%m-%d")
