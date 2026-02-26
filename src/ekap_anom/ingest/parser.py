"""W3C IIS log parser with dynamic field mapping."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ekap_anom.ingest.reader import iter_lines, iter_log_files
from ekap_anom.ingest.schema import (
    REQUIRED_FIELDS,
    coerce_value,
    get_alias,
)


def parse_fields_header(line: str) -> list[str]:
    """Extract field names from a ``#Fields:`` directive.

    Example:
        ``"#Fields: date time c-ip cs-method …"`` → ``["date", "time", "c-ip", "cs-method", …]``
    """
    return line.replace("#Fields:", "").strip().split()


def parse_line(raw: str, fields: list[str]) -> dict[str, Any] | None:
    """Parse a single data line into a dict using the current field mapping.

    Returns ``None`` if the line cannot be parsed (wrong column count, etc.).
    """
    parts = raw.split()
    if len(parts) != len(fields):
        return None

    row: dict[str, Any] = {}
    for field_name, value in zip(fields, parts):
        alias = get_alias(field_name)
        row[alias] = coerce_value(field_name, value)

    # Merge date + time → timestamp
    date_val = row.pop("date", None)
    time_val = row.pop("time", None)
    if date_val and time_val:
        try:
            row["timestamp"] = dt.datetime.strptime(
                f"{date_val} {time_val}", "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            row["timestamp"] = None
    else:
        row["timestamp"] = None

    # Missing indicators
    for alias in list(row.keys()):
        row[f"{alias}_missing"] = 1 if row[alias] is None else 0

    return row


def parse_file(path: str | Path, chunk_size: int = 100_000) -> Iterator[pd.DataFrame]:
    """Parse an IIS W3C log file and yield DataFrames in chunks.

    Handles:
      - Dynamic ``#Fields:`` header (may appear mid-file)
      - Comment lines starting with ``#`` (skipped)
      - ``"-"`` → ``None`` normalization
      - Missing indicator columns

    Args:
        path: Path to log file.
        chunk_size: Number of rows per yielded chunk.

    Yields:
        pd.DataFrame chunks.
    """
    fields: list[str] = []
    buffer: list[dict[str, Any]] = []

    for line in iter_lines(path):
        # Handle directives
        if line.startswith("#"):
            if line.startswith("#Fields:"):
                fields = parse_fields_header(line)
            continue

        if not fields:
            continue  # no header seen yet

        row = parse_line(line, fields)
        if row is not None:
            buffer.append(row)

        if len(buffer) >= chunk_size:
            yield pd.DataFrame(buffer)
            buffer.clear()

    if buffer:
        yield pd.DataFrame(buffer)


def parse_directory(
    directory: str | Path,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Parse all log files in a directory, yielding DataFrames.

    Args:
        directory: Path to directory containing log files.
        chunk_size: Rows per chunk.

    Yields:
        pd.DataFrame chunks.
    """
    for log_path in iter_log_files(directory):
        yield from parse_file(log_path, chunk_size=chunk_size)


def parse_to_parquet(
    input_path: str | Path,
    output_dir: str | Path,
    chunk_size: int = 100_000,
) -> list[Path]:
    """Parse IIS logs and write date-partitioned Parquet files.

    Args:
        input_path: File or directory of log files.
        output_dir: Root output directory (``data/parsed/``).
        chunk_size: Rows per chunk.

    Returns:
        List of written Parquet file paths.
    """
    output = Path(output_dir)
    written: list[Path] = []
    inp = Path(input_path)

    source = parse_file(inp, chunk_size) if inp.is_file() else parse_directory(inp, chunk_size)

    chunk_idx = 0
    for df in source:
        if df.empty:
            continue

        # Partition by date
        if "timestamp" in df.columns:
            df["_date"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
        else:
            df["_date"] = "unknown"

        for date_val, group in df.groupby("_date"):
            part_dir = output / f"date={date_val}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out_path = part_dir / f"chunk_{chunk_idx:06d}.parquet"
            group.drop(columns=["_date"]).to_parquet(out_path, index=False)
            written.append(out_path)
            chunk_idx += 1

    return written
