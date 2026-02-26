"""Streaming file reader for IIS log files (.log, .txt, .gz)."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator


def iter_lines(path: str | Path) -> Iterator[str]:
    """Yield stripped lines from a log file, supporting gzip.

    Args:
        path: Path to ``.log``, ``.txt``, or ``.gz`` file.

    Yields:
        Each non-empty line (stripped).
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[call-overload]
        for line in fh:
            stripped = line.rstrip("\n\r")
            if stripped:
                yield stripped


def iter_log_files(directory: str | Path, recursive: bool = True) -> Iterator[Path]:
    """Yield all log files in a directory.

    Supported extensions: ``.log``, ``.txt``, ``.gz``
    """
    d = Path(directory)
    pattern = "**/*" if recursive else "*"
    for p in sorted(d.glob(pattern)):
        if p.is_file() and p.suffix.lower() in (".log", ".txt", ".gz"):
            yield p
