"""Filter to keep only dynamic page requests (.aspx, .ashx)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ekap_anom.config import FilterConfig
from ekap_anom.logging import get_logger
from ekap_anom.utils.url import extract_extension

logger = get_logger(__name__)


def is_dynamic(uri_stem: str, allowed_extensions: set[str]) -> bool:
    """Check if a URI stem has a dynamic extension."""
    ext = extract_extension(uri_stem)
    if ext is None:
        return False
    return ext in allowed_extensions


def filter_dynamic_only(
    df: pd.DataFrame,
    config: FilterConfig | None = None,
) -> pd.DataFrame:
    """Filter DataFrame to keep only dynamic page requests.

    Removes static content (images, CSS, JS, fonts, etc.) and keeps
    only requests to configured dynamic extensions.

    Args:
        df: Input DataFrame with ``uri_stem`` column.
        config: Filter configuration. Uses defaults if not provided.

    Returns:
        Filtered DataFrame containing only dynamic requests.
    """
    if config is None:
        config = FilterConfig()

    allowed = config.allowed_extensions
    before = len(df)

    if "uri_stem" not in df.columns:
        logger.error("filter_missing_column", column="uri_stem")
        return df

    mask = df["uri_stem"].apply(lambda u: is_dynamic(u, allowed) if pd.notna(u) else False)
    result = df[mask].copy()

    dropped = before - len(result)
    logger.info(
        "dynamic_filter_applied",
        before=before,
        after=len(result),
        dropped=dropped,
        allowed_extensions=sorted(allowed),
    )
    return result


def filter_parquet_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    config: FilterConfig | None = None,
) -> list[Path]:
    """Apply dynamic filter to all Parquet files in a directory.

    Args:
        input_dir: Input parsed parquet directory.
        output_dir: Output for filtered data.
        config: Filter configuration.

    Returns:
        List of written output files.
    """
    if config is None:
        config = FilterConfig()

    inp = Path(input_dir)
    out = Path(output_dir)
    written: list[Path] = []

    for parquet_file in sorted(inp.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        filtered = filter_dynamic_only(df, config)

        if filtered.empty:
            continue

        # Preserve partition structure
        relative = parquet_file.relative_to(inp)
        out_path = out / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(out_path, index=False)
        written.append(out_path)

    return written
