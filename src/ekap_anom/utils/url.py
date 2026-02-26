"""URL normalization, path templating, query-string parsing."""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse, parse_qs
from typing import Any


def normalize_url(
    uri_stem: str,
    *,
    lowercase: bool = True,
    strip_trailing_slash: bool = True,
    numeric_template: bool = True,
    aggressive_template: bool = False,
) -> str:
    """Normalize a ``cs-uri-stem`` value.

    Steps:
      1. URL-decode
      2. Optional lowercase
      3. Strip trailing ``/`` (except root)
      4. Numeric segment templating ``/123/`` → ``/{num}/``
      5. Optional aggressive templating (GUID, hex, etc.)
    """
    path = unquote(uri_stem)
    if lowercase:
        path = path.lower()
    if strip_trailing_slash and len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    if numeric_template:
        # Replace pure numeric segments: /something/12345/other -> /something/{num}/other
        path = re.sub(r"/\d+(?=/|$)", "/{num}", path)

    if aggressive_template:
        # GUIDs
        path = re.sub(
            r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=/|$)",
            "/{guid}",
            path,
            flags=re.IGNORECASE,
        )
        # Long hex strings (>=16 chars)
        path = re.sub(r"/[0-9a-f]{16,}(?=/|$)", "/{hex}", path, flags=re.IGNORECASE)

    return path


def extract_extension(uri_stem: str) -> str | None:
    """Extract file extension from URI stem (lowercase, without dot).

    Returns ``None`` if no extension is present.
    """
    path = unquote(uri_stem).lower()
    # Handle query strings that leaked into stem
    path = path.split("?")[0]
    dot_pos = path.rfind(".")
    if dot_pos == -1:
        return None
    ext = path[dot_pos + 1:]
    # Ignore extensions with slashes (not real extensions)
    if "/" in ext:
        return None
    return ext


def path_depth(uri_stem: str) -> int:
    """Count the number of path segments (``/a/b/c`` → 3)."""
    path = unquote(uri_stem).strip("/")
    if not path:
        return 0
    return len(path.split("/"))


def path_tokens(uri_stem: str) -> list[str]:
    """Split URI stem into path segment tokens."""
    path = unquote(uri_stem).lower().strip("/")
    if not path:
        return []
    return path.split("/")


def parse_query_string(query: str | None) -> dict[str, Any]:
    """Parse query string into structured statistics.

    Returns:
        Dict with ``key_set``, ``key_count``, ``value_count``,
        ``mean_value_len``, ``max_value_len``, ``special_char_ratio``,
        ``decoded_len``.
    """
    if not query or query == "-":
        return {
            "key_set": [],
            "key_count": 0,
            "value_count": 0,
            "mean_value_len": 0.0,
            "max_value_len": 0,
            "special_char_ratio": 0.0,
            "decoded_len": 0,
        }

    decoded = unquote(query)
    parsed = parse_qs(decoded, keep_blank_values=True)

    keys = sorted(parsed.keys())
    all_values = [v for vals in parsed.values() for v in vals]
    value_lens = [len(v) for v in all_values] if all_values else [0]

    # Special character ratio in decoded query
    special_chars = sum(1 for c in decoded if not c.isalnum() and c not in "=&_-.")
    special_ratio = special_chars / max(len(decoded), 1)

    return {
        "key_set": keys,
        "key_count": len(keys),
        "value_count": len(all_values),
        "mean_value_len": sum(value_lens) / max(len(value_lens), 1),
        "max_value_len": max(value_lens) if value_lens else 0,
        "special_char_ratio": round(special_ratio, 4),
        "decoded_len": len(decoded),
    }


def endpoint_group(
    uri_stem: str,
    *,
    lowercase: bool = True,
    strip_trailing_slash: bool = True,
    numeric_template: bool = True,
    aggressive_template: bool = False,
) -> str:
    """Compute the endpoint group key from a URI stem.

    This is the canonical grouping key used for segmentation.
    """
    return normalize_url(
        uri_stem,
        lowercase=lowercase,
        strip_trailing_slash=strip_trailing_slash,
        numeric_template=numeric_template,
        aggressive_template=aggressive_template,
    )
