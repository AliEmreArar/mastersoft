"""IIS W3C log field definitions and schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Canonical field names and their Python types
FIELD_TYPE_MAP: dict[str, type] = {
    "date": str,
    "time": str,
    "c-ip": str,
    "cs-username": str,
    "cs-method": str,
    "cs-uri-stem": str,
    "cs-uri-query": str,
    "sc-status": int,
    "sc-substatus": int,
    "sc-win32-status": int,
    "time-taken": int,
    "sc-bytes": int,
    "cs-bytes": int,
    "cs(User-Agent)": str,
    "cs(Referer)": str,
    "cs(Cookie)": str,
    "cs-host": str,
    "cs-version": str,
    "s-ip": str,
    "s-port": int,
    "s-sitename": str,
    "s-computername": str,
}

# Fields that must be present for a valid row
REQUIRED_FIELDS: set[str] = {
    "date", "time", "cs-method", "cs-uri-stem", "sc-status",
}

# Numeric fields that may need coercion
NUMERIC_FIELDS: set[str] = {
    "sc-status", "sc-substatus", "sc-win32-status",
    "time-taken", "sc-bytes", "cs-bytes", "s-port",
}

# Friendly aliases for DataFrame column names
COLUMN_ALIASES: dict[str, str] = {
    "c-ip": "client_ip",
    "cs-username": "username",
    "cs-method": "method",
    "cs-uri-stem": "uri_stem",
    "cs-uri-query": "uri_query",
    "sc-status": "status",
    "sc-substatus": "substatus",
    "sc-win32-status": "win32_status",
    "time-taken": "time_taken",
    "sc-bytes": "sc_bytes",
    "cs-bytes": "cs_bytes",
    "cs(User-Agent)": "user_agent",
    "cs(Referer)": "referer",
    "cs(Cookie)": "cookie",
    "cs-host": "host",
    "cs-version": "http_version",
    "s-ip": "server_ip",
    "s-port": "server_port",
    "s-sitename": "site_name",
    "s-computername": "server_name",
}


def get_alias(field_name: str) -> str:
    """Get the friendly alias for an IIS field name."""
    return COLUMN_ALIASES.get(field_name, field_name.replace("-", "_").replace("(", "_").replace(")", ""))


def coerce_value(field_name: str, raw_value: str) -> Any:
    """Coerce a raw string value to the appropriate Python type.

    ``"-"`` and empty strings are normalised to ``None``.
    """
    if raw_value in ("-", "", None):
        return None

    if field_name in NUMERIC_FIELDS:
        try:
            return int(raw_value)
        except (ValueError, TypeError):
            return None

    return raw_value
