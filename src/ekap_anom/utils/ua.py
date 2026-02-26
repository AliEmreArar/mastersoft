"""User-Agent parsing, bot detection, and UA novelty tracking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ua_parser import user_agent_parser


@dataclass
class UAInfo:
    """Parsed User-Agent information."""

    browser_family: str = "Other"
    browser_version: str = ""
    os_family: str = "Other"
    os_version: str = ""
    device_family: str = "Other"
    bot_flag: int = 0
    raw: str = ""


# Pre-compiled default bot patterns
_DEFAULT_BOT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p)
    for p in [
        r"(?i)bot", r"(?i)crawler", r"(?i)spider", r"(?i)slurp",
        r"(?i)wget", r"(?i)curl", r"(?i)python-requests",
        r"(?i)httpx", r"(?i)scrapy",
    ]
]


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    """Compile regex patterns for bot detection."""
    return [re.compile(p) for p in patterns]


def is_bot(ua_string: str, patterns: list[re.Pattern[str]] | None = None) -> bool:
    """Check if a User-Agent string matches any bot pattern."""
    if not ua_string or ua_string == "-":
        return False
    check_patterns = patterns or _DEFAULT_BOT_PATTERNS
    return any(p.search(ua_string) for p in check_patterns)


def parse_ua(ua_string: str, bot_patterns: list[re.Pattern[str]] | None = None) -> UAInfo:
    """Parse a User-Agent string into structured components.

    Args:
        ua_string: Raw User-Agent header value.
        bot_patterns: Optional custom bot-detection regex patterns.

    Returns:
        Populated ``UAInfo`` dataclass.
    """
    if not ua_string or ua_string == "-":
        return UAInfo(raw=ua_string or "")

    parsed = user_agent_parser.Parse(ua_string)

    ua = parsed.get("user_agent", {})
    os_info = parsed.get("os", {})
    device = parsed.get("device", {})

    return UAInfo(
        browser_family=ua.get("family", "Other") or "Other",
        browser_version=ua.get("major", "") or "",
        os_family=os_info.get("family", "Other") or "Other",
        os_version=os_info.get("major", "") or "",
        device_family=device.get("family", "Other") or "Other",
        bot_flag=1 if is_bot(ua_string, bot_patterns) else 0,
        raw=ua_string,
    )


@dataclass
class UANoveltyTracker:
    """Track first-seen User-Agent strings over a rolling window.

    Used in offline/batch mode to compute ``ua_novelty`` features.
    """

    known_uas: set[str] = field(default_factory=set)

    def update(self, ua_string: str) -> bool:
        """Register a User-Agent and return True if it was *new*."""
        if ua_string in self.known_uas:
            return False
        self.known_uas.add(ua_string)
        return True

    def is_novel(self, ua_string: str) -> bool:
        """Check if a User-Agent has not been seen before."""
        return ua_string not in self.known_uas

    def reset(self) -> None:
        """Clear the tracker (e.g. at window boundary)."""
        self.known_uas.clear()
