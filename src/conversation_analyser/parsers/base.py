"""Shared types and helpers for conversation parsers.

Every adapter normalises its input into a flat list of `ParsedTurn`. The
`label`/`rationale` fields are usually left None — the taxonomy step fills them
later — except the LLM-segment adapter, which segments and labels in one call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


class ParseError(ValueError):
    """Raised when a parser cannot turn its input into turns."""


@dataclass
class ParsedTurn:
    role: str  # "human" | "assistant"
    content: str
    timestamp: datetime | None = None
    label: str | None = None  # pre-filled only by the LLM-segment adapter
    rationale: str | None = None


@dataclass
class ParseResult:
    turns: list[ParsedTurn]
    format_name: str  # role_content | anythingllm | text | ...
    parse_mode: str  # structured | heuristic | llm-segment | unsegmented
    notes: list[str] = field(default_factory=list)


def parse_timestamp(value: object) -> datetime | None:
    """Best-effort parse of epoch-ms/seconds ints or ISO-8601 strings."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # Heuristic: values past ~year 2001 in seconds are < 1e10; ms are larger.
        seconds = value / 1000.0 if value > 1e11 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None
