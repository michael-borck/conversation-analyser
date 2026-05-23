"""LLM-segment fallback: find and label human turns directly from raw text.

Used only when structured adapters and heuristic markers both fail. Delegates to
taxonomy.classify_transcript, which segments and labels in one Haiku call, so the
returned turns already carry labels/rationales. Content is preview-length only
(the segmenter returns truncated prompts), so analytics in this mode are
approximate — the caller flags this with a note.
"""
from __future__ import annotations

from .base import ParsedTurn


def parse(text: str) -> list[ParsedTurn]:
    """Return labelled human turns, or [] if the LLM is unavailable/empty."""
    from .. import taxonomy

    try:
        items = taxonomy.classify_transcript(text)
    except Exception:
        return []
    return [
        ParsedTurn(
            role="human",
            content=item.get("prompt", ""),
            label=item.get("label"),
            rationale=item.get("rationale"),
        )
        for item in items
        if item.get("prompt")
    ]
