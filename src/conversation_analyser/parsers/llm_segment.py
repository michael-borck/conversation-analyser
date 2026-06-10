"""LLM-segment fallback: find and label human turns directly from raw text.

Used only when structured adapters and heuristic markers both fail. Delegates to
taxonomy.classify_transcript, which segments and labels in one Haiku call, so the
returned turns already carry labels/rationales. Content is preview-length only
(the segmenter returns truncated prompts), so analytics in this mode are
approximate — the caller flags this with a note.

Long transcripts: a single classifier call only sees the first
``SEGMENT_CHUNK_CHARS`` characters, so a long unstructured transcript would get
only its opening labelled (and the band/ratios/score computed on that slice). To
credit the whole thing we split the text on paragraph boundaries into chunks
under that cap, classify each, and concatenate the turns; the pipeline then
aggregates over the full label sequence. A guard (``SEGMENT_MAX_CHUNKS``,
env-overridable) bounds the number of LLM calls and emits a note rather than
silently dropping the tail.
"""

from __future__ import annotations

from typing import Any

from .. import config
from .base import ParsedTurn

_PARA_SEP = "\n\n"
# Headroom so a chunk (<= SEGMENT_CHUNK_CHARS) is never re-truncated by
# classify_transcript's own max_chars cap.
_CLASSIFY_SLACK = 4000


def _chunk_text(text: str, size: int) -> list[str]:
    """Split text into <=size chunks, preferring paragraph boundaries.

    Accumulate paragraphs until the next would overflow ``size``, then start a new
    chunk; any single oversized paragraph is hard-split. Ported from the ISYS6020
    marking pipeline's chunked RoC analysis.
    """
    chunks: list[str] = []
    cur = ""
    for para in text.split(_PARA_SEP):
        if cur and len(cur) + len(para) + len(_PARA_SEP) > size:
            chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}{_PARA_SEP}{para}" if cur else para
    if cur:
        chunks.append(cur)
    out: list[str] = []
    for c in chunks:  # hard-split any single oversized paragraph
        while len(c) > size:
            out.append(c[:size])
            c = c[size:]
        if c:
            out.append(c)
    return out


def _to_turns(items: list[dict[str, Any]]) -> list[ParsedTurn]:
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


def parse(text: str) -> tuple[list[ParsedTurn], list[str]]:
    """Return (labelled human turns, notes).

    Empty turns if the LLM is unavailable/empty. Short transcripts take a single
    classifier call (unchanged behaviour); long ones are chunked so the whole
    transcript is segmented, not just the opening.
    """
    from .. import taxonomy

    body = (text or "").strip()
    if not body:
        return [], []

    chunk_chars = config.SEGMENT_CHUNK_CHARS
    classify_cap = chunk_chars + _CLASSIFY_SLACK
    notes: list[str] = ["llm_segment_preview_only"]

    # Short transcript: one call, no chunking.
    if len(body) <= chunk_chars:
        try:
            items = taxonomy.classify_transcript(body, max_chars=classify_cap)
        except Exception:
            return [], []
        return _to_turns(items), notes

    # Long transcript: chunk → classify each → concatenate.
    chunks = _chunk_text(body, chunk_chars)
    max_chunks = config.SEGMENT_MAX_CHUNKS
    capped = max_chunks > 0 and len(chunks) > max_chunks
    used = chunks[:max_chunks] if capped else chunks

    items: list[dict[str, Any]] = []
    for chunk in used:
        try:
            items.extend(taxonomy.classify_transcript(chunk, max_chars=classify_cap))
        except Exception:
            continue

    turns = _to_turns(items)
    if not turns:
        return [], []

    notes.append(f"chunked_full_transcript ({len(used)} chunks, {len(turns)} turns)")
    if capped:
        notes.append(
            f"segment_chunk_cap_hit (analysed {max_chunks}/{len(chunks)} chunks; "
            "raise CONVERSATION_ANALYSER_SEGMENT_MAX_CHUNKS)"
        )
    return turns, notes
