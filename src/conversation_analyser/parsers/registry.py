"""Adapter registry + the parse strategy chain (design spec §3).

Order in `auto` mode: structured adapters → heuristic markers → LLM-segment →
unsegmented single-blob fallback. New formats are added by registering an
adapter module (with `detect`, `parse`, `FORMAT_NAME`) in `_STRUCTURED`.
"""

from __future__ import annotations

import json

from . import anythingllm, llm_segment, markers, role_content
from .base import ParsedTurn, ParseError, ParseResult

# Structured adapters tried in order. Each exposes detect(payload) and parse(payload).
_STRUCTURED = [role_content, anythingllm]

# Dict wrappers whose list value holds the actual messages.
_UNWRAP_KEYS = ("messages", "chats", "conversation", "turns", "history", "data")

VALID_MODES = ("auto", "structured", "heuristic", "llm-segment")


def _unwrap(payload: object) -> object:
    if isinstance(payload, dict):
        for key in _UNWRAP_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return payload


def _unsegmented(text: str, notes: list[str]) -> ParseResult:
    notes = list(notes)
    notes.append("could_not_segment")
    body = (text or "").strip()
    turns = [ParsedTurn(role="human", content=body)] if body else []
    return ParseResult(turns, "text", "unsegmented", notes=notes)


def parse(
    payload: object, *, mode: str = "auto", allow_llm: bool = True
) -> ParseResult:
    if mode not in VALID_MODES:
        raise ParseError(f"unknown parse mode {mode!r}; expected one of {VALID_MODES}")

    # Structured (non-string) payloads.
    if not isinstance(payload, str):
        candidate = _unwrap(payload)
        for adapter in _STRUCTURED:
            if adapter.detect(candidate):
                return ParseResult(
                    adapter.parse(candidate), adapter.FORMAT_NAME, "structured"
                )
        if mode == "structured":
            raise ParseError("no structured adapter matched the payload")
        # Unrecognised structured payload: stringify and try the text parsers.
        payload = json.dumps(payload, ensure_ascii=False)

    text: str = payload
    notes: list[str] = []

    if mode in ("auto", "structured", "heuristic"):
        turns = markers.parse(text)
        if turns:
            return ParseResult(turns, markers.FORMAT_NAME, "heuristic")
        if mode == "heuristic":
            return _unsegmented(text, notes)

    if mode in ("auto", "llm-segment"):
        if allow_llm:
            turns, segment_notes = llm_segment.parse(text)
            if turns:
                return ParseResult(
                    turns, "text", "llm-segment", notes=notes + segment_notes
                )
        else:
            notes.append("llm_unavailable")

    return _unsegmented(text, notes)
