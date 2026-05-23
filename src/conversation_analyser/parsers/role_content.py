"""Adapter for the de-facto standard role/content message list.

    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

Handles OpenAI/Anthropic shapes, including Anthropic content blocks
([{"type": "text", "text": "..."}]) and common timestamp keys. Non
human/assistant roles (system, tool) are ignored as context (design spec §15.4).
"""
from __future__ import annotations

from .base import ParsedTurn, parse_timestamp

FORMAT_NAME = "role_content"

_HUMAN = {"user", "human"}
_ASSISTANT = {"assistant", "ai", "bot", "model"}
_TS_KEYS = ("timestamp", "created_at", "createdAt", "time", "ts")


def detect(payload: object) -> bool:
    if not isinstance(payload, list) or not payload:
        return False
    return all(
        isinstance(m, dict) and "role" in m and ("content" in m or "text" in m)
        for m in payload
    )


def _content_text(value: object) -> str:
    if isinstance(value, list):  # Anthropic content blocks
        parts = [b.get("text", "") for b in value if isinstance(b, dict)]
        return " ".join(p for p in parts if p).strip()
    return str(value or "").strip()


def parse(payload: list[dict]) -> list[ParsedTurn]:
    turns: list[ParsedTurn] = []
    for msg in payload:
        role = str(msg.get("role", "")).lower()
        if role in _HUMAN:
            norm = "human"
        elif role in _ASSISTANT:
            norm = "assistant"
        else:
            continue  # system / tool / unknown → ignored context
        content = _content_text(msg.get("content", msg.get("text")))
        if not content:
            continue
        ts = None
        for key in _TS_KEYS:
            if key in msg:
                ts = parse_timestamp(msg[key])
                break
        turns.append(ParsedTurn(role=norm, content=content, timestamp=ts))
    return turns
