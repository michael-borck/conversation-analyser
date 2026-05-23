"""Adapter for AnythingLLM-style prompt/response paired rows.

Each row carries one human prompt and one AI response (the `embed_chats` shape).
`response` is often a JSON string like {"text": "...", "type": "chat"}.
"""
from __future__ import annotations

import json

from .base import ParsedTurn, parse_timestamp

FORMAT_NAME = "anythingllm"

_TS_KEYS = ("createdAt", "created_at", "sentAt", "timestamp")


def detect(payload: object) -> bool:
    if not isinstance(payload, list) or not payload:
        return False
    return all(
        isinstance(r, dict) and "prompt" in r and "response" in r for r in payload
    )


def _response_text(value: object) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict) and "text" in obj:
                    return str(obj.get("text") or "").strip()
            except json.JSONDecodeError:
                pass
        return stripped
    if isinstance(value, dict):
        return str(value.get("text") or "").strip()
    return str(value or "").strip()


def parse(payload: list[dict]) -> list[ParsedTurn]:
    turns: list[ParsedTurn] = []
    for row in payload:
        ts = None
        for key in _TS_KEYS:
            if key in row:
                ts = parse_timestamp(row[key])
                break
        prompt = str(row.get("prompt") or "").strip()
        response = _response_text(row.get("response"))
        if prompt:
            turns.append(ParsedTurn(role="human", content=prompt, timestamp=ts))
        if response:
            turns.append(ParsedTurn(role="assistant", content=response, timestamp=ts))
    return turns
