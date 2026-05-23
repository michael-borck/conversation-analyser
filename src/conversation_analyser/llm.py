"""Thin Anthropic SDK wrapper with prompt caching, JSON-only parsing, and retry.

Ported from marking_pipeline/llm.py. All LLM calls go through call_json; the
system prompt is cached (cache_control: ephemeral) so repeated classification of
many conversations reuses the cached taxonomy prompt.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any


class LLMFailure(RuntimeError):
    """Raised when an LLM call cannot return parseable JSON after one retry."""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_CLIENT = None


def llm_available() -> bool:
    """True if the anthropic SDK is importable and an API key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _CLIENT = Anthropic(api_key=api_key)
    return _CLIENT


def call_json(
    *,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 2000,
    cache_system: bool = True,
) -> dict[str, Any]:
    """Single Anthropic call, JSON-only output, one retry on parse failure."""
    client = _get_client()

    if cache_system:
        system_param: Any = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    else:
        system_param = system

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_param,
                messages=[{"role": "user", "content": user}],
            )
            text = message.content[0].text if message.content else ""
            return _parse_json(text)
        except (json.JSONDecodeError, ValueError, IndexError) as exc:
            last_error = exc
            time.sleep(1.0 * (attempt + 1))
            continue

    raise LLMFailure(f"call_json failed after retry: {last_error}")


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_RE.search(cleaned)
        if match is None:
            raise
        return json.loads(match.group(0))
