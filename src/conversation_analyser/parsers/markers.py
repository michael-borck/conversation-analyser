"""Heuristic flat-text parser: split on speaker-label markers.

Ported and extended from marking_pipeline/transcript.py. A turn is a contiguous
block introduced by a label such as 'User:', 'Assistant:', 'Me:', 'ChatGPT:',
'You said:', 'ChatGPT said:', 'Prompt:', 'Response:'. The first label sets the
speaker; following lines belong to it until the next label.
"""
from __future__ import annotations

import re

from .base import ParsedTurn

FORMAT_NAME = "text"

# Label → normalised role. Multi-word labels are matched before single words.
_LABEL_ROLE: dict[str, str] = {
    "you said": "human",
    "chatgpt said": "assistant",
    "user": "human",
    "student": "human",
    "human": "human",
    "prompt": "human",
    "me": "human",
    "you": "human",
    "q": "human",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "claude": "assistant",
    "copilot": "assistant",
    "response": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "ai": "assistant",
    "a": "assistant",
}

# Longest labels first so "you said" wins over "you", "chatgpt said" over "chatgpt".
_LABELS_SORTED = sorted(_LABEL_ROLE, key=len, reverse=True)
_LABEL_RE = re.compile(
    r"^[\s>*#-]*(?P<label>(?:" + "|".join(re.escape(l) for l in _LABELS_SORTED) + r"))\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def parse(text: str) -> list[ParsedTurn]:
    """Split text into speaker turns. Returns [] if no labels are found."""
    matches = list(_LABEL_RE.finditer(text))
    if not matches:
        return []

    turns: list[ParsedTurn] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        role = _LABEL_ROLE[m.group("label").lower()]
        turns.append(ParsedTurn(role=role, content=body))
    return turns
