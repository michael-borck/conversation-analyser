"""Capability manifest for the lens family.

A small, stable descriptor every family member exposes (as a constant, via
`conversation-analyser manifest`, and via `GET /manifest`). auto-analyser reads
these to build its routing table instead of hard-coding extension maps.

`auto_routable=False` marks this as an explicit-only *content interpretation* (a
conversation is not implied by any file extension), so auto-analyser never routes
to it automatically — it must be invoked deliberately.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    try:
        return version("conversation-analyser")
    except PackageNotFoundError:
        return "0.0.0"


MANIFEST: dict = {
    "name": "conversation-analyser",
    "version": _version(),
    "role": "analyser",
    "accepts": ["text", "messages", "transcript"],
    "extensions": [],  # explicit-only: not claimed for auto-routing
    "auto_routable": False,
    "produces": "ConversationAnalysis",
}
