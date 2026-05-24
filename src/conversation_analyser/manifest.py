"""Capability manifest for the lens family (built via lens-contract).

A small, stable descriptor every family member exposes (as a constant, via
`conversation-analyser manifest`, and via `GET /manifest`). auto-analyser reads
these to build its routing table instead of hard-coding extension maps.

`auto_routable=False` marks this as an explicit-only *content interpretation* (a
conversation is not implied by any file extension), so auto-analyser never routes
to it automatically — it must be invoked deliberately.
"""
from __future__ import annotations

from lens_contract import make_manifest

MANIFEST = make_manifest(
    name="conversation-analyser",
    accepts=["text", "messages", "transcript"],
    extensions=[],  # explicit-only: not claimed for auto-routing
    auto_routable=False,
    produces="ConversationAnalysis",
)
