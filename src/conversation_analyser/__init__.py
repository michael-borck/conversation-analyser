"""conversation-analyser: critical-thinking + analytics for human-AI conversations.

Public API:

    from conversation_analyser import ConversationAnalyser, ConversationAnalysis

    result = ConversationAnalyser().analyse("transcript.txt")
    print(result.model_dump_json(indent=2))
"""
from __future__ import annotations

from .manifest import MANIFEST
from .models import ConversationAnalysis
from .pipeline import ConversationAnalyser

__all__ = ["ConversationAnalyser", "ConversationAnalysis", "MANIFEST"]
