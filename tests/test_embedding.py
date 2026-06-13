"""Conversation embedding wiring — field presence + graceful degradation."""
from __future__ import annotations
import importlib.util
import pytest
from conversation_analyser.embedding import embed_document
from conversation_analyser.models import ConversationAnalysis

_TEXT = importlib.util.find_spec("lens_embed") is not None and importlib.util.find_spec("sentence_transformers") is not None

def test_field_default_none():
    assert "embedding" in ConversationAnalysis.model_fields
    assert ConversationAnalysis.model_fields["embedding"].default is None

def test_empty_is_none():
    assert embed_document("") is None
    assert embed_document("  \n ") is None

@pytest.mark.skipif(_TEXT, reason="embeddings extra installed")
def test_none_without_backend():
    assert embed_document("Can you explain gradient descent?") is None

@pytest.mark.skipif(not _TEXT, reason="needs [embeddings]")
def test_vector_with_backend():
    v = embed_document("Can you explain gradient descent?\n\nWhy does the learning rate matter?\n\n" * 3)
    assert isinstance(v, list) and len(v) == 384
