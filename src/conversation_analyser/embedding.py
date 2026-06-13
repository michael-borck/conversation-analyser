"""Conversation-transcript embedding via the family's shared helper (lens-embed).

A single pinned model across the family means this vector is comparable to
other members' vectors — the basis for cross-artefact and cohort-distinctiveness
signals downstream. Distinct from the internal prompt self-similarity in
embeddings.py (that measures within-conversation drift; this is one vector for
the whole transcript). Opt-in and degradable: returns None without the
[embeddings] extra or on any failure.
"""

from __future__ import annotations


def embed_document(text: str) -> list[float] | None:
    """Pooled, L2-normalised vector, or None if embeddings are off."""
    if not text or not text.strip():
        return None
    try:
        from lens_embed import backend_available, embed_long_text
    except ImportError:
        return None
    if not backend_available("text"):
        return None
    try:
        return embed_long_text(text)
    except Exception:
        return None
