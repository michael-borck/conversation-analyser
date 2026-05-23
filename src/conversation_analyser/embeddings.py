"""Prompt self-similarity via local sentence-transformers (optional [embeddings]).

Model: all-MiniLM-L6-v2 (~25 MB, CPU). Ported from marking_pipeline. Imports are
lazy so the core install stays light; callers check `available()` first.
"""
from __future__ import annotations

from .config import EMBED_MODEL

_MODEL = None


def available() -> bool:
    try:
        import numpy  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(EMBED_MODEL)
    return _MODEL


def mean_self_similarity(prompts: list[str]) -> float | None:
    """Mean cosine similarity of consecutive prompts. None if <2 prompts."""
    if len([p for p in prompts if p.strip()]) < 2:
        return None
    import numpy as np

    model = _get_model()
    embs = model.encode(prompts, normalize_embeddings=True, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    sims = [float(np.dot(embs[i], embs[i + 1])) for i in range(len(embs) - 1)]
    return round(float(np.mean(sims)), 3)
