"""Composite 0-100 critical-thinking score, derived from taxonomy labels (spec §5).

No extra LLM call. The reported `band` is the authoritative suggested_band, so the
headline score and the band can never contradict each other; the score is a
continuous companion to that ordinal band.
"""
from __future__ import annotations

from .config import CHAIN_CAP, CT_SCORE_WEIGHTS, PUSHBACK_CAP
from .models import CriticalThinking


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def critical_thinking_score(
    *,
    ratios: dict[str, float],
    longest_chain: int,
    pushback_hits: int,
    band: str,
) -> CriticalThinking:
    w = CT_SCORE_WEIGHTS
    ratio = ratios.get("critical_thinking", 0.0)
    deleg = ratios.get("delegation", 0.0)
    filler = ratios.get("filler", 0.0)
    chain = _clamp(longest_chain / CHAIN_CAP) if CHAIN_CAP else 0.0
    pushback = _clamp(pushback_hits / PUSHBACK_CAP) if PUSHBACK_CAP else 0.0

    raw = (
        w["ratio"] * ratio
        + w["chain"] * chain
        + w["pushback"] * pushback
        - w["deleg"] * deleg
        - w["filler"] * filler
    )
    score = round(100.0 * _clamp(raw), 1)

    components = {
        "ratio": round(w["ratio"] * ratio, 3),
        "chain": round(w["chain"] * chain, 3),
        "pushback": round(w["pushback"] * pushback, 3),
        "deleg_penalty": round(-w["deleg"] * deleg, 3),
        "filler_penalty": round(-w["filler"] * filler, 3),
    }
    return CriticalThinking(score=score, band=band, components=components)
