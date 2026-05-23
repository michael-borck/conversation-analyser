"""Per-prompt taxonomy classifier and derived engagement metrics.

Ported from marking_pipeline/taxonomy.py (then forked). Each human turn is
assigned exactly one of seven mutually-exclusive labels by a Haiku call.
Derived ratios, longest_engaged_chain, and the suggested_band heuristic are
computed deterministically from the label sequence.
"""
from __future__ import annotations

import json
from typing import Any

from .config import CLASSIFIER_MODEL, ENGAGEMENT_BAND_THRESHOLDS, FILLER_HEAVY_RATIO
from .llm import LLMFailure, call_json

LABELS = ("NQ", "FU", "CH", "EX", "DG", "AC", "MT")


SYSTEM_PROMPT = """\
You are classifying student prompts in a transcript between a university student
and an AI assistant. For each user turn, assign exactly ONE label from this set:

NQ (New Query): opens a new topic; does not build on the prior turn.
FU (Follow-up): asks for clarification or elaboration on the AI's prior response.
CH (Challenge): pushes back, asks why, requests alternatives, disagrees, tests the AI.
EX (Extension): extends the AI's response in a new direction; applies to a specific
   organisational context; compares; synthesises.
DG (Delegation): task hand-off with no engagement with the prior response.
AC (Acknowledgement): pure confirmation or thanks; no content.
MT (Meta): about the conversation itself ("let's go back to...", "summarise so far").

Tiebreaker rules:
- CH vs EX: if the turn extends in a new direction, label EX; if pushback is the
  primary move, label CH.
- FU vs CH: if the student is testing the AI's claim, label CH; if asking for
  elaboration, label FU.
- AC vs MT: if it's purely "thanks/good", label AC; if it discusses the conversation,
  label MT.

Respond with ONLY a JSON object of this shape:
{"labels": [{"index": 0, "label": "NQ", "rationale": "one short sentence"}, ...]}
No markdown fences. One entry per user turn, in order.
"""


SEGMENT_SYSTEM_PROMPT = """\
You are given the raw text of a student's saved conversation with an AI assistant,
exported in an arbitrary format. It may use labels like 'User:'/'Assistant:',
'Me:'/'ChatGPT:', 'You said:'/'ChatGPT said:', 'Prompt:'/'Response:', numbered
prompts, or no labels at all. Identify each STUDENT (human) message in order and
ignore the AI responses. For each student message assign exactly ONE label:

NQ (New Query): opens a new topic; does not build on the prior turn.
FU (Follow-up): asks for clarification or elaboration on the AI's prior response.
CH (Challenge): pushes back, asks why, requests alternatives, disagrees, tests the AI.
EX (Extension): extends the AI's response in a new direction; applies to a specific
   organisational context; compares; synthesises.
DG (Delegation): task hand-off with no engagement with the prior response.
AC (Acknowledgement): pure confirmation or thanks; no content.
MT (Meta): about the conversation itself ("let's go back to...", "summarise so far").

Tiebreaker rules:
- CH vs EX: if the turn extends in a new direction, label EX; if pushback is the
  primary move, label CH.
- FU vs CH: if the student is testing the AI's claim, label CH; if asking for
  elaboration, label FU.
- AC vs MT: if it's purely "thanks/good", label AC; if it discusses the conversation,
  label MT.

Respond with ONLY a JSON object of this shape:
{"turns": [{"prompt": "<first ~100 chars of the student message>", "label": "CH", "rationale": "one short sentence"}, ...]}
No markdown fences. One entry per student message, in order. If you find no student
messages, return {"turns": []}.
"""


def classify_turns(user_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify each user turn. Returns a list of {label, rationale} in order.

    Pads/repairs to length len(user_turns); invalid labels collapse to NQ. Raises
    nothing — on LLM failure every turn is returned as NQ with an empty rationale.
    """
    if not user_turns:
        return []

    user_msg = json.dumps({"turns": user_turns}, ensure_ascii=False)
    try:
        result = _call_classifier(system=SYSTEM_PROMPT, user=user_msg, model=CLASSIFIER_MODEL)
    except LLMFailure:
        return [{"label": "NQ", "rationale": ""} for _ in user_turns]

    by_index = {
        item.get("index"): (item.get("label"), item.get("rationale"))
        for item in result.get("labels", [])
    }
    out: list[dict[str, Any]] = []
    for i in range(len(user_turns)):
        label, rationale = by_index.get(i, (None, None))
        if label not in LABELS:
            label = "NQ"
        out.append({"label": label, "rationale": (rationale or "")[:300]})
    return out


def classify_transcript(raw_text: str, *, max_chars: int = 40000) -> list[dict[str, Any]]:
    """Segment and classify student prompts directly from raw transcript text.

    Returns a list of {prompt, label, rationale}. Empty list for empty text or on
    LLM failure. Invalid labels collapse to NQ.
    """
    text = (raw_text or "").strip()
    if not text:
        return []
    try:
        result = _call_classifier(
            system=SEGMENT_SYSTEM_PROMPT, user=text[:max_chars], model=CLASSIFIER_MODEL
        )
    except LLMFailure:
        return []
    out: list[dict[str, Any]] = []
    for item in result.get("turns", []):
        label = item.get("label")
        if label not in LABELS:
            label = "NQ"
        out.append(
            {
                "prompt": (item.get("prompt") or "")[:200],
                "label": label,
                "rationale": (item.get("rationale") or "")[:300],
            }
        )
    return out


def _call_classifier(*, system: str, user: str, model: str) -> dict[str, Any]:
    """Indirection so tests can monkeypatch the LLM call."""
    return call_json(system=system, user=user, model=model, max_tokens=8000)


def compute_ratios(labels: list[str]) -> dict[str, float]:
    """Five ratios computed from the label sequence, rounded to two decimals."""
    n = max(len(labels), 1)
    counts = {code: labels.count(code) for code in LABELS}
    return {
        "critical_thinking": round((counts["CH"] + counts["EX"]) / n, 2),
        "delegation": round(counts["DG"] / n, 2),
        "filler": round((counts["AC"] + counts["MT"]) / n, 2),
        "extension": round(counts["EX"] / n, 2),
        "challenge": round(counts["CH"] / n, 2),
    }


def label_counts(labels: list[str]) -> dict[str, int]:
    return {code: labels.count(code) for code in LABELS}


def longest_engaged_chain(labels: list[str]) -> int:
    """Longest run of consecutive turns labelled CH, EX, or FU."""
    best = 0
    current = 0
    for code in labels:
        if code in ("CH", "EX", "FU"):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def suggested_band(
    *,
    ratios: dict[str, float],
    longest_chain: int,
    pushback_hits: int,
    user_turn_count: int,
    labels: list[str],
) -> str:
    t = ENGAGEMENT_BAND_THRESHOLDS

    # One-Shot: explicit short DG-only conversations.
    if (
        user_turn_count <= t["One-Shot"]["max_user_turns"]
        and labels
        and all(code == "DG" for code in labels)
    ):
        return "One-Shot"

    if (
        ratios["delegation"] >= t["Delegator"]["delegation_min"]
        or ratios["critical_thinking"] < t["Delegator"]["critical_thinking_max"]
    ):
        return "Delegator"

    if (
        ratios["critical_thinking"] >= t["Critical"]["critical_thinking_min"]
        and longest_chain >= t["Critical"]["longest_engaged_chain_min"]
        and pushback_hits >= t["Critical"]["pushback_regex_hits_min"]
    ):
        return "Critical"

    if ratios["critical_thinking"] >= t["Iterative"]["critical_thinking_min"]:
        return "Iterative"

    return "Directed"


def filler_heavy(ratios: dict[str, float]) -> bool:
    return ratios["filler"] >= FILLER_HEAVY_RATIO
