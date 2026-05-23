"""ConversationAnalyser: orchestrates parse → classify → sessionize → score.

One conversation per analyse() call. When turn timestamps are present the
conversation is split into idle-gap sub-sessions; the headline figures live in
`aggregate` (rolled up over all human turns). Everything degrades gracefully:
no LLM → analytics only; no embeddings → self-similarity null.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import analytics, embeddings, scoring, taxonomy
from .config import IDLE_GAP_MIN
from .llm import llm_available
from .models import (
    ConversationAnalysis,
    CriticalThinking,
    SessionAnalysis,
    TaxonomySignals,
    TurnLabel,
)
from .parsers import ParsedTurn
from .parsers import parse as parse_payload

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text", ""}


class ConversationAnalyser:
    def __init__(self, *, idle_gap_min: float = IDLE_GAP_MIN) -> None:
        self.idle_gap_min = idle_gap_min

    # -- public API ---------------------------------------------------------

    def analyse(
        self,
        source: "str | Path | list | dict",
        *,
        llm: bool = False,
        parse_mode: str = "auto",
        with_embeddings: bool = True,
        input_label: str | None = None,
    ) -> ConversationAnalysis:
        payload, label = self._load(source, input_label)
        allow_llm = bool(llm) and llm_available()

        parsed = parse_payload(payload, mode=parse_mode, allow_llm=allow_llm)
        turns = parsed.turns
        notes: list[str] = list(parsed.notes)

        llm_used = self._classify(turns, parsed.parse_mode, allow_llm, notes)

        if llm and not allow_llm:
            notes.append("llm_unavailable")
        if with_embeddings and not embeddings.available():
            notes.append("embeddings_unavailable")
        if turns and not any(t.timestamp for t in turns):
            notes.append("no timestamps")

        sessions_turns = self._split_sessions(turns)
        sessions = [
            self._build_session(s, i, with_embeddings) for i, s in enumerate(sessions_turns)
        ]
        aggregate = self._build_session(turns, -1, with_embeddings)

        return ConversationAnalysis(
            input=label,
            format_detected=parsed.format_name,
            parse_mode=parsed.parse_mode,
            llm_used=llm_used,
            notes=_dedup(notes),
            session_count=len(sessions),
            aggregate=aggregate,
            sessions=sessions,
        )

    # -- loading ------------------------------------------------------------

    def _load(self, source: Any, input_label: str | None) -> tuple[Any, str]:
        if isinstance(source, (list, dict)):
            return source, input_label or "<in-memory>"
        if isinstance(source, Path) or (isinstance(source, str) and _looks_like_path(source)):
            path = Path(source)
            label = input_label or str(path)
            if path.suffix.lower() == ".json":
                return json.loads(path.read_text(encoding="utf-8", errors="replace")), label
            return _extract_text(path), label
        return source, input_label or "<text>"

    # -- classification -----------------------------------------------------

    def _classify(
        self, turns: list[ParsedTurn], parse_mode: str, allow_llm: bool, notes: list[str]
    ) -> bool:
        """Attach taxonomy labels to human turns. Returns whether an LLM was used."""
        if parse_mode == "llm-segment":
            return True  # segmentation already labelled the turns in one call
        human = [t for t in turns if t.role == "human"]
        if not (allow_llm and human) or any(t.label for t in human):
            return False
        try:
            results = taxonomy.classify_turns([{"prompt": t.content} for t in human])
        except Exception:
            notes.append("classification_failed")
            return False
        for turn, res in zip(human, results):
            turn.label = res["label"]
            turn.rationale = res["rationale"]
        return True

    # -- sessionization -----------------------------------------------------

    def _split_sessions(self, turns: list[ParsedTurn]) -> list[list[ParsedTurn]]:
        if not turns:
            return [[]]
        if any(t.timestamp is None for t in turns) or len(turns) == 1:
            return [turns]
        sessions: list[list[ParsedTurn]] = []
        current = [turns[0]]
        for prev, cur in zip(turns, turns[1:]):
            gap_min = (cur.timestamp - prev.timestamp).total_seconds() / 60.0
            if gap_min >= self.idle_gap_min:
                sessions.append(current)
                current = [cur]
            else:
                current.append(cur)
        sessions.append(current)
        return sessions

    # -- per-session assembly ----------------------------------------------

    def _build_session(
        self, turns: list[ParsedTurn], index: int, with_embeddings: bool
    ) -> SessionAnalysis:
        human = [t for t in turns if t.role == "human"]
        metrics = analytics.compute_analytics(turns, with_embeddings=with_embeddings)
        signals, ct = _taxonomy_signals(human)

        turn_labels = [
            TurnLabel(
                index=i,
                role=t.role,  # type: ignore[arg-type]
                text_preview=t.content[:200],
                label=t.label if t.role == "human" else None,
                rationale=t.rationale if t.role == "human" else None,
            )
            for i, t in enumerate(turns)
        ]
        stamps = [t.timestamp for t in turns if t.timestamp is not None]
        return SessionAnalysis(
            session_index=index,
            started_at=min(stamps) if stamps else None,
            ended_at=max(stamps) if stamps else None,
            analytics=metrics,
            taxonomy=signals,
            critical_thinking=ct,
            turns=turn_labels,
        )


# -- module helpers ---------------------------------------------------------


def _taxonomy_signals(
    human_turns: list[ParsedTurn],
) -> tuple[TaxonomySignals | None, CriticalThinking | None]:
    labels = [t.label for t in human_turns if t.label]
    if not labels:
        return None, None
    ratios = taxonomy.compute_ratios(labels)
    chain = taxonomy.longest_engaged_chain(labels)
    pushback = analytics.pushback_count(human_turns)
    band = taxonomy.suggested_band(
        ratios=ratios,
        longest_chain=chain,
        pushback_hits=pushback,
        user_turn_count=len(human_turns),
        labels=labels,
    )
    signals = TaxonomySignals(
        label_counts=taxonomy.label_counts(labels),
        ratios=ratios,
        longest_engaged_chain=chain,
        band=band,
        filler_heavy=taxonomy.filler_heavy(ratios),
    )
    ct = scoring.critical_thinking_score(
        ratios=ratios, longest_chain=chain, pushback_hits=pushback, band=band
    )
    return signals, ct


def _looks_like_path(s: str) -> bool:
    if not s or len(s) > 4096 or "\n" in s:
        return False
    try:
        return Path(s).is_file()
    except OSError:
        return False


def _extract_text(path: Path) -> str:
    # Plain text we read ourselves (no dependency). Binary documents delegate to
    # the family's canonical extractor in document-analyser rather than
    # re-implementing extraction here.
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        from document_analyser import extract_text
    except ImportError as exc:  # pragma: no cover
        raise ValueError(
            f"{suffix} input needs text extraction. Install document-analyser "
            "(pip install -e ../document-analyser) or pre-extract the text and pass "
            "it (or a .txt/.json transcript) instead."
        ) from exc
    return extract_text(path)  # pragma: no cover


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
