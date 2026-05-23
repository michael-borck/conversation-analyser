"""Domain-neutral analytics tier (design spec §7).

All metrics derive deterministically from a session's turns. Optional/heavy
metrics (self-similarity) degrade to None when their dependency is absent; the
pipeline records a single note. Sentiment/readability/typo use the light core
deps and are wrapped defensively.
"""
from __future__ import annotations

import re
from collections import Counter
from statistics import fmean

from . import embeddings
from .models import AnalyticsMetrics
from .parsers.base import ParsedTurn

# Pushback cue regex, ported verbatim from marking_pipeline/transcript.py.
_PUSHBACK_RE = re.compile(
    r"\b(no,|actually|but\b|wait\b|are you sure|that's wrong|incorrect|"
    r"i disagree|not right|you're wrong|why\b|why is|why does)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\w+")


def _words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def pushback_count(human_turns: list[ParsedTurn]) -> int:
    return sum(len(_PUSHBACK_RE.findall(t.content)) for t in human_turns)


def _question_ratio(prompts: list[str]) -> float:
    if not prompts:
        return 0.0
    asked = sum(1 for p in prompts if "?" in p)
    return round(asked / len(prompts), 2)


def _flesch(text: str) -> float | None:
    if not text.strip():
        return None
    try:
        import textstat

        return round(float(textstat.flesch_reading_ease(text)), 1)
    except Exception:
        return None


def _typo_rate(prompts: list[str]) -> float | None:
    try:
        from spellchecker import SpellChecker
    except Exception:
        return None
    checker = SpellChecker()
    rates: list[float] = []
    for p in prompts:
        words = [w.lower() for w in _WORD_RE.findall(p) if w.isalpha()]
        if not words:
            continue
        unknown = checker.unknown(words)
        rates.append(len(unknown) / len(words))
    return round(fmean(rates), 3) if rates else None


def _sentiment(prompts: list[str]) -> tuple[float | None, float | None, float | None]:
    if not prompts:
        return None, None, None
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        return None, None, None
    analyzer = SentimentIntensityAnalyzer()
    start = round(analyzer.polarity_scores(prompts[0])["compound"], 3)
    end = round(analyzer.polarity_scores(prompts[-1])["compound"], 3)
    return start, end, round(end - start, 3)


def _temporal(turns: list[ParsedTurn]) -> tuple[float | None, int | None, int | None]:
    stamps = [t.timestamp for t in turns if t.timestamp is not None]
    if len(stamps) < 2:
        return None, None, None
    duration = (max(stamps) - min(stamps)).total_seconds() / 60.0
    hour_mode = Counter(s.hour for s in stamps).most_common(1)[0][0]
    weekday_mode = Counter(s.weekday() for s in stamps).most_common(1)[0][0]
    return round(duration, 1), hour_mode, weekday_mode


def compute_analytics(turns: list[ParsedTurn], *, with_embeddings: bool = True) -> AnalyticsMetrics:
    human = [t for t in turns if t.role == "human"]
    assistant = [t for t in turns if t.role == "assistant"]
    prompts = [t.content for t in human]
    responses = [t.content for t in assistant]

    prompt_words = [_words(p) for p in prompts]
    response_words = [_words(r) for r in responses]

    similarity = None
    if with_embeddings and embeddings.available():
        similarity = embeddings.mean_self_similarity(prompts)

    sent_start, sent_end, sent_delta = _sentiment(prompts)
    duration_min, hour_mode, weekday_mode = _temporal(turns)

    return AnalyticsMetrics(
        turn_count=len(turns),
        human_turn_count=len(human),
        assistant_turn_count=len(assistant),
        total_words=sum(prompt_words) + sum(response_words),
        mean_prompt_len=round(fmean(prompt_words), 2) if prompt_words else 0.0,
        max_prompt_len=max(prompt_words) if prompt_words else 0,
        mean_response_len=round(fmean(response_words), 2) if response_words else 0.0,
        question_ratio=_question_ratio(prompts),
        pushback_count=pushback_count(human),
        prompt_self_similarity=similarity,
        flesch_reading_ease=_flesch("\n\n".join(prompts)),
        mean_typo_rate=_typo_rate(prompts),
        sentiment_start=sent_start,
        sentiment_end=sent_end,
        sentiment_delta=sent_delta,
        duration_min=duration_min,
        hour_of_day_mode=hour_mode,
        weekday_mode=weekday_mode,
    )
