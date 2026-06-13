"""Pydantic result models for conversation-analyser (design spec §8).

`ConversationAnalysis` is the public Result. It holds an `aggregate`
SessionAnalysis (rolled up over all human turns — the headline) plus one
SessionAnalysis per idle-gap sub-session.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TurnLabel(BaseModel):
    index: int
    role: Literal["human", "assistant"]
    text_preview: str
    label: str | None = None  # NQ|FU|CH|EX|DG|AC|MT (None for assistant / LLM unavailable)
    rationale: str | None = None


class AnalyticsMetrics(BaseModel):
    turn_count: int
    human_turn_count: int
    assistant_turn_count: int
    total_words: int
    mean_prompt_len: float
    max_prompt_len: int
    mean_response_len: float
    question_ratio: float
    pushback_count: int
    prompt_self_similarity: float | None = None
    flesch_reading_ease: float | None = None
    mean_typo_rate: float | None = None
    sentiment_start: float | None = None
    sentiment_end: float | None = None
    sentiment_delta: float | None = None
    duration_min: float | None = None
    hour_of_day_mode: int | None = None
    weekday_mode: int | None = None


class TaxonomySignals(BaseModel):
    label_counts: dict[str, int]
    ratios: dict[str, float]
    longest_engaged_chain: int
    band: str
    filler_heavy: bool


class CriticalThinking(BaseModel):
    score: float  # 0-100 composite
    band: str
    components: dict[str, float]


class SessionAnalysis(BaseModel):
    session_index: int
    started_at: datetime | None = None
    ended_at: datetime | None = None
    analytics: AnalyticsMetrics
    taxonomy: TaxonomySignals | None = None
    critical_thinking: CriticalThinking | None = None
    turns: list[TurnLabel] = Field(default_factory=list)


class ConversationAnalysis(BaseModel):
    input: str
    format_detected: str
    parse_mode: str  # structured | heuristic | llm-segment
    llm_used: bool
    notes: list[str] = Field(default_factory=list)
    session_count: int
    aggregate: SessionAnalysis
    sessions: list[SessionAnalysis] = Field(default_factory=list)
    # Pooled, L2-normalised transcript vector from lens-embed (pinned
    # all-MiniLM-L6-v2). Comparable across members; None unless [embeddings]
    # installed or when with_embeddings=False.
    embedding: list[float] | None = None
