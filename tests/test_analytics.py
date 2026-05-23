from datetime import datetime, timezone

from conversation_analyser.analytics import compute_analytics, pushback_count
from conversation_analyser.parsers.base import ParsedTurn


def _turns():
    return [
        ParsedTurn("human", "Why is this wrong? I disagree with that."),
        ParsedTurn("assistant", "Here is a fairly long response with several words indeed."),
        ParsedTurn("human", "Write me an essay about it."),
    ]


def test_counts_and_pushback():
    m = compute_analytics(_turns(), with_embeddings=False)
    assert m.turn_count == 3
    assert m.human_turn_count == 2
    assert m.assistant_turn_count == 1
    assert m.pushback_count >= 2  # "why", "i disagree"
    assert m.prompt_self_similarity is None  # embeddings off
    assert m.question_ratio == 0.5


def test_pushback_count_helper():
    assert pushback_count([ParsedTurn("human", "actually, are you sure? that's wrong")]) >= 2


def test_temporal_present_with_timestamps():
    t0 = datetime(2024, 5, 18, 9, 0, tzinfo=timezone.utc)
    turns = [ParsedTurn("human", "hi", t0), ParsedTurn("assistant", "yo", t0)]
    m = compute_analytics(turns, with_embeddings=False)
    assert m.duration_min == 0.0
    assert m.hour_of_day_mode == 9


def test_temporal_absent_without_timestamps():
    m = compute_analytics(_turns(), with_embeddings=False)
    assert m.duration_min is None
    assert m.hour_of_day_mode is None
