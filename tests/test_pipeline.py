from pathlib import Path

from conversation_analyser import pipeline
from conversation_analyser.pipeline import ConversationAnalyser

FIX = Path(__file__).parent / "fixtures"


def _mock_llm(monkeypatch):
    monkeypatch.setattr(pipeline, "llm_available", lambda: True)
    monkeypatch.setattr(pipeline.embeddings, "available", lambda: False)

    def fake_classify(user_turns):
        seq = ["NQ", "CH", "EX", "FU", "DG", "AC", "MT"]
        return [{"label": seq[i % len(seq)], "rationale": "x"} for i in range(len(user_turns))]

    monkeypatch.setattr(pipeline.taxonomy, "classify_turns", fake_classify)


def test_analytics_only_by_default(monkeypatch):
    # llm is opt-in; the default run is analytics-only with no llm note.
    monkeypatch.setattr(pipeline.embeddings, "available", lambda: False)
    res = ConversationAnalyser().analyse(FIX / "role_content.json")
    assert res.llm_used is False
    assert "llm_unavailable" not in res.notes
    assert res.aggregate.taxonomy is None
    assert res.aggregate.critical_thinking is None
    assert res.aggregate.analytics.human_turn_count == 4


def test_llm_requested_but_unavailable(monkeypatch):
    monkeypatch.setattr(pipeline, "llm_available", lambda: False)
    monkeypatch.setattr(pipeline.embeddings, "available", lambda: False)
    res = ConversationAnalyser().analyse(FIX / "role_content.json", llm=True)
    assert res.llm_used is False
    assert "llm_unavailable" in res.notes
    assert res.aggregate.taxonomy is None


def test_with_llm_assigns_labels_and_score(monkeypatch):
    _mock_llm(monkeypatch)
    res = ConversationAnalyser().analyse(FIX / "role_content.json", llm=True)
    assert res.llm_used is True
    assert res.aggregate.taxonomy is not None
    assert res.aggregate.critical_thinking is not None
    human_labels = [t.label for t in res.aggregate.turns if t.role == "human"]
    assert all(label is not None for label in human_labels)
    assert sum(res.aggregate.taxonomy.label_counts.values()) == 4


def test_idle_gap_splits_sessions(monkeypatch):
    _mock_llm(monkeypatch)
    payload = [
        {"role": "user", "content": "first topic question?", "timestamp": "2024-05-18T09:00:00Z"},
        {"role": "assistant", "content": "answer one", "timestamp": "2024-05-18T09:00:30Z"},
        {"role": "user", "content": "much later question?", "timestamp": "2024-05-18T11:00:00Z"},
        {"role": "assistant", "content": "answer two", "timestamp": "2024-05-18T11:00:30Z"},
    ]
    res = ConversationAnalyser().analyse(payload)
    assert res.session_count == 2
    assert "no timestamps" not in res.notes


def test_no_split_without_timestamps(monkeypatch):
    _mock_llm(monkeypatch)
    res = ConversationAnalyser().analyse(FIX / "transcript.txt")
    assert res.parse_mode == "heuristic"
    assert res.session_count == 1
    assert "no timestamps" in res.notes
    assert res.aggregate.analytics.human_turn_count == 4
