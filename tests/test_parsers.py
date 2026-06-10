import json
from pathlib import Path

from conversation_analyser.parsers import anythingllm, parse, role_content

FIX = Path(__file__).parent / "fixtures"


def test_role_content_detect_and_parse():
    payload = json.loads((FIX / "role_content.json").read_text())
    assert role_content.detect(payload)
    res = parse(payload, allow_llm=False)
    assert res.parse_mode == "structured"
    assert res.format_name == "role_content"
    assert res.turns[0].role == "human"
    assert res.turns[1].role == "assistant"
    assert res.turns[-1].content == "Thanks, that helps."


def test_role_content_ignores_system_and_tool():
    payload = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "result"},
    ]
    res = parse(payload, allow_llm=False)
    assert [t.role for t in res.turns] == ["human"]


def test_role_content_handles_anthropic_blocks():
    payload = [{"role": "user", "content": [{"type": "text", "text": "block one"}]}]
    res = parse(payload, allow_llm=False)
    assert res.turns[0].content == "block one"


def test_anythingllm_unwraps_response_json_and_timestamp():
    payload = json.loads((FIX / "anythingllm.json").read_text())
    assert anythingllm.detect(payload)
    res = parse(payload, allow_llm=False)
    assert res.format_name == "anythingllm"
    assert res.turns[0].role == "human"
    assert res.turns[1].role == "assistant"
    assert "SECI" in res.turns[1].content
    assert res.turns[0].timestamp is not None


def test_markers_parse_flat_text():
    text = (FIX / "transcript.txt").read_text()
    res = parse(text, allow_llm=False)
    assert res.parse_mode == "heuristic"
    human = [t for t in res.turns if t.role == "human"]
    assert len(human) == 4  # two "User:" + two "Me:"


def test_dict_wrapper_unwrapped():
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    res = parse(payload, allow_llm=False)
    assert res.format_name == "role_content"


def test_unsegmented_fallback_without_llm():
    res = parse("just a blob of text with no markers", allow_llm=False)
    assert res.parse_mode == "unsegmented"
    assert "could_not_segment" in res.notes
    assert len(res.turns) == 1
    assert res.turns[0].role == "human"


# --- LLM-segment chunking (long unstructured transcripts) --------------------

from conversation_analyser import config as ca_config  # noqa: E402
from conversation_analyser import taxonomy  # noqa: E402
from conversation_analyser.parsers.llm_segment import _chunk_text  # noqa: E402

_PARAS = "\n\n".join(
    f"paragraph number {i} here" for i in range(6)
)  # 6 paras, ~23 chars each


def _stub_classifier(monkeypatch):
    """Stub the segmenter: one labelled turn per call, echoing the chunk."""
    calls: list[str] = []

    def fake(text, *, max_chars=40000):
        calls.append(text)
        return [{"prompt": text.strip()[:20], "label": "EX", "rationale": "r"}]

    monkeypatch.setattr(taxonomy, "classify_transcript", fake)
    return calls


def test_llm_segment_short_text_single_call(monkeypatch):
    calls = _stub_classifier(monkeypatch)
    res = parse("a short unstructured blob with no markers", allow_llm=True)
    assert res.parse_mode == "llm-segment"
    assert len(calls) == 1  # no chunking
    assert "llm_segment_preview_only" in res.notes
    assert not any("chunked" in n for n in res.notes)


def test_llm_segment_long_text_is_chunked_full_coverage(monkeypatch):
    calls = _stub_classifier(monkeypatch)
    monkeypatch.setattr(ca_config, "SEGMENT_CHUNK_CHARS", 30)
    res = parse(_PARAS, allow_llm=True)
    assert res.parse_mode == "llm-segment"
    assert len(calls) == 6  # one chunk per paragraph
    assert len(res.turns) == 6  # all chunks' turns concatenated, nothing dropped
    assert any(n.startswith("chunked_full_transcript") for n in res.notes)


def test_llm_segment_chunk_cap_emits_note(monkeypatch):
    calls = _stub_classifier(monkeypatch)
    monkeypatch.setattr(ca_config, "SEGMENT_CHUNK_CHARS", 30)
    monkeypatch.setattr(ca_config, "SEGMENT_MAX_CHUNKS", 2)
    res = parse(_PARAS, allow_llm=True)
    assert len(calls) == 2  # only the first two chunks analysed
    assert any(n.startswith("segment_chunk_cap_hit") for n in res.notes)


def test_llm_segment_chunk_cap_zero_means_unlimited(monkeypatch):
    calls = _stub_classifier(monkeypatch)
    monkeypatch.setattr(ca_config, "SEGMENT_CHUNK_CHARS", 30)
    monkeypatch.setattr(ca_config, "SEGMENT_MAX_CHUNKS", 0)
    res = parse(_PARAS, allow_llm=True)
    assert len(calls) == 6  # cap lifted
    assert not any("segment_chunk_cap_hit" in n for n in res.notes)


def test_chunk_text_hard_splits_oversized_paragraph():
    assert _chunk_text("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]
