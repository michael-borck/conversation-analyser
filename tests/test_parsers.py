import json
from pathlib import Path

from conversation_analyser.parsers import anythingllm, markers, parse, role_content

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
