import json
import sys
from pathlib import Path

import pytest

from conversation_analyser import cli

FIX = Path(__file__).parent / "fixtures"


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["conversation-analyser", *argv])
    cli.main()


def test_cli_json_emits_valid_json(monkeypatch, capsys):
    _run(monkeypatch, [str(FIX / "role_content.json"), "--json", "--no-embeddings"])
    data = json.loads(capsys.readouterr().out)
    assert data["format_detected"] == "role_content"
    assert data["aggregate"]["analytics"]["human_turn_count"] == 4
    assert data["llm_used"] is False


def test_cli_human_default(monkeypatch, capsys):
    _run(monkeypatch, [str(FIX / "transcript.txt"), "--no-embeddings"])
    out = capsys.readouterr().out
    assert "Input:" in out
    assert "Critical-thinking tier skipped" in out  # no --llm flag


def test_cli_missing_file_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["conversation-analyser", "/no/such/file.json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_cli_requires_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["conversation-analyser"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2  # argparse: missing required positional
