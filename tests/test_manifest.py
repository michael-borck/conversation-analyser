import json
import sys

from fastapi.testclient import TestClient

from conversation_analyser import MANIFEST, cli
from conversation_analyser.api import app

client = TestClient(app)


def test_manifest_shape():
    assert MANIFEST["name"] == "conversation-analyser"
    assert MANIFEST["auto_routable"] is False  # explicit-only content lens
    assert MANIFEST["extensions"] == []  # not auto-routed by extension
    assert {"name", "version", "role", "accepts", "extensions", "auto_routable", "produces"} <= set(
        MANIFEST
    )


def test_manifest_endpoint():
    r = client.get("/manifest")
    assert r.status_code == 200
    assert r.json()["name"] == "conversation-analyser"


def test_manifest_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["conversation-analyser", "manifest"])
    cli.main()
    data = json.loads(capsys.readouterr().out)
    assert data["auto_routable"] is False
