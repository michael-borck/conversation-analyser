from pathlib import Path

from fastapi.testclient import TestClient

from conversation_analyser.api import app

FIX = Path(__file__).parent / "fixtures"
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_analyse_upload_role_content():
    with open(FIX / "role_content.json", "rb") as f:
        r = client.post("/analyse", files={"file": ("chat.json", f, "application/json")})
    assert r.status_code == 200
    data = r.json()
    assert data["format_detected"] == "role_content"
    assert data["aggregate"]["analytics"]["human_turn_count"] == 4
    assert data["llm_used"] is False


def test_analyse_empty_file_422():
    r = client.post("/analyse", files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code == 422
