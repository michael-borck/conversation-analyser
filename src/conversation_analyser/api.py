"""FastAPI app for conversation-analyser, mirroring the lens family contract.

Module-level `app` so the CLI can launch it with
`uvicorn.run("conversation_analyser.api:app", ...)` and tests can drive it with
fastapi.testclient.TestClient. Endpoints: GET /health, POST /analyse (file upload).
"""
from __future__ import annotations

import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .manifest import MANIFEST
from .models import ConversationAnalysis
from .pipeline import ConversationAnalyser

_start_time = time.time()

app = FastAPI(title="conversation-analyser", version=version("conversation-analyser"))

_analyser = ConversationAnalyser()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "uptime": round(time.time() - _start_time, 1),
        "version": version("conversation-analyser"),
    }


@app.get("/manifest")
def manifest() -> dict:
    return MANIFEST


@app.post("/analyse", response_model=ConversationAnalysis)
async def analyse(
    file: UploadFile = File(...),
    llm: bool = Form(False),
) -> ConversationAnalysis:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Empty file")

    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(content)

    try:
        return _analyser.analyse(tmp_path, llm=llm, input_label=file.filename or "<upload>")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)
