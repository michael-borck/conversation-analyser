"""FastAPI app for conversation-analyser, built on the lens-contract scaffolding.

Module-level `app` so the CLI can launch it with
`uvicorn.run("conversation_analyser.api:app", ...)` and tests can drive it with
fastapi.testclient.TestClient.

`GET /health` and `GET /manifest` come from lens-contract. `POST /analyse` stays
bespoke here because it takes an extra `llm` form field and a typed response_model.
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from lens_contract import add_contract_routes, add_cors, upload_tempfile

from .manifest import MANIFEST
from .models import ConversationAnalysis
from .pipeline import ConversationAnalyser

app = FastAPI(title=MANIFEST["name"], version=MANIFEST["version"])
add_contract_routes(app, MANIFEST)
# CORS — env-driven: CONVERSATION_ANALYSER_MODE=desktop (Electron) or
# CONVERSATION_ANALYSER_ALLOWED_ORIGINS. Lets any consumer front it from a browser.
add_cors(app, env_prefix="CONVERSATION_ANALYSER")

_analyser = ConversationAnalyser()


@app.post("/analyse", response_model=ConversationAnalysis)
async def analyse(
    file: UploadFile = File(...),
    llm: bool = Form(False),
) -> ConversationAnalysis:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Empty file")

    with upload_tempfile(content, file.filename) as tmp_path:
        try:
            return _analyser.analyse(
                tmp_path, llm=llm, input_label=file.filename or "<upload>"
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e)) from e
