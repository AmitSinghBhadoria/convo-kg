"""Thin FastAPI surface for the Atyx demo. Adapts existing functions (qa.answer,
graph.read_graph, file reads) — NO business logic. Serves the wired dc-app from
frontend/. The pre-built graph stays authoritative; clip-specificity is config only.
"""
import json
import os
import uuid
from pathlib import Path

import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import src.audioprep as audioprep
from src.config import load_config
from src.contracts import Transcript
from src.graph import connect, read_graph
from src.qa import answer as qa_answer

app = FastAPI(title="Atyx Convo-KG")
CFG = load_config()
DB = os.environ.get("NEO4J_DATABASE", "neo4j")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class AskBody(BaseModel):
    question: str


@app.get("/api/graph")
def api_graph() -> dict:
    drv = connect()
    try:
        return read_graph(drv, DB)
    finally:
        drv.close()


@app.post("/api/ask")
def api_ask(body: AskBody) -> dict:
    return qa_answer(body.question).model_dump()   # QAResult verbatim; found=False is a 200


@app.get("/api/experiment")
def api_experiment() -> dict:
    p = Path(CFG.paths.ground_truth) / "snr_results.json"
    if not p.exists():
        raise HTTPException(status_code=404,
                            detail="snr_results.json not found — run `python -m src.evaluate`")
    return json.loads(p.read_text())


_RUNS: dict[str, str] = {}   # run_id -> clip (demo pointer)


@app.post("/api/run")
def api_run() -> dict:
    run_id = uuid.uuid4().hex[:12]
    _RUNS[run_id] = CFG.demo.clip
    return {"run_id": run_id}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/run/{run_id}/stream")
def api_run_stream(run_id: str, replay: int = 0):
    clip = _RUNS.get(run_id, CFG.demo.clip)
    work = Path(CFG.paths.work)

    def gen():
        try:
            tr = Transcript.model_validate_json((work / f"{clip}.transcript.json").read_text())
            stages = [("Speech enhancement", "DeepFilterNet"), ("Diarization", "pyannote 3.x"),
                      ("Transcribe · Hinglish→EN", "Whisper large-v3")]
            for i, (label, sub) in enumerate(stages):
                yield _sse("stage", {"index": i, "label": label, "sub": sub,
                                     "status": "done", "replayed": True})
            for u in tr.utterances:
                yield _sse("transcript_line", {"speaker": u.speaker, "t": round(u.start, 1),
                                               "text": u.text})
            # Extract stage: live display-only proof (or pure replay if ?replay=1).
            yield _sse("stage", {"index": 3, "label": "Fact extraction", "sub": "Qwen 9B · live",
                                 "status": "active", "replayed": False})
            if replay:
                facts = json.loads((work / f"{clip}.facts.json").read_text()).get("facts", [])
                for f in facts:
                    yield _sse("fact", {"text": f.get("statement", "")})
            else:
                from src.extract import extract
                fs = extract(clip)                       # live; display-only, NOT upserted
                for f in fs.facts:
                    yield _sse("fact", {"text": f.statement})
            yield _sse("stage", {"index": 4, "label": "Graph build", "sub": "Neo4j · authoritative",
                                 "status": "done", "replayed": False})
            yield _sse("done", {"clip": clip})
        except Exception as e:                            # transport/processing error -> honest event
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict:
    """Accept a multipart audio upload, validate duration ≤ 10 min, prep to 16 kHz mono.

    Returns ``{"clip_id": str}`` on success; HTTP 400 on non-audio or > 600 s.
    No Neo4j write; no src.qa import.
    """
    uploads_dir = Path(CFG.paths.uploads)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    clip_id = "upload_" + uuid.uuid4().hex[:10]
    out_path = uploads_dir / f"{clip_id}.wav"

    # Write upload to a temp file so ffprobe/ffmpeg can open it by path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload").suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        try:
            duration = audioprep.probe_duration(tmp_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="could not read audio — upload a valid audio file")

        if duration > 600:
            raise HTTPException(status_code=400, detail="clip too long — 10 min max")

        try:
            audioprep.to_16k_mono(tmp_path, str(out_path))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=f"audio re-encode failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"clip_id": clip_id}


# Static mount LAST so /api/* routes win.
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
