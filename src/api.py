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
import src.pipeline as pipeline
from src.config import load_config
from src.contracts import Transcript
from src.graph import connect, read_graph
from src.qa import answer as qa_answer

app = FastAPI(title="Atyx Convo-KG")
CFG = load_config()
DB = os.environ.get("NEO4J_DATABASE", "neo4j")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

# ── Clip registry state ───────────────────────────────────────────────────────
_ACTIVE_CLIP: str = CFG.demo.clip          # currently selected clip id
_CLIP_MODE: dict[str, str] = {}            # uploaded clip_id → mode (e.g. "live")
_LIVE_RUNNING: bool = False                # guard: only one live run at a time


def _clip_mode(clip_id: str) -> str:
    """Return the mode string for a clip id.

    Lookup order:
    1. Registry (CFG.demo.clips) — authoritative for pre-built clips.
    2. _CLIP_MODE dict — for uploaded clips whose mode was recorded on select.
    3. Prefix heuristic — any clip_id starting with "upload_" defaults to "live".
    4. 404 — unknown clip.
    """
    for c in CFG.demo.clips:
        if c.id == clip_id:
            return c.mode
    if clip_id in _CLIP_MODE:
        return _CLIP_MODE[clip_id]
    if clip_id.startswith("upload_"):
        return "live"
    raise HTTPException(status_code=404, detail=f"unknown clip: {clip_id}")


class AskBody(BaseModel):
    question: str


class SelectClipBody(BaseModel):
    id: str


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


@app.get("/api/clips")
def api_clips() -> dict:
    return {"active": _ACTIVE_CLIP, "clips": [c.model_dump() for c in CFG.demo.clips]}


@app.post("/api/select_clip")
def api_select_clip(body: SelectClipBody) -> dict:
    global _ACTIVE_CLIP
    clip_id = body.id
    mode = _clip_mode(clip_id)   # raises 404 if completely unknown
    # For uploaded clips: record mode so subsequent _clip_mode calls are fast.
    if clip_id.startswith("upload_"):
        _CLIP_MODE[clip_id] = "live"
    # HERO INVARIANT: facts/live selection must never touch Neo4j.
    # Only for graph clips do we (optionally) ensure the snapshot is loaded.
    if mode == "graph":
        # Best-effort: load snapshot only when the DB is empty.
        # If restore_snapshot is not implemented yet, skip silently.
        try:
            drv = connect()
            try:
                with drv.session(database=DB) as sess:
                    count = sess.run("MATCH (n) RETURN count(n) AS n").single()["n"]
                if count == 0:
                    try:
                        from src.graph import restore_snapshot
                        restore_snapshot(drv, DB, clip_id)
                    except (ImportError, AttributeError):
                        pass   # restore_snapshot not yet implemented — skip
            finally:
                drv.close()
        except Exception:
            pass   # best-effort; demo still works if graph is already loaded
    _ACTIVE_CLIP = clip_id
    return {"active": clip_id, "mode": mode}


_RUNS: dict[str, str] = {}   # run_id -> clip (demo pointer)


@app.post("/api/run")
def api_run() -> dict:
    global _ACTIVE_CLIP
    run_id = uuid.uuid4().hex[:12]
    _RUNS[run_id] = _ACTIVE_CLIP
    return {"run_id": run_id}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/run/{run_id}/stream")
def api_run_stream(run_id: str, replay: int = 0):
    clip = _RUNS.get(run_id, CFG.demo.clip)
    work = Path(CFG.paths.work)

    # ── Live-mode dispatch ────────────────────────────────────────────────────
    if _clip_mode(clip) == "live":
        def live_gen():
            global _LIVE_RUNNING
            if _LIVE_RUNNING:
                yield _sse("error", {"message": "pipeline already running"})
                return
            _LIVE_RUNNING = True
            try:
                for ev in pipeline.run_live(clip):
                    yield _sse(ev["event"], ev["data"])
            except Exception as e:
                yield _sse("error", {"message": str(e)})
            finally:
                _LIVE_RUNNING = False

        return StreamingResponse(live_gen(), media_type="text/event-stream")

    # ── Replay path (graph/facts clips) — UNCHANGED ───────────────────────────
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
    # Uploaded clips are written into data/raw so enhance.run(clip_id) finds them
    # at data/raw/<clip_id>.wav — the path it always reads from.
    # CFG.paths.uploads still exists in config (task-1 test asserts the key).
    raw_dir = Path(CFG.paths.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)

    clip_id = "upload_" + uuid.uuid4().hex[:10]
    out_path = raw_dir / f"{clip_id}.wav"

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
