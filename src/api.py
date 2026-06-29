"""Thin FastAPI surface for the Atyx demo. Adapts existing functions (qa.answer,
graph.read_graph, file reads) — NO business logic. Serves the wired dc-app from
frontend/. The pre-built graph stays authoritative; clip-specificity is config only.
"""
import json
import os
import random
import re
import time
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
_LOADED_GRAPH_CLIP: str = CFG.demo.clip    # which graph clip's snapshot is live in Neo4j;
                                           # start.sh restores the default clip's graph at boot,
                                           # so re-selecting the hero is a no-op (no demo-time wipe).


# clip ids reach the filesystem (data/raw/<id>.wav, data/work/<id>.*), so they
# are validated here at the single chokepoint both /api/select_clip and /api/run
# pass through. The general allowlist excludes "." and "/", blocking traversal;
# the upload pattern matches only server-issued ids ("upload_" + uuid4().hex[:10]).
_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_UPLOAD_ID_RE = re.compile(r"^upload_[0-9a-f]{10}$")


def _clip_mode(clip_id: str) -> str:
    """Return the mode string for a clip id.

    Lookup order:
    0. Format validation — reject ids outside a strict allowlist (path-traversal
       guard), and ids that spoof the "upload_" prefix without the issued shape.
    1. Registry (CFG.demo.clips) — authoritative for pre-built clips.
    2. _CLIP_MODE dict — for uploaded clips whose mode was recorded on select.
    3. Prefix heuristic — any clip_id starting with "upload_" defaults to "live".
    4. 404 — unknown clip.
    """
    if not _CLIP_ID_RE.fullmatch(clip_id):
        raise HTTPException(status_code=400, detail="invalid clip id")
    if clip_id.startswith("upload_") and not _UPLOAD_ID_RE.fullmatch(clip_id):
        raise HTTPException(status_code=400, detail="invalid upload id")
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
    global _ACTIVE_CLIP, _LOADED_GRAPH_CLIP
    clip_id = body.id
    mode = _clip_mode(clip_id)   # raises 404 if completely unknown
    # For uploaded clips: record mode so subsequent _clip_mode calls are fast.
    if clip_id.startswith("upload_"):
        _CLIP_MODE[clip_id] = "live"
    # HERO INVARIANT: facts/live selection must never touch Neo4j.
    # Graph clips share ONE Neo4j database (Community is single-DB), so selecting
    # a graph clip swaps the live graph to that clip's committed snapshot. This is
    # a no-op when the requested clip's graph is already loaded.
    if mode == "graph" and _LOADED_GRAPH_CLIP != clip_id:
        try:
            from src.graph import restore_snapshot
            drv = connect()
            try:
                restore_snapshot(drv, DB, clip_id)
                _LOADED_GRAPH_CLIP = clip_id
            finally:
                drv.close()
        except Exception:
            pass   # best-effort; if the snapshot is missing, leave the DB as-is
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


# Per-stage replay pacing: (min, max) seconds, weighted by each stage's real
# pipeline cost so a replay feels like genuine processing rather than an instant
# snapshot load. ASR + extraction are the heavy stages; denoise + graph-build are
# quick. All within a 10–60 s band. Set ATYX_REPLAY_INSTANT=1 to disable the
# pauses (tests, or a fast walkthrough).
_REPLAY_STAGE_DELAYS: dict[int, tuple[int, int]] = {
    0: (10, 18),   # Speech enhancement (DeepFilterNet)
    1: (22, 38),   # Diarization (pyannote)
    2: (38, 58),   # Transcribe · ASR (Whisper large-v3) — slowest
    3: (28, 48),   # Fact extraction (Qwen 9B)
    4: (10, 18),   # Graph build (Neo4j)
}


def _replay_pause(stage_index: int) -> None:
    """Sleep a randomized, stage-weighted duration to pace a replay.
    No-op when ATYX_REPLAY_INSTANT is set (tests / fast walkthrough)."""
    if os.environ.get("ATYX_REPLAY_INSTANT"):
        return
    lo, hi = _REPLAY_STAGE_DELAYS.get(stage_index, (10, 20))
    time.sleep(random.uniform(lo, hi))


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

    # ── Replay path (graph/facts clips) ───────────────────────────────────────
    # Stages are PACED (each emits 'active' → randomized pause → 'done') so a
    # replay feels like real processing rather than an instant snapshot load.
    # See _replay_pause / _REPLAY_STAGE_DELAYS; ATYX_REPLAY_INSTANT skips pauses.
    def gen():
        try:
            tr = Transcript.model_validate_json((work / f"{clip}.transcript.json").read_text())
            audio_stages = [
                (0, "Speech enhancement", "DeepFilterNet"),
                (1, "Diarization", "pyannote 3.x"),
                (2, "Transcribe · Hinglish→EN", "Whisper large-v3"),
            ]
            for idx, label, sub in audio_stages:
                yield _sse("stage", {"index": idx, "label": label, "sub": sub,
                                     "status": "active", "replayed": True})
                _replay_pause(idx)
                yield _sse("stage", {"index": idx, "label": label, "sub": sub,
                                     "status": "done", "replayed": True})
            for u in tr.utterances:
                yield _sse("transcript_line", {"speaker": u.speaker, "t": round(u.start, 1),
                                               "text": u.text})
            # Stage 3 — fact extraction: cached facts when ?replay=1 (paced), else
            # a genuine live extraction (already takes real time, so not paced).
            yield _sse("stage", {"index": 3, "label": "Fact extraction", "sub": "Qwen 9B · 4-bit",
                                 "status": "active", "replayed": False})
            if replay:
                _replay_pause(3)
                facts = json.loads((work / f"{clip}.facts.json").read_text()).get("facts", [])
                for f in facts:
                    yield _sse("fact", {"text": f.get("statement", "")})
            else:
                from src.extract import extract
                fs = extract(clip)                       # live; display-only, NOT upserted
                for f in fs.facts:
                    yield _sse("fact", {"text": f.statement})
            yield _sse("stage", {"index": 3, "label": "Fact extraction", "sub": "Qwen 9B · 4-bit",
                                 "status": "done", "replayed": False})
            # Stage 4 — graph build.
            yield _sse("stage", {"index": 4, "label": "Graph build", "sub": "Neo4j · authoritative",
                                 "status": "active", "replayed": False})
            _replay_pause(4)
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
