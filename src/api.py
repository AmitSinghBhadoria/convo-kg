"""Thin FastAPI surface for the Atyx demo. Adapts existing functions (qa.answer,
graph.read_graph, file reads) — NO business logic. Serves the wired dc-app from
frontend/. The pre-built graph stays authoritative; clip-specificity is config only.
"""
import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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


# Static mount LAST so /api/* routes win.
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
