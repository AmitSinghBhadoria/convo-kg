"""Thin FastAPI surface for the Atyx demo. Adapts existing functions (qa.answer,
graph.read_graph, file reads) — NO business logic. Serves the wired dc-app from
frontend/. The pre-built graph stays authoritative; clip-specificity is config only.
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import load_config
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


# Static mount LAST so /api/* routes win.
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
