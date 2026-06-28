"""Live pipeline orchestrator: audio clip → SSE event dicts.

Stages (no Neo4j, no graph.py, no qa.py):
  0  Speech enhancement  (enhance_run)
  1  Diarization         (diarize_asr_run — also covers Transcribe)
  2  Transcribe · EN     (emitted as done immediately after diarize_asr_run)
  3  Fact extraction     (extract)

Yields dicts  {"event": str, "data": dict}  in this order:
  stage(0, active) → [enhance] → stage(0, done)
  → stage(1, active) → [diarize_asr] → stage(1, done) + stage(2, done)
  → transcript_line × N
  → stage(3, active) → [extract] → fact × M → stage(3, done)
  → done

On any stage exception:
  → error{stage: <label>, message: str(e)}  (generator stops)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

# Import under module-level names so tests can monkeypatch them.
from src.enhance import run as enhance_run          # noqa: F401 (patched by tests)
from src.diarize_asr import run as diarize_asr_run  # noqa: F401 (patched by tests)
from src.extract import extract                     # noqa: F401 (patched by tests)
from src.contracts import Transcript
from src.config import load_config

# Stage metadata table: (index, label, sub)
_STAGES = [
    (0, "Speech enhancement", "DeepFilterNet"),
    (1, "Diarization",        "pyannote 3.x"),
    (2, "Transcribe · EN",    "Whisper large-v3"),
    (3, "Fact extraction",    "Qwen 9B · live"),
]


def _stage_event(index: int, status: str) -> dict:
    idx, label, sub = _STAGES[index]
    return {"event": "stage", "data": {"index": idx, "label": label, "sub": sub, "status": status}}


def run_live(clip: str) -> Iterator[dict]:
    """Generator: run pipeline stages on *clip*, yielding SSE event dicts."""

    cfg = load_config()
    work_dir = Path(cfg.paths.work)

    # ── Stage 0: Speech enhancement ──────────────────────────────────────────
    yield _stage_event(0, "active")
    try:
        enhance_run(clip)
    except Exception as e:
        yield {"event": "error", "data": {"stage": "Speech enhancement", "message": str(e)}}
        return
    yield _stage_event(0, "done")

    # ── Stage 1+2: Diarization + Transcription (single subprocess) ───────────
    yield _stage_event(1, "active")
    try:
        diarize_asr_run(clip)
        transcript_path = work_dir / f"{clip}.transcript.json"
        transcript = Transcript.model_validate_json(transcript_path.read_text())
    except Exception as e:
        yield {"event": "error", "data": {"stage": "Diarization", "message": str(e)}}
        return
    yield _stage_event(1, "done")
    yield _stage_event(2, "done")

    # Emit transcript lines
    for u in transcript.utterances:
        yield {
            "event": "transcript_line",
            "data": {"speaker": u.speaker, "t": round(u.start, 1), "text": u.text},
        }

    # ── Stage 3: Fact extraction ──────────────────────────────────────────────
    yield _stage_event(3, "active")
    try:
        fs = extract(clip)
    except Exception as e:
        yield {"event": "error", "data": {"stage": "Fact extraction", "message": str(e)}}
        return
    for f in fs.facts:
        yield {"event": "fact", "data": {"text": f.statement}}
    yield _stage_event(3, "done")

    # ── Done ──────────────────────────────────────────────────────────────────
    yield {"event": "done", "data": {"clip": clip}}
