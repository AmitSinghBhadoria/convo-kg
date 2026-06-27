# Live Upload Pipeline (Path A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user upload an arbitrary audio clip (≤10 min) and watch the whole pipeline run live — denoise → diarize → ASR → extract — streamed with real per-stage progress, the diarized English transcript, and the extracted facts, in a facts-mode UI (no graph, no chat). Then add a clip picker with two pre-processed replay clips.

**Architecture:** A new `src/pipeline.run_live(clip)` generator runs the existing per-stage venv drivers (`enhance.run`, `diarize_asr.run`, `extract`) in order and yields SSE event dicts. `POST /api/upload` validates + preps an uploaded file; `/api/run` dispatches by clip mode (live orchestrator vs. existing replay) over the existing SSE channel. The frontend's empty-state becomes a real upload control and panels render by clip mode. Uploads never write Neo4j and never import `qa.py`, so the verified PMS hero is untouched.

**Tech Stack:** Python 3.12 (main `.venv`), FastAPI, pydantic, Server-Sent Events, ffmpeg/ffprobe (already present), the dc-app runtime in `frontend/` (do-NOT-edit `support.js`).

## Global Constraints

- **Hero invariant:** no upload/live code path writes to Neo4j or imports `src/qa.py`. `data/ground_truth/pms_graph_snapshot.json` and `src/qa.py` are unchanged. (spec §6)
- **Duration cap:** uploaded audio **> 600 s is rejected** with a clear message. (spec §4.1)
- **Honesty:** live/facts output is the real extraction shown as-is, never curated; the UI carries a "live / unverified extraction" label and no UI implies a capability it lacks (remove the decorative `▾`/“drop a conversation” copy by wiring it). (spec §5)
- **Reuse SSE shapes verbatim:** events are `stage` `{index,label,sub,status}`, `transcript_line` `{speaker,t,text}`, `fact` `{text}`, `done` `{clip}`, `error` `{message}` (+ optional `stage`). The frontend already handles these. (spec §4.3, §2)
- **No `support.js` edits.** Frontend changes live only in the `frontend/index.html` `data-dc-script` controller + `<x-dc>` markup. Reuse the foreignObject/label + `liveFacts` patterns; do not touch the graph/chat logic from commit `8832817` for `graph` mode.
- **One live run at a time:** a module guard rejects a second concurrent live run.
- **venv:** run everything in the activated main `.venv` (`source .venv/bin/activate`), not `uv run`.
- **Commit footer (verbatim):** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

- `src/config.py` (modify) — add `ClipCfg`, `DemoCfg.clips`, `Paths.uploads`.
- `config.yaml` (modify) — `demo.clips` registry, `paths.uploads`.
- `src/pipeline.py` (create) — `run_live(clip)` orchestrator generator. The only new business logic.
- `src/audioprep.py` (create) — `probe_duration(path)`, `to_16k_mono(src,dst)`. Thin, testable, mockable audio helpers.
- `src/api.py` (modify) — `POST /api/upload`, `GET /api/clips`, `POST /api/select_clip`; `/api/run` + stream mode-dispatch; module state `_ACTIVE_CLIP`, `_LIVE_RUNNING`.
- `frontend/index.html` (modify) — upload control, `clipMode` state, mode-driven panels, facts panel, relaxed live watchdog, clip picker.
- `tests/test_config.py` (create) — registry parsing.
- `tests/test_pipeline.py` (create) — `run_live` sequencing + failure.
- `tests/test_api.py` (modify) — upload validation, run mode-dispatch, clips/select_clip, no-Neo4j-write.
- `tests/test_frontend_audit.py` (modify) — upload wiring, facts-mode hiding, picker.
- `scripts/process_clip.py` (create, Phase 2) — offline pipeline runner for example clips.

---

# Phase 1 — Live upload (headline)

### Task 1: Clip registry + uploads path (config)

**Files:**
- Modify: `src/config.py`
- Modify: `config.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ClipCfg(id:str, label:str, mode:str, domain:str="", speakers:int=0)`; `DemoCfg.clip:str="pms"`, `DemoCfg.clips:list[ClipCfg]=[]`; `Paths.uploads:str`. `load_config()` returns `Config` with `.demo.clips` and `.paths.uploads`.

- [ ] **Step 1: Write the failing test** — `tests/test_config.py`

```python
from src.config import load_config

def test_clip_registry_parses_with_modes():
    cfg = load_config()
    clips = cfg.demo.clips
    assert len(clips) >= 1
    pms = next(c for c in clips if c.id == "pms")
    assert pms.mode == "graph"
    assert pms.label and pms.domain
    for c in clips:
        assert c.mode in {"graph", "facts", "live"}, c.mode

def test_uploads_path_present():
    assert load_config().paths.uploads  # non-empty path string
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_config.py -q`
Expected: FAIL (`AttributeError`/`ValidationError`: `clips`/`uploads` missing).

- [ ] **Step 3: Implement** — in `src/config.py` add the `ClipCfg` model and extend `DemoCfg`/`Paths`:

```python
class ClipCfg(BaseModel):
    id: str
    label: str
    mode: str            # "graph" | "facts" | "live"
    domain: str = ""
    speakers: int = 0

class DemoCfg(BaseModel):
    clip: str = "pms"
    clips: list[ClipCfg] = Field(default_factory=list)
```
Add `uploads: str` to `Paths` (with the existing fields). In `config.yaml`, under `paths:` add `uploads: data/uploads`; under `demo:` add:
```yaml
  clips:
    - {id: pms, label: PMS-advisory.wav, mode: graph, domain: "Private-wealth advisory (Hinglish)", speakers: 2}
```
(Phase 2 appends `call_100`/`call_103`.) Import `Field` if not already imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_config.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/config.py config.yaml tests/test_config.py
git commit -m "feat(config): clip registry + uploads path"   # + footer
```

---

### Task 2: Live orchestrator `src/pipeline.py`

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `src.enhance.run(clip:str)->Path`, `src.diarize_asr.run(clip:str)->Path` (writes `data/work/<clip>.transcript.json`), `src.extract.extract(clip:str,cfg,llm)->FactSet`, `src.contracts.Transcript`.
- Produces: `run_live(clip:str) -> Iterator[dict]`, each item `{"event": str, "data": dict}` with events/data exactly matching the SSE shapes (Global Constraints). Stage indices: `0` Speech enhancement, `1` Diarization, `2` Transcribe, `3` Fact extraction. **No graph-build stage and no Neo4j.** Order: enhancement(active→done) → diarization(active) → [run diarize_asr] → diarization(done)+transcribe(done)+one `transcript_line` per utterance → extraction(active) → [run extract] → one `fact` per `fs.facts` (use `f.statement`) → extraction(done) → `done`. Any stage raising → yield `error` `{message, stage}` and stop. Diarization+transcribe both complete when `diarize_asr.run` returns (single subprocess) — emit both `done` then the transcript lines.

- [ ] **Step 1: Write the failing test** — `tests/test_pipeline.py`

```python
import src.pipeline as P

class _FS:  # minimal FactSet stand-in
    def __init__(self, facts): self.facts = facts
class _F:
    def __init__(self, s): self.statement = s

def _patch(monkeypatch, *, fail=None):
    calls = []
    def enh(clip): calls.append("enh"); 
    def dia(clip):
        calls.append("dia")
        # write a 2-utterance transcript the orchestrator will read
        from src.contracts import Transcript, Utterance
        import json, pathlib
        from src.config import load_config
        w = pathlib.Path(load_config().paths.work); w.mkdir(parents=True, exist_ok=True)
        tr = Transcript(clip=clip, utterances=[
            Utterance(speaker="SPEAKER_00", text="hello", start=0.0, end=1.0),
            Utterance(speaker="SPEAKER_01", text="world", start=1.0, end=2.0)])
        (w / f"{clip}.transcript.json").write_text(tr.model_dump_json())
    def ext(clip, cfg=None, llm=None): calls.append("ext"); return _FS([_F("fact one"), _F("fact two")])
    if fail == "dia":
        def dia(clip): calls.append("dia"); raise RuntimeError("pyannote boom")
    monkeypatch.setattr(P, "enhance_run", enh)
    monkeypatch.setattr(P, "diarize_asr_run", dia)
    monkeypatch.setattr(P, "extract", ext)
    return calls

def test_run_live_sequences_and_emits(monkeypatch):
    _patch(monkeypatch)
    evs = list(P.run_live("uploadX"))
    kinds = [e["event"] for e in evs]
    assert kinds[0] == "stage" and kinds[-1] == "done"
    assert kinds.count("transcript_line") == 2
    assert [e["data"]["text"] for e in evs if e["event"] == "fact"] == ["fact one", "fact two"]
    # extraction stage announced before facts; done last
    assert "fact" in kinds and kinds.index("fact") < kinds.index("done")

def test_run_live_stage_failure_emits_error_and_stops(monkeypatch):
    _patch(monkeypatch, fail="dia")
    evs = list(P.run_live("uploadX"))
    assert evs[-1]["event"] == "error"
    assert evs[-1]["data"]["stage"] == "Diarization"
    assert "pyannote" in evs[-1]["data"]["message"]
    assert not any(e["event"] == "fact" for e in evs)  # stopped before extraction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_pipeline.py -q`
Expected: FAIL (`ModuleNotFoundError: src.pipeline`).

- [ ] **Step 3: Implement** — create `src/pipeline.py`. Import the stage drivers under the names the test patches (so they are monkeypatchable module attributes):

```python
from src.enhance import run as enhance_run
from src.diarize_asr import run as diarize_asr_run
from src.extract import extract
from src.contracts import Transcript
from src.config import load_config
```
Implement `run_live(clip)` as a generator following the **Produces** contract: wrap each stage in try/except; on exception `yield {"event":"error","data":{"stage":<name>,"message":str(e)}}` then `return`. Emit `stage` dicts with `{index,label,sub,status}` (labels: "Speech enhancement"/"DeepFilterNet", "Diarization"/"pyannote 3.x", "Transcribe · EN"/"Whisper large-v3", "Fact extraction"/"Qwen 9B · live"). After `diarize_asr_run`, read `Path(load_config().paths.work)/f"{clip}.transcript.json"` via `Transcript.model_validate_json` and yield one `transcript_line` per utterance `{speaker, t:round(u.start,1), text:u.text}`. Run `extract(clip)`; yield one `fact` `{text:f.statement}` per fact. End with `{"event":"done","data":{"clip":clip}}`. **Do not import or call anything from `src.graph` or `src.qa`.**

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_pipeline.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): live run_live orchestrator (no Neo4j)"   # + footer
```

---

### Task 3: Upload endpoint + audio helpers

**Files:**
- Create: `src/audioprep.py`
- Modify: `src/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `src.audioprep.probe_duration(path:str)->float` (seconds; raises `ValueError` if not decodable audio); `src.audioprep.to_16k_mono(src:str, dst:str)->None`. `POST /api/upload` (multipart field `file`) → `{"clip_id": str}`; `400` with `detail` on non-audio or `> 600 s`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_api.py`

```python
import io
from fastapi.testclient import TestClient
import src.api as api
from src.api import app

client = TestClient(app)

def test_upload_rejects_too_long(monkeypatch):
    monkeypatch.setattr(api.audioprep, "probe_duration", lambda p: 601.0)
    monkeypatch.setattr(api.audioprep, "to_16k_mono", lambda s, d: None)
    r = client.post("/api/upload", files={"file": ("x.wav", io.BytesIO(b"RIFFxxxx"), "audio/wav")})
    assert r.status_code == 400 and "10 min" in r.json()["detail"]

def test_upload_rejects_non_audio(monkeypatch):
    def boom(p): raise ValueError("not audio")
    monkeypatch.setattr(api.audioprep, "probe_duration", boom)
    r = client.post("/api/upload", files={"file": ("x.txt", io.BytesIO(b"hello"), "text/plain")})
    assert r.status_code == 400

def test_upload_accepts_valid(monkeypatch, tmp_path):
    monkeypatch.setattr(api.audioprep, "probe_duration", lambda p: 42.0)
    written = {}
    monkeypatch.setattr(api.audioprep, "to_16k_mono", lambda s, d: written.update(dst=d))
    r = client.post("/api/upload", files={"file": ("x.wav", io.BytesIO(b"RIFFxxxx"), "audio/wav")})
    assert r.status_code == 200
    cid = r.json()["clip_id"]
    assert cid and written["dst"].endswith(f"{cid}.wav")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_api.py -q -k upload`
Expected: FAIL (no `/api/upload`; `api.audioprep` missing).

- [ ] **Step 3: Implement**

`src/audioprep.py`: `probe_duration(path)` shells `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 <path>` (subprocess), parse float, raise `ValueError` on non-zero/empty. `to_16k_mono(src,dst)` shells `ffmpeg -nostdin -y -loglevel error -i <src> -ac 1 -ar 16000 <dst>`.

`src/api.py`: `import src.audioprep as audioprep`, add `from fastapi import UploadFile, File`. Add:
```python
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict:
    # save bytes to a temp file; probe duration; reject >600s or non-audio (400);
    # to_16k_mono -> Path(CFG.paths.uploads)/f"{clip_id}.wav"; return {"clip_id": clip_id}
```
`clip_id = "upload_" + uuid.uuid4().hex[:10]`. Ensure `Path(CFG.paths.uploads)` exists. Wrap `probe_duration` `ValueError` → `HTTPException(400, "could not read audio")`; duration > 600 → `HTTPException(400, "clip too long — 10 min max")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_api.py -q -k upload`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/audioprep.py src/api.py tests/test_api.py
git commit -m "feat(api): /api/upload with audio + 10-min validation"   # + footer
```

---

### Task 4: Run mode-dispatch + clips/select_clip endpoints

**Files:**
- Modify: `src/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `src.pipeline.run_live`, `CFG.demo.clips`, `CFG.demo.clip`.
- Produces: module state `_ACTIVE_CLIP:str` (default `CFG.demo.clip`), `_LIVE_RUNNING:bool`. `GET /api/clips` → `{"active":str,"clips":[{id,label,mode,domain,speakers}]}`. `POST /api/select_clip {"id":str}` → `{"active":str,"mode":str}` (404 if id not registered and not an `upload_` id; `facts`/`live` selection performs **no** Neo4j call). `POST /api/run` → `{"run_id":str}` recording the active clip; the stream dispatches `live` mode → `pipeline.run_live`, else the existing replay.

- [ ] **Step 1: Write the failing test** — add to `tests/test_api.py`

```python
def test_clips_lists_registry():
    r = client.get("/api/clips")
    assert r.status_code == 200
    body = r.json()
    assert "active" in body and any(c["id"] == "pms" for c in body["clips"])
    assert all({"id","label","mode"} <= set(c) for c in body["clips"])

def test_select_facts_clip_no_neo4j_write(monkeypatch):
    # selecting a non-graph clip must not call connect()/restore
    monkeypatch.setattr(api, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("neo4j touched")))
    r = client.post("/api/select_clip", json={"id": "upload_abc"})
    assert r.status_code == 200 and r.json()["mode"] == "live"

def test_run_live_mode_drives_orchestrator(monkeypatch):
    api._ACTIVE_CLIP = "upload_abc"; api._CLIP_MODE = {"upload_abc": "live"}
    def fake_live(clip):
        yield {"event": "stage", "data": {"index": 0, "label": "Speech enhancement", "sub": "x", "status": "active"}}
        yield {"event": "done", "data": {"clip": clip}}
    monkeypatch.setattr(api.pipeline, "run_live", fake_live)
    rid = client.post("/api/run").json()["run_id"]
    body = client.get(f"/api/run/{rid}/stream").text
    assert "event: stage" in body and "event: done" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_api.py -q -k "clips or select or live_mode"`
Expected: FAIL (endpoints/state missing).

- [ ] **Step 3: Implement** — in `src/api.py`:
- `import src.pipeline as pipeline`.
- Module state: `_ACTIVE_CLIP = CFG.demo.clip`; a `_clip_mode(clip_id)->str` helper: registry lookup, else `"live"` for `upload_*` ids, else `404`.
- `GET /api/clips`: return `{"active":_ACTIVE_CLIP, "clips":[c.model_dump() for c in CFG.demo.clips]}`.
- `POST /api/select_clip`: validate id (registry or `upload_*`), set `_ACTIVE_CLIP`, return `{"active":id,"mode":_clip_mode(id)}`. For `graph` mode, ensure the snapshot is loaded (call into existing restore only if Neo4j is empty); for `facts`/`live`, **no DB call**.
- `POST /api/run`: `_RUNS[run_id] = _ACTIVE_CLIP`.
- In `api_run_stream`: if `_clip_mode(clip) == "live"`, stream `for ev in pipeline.run_live(clip): yield _sse(ev["event"], ev["data"])` (guarded by `_LIVE_RUNNING` — reject concurrent with an `error` event); else keep the existing replay `gen()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_api.py -q`
Expected: PASS (all api tests, incl. existing).

- [ ] **Step 5: Commit**

```bash
git add src/api.py tests/test_api.py
git commit -m "feat(api): clips/select_clip + run mode-dispatch (live vs replay)"   # + footer
```

---

### Task 5: Frontend — upload control + mode-driven panels + live run

**Files:**
- Modify: `frontend/index.html` (`data-dc-script` controller + `<x-dc>` markup only)
- Test: `tests/test_frontend_audit.py`

**Interfaces:**
- Consumes: `GET /api/clips`, `POST /api/upload`, `POST /api/select_clip`, `POST /api/run` + stream.
- Produces: controller state `clipMode` (`"graph"|"facts"|"live"`); render flags `isGraphMode`/`isFactsMode`; an upload handler; a facts panel; a relaxed watchdog for live runs.

- [ ] **Step 1: Write the failing test** — add to `tests/test_frontend_audit.py`

```python
def test_upload_control_wired():
    assert "/api/upload" in HTML            # real upload, not decorative
    assert "type=\"file\"" in HTML or "type='file'" in HTML

def test_facts_mode_hides_graph_and_chat():
    # graph + Ask panels gated behind graph-mode flag; facts panel gated behind facts/live flag
    assert "isGraphMode" in HTML and "isFactsMode" in HTML

def test_picker_reads_clips_endpoint():
    assert "/api/clips" in HTML and "/api/select_clip" in HTML
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_frontend_audit.py -q -k "upload or facts_mode or picker"`
Expected: FAIL (strings absent).

- [ ] **Step 3: Implement** (controller + markup; **no `support.js` edits**):
- `componentDidMount`: also `fetch('/api/clips')` → store `this.clips`, set `clipMode` for the active clip. Keep the `/api/graph` fetch (only meaningful in graph mode).
- **Upload control:** replace the decorative empty-state headline with a real control — an `<input type="file" accept="audio/*">` (+ the drop target styling). Handler `onUpload(file)`: `POST /api/upload` (FormData) → on `{clip_id}` → `POST /api/select_clip {id:clip_id}` → set `clipMode:'live'` → `startRun()`. On a `400`, surface the `detail` message honestly (reuse the liveFacts/error display).
- **Mode flags in `renderVals`:** `isGraphMode = clipMode==='graph'`, `isFactsMode = clipMode!=='graph'`. Wrap the **knowledge-graph panel** and the **Ask-chat panel** in `<sc-if value="{{ isGraphMode }}">`; add a **facts panel** in `<sc-if value="{{ isFactsMode }}">` that renders `liveFacts` (reuse the existing `liveFacts` markup/sizing) under a "live / unverified extraction" label, where the graph used to sit; widen the centre when chat is hidden.
- **Pipeline rail:** in facts/live mode show 4 stages (drop "Graph build"); `finishRun` caps `step` at 4 for non-graph mode.
- **Watchdog:** in `startRun`, if `clipMode!=='graph'` (live), set the watchdog to a large value (e.g. 30 min) or clear it and rely on `done`/`error`; keep 90 s only for graph-mode replay.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_frontend_audit.py -q`
Expected: PASS (all audits).

- [ ] **Step 5: Manual end-to-end verification (documented, not automated)**

Trim a short test clip: `ffmpeg -y -ss 30 -t 75 -i data/raw/call_112.mp3 -ac 1 -ar 16000 /tmp/call112_75s.wav`. Start the app (`./start.sh`), drop the file, confirm: stages animate as real work runs, transcript appears after ASR, facts appear after extraction, no graph/chat shown, Neo4j PMS node count unchanged (`MATCH (n) RETURN count(n)` before/after). Record timing + any stage failures in the report.

- [ ] **Step 6: Commit**

```bash
git add frontend/index.html tests/test_frontend_audit.py
git commit -m "feat(frontend): real upload control + facts-mode live run"   # + footer
```

---

# Phase 2 — Picker + replay clips (additive; build only after Phase 1 works)

### Task 6: Process + commit call_100 / call_103 + register them

**Files:**
- Create: `scripts/process_clip.py`
- Create (committed artifacts): `data/work/call_100.transcript.json`, `call_100.facts.json`, `call_103.transcript.json`, `call_103.facts.json`
- Modify: `config.yaml` (append two `facts` clips)
- Test: `tests/test_config.py` (extend)

**Interfaces:**
- `scripts/process_clip.py <clip_id> <src.mp3> <start_s> <dur_s>` — trims to ≤90 s 16 kHz mono → `data/raw/<clip_id>.wav`, runs `enhance.run` → `diarize_asr.run` → `extract`, leaving `data/work/<clip_id>.transcript.json` + `.facts.json`.

- [ ] **Step 1: Write the failing test** — extend `tests/test_config.py`

```python
import json, pathlib
def test_example_clips_have_committed_artifacts():
    cfg = load_config()
    work = pathlib.Path(cfg.paths.work)
    for cid in ("call_100", "call_103"):
        c = next(c for c in cfg.demo.clips if c.id == cid)
        assert c.mode == "facts"
        assert (work / f"{cid}.transcript.json").exists()
        facts = json.loads((work / f"{cid}.facts.json").read_text())
        assert len(facts.get("facts", [])) >= 1   # non-empty extraction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_config.py -q -k example_clips`
Expected: FAIL (clips/artifacts absent).

- [ ] **Step 3: Implement + run the offline processing**

Write `scripts/process_clip.py` (chains the existing drivers; prints entity/fact counts). Then process both (pick the most fact-dense ≤90 s window — start near 30 s per the content checks):
```bash
source .venv/bin/activate; set -a; . ./.env; set +a
python -m scripts.process_clip call_100 data/raw/call_100.mp3 30 90
python -m scripts.process_clip call_103 data/raw/call_103.mp3 30 90
```
**Eyeball** each `data/work/<clip>.facts.json` — confirm the facts are recognisably about the call (location, incident, people, weapon), not garbage. If a window extracts poorly, try another and note it. Append to `config.yaml` `demo.clips`:
```yaml
    - {id: call_100, label: call_100.wav, mode: facts, domain: "911 dispatch — water rescue", speakers: 2}
    - {id: call_103, label: call_103.wav, mode: facts, domain: "911 dispatch — active shooter", speakers: 2}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/process_clip.py config.yaml \
  data/work/call_100.transcript.json data/work/call_100.facts.json \
  data/work/call_103.transcript.json data/work/call_103.facts.json tests/test_config.py
git commit -m "feat(clips): process + register call_100/call_103 (facts replay)"   # + footer
```

---

### Task 7: Clip picker UI

**Files:**
- Modify: `frontend/index.html`
- Test: `tests/test_frontend_audit.py`

**Interfaces:**
- Consumes: `this.clips` (from `/api/clips`), `POST /api/select_clip`.
- Produces: a real picker over registry clips; selecting one calls `select_clip`, sets `clipMode`, **resets** (clear chat, run→idle, `liveFacts`/`streamLines` empty) and in graph mode reloads `/api/graph`.

- [ ] **Step 1: Write the failing test** — extend `tests/test_frontend_audit.py`

```python
def test_picker_renders_registry_and_switches():
    assert "onSelectClip" in HTML or "selectClip" in HTML   # a handler exists
    # the decorative caret is now backed by data (clips list), not a static label only
    assert "this.clips" in HTML or "clips" in HTML
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_frontend_audit.py -q -k picker_renders`
Expected: FAIL.

- [ ] **Step 3: Implement** — replace the static Clip box with a control listing `this.clips` (label + domain). Handler `selectClip(id)`: `POST /api/select_clip {id}` → set `clipMode` from the response → reset state (`messages:[]`, `run:'idle'`, `liveFacts:[]`, `streamLines:[]`, `step:-1`, `gsel:null`) → if `graph` mode, re-`fetch('/api/graph')` and repopulate `graphNodes/graphEdges`. The upload control stays available; selecting `pms` returns to the verified hero.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_frontend_audit.py -q && pytest -q`
Expected: PASS (full suite green).

- [ ] **Step 5: Manual verification** — `./start.sh`; switch pms → call_103 → pms; confirm graph+chat only on pms, facts replay on the calls, and the PMS graph/Q&A still works after switching back.

- [ ] **Step 6: Commit**

```bash
git add frontend/index.html tests/test_frontend_audit.py
git commit -m "feat(frontend): real clip picker over registry"   # + footer
```

---

## Self-Review (against the spec)

- **§4.1 upload + ≤10 min + 16k mono** → Task 3. **§4.2 run_live orchestrator, no Neo4j** → Task 2. **§4.3 mode-dispatch over existing SSE** → Task 4. **§4.4 honest per-stage failure** → Task 2 (error event) + Task 5 (surfaced). **§5 honesty label + remove decorative copy** → Task 5. **§6 hero invariant** → enforced by Tasks 2/4 (no `qa.py`/Neo4j) + `test_select_facts_clip_no_neo4j_write`. **§7 endpoints** → Tasks 3,4. **§8 frontend modes + upload** → Task 5; picker → Task 7. **§9 replay clips** → Task 6. **§12 tests** → each task. **§11 limitations** → documented in Phase 6 (out of this plan).
- **Type consistency:** `run_live` yields `{"event","data"}` (Task 2) consumed identically in Task 4; `ClipCfg` fields (Task 1) used in `/api/clips` (Task 4) and frontend (Tasks 5,7); `clip_id` `upload_*` convention shared by Tasks 3,4,5.
- **Placeholder scan:** none — tests are concrete; implementations are signature+contract+outline per the project's plan-style (full bodies are written during the build).
- **Out of scope (no task, intentional):** `src/qa.py`, PMS snapshot, Experiment tab, `support.js`, Phase 6 packaging.
