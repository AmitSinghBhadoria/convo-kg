# Live Upload Pipeline (Path A) + Clip Modes — Design

**Date:** 2026-06-27
**Status:** approved-for-planning
**Author:** pairing session (Amit + Claude)

## 1. Goal

Make the demo **prove it works on a real, arbitrary conversation, live**: the user
uploads an audio clip and watches the **whole pipeline run end-to-end** — denoise →
diarize → ASR → extract — with real progress, the diarized English transcript, and
the automatically extracted facts. This is **Path A**, built **first**, as the
headline capability.

The verified PMS hero (knowledge graph + grounded Q&A) is **never touched**: uploads
never write to Neo4j and never use `qa.py`. A flaky live run can never damage the
working demo.

## 2. Context / what exists

- `frontend/index.html` — dc-app UI. The "Drop a conversation to begin" headline is
  currently decorative; **this work makes it the real upload control.**
- `src/api.py` — FastAPI: `/api/graph`, `/api/ask`, `/api/experiment`, `/api/run`
  (+ `/api/run/{id}/stream` SSE). The SSE already emits
  `stage` / `transcript_line` / `fact` / `done` / `error` events (today by *replaying*
  committed JSON with artificial timing) and the frontend already renders them
  (`step`, `streamLines`, `liveFacts`) with a 90 s watchdog + error handler. **Live
  mode reuses these exact event shapes**, driven by real stage completion instead of
  replay.
- Pipeline stages are clip-parameterised and chain across three venvs:
  `src/enhance.run(clip)` (.venv-denoise, writes `{clip}.clean.wav`) →
  `src/diarize_asr.run(clip)` (.venv-asr, writes `{clip}.transcript.json`) →
  `src/extract` (main venv + LM Studio, writes `{clip}.facts.json`). Extraction uses
  an **induced per-clip ontology** (`src/ontology.propose_ontology`), so it adapts to
  new domains. **There is no end-to-end orchestrator yet** — this spec adds it.
- Neo4j Community = single database. Uploaded clips therefore get **no graph and no
  Q&A** (that would require namespacing + editing the hero's query path). They render
  in **facts mode**.

## 3. Clip modes

| Clip | Mode | Run | Graph | Facts | Ask chat |
|------|------|-----|-------|-------|----------|
| `pms` | `graph` | replay (committed) | yes (Neo4j, verified) | (in graph) | yes |
| uploaded clip | `live` | **real pipeline** | no | yes (produced live) | no |
| `call_100`, `call_103` | `facts` | replay (committed) | no | yes | no |

- **`graph`** — verified-depth hero. Unchanged.
- **`live`** — the headline: real end-to-end processing of an uploaded clip, streamed.
- **`facts`** — pre-processed example clips replayed fast/reliably (Phase 2, additive).

`live` and `facts` are the same UI (pipeline rail + transcript + facts, no graph, no
chat); they differ only in whether the run is real-time or a replay.

## 4. Primary feature — upload + live processing

### 4.1 Upload + validation
- `POST /api/upload` (multipart, one audio file). Validates:
  - content is decodable audio (probe with the same tooling the prep step uses);
  - **duration ≤ 10 min (hard cap)** — longer is rejected with a clear message;
  - a sane size limit (reject absurd files before probing).
- On success: prep to 16 kHz mono → `data/uploads/{clip_id}.wav` (ephemeral;
  `clip_id` is a server-generated handle, not user input). Returns `{ clip_id }`.

### 4.2 Live orchestrator (new: `src/pipeline.py`)
- `run_live(clip_id)` — a **generator** that runs the real stages in order and yields
  progress events, so the SSE layer can stream them as they happen:
  1. yield `stage(denoise, active)` → `enhance.run(clip_id)` → yield `stage(denoise, done)`
  2. yield `stage(diarize_asr, active)` → `diarize_asr.run(clip_id)` → read the written
     transcript → yield each `transcript_line` → yield `stage(diarize_asr, done)`
  3. yield `stage(extract, active)` → run extraction → yield each `fact` →
     yield `stage(extract, done)`
  4. yield `done`
- Progress is **stage-level and honest**: the stage functions are blocking subprocess
  calls, so the transcript appears when ASR finishes and facts appear as extraction
  completes — not fabricated word-by-word streaming.
- **No Neo4j writes.** `run_live` stops at facts; it never calls `graph.upsert`.
- Runs in a background task; **one live run at a time** (a module guard rejects a
  second concurrent start with a clear message).

### 4.3 Wiring into the run/SSE
- `POST /api/run` dispatches by the active clip's mode: `live` → drive `run_live`;
  `graph`/`facts` → existing replay. Same `/api/run/{run_id}/stream` channel and event
  shapes for both.
- The frontend's existing `stage`/`transcript_line`/`fact`/`done`/`error` handling is
  reused; the only new need is tolerating **minutes-long** runs (raise/disable the
  90 s watchdog for live mode; rely on the `error` event + a heartbeat instead).

### 4.4 Failure handling (honest, never hangs)
- Each stage is wrapped: on exception, yield `error` with the **stage name + a clean
  message** (e.g. "diarization failed — HF token / pyannote", "LM Studio unreachable")
  and stop. The UI shows which stage failed; partial results already streamed remain
  visible. No silent hang, no fake success.

## 5. Honesty rules

- Live/facts output is the **real automatic extraction at the documented ceiling**,
  shown as-is, **never curated** — labelled "live / unverified extraction". Messy-but-
  real is a demonstrated capability boundary, not a glitch.
- The **PMS hero is the only verified clip** (graph + grounded Q&A). Untouched.
- The upload control replaces the decorative "drop a conversation" copy — the UI no
  longer implies any capability it lacks.
- The **≤10 min cap is a ceiling, not the recommended demo length**: a ~60–90 s trim
  processes in a watchable window; longer clips take 10–25 min and are more failure-
  prone. This trade-off is stated in the design note.

## 6. Hero safety (invariant)

- No upload/live path writes to Neo4j or imports `qa.py`.
- The PMS snapshot and `qa.py` are unchanged; `start.sh` still restores PMS.
- The graph/chat UI logic from commit `8832817` is unchanged for `graph` mode.

## 7. Backend interfaces (`src/api.py`, additive)

- `POST /api/upload` → `{ clip_id }` (or 4xx with message on invalid/oversized/too-long).
- `GET /api/clips` → registry: `{ active, clips:[{id,label,mode,domain,speakers}] }`
  (includes `pms` + example clips; uploaded clips are transient, not in the registry).
- `POST /api/select_clip {id}` → set active clip + mode; `graph` ensures PMS snapshot
  present (no-op when loaded), `facts` no DB op.
- `POST /api/run` → mode-dispatch (live orchestrator vs replay); returns `run_id`.
- `GET /api/run/{run_id}/stream` → SSE (shapes unchanged).
- `GET /api/graph`, `POST /api/ask` → unchanged; only used in `graph` mode.
- `src/config.py` gains `ClipCfg` + `DemoCfg.clips`; `src/pipeline.py` is new.

## 8. Frontend (`frontend/index.html`, no `support.js` edits)

- **Upload control:** the empty-state becomes a real file picker / drop target →
  `POST /api/upload` → on success, `select_clip`(live) + start the run → stream.
- **Mode-driven panels (`clipMode` state):**
  - `graph` → today's layout (pipeline rail, transcript, knowledge graph, Ask chat).
  - `live`/`facts` → pipeline rail + transcript + **facts panel** (where the graph
    was); **Ask chat hidden**; centre widens. Facts populate from `fact` SSE events
    under the honesty label. A "live / unverified" badge shows in `live` mode.
- Long-run UX: the run button shows real elapsed time; watchdog relaxed for `live`.
- Reuse the foreignObject/label + `liveFacts` patterns; graph/chat fixes untouched.

## 9. Secondary (Phase 2, additive after live works) — picker + replay clips

- Clip registry picker (the `▾` becomes real) listing `pms` + `call_100` + `call_103`.
- Process `call_100`/`call_103` offline → commit `{clip}.transcript.json` +
  `{clip}.facts.json` (no Neo4j, no snapshot). They replay in `facts` mode.
- These give a **reliable** second-domain story even if a live run is flaky on the day.
- Eyeball each clip's facts for basic sanity before it ships (not a correctness claim).

## 10. Build order

1. **Live upload + orchestrator** (headline): `POST /api/upload`, `src/pipeline.run_live`,
   run/SSE mode-dispatch, facts-mode UI + upload control, failure handling. Verify
   end-to-end on `call_112` (trimmed) by hand.
2. **Picker + replay clips** (`call_100`/`call_103`) — additive reliability.
3. **Phase 6** packaging (README, design note incl. the Path A write-up, walkthrough).

If time compresses, (2)/(3) shrink first; the verified hero always remains intact.

## 11. Remaining limitations (state in the design note)

- **No graph / Q&A for uploaded clips** — Neo4j Community single-DB; isolating an
  uploaded graph would mean namespacing + scoping `qa.py` (hero query path) or a
  parallel temp-graph + second QA path. Deferred by design; facts mode is the honest
  surface.
- **Coarse progress** — stage-level, because stages are blocking cross-venv
  subprocesses. Within-stage streaming would require modifying the workers.
- **Processing time / fragility** — ~10–25 min for a 10-min clip; cold model loads +
  memory pressure (Whisper-large-v3 + pyannote + LM Studio 9B) are real failure
  surfaces, surfaced honestly per stage.

## 12. Testing

- `tests/test_api.py` — `/api/upload` rejects non-audio and >10 min (mocked
  duration), accepts a valid short clip; `/api/run` in `live` mode drives the
  orchestrator (with stage fns mocked) and emits `stage`/`transcript_line`/`fact`/
  `done`; a mocked stage failure emits `error` (no hang); selecting a `facts`/`live`
  clip performs **no** Neo4j write (PMS node count unchanged).
- `tests/test_pipeline.py` (new) — `run_live` sequences denoise→diarize_asr→extract
  in order, yields the right events, and on a stage raising yields `error` and stops
  (stage fns mocked; no real models in tests).
- `tests/test_frontend_audit.py` — decorative `▾`/"drop a conversation" literals are
  gone or wired; upload posts to `/api/upload`; facts mode hides graph + Ask.
- PMS golden Q&A + snapshot tests stay green (hero unchanged).
- Real end-to-end live run on `call_112` is a **manual** verification (documented),
  not an automated test.

## 13. Out of scope / not changing

- `src/qa.py`, the PMS snapshot, the Experiment tab, the graph/chat UI logic from
  `8832817`.
- Graph/Q&A for uploaded clips (single-DB constraint — documented).
- Hinglish-specific handling for uploaded clips (pipeline is language-general via
  Whisper translate).

## 14. Risks

- **Live reliability** — the headline is the most fragile piece (cold loads, MPS,
  memory, HF auth, LM Studio). Mitigations: real per-stage progress + honest errors;
  recommend a short demo clip; Phase 2 replay clips as a reliable fallback story.
- **Memory pressure** running Whisper-large-v3 + pyannote alongside LM Studio's 9B on
  24 GB. Stages are sequential (each model released before the next); peak is the ASR
  venv — workable for short clips, tighter for 10-min ones.
- **Extraction quality on new domains** — induced ontology adapts, but English
  emergency register is new; shown honestly, not hidden.
