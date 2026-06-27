# Multi-Clip Picker (Path B) — Design

**Date:** 2026-06-27
**Status:** approved-for-planning
**Author:** pairing session (Amit + Claude)

## 1. Goal

Make the demo's clip selector **real and honest**: prove the pipeline generalises
beyond the single hero clip by letting the user switch, live, between the verified
PMS conversation and **911 emergency-dispatch calls** from a completely different
domain — without putting the verified hero at any risk.

This is **Path B** from the live-upload discussion. Path A (real in-browser upload
→ live processing) is deferred and documented as build-ready in §9.

## 2. Context / what exists

- `frontend/index.html` — dc-app UI. Today the "Clip" box (`PMS-advisory.wav ▾`) is
  **decorative** (no handler) and the "Drop a conversation" headline implies an
  upload that does not exist. Removing that dishonesty is part of this work.
- `src/api.py` — FastAPI surface: `/api/graph`, `/api/ask`, `/api/experiment`,
  `/api/run` (+ `/api/run/{id}/stream` SSE replaying `data/work/{clip}.transcript.json`
  and `{clip}.facts.json`). Static mount last.
- Pipeline stages are clip-parameterised and already chain across the three venvs:
  `src/enhance.run(clip)` (.venv-denoise) → `src/diarize_asr.run(clip)` (.venv-asr) →
  `src/extract.py` → `src/graph.py`. Extraction uses an **induced per-clip ontology**
  (`src/ontology.propose_ontology`), so it adapts to non-finance domains.
- Neo4j is **Community edition = single database** (no `CREATE DATABASE`). This is the
  central constraint that shapes the architecture.
- The PMS graph snapshot is committed at `data/ground_truth/pms_graph_snapshot.json`
  and is always restorable; `start.sh` restores it when Neo4j is empty.

## 3. The two clip modes

A clip declares a **mode** that selects which panels render:

| Clip | Mode | Run+transcript | Knowledge graph | Extracted facts | Ask chat |
|------|------|----------------|-----------------|-----------------|----------|
| `pms` | `graph` | yes | yes (Neo4j) | (in graph) | yes |
| `call_100` | `facts` | yes | **no** | yes | **no** |
| `call_103` | `facts` | yes | **no** | yes | **no** |

- **`graph` mode** is the verified-depth hero: knowledge graph + grounded Q&A,
  backed by the committed Neo4j snapshot. Unchanged from today.
- **`facts` mode** is the generality demo: the same pipeline (denoise → diarize →
  ASR → extract) shown running on a 911 call, the diarized English transcript
  streaming in, and the **automatically extracted facts** displayed. No graph
  visualisation and no Ask chat — there is no verified graph or trustworthy Q&A
  for these clips, and we do not pretend there is.

### Why facts mode de-risks the work

`facts`-mode clips **never use Neo4j**. They are served entirely from committed
`data/work/{clip}.transcript.json` + `{clip}.facts.json`. Consequences:

- The hero's Neo4j graph is **never wiped or restored** — selecting a 911 clip is a
  pure UI mode switch. The PMS graph stays loaded permanently.
- **No per-call snapshot, no restore-on-switch, no second QA path, and `qa.py` is
  untouched.** This is the lowest-risk way to add the clips.
- No Q&A-verification gate for the calls (there is no chatbot to answer wrong).

## 4. Honesty rules

- `facts`-mode facts are the **real, automatic extraction** at the documented model
  ceiling — shown as-is, **not curated or hand-fixed**. The existing "live
  extraction — illustrative" honesty label applies. Messy-but-real reads as a
  demonstrated capability boundary, not a glitch.
- Before a call clip ships in the picker, its extracted facts are **eyeballed for
  basic sanity** (recognisably about the call, not garbage). A clip that extracts
  to noise is flagged, not shipped. This is a sanity check, not a correctness claim.
- The PMS hero stays the only **verified** clip (graph + grounded Q&A). Its snapshot
  and `qa.py` are unchanged.
- The decorative `▾` and "drop a conversation" copy are removed/made real — no UI
  implies a capability that does not exist.

## 5. Clips & data

Three clips total (scope guard: 2 additional, one new domain — not five):

- `pms` — finance advisory, Hinglish. Verified hero. Unchanged.
- `call_100` — 911 EMS, vehicle-in-water. English, real noisy phone audio.
- `call_103` — 911 active-shooter / hostage. English, real noisy phone audio.

Each call clip: one **≤90 s fact-dense segment** (selected from the full recording),
prepped to 16 kHz mono, run through the **full offline pipeline** (denoise → diarize
→ ASR-translate → extract), with `data/work/{clip}.transcript.json` and
`{clip}.facts.json` **committed**. No graph snapshot, no Neo4j for these.

Caveats accepted: the calls are **English, not Hinglish** (the domain contrast and
the genuinely-noisy audio outweigh this), and both are the **same new domain**
(emergency dispatch) — different incidents, not two distinct new domains.

## 6. Clip registry

A single source of truth in `config.yaml` under `demo.clips`, e.g.:

```yaml
demo:
  clip: pms            # default active clip
  clips:
    - id: pms
      label: PMS-advisory.wav
      mode: graph
      domain: Private-wealth advisory (Hinglish)
      speakers: 2
    - id: call_100
      label: call_100.wav
      mode: facts
      domain: 911 dispatch — water rescue
      speakers: 2
    - id: call_103
      label: call_103.wav
      mode: facts
      domain: 911 dispatch — active shooter
      speakers: 2
```

`src/config.py` gains a `ClipCfg` model and `DemoCfg.clips: list[ClipCfg]`. The
existing `demo.clip` stays as the default active clip.

## 7. Backend (`src/api.py`, additive)

Active clip is server-side module state (`_ACTIVE_CLIP`, default `CFG.demo.clip`).

- `GET /api/clips` → `{ "active": "<id>", "clips": [ {id,label,mode,domain,speakers}, ... ] }`
  from the registry.
- `POST /api/select_clip` body `{ "id": "<id>" }` → validates id is in the registry,
  sets `_ACTIVE_CLIP`, returns `{ "active": "<id>", "mode": "<mode>" }`. For
  `graph`-mode it ensures the PMS snapshot is present in Neo4j (no-op when already
  loaded); for `facts`-mode it does **nothing to the DB**.
- `POST /api/run` → uses `_ACTIVE_CLIP` (the existing `_RUNS[run_id] = clip` map
  already carries the clip into the stream). The SSE replay already emits
  `stage` / `transcript_line` / `fact` events from the active clip's committed JSON;
  no shape change.
- `GET /api/graph`, `POST /api/ask` → unchanged; only ever exercised in `graph` mode.

No new facts endpoint: in `facts` mode the extracted facts arrive via the existing
run `fact` SSE events and accumulate in the UI's `liveFacts`.

## 8. Frontend (`frontend/index.html`)

- **Picker:** the Clip box becomes a real control listing `clips` from `/api/clips`
  with per-clip `label` + `domain` + `speakers`. Selecting one calls
  `/api/select_clip`, then **resets**: clears chat, resets run to idle, and for
  `graph` mode reloads `/api/graph`. `componentDidMount` fetches `/api/clips` and
  sets the active clip.
- **Mode-driven panels (controller state `clipMode`):**
  - `graph` mode → current layout: pipeline rail, transcript, knowledge graph, Ask
    chat. Unchanged.
  - `facts` mode → pipeline rail + transcript stream + a **facts panel** where the
    graph used to be; the Ask chat panel is **hidden** and the centre/transcript
    widen to fill. Facts populate from the run `fact` stream (`liveFacts`), shown as
    readable entity / relation lines under the honesty label.
- No `support.js` edits. Reuse the foreignObject/label patterns and the existing
  `liveFacts` rendering. The graph/chat fixes from commit `8832817` are untouched.

## 9. Path A — documented as build-ready (deferred)

A section in `design_note.md` (and retained here) records the live-upload path so the
deferral reads as scoped engineering, not a gap:

- **Blocker:** Neo4j Community is single-database, so an uploaded clip's graph cannot
  be isolated in its own DB alongside the verified PMS graph.
- **Two real isolation paths:** (a) **namespace + scope every Q&A query** by clip —
  smallest data change but it edits `qa.py`, the verified hero's query path; or
  (b) a **parallel temp graph + a second QA path** — isolated from the hero but more
  code. Recommendation: (b), to keep the hero query path untouched.
- **Timing reality:** sequential cold model loads (DeepFilterNet, then pyannote +
  Whisper-large-v3, then the extraction LLM) make a 60–90 s clip a **~2–5 min**
  end-to-end wall-clock job on the M4 — fine offline, fragile live.
- **≤90 s cap rationale:** short clips process in a watchable window and keep cold
  loads + memory pressure bounded; a 10-min clip is a coffee break and far more
  failure-prone in a live demo.
- **Unverified UI state:** an uploaded clip would render in `facts` mode with an
  explicit "live / unverified extraction" banner — exactly the mode this spec builds,
  which is why facts mode is the natural foundation for Path-A-lite later.

## 10. Testing

- `tests/test_api.py` — `/api/clips` shape (active + per-clip fields); `/api/select_clip`
  sets active and returns the right mode; selecting a `facts` clip performs **no**
  Neo4j write (PMS graph node count unchanged before/after).
- `tests/test_config.py` (or extend) — registry parses; every clip has a committed
  `{clip}.transcript.json` and (for `facts` clips) `{clip}.facts.json`; modes are in
  `{graph, facts}`.
- `tests/test_frontend_audit.py` — the decorative `▾`/“drop a conversation” literals
  are gone or wired; picker reads `/api/clips`.
- The PMS Q&A golden checks and snapshot tests stay green (hero unchanged).

## 11. Out of scope / not changing

- `src/qa.py`, the PMS snapshot, the Experiment tab, the graph/chat UI logic from
  `8832817`.
- Live upload / in-browser processing (Path A) — documented only.
- Hinglish 911 audio, additional clips beyond the two calls.

## 12. Open risks

- **Extraction quality on 911 content.** Induced ontology should adapt, but the
  English emergency register is new. Mitigation: the §4 eyeball gate — a clip that
  extracts to garbage is flagged and not shipped; we surface the limit rather than
  hide it.
- **Diarization on noisy phone audio** may merge/oversplit speakers. Acceptable: the
  transcript is shown honestly; perfect diarization is not claimed.
