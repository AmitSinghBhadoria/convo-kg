# Atyx Convo-KG — Sequence Diagrams

This document shows the key runtime interaction sequences for Atyx Convo-KG.
There is **no authentication or login flow** — the system is a local, single-user
research prototype with no user accounts, no sessions, and no auth middleware.
The real entry flow is the browser loading the static frontend and immediately
fetching graph/clips/experiment metadata from the FastAPI backend.

Related docs: [product overview](./product-overview.md) ·
[system architecture](./system-architecture.md) ·
[API specification](./api-specification.md) ·
[entity-relationship](./entity-relationship.md) ·
[wireflows](./wireflows.md) · [deployment guide](./deployment-guide.md)

---

## 1. App Load / Initialization

When the browser loads `http://localhost:8000` it receives the static
`frontend/index.html`. The dc-app runtime immediately fires three parallel
fetches to bootstrap state. There is no login step — the app is always
"authenticated" by virtue of running locally.

```mermaid
sequenceDiagram
    participant Browser
    participant FastAPI
    participant Neo4j

    Browser->>FastAPI: GET /api/clips
    FastAPI-->>Browser: {active: "pms", clips: [{id, label, mode, domain, speakers}, ...]}

    Browser->>FastAPI: GET /api/graph
    FastAPI->>Neo4j: MATCH concept nodes (Entity/Claim/Attribute) + fact edges
    Neo4j-->>FastAPI: nodes[], edges[]
    FastAPI-->>Browser: {nodes: [...], edges: [...]}

    Browser->>FastAPI: GET /api/experiment
    alt snr_results.json present in data/ground_truth/
        FastAPI-->>Browser: EvalResult {curve: [CurvePoint], spotcheck: [...], meta}
    else file absent
        FastAPI-->>Browser: HTTP 404 — run python -m src.evaluate first
    end

    Note over Browser: Renders Console tab on active clip (default: pms, graph mode)
    Note over Browser: Graph mode — 3-column layout: pipeline rail | transcript | KG + Ask Atyx
    Note over Browser: Experiment tab populated only when SNR curve data is present
```

---

## 2. Ask a Question (Q&A)

Q&A is only available for `graph`-mode clips (the pre-built PMS hero). The
pipeline: introspect live Neo4j schema → text-to-Cypher via LM Studio (up to
3 attempts, `1 + MAX_RETRIES=2`) → run read-only Cypher → compose answer. If
Cypher fails or returns no rows, the system falls back to embedding-based
retrieval over `:Statement` nodes with a hard cosine floor of `0.40` — scores
below that floor return `found:false` without calling the LLM, ensuring no
hallucinated answers. A `found:false` result is still HTTP 200; the browser
renders the decline message without an error state.

```mermaid
sequenceDiagram
    participant Browser
    participant FastAPI
    participant Neo4j
    participant LMStudio

    Browser->>FastAPI: POST /api/ask {question: str}
    FastAPI->>Neo4j: introspect schema (CALL apoc.meta.schema + label query)
    Neo4j-->>FastAPI: schema JSON + known_labels set

    loop Up to 3 attempts (1 + MAX_RETRIES)
        FastAPI->>LMStudio: generate_cypher — schema-aware text-to-Cypher prompt
        LMStudio-->>FastAPI: Cypher string
        FastAPI->>FastAPI: is_read_only guard — reject MERGE/CREATE/DELETE/SET
        FastAPI->>Neo4j: EXPLAIN <cypher> — syntax + plan validation
        Neo4j-->>FastAPI: plan OK or error message
        FastAPI->>FastAPI: label guard — reject undefined node labels
        Note over FastAPI: On any failure: feed error msg back, retry
    end

    alt Valid Cypher produced
        FastAPI->>Neo4j: read-only Cypher (read-access transaction)
        Neo4j-->>FastAPI: rows[]

        alt rows returned
            FastAPI->>Neo4j: resolve provenance — MATCH (:Statement) by source_statement_id
            Neo4j-->>FastAPI: Provenance[] {statement_id, speaker, text, kind}
            FastAPI->>LMStudio: compose_answer (question + rows + provenance context)
            LMStudio-->>FastAPI: English answer string
            FastAPI-->>Browser: QAResult {mode:"cypher", found:true, answer, cypher, provenance, graph_node_ids, hops}
        else no rows — fall through to semantic fallback
            FastAPI->>Neo4j: MATCH (s:Statement) RETURN id, speaker, text
            Neo4j-->>FastAPI: all statement texts
            FastAPI->>LMStudio: embed(statements) + embed(question) [nomic-embed-text-v2-moe]
            LMStudio-->>FastAPI: embedding vectors
            FastAPI->>FastAPI: top_k cosine ranking — best_score = top[0].score
            alt best_score >= 0.40 (FALLBACK_MIN_COSINE)
                FastAPI->>LMStudio: compose_answer (question + top-k statement context)
                LMStudio-->>FastAPI: English answer string
                FastAPI-->>Browser: QAResult {mode:"semantic-fallback", found:true, answer, provenance}
            else best_score < 0.40 — no-hallucination decline
                Note over FastAPI: LLM never called — decline is score-gated, not LLM judgment
                FastAPI-->>Browser: QAResult {mode:"semantic-fallback", found:false} [HTTP 200]
            end
        end

    else All retries exhausted — no valid Cypher
        FastAPI->>Neo4j: MATCH (s:Statement) RETURN id, speaker, text
        Neo4j-->>FastAPI: all statement texts
        FastAPI->>LMStudio: embed(statements) + embed(question) [nomic-embed-text-v2-moe]
        LMStudio-->>FastAPI: embedding vectors
        FastAPI->>FastAPI: top_k cosine ranking — best_score = top[0].score
        alt best_score >= 0.40
            FastAPI->>LMStudio: compose_answer (question + top-k statement context)
            LMStudio-->>FastAPI: English answer string
            FastAPI-->>Browser: QAResult {mode:"semantic-fallback", found:true, answer, provenance}
        else best_score < 0.40 — no-hallucination decline
            FastAPI-->>Browser: QAResult {mode:"semantic-fallback", found:false} [HTTP 200]
        end
    end

    Note over Browser: found:true → render answer + ◆ provenance quote + highlight graph_node_ids
    Note over Browser: found:false → render polite decline — no error state
```

---

## 3. Run Replay (graph / facts clip)

For pre-built clips (`pms` in graph mode, `call_100`/`call_103` in facts mode),
"Run" replays committed data. Stages 0–2 (Speech enhancement, Diarization,
Transcribe) are emitted immediately as `replayed:true` — no subprocess is
launched. The Fact extraction stage (index 3) is re-run live against LM Studio
by default (`?replay=0`) so the extraction proof is observable; the result is
display-only and **never upserted to Neo4j**. Passing `?replay=1` skips live
extraction and reads the committed `.facts.json` instead. The authoritative graph
in Neo4j remains untouched either way.

```mermaid
sequenceDiagram
    participant Browser
    participant FastAPI
    participant LMStudio

    Browser->>FastAPI: POST /api/run
    FastAPI-->>Browser: {run_id: "<12-char hex>"}
    Browser->>FastAPI: GET /api/run/{run_id}/stream [?replay=0 default]
    Note over FastAPI: Opens text/event-stream (SSE) — gen() generator starts

    FastAPI-->>Browser: SSE stage(0, "Speech enhancement", replayed:true, done)
    FastAPI-->>Browser: SSE stage(1, "Diarization", replayed:true, done)
    FastAPI-->>Browser: SSE stage(2, "Transcribe · Hinglish→EN", replayed:true, done)
    FastAPI-->>Browser: SSE transcript_line × N (from committed .transcript.json)

    FastAPI-->>Browser: SSE stage(3, "Fact extraction", replayed:false, active)

    alt ?replay=0 (default) — live extraction, display-only
        FastAPI->>LMStudio: extract(clip) — Qwen 9B JSON extraction
        LMStudio-->>FastAPI: FactSet (display-only — NOT upserted to Neo4j)
        FastAPI-->>Browser: SSE fact{text} × M
    else ?replay=1 — read committed .facts.json
        FastAPI-->>Browser: SSE fact{text} × M (from data/work/<clip>.facts.json)
    end

    FastAPI-->>Browser: SSE stage(4, "Graph build", Neo4j · authoritative, done)
    FastAPI-->>Browser: SSE done {clip}

    alt any exception during gen()
        FastAPI-->>Browser: SSE error {message: str(e)}
    end

    Note over FastAPI: Neo4j graph stays authoritative throughout — replay never writes
    Note over Browser: graph mode — Graph panel + Ask Atyx chat remain active after run
    Note over Browser: facts mode — extracted-facts panel shown — no graph, no chat
```

---

## 4. Live Upload

When a user uploads a new audio file, the full pipeline runs for real: denoise →
diarize → ASR → fact extraction, each stage as a subprocess in its own venv
(cross-venv via disk artifacts in `data/work/`). LM Studio handles fact
extraction in the main venv. The `_LIVE_RUNNING` boolean prevents concurrent
live runs. Uploaded clips are `live` mode — they show the facts panel only; no
Neo4j write, no graph, no Ask Atyx.

```mermaid
sequenceDiagram
    participant Browser
    participant FastAPI
    participant SubProc
    participant LMStudio

    Browser->>FastAPI: POST /api/upload (multipart audio file)
    FastAPI->>FastAPI: probe_duration via ffprobe
    alt non-audio file or duration > 600 s
        FastAPI-->>Browser: HTTP 400 {detail: "..."}
    else valid audio ≤ 10 min
        FastAPI->>FastAPI: to_16k_mono via ffmpeg → data/raw/upload_<10hex>.wav
        FastAPI-->>Browser: {clip_id: "upload_<10hex>"}
    end

    Browser->>FastAPI: POST /api/select_clip {id: clip_id}
    FastAPI->>FastAPI: _clip_mode validates _UPLOAD_ID_RE (path-traversal guard)
    Note over FastAPI: HERO INVARIANT — uploaded/live clips never touch Neo4j
    FastAPI-->>Browser: {active: clip_id, mode: "live"}

    Browser->>FastAPI: POST /api/run
    FastAPI-->>Browser: {run_id}

    Browser->>FastAPI: GET /api/run/{run_id}/stream
    Note over FastAPI: live_gen() generator starts

    opt _LIVE_RUNNING already true — single-run guard
        FastAPI-->>Browser: SSE error {message: "pipeline already running"}
        Note over FastAPI: Generator returns immediately — no stages run
    end

    FastAPI->>FastAPI: _LIVE_RUNNING = true
    FastAPI-->>Browser: SSE stage(0, "Speech enhancement", active)
    FastAPI->>SubProc: enhance_run(clip_id) [DeepFilterNet · .venv-denoise subprocess]
    alt stage 0 fails
        FastAPI-->>Browser: SSE error {stage: "Speech enhancement", message}
        Note over FastAPI: Generator stops — _LIVE_RUNNING reset in finally block
    else enhanced WAV written to data/work/
        FastAPI-->>Browser: SSE stage(0, done)

        FastAPI-->>Browser: SSE stage(1, "Diarization", active)
        FastAPI->>SubProc: diarize_asr_run(clip_id) [pyannote 3.x + Whisper large-v3 · .venv-asr subprocess]
        alt stage 1/2 fails
            FastAPI-->>Browser: SSE error {stage: "Diarization", message}
            Note over FastAPI: Generator stops — _LIVE_RUNNING reset
        else .transcript.json written to data/work/
            FastAPI-->>Browser: SSE stage(1, done)
            FastAPI-->>Browser: SSE stage(2, "Transcribe · EN", done)
            FastAPI-->>Browser: SSE transcript_line × N

            FastAPI-->>Browser: SSE stage(3, "Fact extraction", active)
            FastAPI->>LMStudio: extract(clip_id) — Qwen 9B [main .venv, no subprocess]
            alt stage 3 fails
                FastAPI-->>Browser: SSE error {stage: "Fact extraction", message}
                Note over FastAPI: Generator stops — _LIVE_RUNNING reset
            else FactSet returned
                LMStudio-->>FastAPI: FactSet (display-only — no Neo4j write)
                FastAPI-->>Browser: SSE fact{text} × M
                FastAPI-->>Browser: SSE stage(3, done)
                FastAPI-->>Browser: SSE done {clip}
            end
        end
    end

    Note over FastAPI: _LIVE_RUNNING = false (always, in finally block)
    Note over Browser: live clip — facts panel only — no Knowledge Graph, no Ask Atyx
    Note over Browser: Single-speaker phone audio: diarization may collapse to 1 speaker (expected)
```

---

## 5. Select Clip / Graph Snapshot Restore

Clip selection updates the server-side active clip and, for `graph`-mode clips
only, triggers a best-effort Neo4j snapshot restore when the database is found
empty. `facts` and `live` clips never touch Neo4j — this is the **hero
invariant** that keeps the verified PMS graph intact regardless of other
activity. The clip picker in the UI is a click-to-toggle dropdown over the
registered clip registry (`config.yaml`).

```mermaid
sequenceDiagram
    participant Browser
    participant FastAPI
    participant Neo4j

    Browser->>FastAPI: POST /api/select_clip {id: "<clip_id>"}
    FastAPI->>FastAPI: _clip_mode(clip_id) — validate _CLIP_ID_RE regex

    alt invalid format or completely unknown id
        FastAPI-->>Browser: HTTP 404 {detail: "unknown clip"}
    else graph mode (e.g. id = "pms")
        FastAPI->>Neo4j: MATCH (n) RETURN count(n) AS n
        Neo4j-->>FastAPI: node count
        opt count == 0 — database empty
            FastAPI->>Neo4j: restore_snapshot(driver, DB, clip_id)
            Note over FastAPI,Neo4j: Best-effort — skipped silently if restore_snapshot absent
        end
        FastAPI->>FastAPI: _ACTIVE_CLIP = clip_id
        FastAPI-->>Browser: {active: clip_id, mode: "graph"}
        Note over Browser: UI enables Knowledge Graph panel + Ask Atyx chat
    else facts mode (e.g. id = "call_100")
        Note over FastAPI: HERO INVARIANT — no Neo4j interaction
        FastAPI->>FastAPI: _ACTIVE_CLIP = clip_id
        FastAPI-->>Browser: {active: clip_id, mode: "facts"}
        Note over Browser: UI shows extracted-facts panel only — graph + chat hidden
    else live mode (uploaded clip)
        Note over FastAPI: HERO INVARIANT — no Neo4j interaction
        FastAPI->>FastAPI: _CLIP_MODE[clip_id] = "live" — _ACTIVE_CLIP = clip_id
        FastAPI-->>Browser: {active: clip_id, mode: "live"}
        Note over Browser: UI shows Run live button + facts panel — graph + chat hidden
    end
```
