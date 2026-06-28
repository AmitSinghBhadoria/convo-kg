# Atyx Convo-KG — Product Overview

## In one line

Turn every advisor–client conversation into a structured, queryable record of the **advice
given** — products, strategies, fees, and suitability — automatically, and **without a word
leaving the firm**.

## Who it's for

Atyx Convo-KG is built for a **private-wealth firm** — a PMS / AIF / RIA whose Relationship
Managers are in client conversations all day. The advice given on those calls (what was
recommended, why, and for whom) is the firm's most valuable and most perishable asset; today
it lives in an RM's memory or hand-typed CRM notes. The product captures it automatically and
makes it queryable, with a source quote behind every answer for compliance.

**Why it has to run locally.** These calls carry client PII, live portfolio positions, and
sometimes material non-public information. They cannot be shipped to a cloud frontier API.
Running extraction and Q&A on a **local open-weight LLM** is not a constraint we merely
tolerated — it is the reason the product can exist inside a regulated wealth firm at all.
Data residency is the moat. See [User Stories](./user-stories.md) for the Relationship Manager
and Compliance Officer personas.

## Executive summary

Atyx Convo-KG is a local, batch-mode prototype that turns recorded multi-party conversations
— spoken in Hinglish (Hindi-English code-mix), often over background noise — into a structured
knowledge graph and answers natural-language questions about what was said. Every stage of the
pipeline runs on-device using open-weight models; no data leaves the machine and no frontier
API is called.

The system chains five sequential stages: speech enhancement (DeepFilterNet), speaker
diarization (pyannote.audio), Hinglish-to-English ASR (mlx-whisper large-v3), LLM-driven
fact extraction with an induced-per-clip ontology (Qwen3.5-9B 4-bit via LM Studio), and
single-hop natural-language Q&A grounded against the populated Neo4j graph. Each answer
carries a verbatim source quote and the graph nodes it touched; the system will decline to
answer rather than hallucinate when no evidence is found.

The verified demo runs on a real ten-minute private-wealth advisory call (a PMS product
walk-through): the graph captures the products, strategies, fee structures, and suitability
segment discussed, and the Q&A answers questions about them with provenance.

This is an honest-scope prototype. It is single-user and local-only; there is no
authentication, no cloud deployment, no real-time streaming, and no multi-tenancy. The
design is measurement-driven: a controlled-SNR evaluation suite (café-babble sweep,
transcript-similarity curve, and spotcheck rows) makes accuracy vs noise directly
observable.

---

## The problem

A wealth advisor's client conversations — advisory calls, product walk-throughs, suitability
discussions, held in mixed Hindi-English — contain the firm's highest-value information: what
was recommended, how it was positioned against alternatives, the fees disclosed, and who it
was deemed suitable for. That information is buried in noisy audio and never extracted. It
evaporates into memory or gets hand-typed into a CRM, and the compliance trail (what advice
was given, and was it appropriate) has to be reconstructed after the fact.

Existing tools don't fit this setting: they require cloud APIs (a non-starter for PII /
portfolio / MNPI data), do not handle code-mixed Hinglish, flatten facts into raw transcripts
(losing the structure a graph needs), or conflate multiple speakers.

Atyx Convo-KG addresses the full chain: denoise the audio, separate speakers, transcribe and
translate Hinglish to English, extract structured facts into a queryable knowledge graph, and
answer questions with grounded, citable answers — all on a local open-weight LLM running on a
single laptop, with nothing leaving the firm.

---

## Key features

- **Speech enhancement** — DeepFilterNet suppresses background noise before ASR, making the
  transcript quality observable as a function of SNR.
- **Speaker diarization** — pyannote.audio 3.x segments and labels speakers (who said what);
  attribution is preserved all the way to the Q&A answer.
- **Hinglish-to-English ASR** — mlx-whisper large-v3 with the `translate` task transcribes
  code-mixed Hinglish and outputs English directly; Apple-Silicon optimized.
- **Induced-ontology fact extraction** — Qwen3.5-9B (4-bit, LM Studio) reads
  speaker-attributed chunks and emits schema-enforced JSON (entities, relations, facts);
  the ontology is induced per clip, not hand-coded.
- **Neo4j knowledge graph** — entities and relations stored as first-class nodes and typed
  edges; idempotent MERGE upserts; entity resolution merges near-duplicates by normalized
  name or embedding similarity (cosine ≥ 0.85, same label+type).
- **Grounded single-hop Q&A** — text-to-Cypher path with a semantic-embedding fallback;
  every answer carries `◆ source quote` provenance and highlighted graph nodes; the
  no-hallucination floor declines rather than fabricates when best-match cosine < 0.40.
- **Live audio upload** — drop any audio file (≤ 10 min) into the UI; the full pipeline
  runs live and produces a transcript + extracted-facts panel (no Neo4j write; graph/Q&A
  not available for uploaded clips — see Limitations).
- **Controlled-SNR evaluation** — the Experiment tab shows transcript-similarity vs SNR
  over a café-babble sweep and a spotcheck question grid; the accuracy-vs-noise degradation
  curve is directly inspectable.
- **Local open-weight LLM** — Qwen3.5-9B 4-bit via LM Studio OpenAI-compatible endpoint;
  no frontier API call; no data leaves the machine.

---

## Target users

**Who the product serves** (domain personas — see [User Stories](./user-stories.md) for the
full stories and acceptance criteria):

| Persona | What they get |
|---------|---------------|
| **Relationship Manager / Wealth Advisor** | Recall what was recommended on any call — strategy, comparison basis, suitability — in seconds, with the exact quote, instead of re-listening to recordings or typing CRM notes. |
| **Compliance Officer** | An audit-ready, grounded record of the advice given: every answer cites its source, the system declines rather than fabricate, and nothing ever leaves the firm's infrastructure. |

**How the prototype is driven.** This v1 is a single-user local prototype. There is **no
authentication, no login, no user accounts, no admin panel, and no role system** — in a real
deployment the personas above are served through the same surfaces.

| Prototype role | Description |
|-----------|-------------|
| **Analyst / end-user** | Opens the browser UI at `http://localhost:8000`, selects a clip (or uploads one), runs the pipeline, reads the transcript, browses the knowledge graph, and asks questions via Ask-Atyx chat. |
| **Operator / developer** | Runs `./setup.sh` and `./start.sh` to install dependencies and start the server; edits `.env` for Neo4j password and HF token; loads models in LM Studio; may add clips to `config.yaml`. |

---

## Value proposition

| Property | What it means in practice |
|----------|--------------------------|
| **Data residency is the moat** | All audio, transcripts, and extracted facts stay on the firm's machine. No cloud API, no telemetry — the one thing that lets PII / portfolio / MNPI conversations be processed at all in a regulated wealth firm. |
| **Grounded, honest answers** | Every Q&A answer cites the verbatim statement it came from. The system returns `found: false` rather than hallucinate when evidence is absent — an audit-ready, defensible record. |
| **Captures the advice, not just the audio** | The graph holds the products, strategies, fees, and suitability segment that were actually discussed — as first-class entity nodes and typed relation edges, not text blobs. Single-hop queries work today; multi-hop traversals (e.g. across clients) are a graph query away (constrained by local ~9B LLM Cypher quality, not architecture). |
| **Measurement-driven** | SNR degradation curves and spotcheck rows make the accuracy ceiling visible and reproducible, not claimed. |
| **Reproducible** | `./setup.sh` + `.env` edit + `./start.sh` is the full setup; no Docker for the app itself; three isolated venvs handle irreconcilable torch pins cleanly. |

---

## Technology stack

| Layer | Technology | Why chosen |
|-------|-----------|------------|
| Speech enhancement | DeepFilterNet (`.venv-denoise`, Python 3.11, torch 2.0.1) | Proven real-time-class speech denoising; isolated venv for torch pin compatibility |
| Diarization | pyannote.audio 3.x (`.venv-asr`) | State-of-the-art speaker segmentation; HF-gated model via `HF_TOKEN` |
| ASR + translation | mlx-whisper large-v3 (`translate` task, `.venv-asr`) | Apple-Silicon MLX backend; `translate` task gives Hinglish→English in one pass |
| LLM — extraction + Q&A | Qwen3.5-9B 4-bit via LM Studio (`http://localhost:1234/v1`) | Strong structured JSON output; Cypher/tool generation; multilingual; fits 24 GB sequentially alongside ASR |
| Embeddings — semantic fallback | `text-embedding-nomic-embed-text-v2-moe` via LM Studio | Local embedding model for statement-cosine semantic fallback in Q&A |
| Graph DB | Neo4j 5.x Community (local, single database) | Cypher for single/multi-hop traversal; idempotent MERGE upserts; open-source |
| API | FastAPI + uvicorn + SSE | Thin Python API; Server-Sent Events for real-time pipeline progress events |
| Frontend | dc-app (single-file `frontend/index.html` + vendored `support.js`) | No build step; served statically by FastAPI; custom lightweight React-like runtime |
| Orchestration | Python 3.12 main `.venv`; cross-venv via subprocess + disk artifacts | Three-venv isolation for irreconcilable torch pins; main venv is torch-free |
| Setup / env | uv (`setup.sh`, `start.sh`) | Fast, reproducible venv creation; `SKIP_AUDIO=1` for demo-only install |

---

## End-to-end pipeline

```mermaid
flowchart LR
    A([Audio file]) --> B[Denoise\nDeepFilterNet]
    B --> C[Diarize\npyannote 3.x]
    C --> D[ASR + Translate\nmlx-whisper v3]
    D --> E[Fact Extraction\nQwen3.5-9B]
    E --> F[(Neo4j Graph)]
    F --> G[Q&A\ntext-to-Cypher\n+ semantic fallback]
    G --> H([Grounded answer\n+ provenance])
```

### Feature grouping by value

```mermaid
flowchart LR
    subgraph Audio["Audio → Text"]
        B2[Denoise] --> C2[Diarize] --> D2[ASR / Translate]
    end
    subgraph Knowledge["Text → Knowledge"]
        E2[Extract facts] --> F2[(Graph DB)]
    end
    subgraph QA["Knowledge → Answers"]
        F2 --> G2[Cypher Q&A]
        F2 --> H2[Semantic fallback]
        G2 --> I2([Grounded answer])
        H2 --> I2
    end
    subgraph Eval["Evaluation"]
        J2[SNR sweep] --> K2[Fidelity curve]
        J2 --> L2[Spotcheck grid]
    end
    Audio --> Knowledge
    Knowledge --> QA
```

---

## Scope and honest limitations

| Area | Current state | Extension path |
|------|--------------|----------------|
| Q&A depth | **Single-hop** — one graph traversal per question | Graph and schema already support multi-hop; constrained by local ~9B Cypher generation quality |
| Uploaded clips | Pipeline runs live; transcript + facts panel shown. **No Neo4j write, no graph view, no Ask-Atyx chat** | Neo4j Community is a single-database server; per-clip graph isolation needs namespacing or a graph-per-database approach |
| Diarization on phone audio | Single-channel phone recordings (e.g. 911 clips) collapse to 1 speaker — the diarizer cannot separate what it cannot observe | Collapses gracefully with an honest UI note; does not affect far-field mic recordings |
| Extraction quality | Local ~9B LLM on noisy/code-mixed Hinglish is the measured accuracy ceiling — not a pipeline gap | Larger quantized models (14B+) or fine-tuning on Hinglish extraction improve this directly |
| Processing mode | **Batch only** — full pipeline runs end-to-end on a finished recording | Streaming ingestion would need chunked ASR + incremental graph upserts |
| Deployment | **Local only** — `localhost:8000`; no Docker Compose for the full stack, no cloud path | Auth, multi-tenancy, and cloud deployment are out of scope for v1 |
| Authentication | **None** — single-user local tool; no login, no roles, no admin panel | Not planned for this prototype |
| Target hardware | Apple M4, 24 GB unified memory; stages run **sequentially** to fit memory | Parallel execution feasible on higher-memory machines |

---

## Documentation map

| Document | Description |
|----------|-------------|
| [Product Overview](./product-overview.md) | This document — executive summary, features, stack, limitations |
| [System Architecture](./system-architecture.md) | Three-venv pipeline design, component interactions, data flow |
| [Entity Relationship](./entity-relationship.md) | Neo4j graph schema — node labels, relationship types, Pydantic contracts |
| [User Stories](./user-stories.md) | Domain personas (Relationship Manager, Compliance Officer) and prototype interaction roles (Analyst, Operator); acceptance criteria |
| [Wireflows](./wireflows.md) | Screen-level user flows — clip selection, run, upload, Q&A, Experiment tab |
| [Wireframes](./wireframes.md) | UI layout annotations — Console and Experiment screens |
| [Sequence Diagrams](./sequence-diagrams.md) | Request/response sequences for every API call + SSE stream |
| [API Specification](./api-specification.md) | Full REST API reference — endpoints, request/response shapes, error model |
| [Deployment Guide](./deployment-guide.md) | Prerequisites, setup steps, `.env` config, start/stop, troubleshooting |
