# Atyx — Conversational Knowledge Graph: Design Spec

**Date:** 2026-06-25 · **Target demo:** Mon 29 Jun 2026 · **Reviewer:** Debopam Bhattacherjee
**Status:** design approved in brainstorming; pending final user review before implementation planning.

---

## 1. Goal (one line)
From recorded **multi-party Hinglish audio**, extract facts into a **knowledge graph** and answer **natural-language questions** about what was said — running extraction and Q&A on a **local open-weight LLM**, built to work on **arbitrary, noisy real-world audio**.

## 2. Guiding principle: build general, demo specific
The system is built for **arbitrary input audio** — no hardcoded speaker count, domain, topic, or schema. The financial **PMS-vs-Mutual-Fund** clip is the *showcase instance*, not a baked-in assumption. The pipeline must accept any clip (ideally live in the demo), induce its ontology from the conversation itself, and **degrade gracefully** (fewer, lower-confidence, source-backed facts on hard audio — never a crash or a hallucinated graph).

## 3. Scope

**In scope (one complete working spine, end to end):**
`Audio → Denoise → Diarization → Hinglish→English ASR → Induced-ontology fact extraction → Neo4j graph → single-hop NL Q&A`, plus a **controlled-SNR evaluation harness**.

**Out of scope for v1 (architect for, don't build):** clips > 15 min (chunk-and-merge is the noted extension), streaming / large-scale ingestion, multi-hop Q&A as a metric (one worked example only), guaranteed accuracy under adversarial noise, auth / multi-tenancy / UI polish.

**Hard constraints:**
- Extraction + Q&A on a **local open-weight LLM** (no frontier API in the product path).
- **24 GB unified memory (Apple M4):** models loaded/released sequentially, never concurrently.
- **15-minute clip cap** for v1.

## 4. Architecture — sequential pipeline of pure stages over disk artifacts

Each stage is a pure function `input file → one job → output file`. Stages never call each other; they communicate through **typed artifacts on disk**. The orchestrator runs them in order.

```
data/raw/<clip>.wav
  └─ enhance.py     → data/work/<clip>.clean.wav
       └─ diarize_asr.py → data/work/<clip>.transcript.json   (who said what, English)
            └─ extract.py    → data/work/<clip>.facts.json    (induced-ontology facts)
                 └─ graph.py      → Neo4j (nodes + edges)
                      └─ qa.py     → English answers (reads Neo4j)
```

**Why this pattern (the tradeoffs chosen):**
1. **Memory safety (dominant constraint).** Whisper + pyannote + the 9B LLM cannot co-reside in 24 GB. Disk artifacts let each stage load → run → fully release → exit before the next loads. In-memory pipelining would OOM.
2. **Auditability.** Every stage output is a human-readable file; you can see exactly where an error enters (e.g. a mangled Hindi word in `transcript.json`) instead of guessing. This is what makes the SNR experiment measurable — re-run one stage, diff the artifact.
3. **Resumability & fast iteration.** A bug in `extract.py` re-runs only that stage against the cached `transcript.json` — no re-transcription. Each stage is independently CLI-runnable.
4. **Testability & isolation.** One job, typed I/O per stage → testable with tiny fixtures, reasoned about alone.

**Design patterns:**
- **Data contracts (typed schemas):** every artifact shape is one Pydantic model in `contracts.py`. The schemas *are* the inter-stage interfaces; mismatches fail loudly at the boundary.
- **Thin uniform stage interface:** every stage is `run(clip_name) -> writes artifact`; the orchestrator is a list of stages. Adding/reordering/stubbing a stage is a one-line change.

**Explicitly NOT used (YAGNI):** no streaming, async, message queue, or DAG framework (Airflow/Prefect); **no LangChain/LangGraph** (linear deterministic flow needs none, and in-memory framework state fights the memory model; hand-written prompts + Pydantic + the Neo4j driver are more legible and defensible). Noted in the design note as the scaling path: LangGraph becomes worth it only if Q&A grows into agentic multi-hop retrieval.

## 5. Data contracts (`contracts.py`)

```python
# Transcript (diarize_asr.py output) — merge of "who" (diarization) + "what" (ASR)
class Word(BaseModel):
    text: str; start: float; end: float; speaker: str
class Utterance(BaseModel):
    speaker: str            # anonymous label e.g. "SPEAKER_01"
    text: str               # English
    start: float; end: float
    words: list[Word]
class Transcript(BaseModel):
    clip: str
    snr: str | None         # "10dB" — set on the eval path, None on the product path
    utterances: list[Utterance]

# Facts (extract.py output) — entities are FIRST-CLASS nodes with stable IDs (multi-hop-ready)
class Entity(BaseModel):
    id: str                 # "entity:pms"  (stable dedupe key)
    label: str              # structural backbone (fixed): Speaker | Statement | Entity | Claim | Attribute
    type: str               # INDUCED open vocabulary: "FinancialProduct", "RegulatoryComparison", ...
    name: str
    attrs: dict
class Fact(BaseModel):
    subject_id: str
    relation: str           # induced open vocabulary: WORKS_AT, HAS_MIN_INVESTMENT, ALLOWS, CONTRASTS_WITH, ...
    object_id: str
    statement: str          # MANDATORY — the exact source utterance this came from (grounding)
    speaker: str
    confidence: float       # LLM self-rated 0–1 — coarse secondary filter only
class FactSet(BaseModel):
    clip: str
    entities: list[Entity]
    facts: list[Fact]
```

Each `Fact` carries its **source statement + speaker** so every answer is traceable to *"X said this at 0:42."*

## 6. Induced ontology (option c + optional proposal pass)

- **Stable structural backbone** = Neo4j *labels*: `:Speaker`, `:Statement`, `:Entity`, `:Claim`, `:Attribute`.
- **Open vocabulary** = the fine `type` (on entities) and `relation` (on edges) are free strings the LLM fills per conversation (`Entity {type:"FinancialProduct", name:"PMS"}`, `Claim {type:"RegulatoryComparison"}`).
- **Proposal pass — ON by default, with base-schema fallback** (`ontology.py`): pass 1 — LLM proposes a small candidate type/relation vocabulary from the transcript; pass 2 — extraction is biased toward that vocabulary for consistency. If induction misfires on a given clip (empty/degenerate proposal, or a hard/noisy transcript), extraction **falls back to the fixed base backbone types** so it can never produce a broken graph. Induction enriches; the base schema guarantees a floor.

Rationale: the vocabulary is genuinely induced from whatever audio arrives, but the graph *shape* is stable — which keeps text-to-Cypher tractable and provenance uniform. Fully-open extraction (no backbone) makes Q&A fight a different schema every run.

## 7. LLM integration (`llm.py`) — runner-agnostic

- Talk to a **local OpenAI-compatible endpoint** via the standard OpenAI client. `base_url`, `model`, and `embed_model` live in `config.yaml`. Default: LM Studio `http://localhost:1234/v1`, model `qwen/qwen3.5-9b`; swappable to Ollama without touching pipeline code.
- **Extraction** uses **JSON-schema / structured-output enforcement** (derived from the Pydantic models) so facts return as clean, valid JSON.
- **Embeddings** (`nomic-embed-text`, same endpoint) power the Q&A semantic fallback and cross-chunk entity matching — no extra infra.

## 8. Extraction (`extract.py`) — chunk → extract → consolidate

1. **Chunk — speaker turn is the HARD boundary; ~1800 tokens is a SOFT target.** Accumulate
   whole speaker turns until near the target, then cut at the **next** turn boundary — **never
   mid-turn**. Chunks flex (~1600–2000) to land on clean boundaries. **Oversized single turn:**
   if one turn alone exceeds the target, keep it intact as one over-target chunk (attribution
   beats hitting the size; the 16K context absorbs it). **Length is measured with `tiktoken`
   (`cl100k_base`)** — a pure-Python, torch-free *ruler* only (not exact qwen counts; chunk size
   is a tunable heuristic in `config.yaml`). **Overlap:** prepend the previous chunk's last turn
   as **read-only context**, explicitly marked context-only in the prompt so facts are **not**
   extracted from or attributed to the overlap turn (prevents a Q→A fact being lost at a cut);
   consolidation dedups any repeats.
2. **Extract per chunk** against the induced schema, structured-output enforced; every fact must
   cite its **source statement**. Facts below the **confidence threshold (~0.6, precision-biased)**
   are dropped — a coarse secondary filter on top of mandatory grounding.
3. **Consolidate — entity resolution is precision-biased.** Primary merge is **exact
   normalized-name / alias match** (lowercase, strip, collapse whitespace) — safe and
   high-precision. Embedding similarity is a **fallback only for non-matching names**, gated at
   **cosine ≥ 0.85 AND same backbone label + induced `type`**. **When uncertain, do NOT merge:**
   a duplicate node is a minor cosmetic issue, but wrongly merging two distinct entities (e.g.
   PMS vs AIF) would break the comparison demo — so the **same-type guard is mandatory** and the
   threshold stays high. **Stable ID = `slug(canonical_name)`.** Embeddings come from the
   config `embed_model` over the OpenAI-compatible endpoint (torch-free main env). Facts are
   deduped on `(subject_id, relation, object_id)`.
   - **Relation canonicalization** mirrors entity resolution so near-synonym relations don't
     fragment comparison edges: lexical-normalize (lowercase, expand common abbreviations e.g.
     `min`→`minimum`, drop low-content words e.g. `amount`, snake_case) → exact match against the
     induced relation vocab → embedding fallback (precision-gated) for the rest. **Acceptance:**
     `"min investment"` and `"minimum investment amount"` must map to the **same** relation type.

## 9. Graph build (`graph.py`)
Upsert the `FactSet` into Neo4j via the official `neo4j` Python driver (torch-free; creds from
`.env` — `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD`/`NEO4J_DATABASE=neo4j`; local Neo4j 5.26
Desktop instance). **Parameterized Cypher only** (no string interpolation). All writes are
idempotent `MERGE`s keyed by stable ID — re-running the pipeline never duplicates nodes/edges.

**Concrete schema mapping:**
- **`:Statement`** node per source utterance — `{id, text, speaker, clip, start, end}`. The
  grounding anchor every fact traces back to.
- **`:Speaker`** node per speaker — `(:Speaker)-[:SAID]->(:Statement)`.
- **`:Entity` / `:Claim` / `:Attribute`** nodes (backbone label) — `{id, name, type, attrs…}`,
  where `type` is the open induced subtype.
- **Each `Fact`** → a relationship `(subject)-[:REL {confidence, speaker, source_statement_id}]->(object)`,
  where `REL` is the canonicalized induced `relation` sanitized to a valid Neo4j type
  (e.g. `HAS_MINIMUM_INVESTMENT`). Grounding is stored as **`source_statement_id` on the edge**
  (single-hop); the `:Statement` node stays queryable so every edge traces to who said it.
  *(Reify grounding as a separate `(:Statement)-[:SUPPORTS]->` node only later, if multi-hop needs it.)*
- **Phase 3 wiring note:** Q&A reads `source_statement_id` off the answering edge and joins back to
  the `:Statement` node to surface the verbatim quote — that's the grounding-to-UI path.

Neo4j Browser (`localhost:7474`) gives the live, clickable graph for the demo. **Phase 2 bar:** a
populated, browsable, traceable graph — don't gold-plate the schema at the expense of the
end-to-end run.

## 10. Q&A (`qa.py`) — schema-aware text-to-Cypher with fallback
1. **Introspect the live Neo4j schema** (labels, relationship types, property keys) and inject it into the Cypher-generation prompt — Q&A adapts to whatever ontology was induced.
2. LLM generates Cypher → **validate via `EXPLAIN`** → on error, **retry** with the error fed back.
3. Run the query; LLM composes an English answer **grounded in the returned rows** (with provenance).
4. **Embedding fallback**: if no clean Cypher resolves, retrieve relevant statements by semantic similarity and answer from those.
5. **Scope:** single-hop is the v1 hero path; the graph is multi-hop-ready and one worked multi-hop example (the PMS-vs-MF *regulation comparison*) is demonstrated.

## 11. Robustness & graceful degradation
- **Source-grounding is the load-bearing anti-hallucination guarantee** — every fact must cite the transcript statement it came from; structurally prevents fabrication and makes facts auditable.
- **Confidence threshold** (configurable, **default ~0.6, precision-biased**) is a **coarse secondary filter** on top of mandatory grounding — not treated as a calibrated probability (models are often miscalibrated). Lives in `config.yaml`; bias toward dropping shaky facts over keeping them.
- **Precision over recall by design:** on hard/noisy audio the system yields fewer, lower-confidence, source-backed facts rather than a hallucinated graph. In the SNR experiment this degradation is **reported as intended behavior** (the threshold rejecting low-confidence facts), not silent failure.
- No hardcoded speaker count (assume 2–4), domain, or format; **denoise always runs** (real input noise ≠ our synthetic mix).

## 12. Evaluation harness (`evaluate.py`) — controlled-SNR, three curves
`add_noise.py` mixes controlled noise at known SNRs over the clean baseline (a true single-variable experiment). At each SNR, measure the **whole chain** and plot **three curves on one SNR axis**:
1. **Transcript similarity** vs the reference transcript — the *ceiling* (if ASR degrades, extraction can't recover facts that aren't in the transcript).
2. **Fact-recall** vs ground-truth facts — the **hero metric**.
3. **Q&A correctness** vs the answer key — the user-facing outcome.

The **gaps between curves** are the finding: facts robust while transcript degrades = "facts survive rough words"; falling together = front-end bottleneck.

**Ground truth** (eval oracle, **never** the product path): scoped to the demo clip, **fact-level only** (~15–25 key facts + the demo-question answers). Built with the **best available oracle — a frontier model is permitted here** (a more accurate oracle = a more trustworthy metric; a weak tool would inject errors into the very thing we grade against) — to draft a clean transcript + fact list, then a **mandatory ~15-min human verification** pass correcting facts and spot-checking numbers/names against the audio (so we're never grading one model against another unchecked). Speaker noted only where attribution changes the fact. **Non-negotiable firewall:** all oracle/ground-truth code lives strictly in the eval path (`scripts/make_ground_truth.py`, `evaluate.py`) and is **never importable by the product path** — enforces the G4 product/eval separation.

## 13. Demo question set (tagged; draft answers from the reference transcript, pending verification)

**Single-hop (v1 hero metric):**
1. Minimum investment for a PMS? → ₹50 lakh
2. Minimum investment for an AIF? → ₹1 crore
3. Which firm offers both PMS and mutual funds? → White Oak
4. What did the expert say about transparency in a PMS? → cuts both ways; you see every transaction, which can be stressful
5. How are fees structured differently in a PMS? → performance/alpha-linked fees allowed
6. What happened in March 2020 per the speaker? → investors panicked and pulled money
7. India's equity mutual fund corpus mentioned? → ₹20–21 lakh crore
8. Does corpus size determine whether to buy a PMS? → no; it's about fit/engagement

**Multi-hop (one worked example — graph pays off):**
9. How does a PMS differ from a mutual fund in regulation? → PMS/AIF light-touch (more manager latitude, performance fees); MF tightly defined. *(traverses PMS ∩ MF attributes on the regulation dimension)*

## 14. Frontend & API architecture (the demo surface)
The demo UI is the approved **Atyx dark-editorial design** (`Ui-design/Atyx Convo-KG.html` prototype): Cormorant Garamond display + Inter UI + JetBrains Mono code, near-black canvas, gold accent. The prototype is a self-contained React/D3 bundle with **mock data + simulated timing**; we **rebuild it as clean, no-build source** and wire it to the real pipeline.

**Stack:** vanilla JS + **D3** (graph + SNR curves), a single static page **served by a thin FastAPI backend** (no Node toolchain). Fonts vendored locally for offline reproducibility.

**Two surfaces** (tabs): **Console** (clip picker · Run live · vertical pipeline timeline · streaming transcript · interactive knowledge graph · Ask Atyx) and **Experiment** (3-curve SNR chart + facts/WER table).

**FastAPI endpoints (the contract):**
- `POST /api/run` — start the pipeline on a selected/uploaded clip → `{run_id}`.
- `GET /api/run/{run_id}/stream` — **SSE** stream of real progress: `stage` (index, status, elapsed), `transcript_line` ({speaker, t, text}), `graph_node`/`graph_edge` as extraction emits them, `done`/`error`.
- `POST /api/ask` — `{question}` → `{answer, explanation, source:{speaker,t}, cypher, node_ids:[subgraph], src_line}`. Free text → real schema-aware text-to-Cypher; the prototype's 3 questions become **suggested prompts**.
- `GET /api/experiment` — `{snr[], sim[], recall[], qa[], table[]}` from `evaluate.py` artifacts.
- `GET /api/graph` — current graph snapshot (nodes/edges) for initial render.
- Static mount serves the frontend.

**Frontend data model** (mirrors `contracts.py`): transcript line = `Utterance`; graph node = `Entity` (induced `type` → node size/color via a small mapping with a default for unknown types); edge = `Fact.relation`; a Q&A message carries answer + cypher + source + highlighted `node_ids`. Real graph uses a **D3 force layout** (the prototype's hand-placed coords are dropped).

**New backend requirement this surfaces:** `qa.py` must return, per answer, the **subgraph node IDs used** and the **source statement(s)** — already implied by our "full explainability" commitment; now contractually required by the UI.

**Honesty notes:** the prototype fakes an ~8.8 s run; **real runs are slower and messier**. The UI must handle **variable timing, stage errors, and graceful degradation** (fewer graph nodes / lower-confidence facts on noisy audio) — the streaming model already fits this. Stage sub-labels show the **real stack** (DeepFilterNet · pyannote 3.x · mlx-whisper translate · qwen3.5-9b via LM Studio · Neo4j).

**Third entrypoint:** `serve` — launch FastAPI + frontend for the live demo (alongside `run` and `eval`).

## 15. Tech stack & environment
- **Python 3.12 pinned via `uv`** (3.14 has no ML wheels). Lockfile for reproducibility.
- **Audio ML runs in two isolated, exact-pinned venvs** (subprocess workers; the audio libs have irreconcilable `torchaudio` needs):
  - `.venv-asr` (`requirements-asr.txt`): **ASR** mlx-whisper (Metal) `task="translate"` (Hinglish→English) · **Diarization** pyannote 3.3.2 · torch 2.2.2 / torchaudio 2.2.2 · transformers 4.40 · huggingface_hub 0.25.2 (HF token). *(WhisperX is installed in this venv but no longer used in the live path — its forced-alignment is language-matched and incompatible with translate-to-English on Hinglish; see design_note.md §ASR.)*
  - `.venv-denoise` (`requirements-denoise.txt`): **Denoise** DeepFilterNet 0.5.6 · torch 2.0.1 / torchaudio 2.0.2.
- **LLM:** `qwen/qwen3.5-9b` via LM Studio OpenAI-compatible endpoint (runner-agnostic). **Embeddings:** nomic-embed (local).
- **Graph:** Neo4j Community via Docker. **Driver:** neo4j-python.
- **Main-env deps (no torch):** `openai`, `pydantic`, `neo4j`, `soundfile`/`librosa`, `numpy`, `python-dotenv`.
- **API/UI:** `fastapi` + `uvicorn` (SSE); frontend = static vanilla JS + **D3** (vendored), fonts vendored (Cormorant Garamond, Inter, JetBrains Mono). No Node build.

## 16. Repo structure
```
pyproject.toml (main, torch-free) · requirements-asr.txt · requirements-denoise.txt
src/
  contracts.py · config.py · llm.py · evaltools.py · asr_merge.py (pure merge) ·
  enhance.py (->subprocess .venv-denoise) · diarize_asr.py (->subprocess .venv-asr) ·
  ontology.py · extract.py · graph.py · qa.py ·
  pipeline.py (run <audio>) · evaluate.py (eval <audio> <ref>) ·
  api.py (FastAPI: /api/run·stream·ask·experiment·graph + serves frontend/)
scripts/ prep_clips.py · add_noise.py · make_ground_truth.py ·
  denoise_worker.py (.venv-denoise) · asr_worker.py (.venv-asr, mlx-whisper translate + pyannote)
frontend/ index.html · app.js · styles.css · assets/fonts/   (rebuilt from the prototype)
notebooks/demo.ipynb · design_note.md · Ui-design/Atyx Convo-KG.html (design reference)
data/{raw,noisy,ground_truth,work}/   (media + noisy/work gitignored)
.venv (main) · .venv-asr · .venv-denoise   (all gitignored; built per README)
```
Three entrypoints: **`run <audio>`** = product path (any clip → graph → Q&A, no reference); **`eval <audio> <ref>`** = SNR sweep → three curves; **`serve`** = FastAPI + frontend for the live demo.

## 17. Data assets
- **PMS-vs-MF (~10 min)** — rich demo clip; reference transcript available (clean, semantic, not verbatim).
- **sample2 (48 s, heavier Hindi)** — generality + translation stress test; reference transcript available.
- Source media is gitignored (exceeds GitHub limits); README documents sourcing.

## 18. Build order / milestones
1. **Env + infra**: uv Python 3.12, deps, Neo4j up, LM Studio reachable, `config.yaml`, `contracts.py`.
2. **Audio → English transcript** (denoise → diarize → ASR+translate); validate vs reference on `dev`/sample2, multiple SNRs. *(hardest; do first)*
3. **Transcript → graph** (induced ontology, schema-enforced extraction, chunk+consolidate, Neo4j upsert).
4. **Graph → answers** (schema-aware text-to-Cypher, single-hop; one worked multi-hop; returns subgraph node IDs + source).
5. **Eval harness** (SNR sweep → three curves) + ground-truth build.
6. **Frontend + API**: FastAPI endpoints + SSE; rebuild the Atyx UI as vanilla JS/D3 from the prototype; wire Run live / Ask / Experiment to real streamed data.
7. **Package** (README, demo walkthrough, design note).

## 19. Risks, what's stubbed, scaling path
- **Biggest risk:** Hinglish ASR on noisy multi-speaker audio (esp. rapid interjections). Mitigation: tune denoise + diarization; track accuracy vs SNR; report honestly.
- **Stubbed/limited in v1:** > 15-min clips (chunk-and-merge noted), multi-hop Q&A (one example), exhaustive ground truth (scoped to demo clip).
- **Scaling path:** chunk-and-merge for long audio; agentic multi-hop retrieval (LangGraph) if Q&A grows; streaming ingestion; richer entity resolution.

## 20. Resolved defaults (tune empirically; all in `config.yaml`)
- **Ontology proposal pass:** ON by default, with base-schema fallback (can't produce a broken graph).
- **Chunk size:** ~1500–2000 tokens with overlap (config value).
- **Confidence threshold:** ~0.6, precision-biased (config value).
- **Ground-truth oracle:** best available, frontier permitted; strictly eval-path, never importable by product; mandatory human-verify.

All four are starting baselines, not frozen — each is a config value so tuning is a one-line change, never a code change.
