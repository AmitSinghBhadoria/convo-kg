# Atyx Convo-KG — Design Note

> Architecture, tool choices + rationale, what's stubbed, scaling path, and
> accuracy-vs-noise observations. This note grows as phases land; sections marked
> _(pending)_ are not yet built and are not claimed to work.

## Pipeline (one line)

`Audio → Denoise (DeepFilterNet) → Diarization (pyannote) → Hinglish→English ASR
(mlx-whisper translate) → Induced-ontology fact extraction (qwen3.5-9b, local) →
Neo4j graph → single-hop NL Q&A`, with a controlled-SNR evaluation harness.

Stages are **pure, sequential, and communicate via typed disk artifacts** (Pydantic
v2 contracts in `src/contracts.py`). On a 24 GB M4, only one heavy model is resident at
a time: each stage loads → runs → fully releases → exits before the next loads.
Artifacts also make the pipeline **auditable** (you can see exactly which stage
introduced an error) and **resumable** (re-run one stage against cached inputs).

### Three-venv isolation

Heavy audio ML has irreconcilable `torchaudio` needs, so the main env is **torch-free**
and audio stages run in pinned, isolated venvs invoked as subprocess workers:

- `.venv` (main): Pydantic, OpenAI client → LM Studio, FastAPI, pytest. No torch.
- `.venv-asr`: mlx-whisper + pyannote (torch 2.2.2). Worker: `scripts/asr_worker.py`.
- `.venv-denoise`: DeepFilterNet (torch 2.0.1). Worker: `scripts/denoise_worker.py`.

Exact pins live in `requirements-asr.txt` / `requirements-denoise.txt`; rationale is in
the README. This is a deliberate cost: subprocess hand-offs over disk, in exchange for
each stack staying on a version set we verified imports together.

## ASR: why a single mlx-whisper translate path, and not WhisperX forced-alignment

**Decision.** The one live ASR path is **mlx-whisper with `task="translate"`** (Hinglish
audio → English text), with **segment-level** speaker attribution from pyannote turns
merged against Whisper's own word timestamps (`src/asr_merge.py`). We evaluated and
**dropped** a WhisperX "final path" that added wav2vec **forced-alignment**.

**What we tried and measured.** A WhisperX worker did `task="translate"` then aligned
the English output with the English (`language_code="en"`) wav2vec model, then diarized.
On the sample2 clip (47.6 s, Hindi-heavy):

| Metric              | WhisperX translate + en-align | mlx-whisper translate |
|---------------------|-------------------------------|-----------------------|
| English-only output | ❌ 41% Devanagari             | ✅ **0% Devanagari**  |
| Clip coverage       | ❌ 33% (0–15.9 s of 47.6 s)   | ✅ **100% (47.6 s)**  |
| Contract-valid      | yes                           | yes                   |

**Root cause — architectural, not a bug.** WhisperX forced-alignment uses a wav2vec
aligner that requires the transcript to be in the **same language as the audio**. The
translate task yields **English text** for **Hindi audio**; there is no English audio to
align the English words against. The aligner therefore drops the segments it cannot
align (the 67% coverage loss) and lets untranslated Devanagari pass through (the 41%).
"Translate to English" and "forced-align against the source audio" are fundamentally
incompatible for code-switched Hinglish input.

**Why segment-level grounding is sufficient.** Single-hop Q&A traces each answer to a
*statement* — speaker + segment + verbatim quote. That needs turn/segment boundaries,
not per-word audio offsets. mlx-whisper provides clean English segments with
speaker attribution; word timestamps (from Whisper itself, not wav2vec) remain available
inside each utterance if ever needed.

**Future work (the principled word-level path).** If word-level timing on translated
text is ever required, the correct redesign is **align-in-source-then-translate**:
(1) WhisperX transcribe Hindi + align in Hindi (valid — same language, real word
offsets), (2) diarize, (3) translate the aligned text to English carrying the
timestamps over. This is more build + verification than the deadline allows; it is noted
here, not implemented. WhisperX remains installed in `.venv-asr` (for the pinned pyannote
it pulls in) but is **not used** in the live path.

## Local LLM (extraction + Q&A)

Runner-agnostic client (`src/llm.py`) talks to any OpenAI-compatible endpoint
(`base_url`/`model` from `config.yaml`, env-overridable); default LM Studio,
`qwen/qwen3.5-9b`. qwen3.5 is a **thinking model** — reasoning must be **disabled** in
LM Studio so structured JSON lands in `content` (deterministic, faster); the client also
falls back to `reasoning_content` as a safety net. See README → Local LLM. Extraction
showed the local model returns schema-compliant, fully-grounded JSON on the first attempt
(sample2: 13 entities / 12 facts, 0 ungrounded) — no prompt/schema workarounds needed.

## Induced ontology + graph (Phase 2 — built)

`transcript.json → ontology proposal → chunk → per-chunk extraction → consolidation →
Neo4j`. The stable label backbone is `:Speaker/:Statement/:Entity/:Claim/:Attribute` with
an open `type`/`relation` vocabulary; entities are first-class nodes so single-hop extends
to multi-hop.

- **Chunking** (`chunking.py`): speaker turn is a HARD boundary, ~1800 tokens a SOFT target
  (tiktoken ruler); an oversized turn is kept intact; the previous turn rides along as
  read-only context so cross-boundary facts survive.
- **Ontology** (`ontology.py`): a proposal pass induces a small per-clip type/relation
  vocabulary, with a fixed `BASE_ONTOLOGY` fallback on an empty/degenerate/errored proposal.
- **Extraction** (`extract.py`): schema-enforced per-chunk JSON; every fact must cite the
  `statement_id` of its source turn. Consolidation is **precision-biased** — drop facts that
  are ungrounded (empty `statement_id`), below the confidence threshold (0.6), or reference
  an unresolved entity; then dedup.
- **Resolution** (`resolve.py`): entities merge by exact normalized name first, then an
  embedding fallback gated by **cosine ≥ 0.85 AND same label+type** — so PMS and AIF never
  collapse. Relations canonicalize (near-synonyms like "min investment" / "minimum investment
  amount" → one type) while the canonical vocabulary stays a fixed point.
- **Graph** (`graph.py`): idempotent `MERGE` upserts via the Neo4j driver; **parameterized
  Cypher only**, with node labels gated by a closed allowlist and relationship types by a
  charset-validated `safe_rel_type` — no LLM string is ever interpolated into a query.
  Grounding is a `source_statement_id` on each fact edge, joinable back to its `:Statement`.
- **End-to-end** (verified on sample2): a non-empty, fully source-traceable, idempotent
  graph (re-running the upsert does not change node counts).

## Q&A — _(pending: Phase 3)_

NL question → text-to-Cypher → run → LLM composes an English answer, with a
keyword/embedding fallback. Source-grounding (every fact cites its statement) is the
load-bearing anti-hallucination guarantee.

## Evaluation: accuracy vs noise — _(pending: Phase 4)_

Controlled-SNR harness with three curves: transcript-similarity (ceiling) → fact-recall
(hero) → Q&A-correctness (outcome). Ground truth is fact-level for the demo clip, built
with a frontier oracle in the eval path only (never importable by the product path), then
human-verified. Numbers go here once the harness runs.

## Measured capability boundary: single-hop generalizes, multi-hop does not (local ~9B)

This is the most instructive result, so it is reported in full rather than hidden.

**Single-hop text-to-Cypher generalizes genuinely.** Given only *live, data-driven*
schema grounding — sampled relationship directions, sampled entity names by type, the
live label set, and the property-graph rules (`:Entity {type:'X'}`, not `:X`; fact edges
connect Entity→Entity; `source_statement_id` is an edge property) — the local ~9B model
writes correct single-hop Cypher across rephrasings of a question, returning the right
rows with real edge-level provenance. Nothing about the question is encoded in the prompt;
the same grounding code produces correct grounding for any induced graph.

**Multi-hop does not — and the honest path to that finding matters.** An early version of
the multi-hop acceptance test passed, but only because the prompt contained worked-example
queries spelling out the exact `AssetClass-[HAS_STRATEGY]->WealthStrategy-[ACHIEVES_GOAL]->
FinancialGoal` chain that answered the demo question. That is the demo answer smuggled into
the prompt: the test proved the plumbing, not the model's reasoning. We caught it with a
single test — *would this prompt text be equally correct and helpful for a completely
different conversation's induced graph?* — and removed it. With the overfit gone, we then
tried a **generic** fix: sample the live graph's actual 2-hop connectivity *patterns*
(`MATCH (a)-[r1]->(b)-[r2]->(c) RETURN DISTINCT type-shapes`) and show those to the model.
That is legitimate (graph-derived, question-agnostic) and it helped — but only to **~3/5
across rephrasings**, below our "robust, not rehearsed" bar, so it was reverted rather than
shipped as a demo that works only on the rehearsed phrasing (which would be worse than no
multi-hop, because it looks like the overfit we just removed).

**Root cause is entity resolution, not path ignorance.** The residual failures are not the
model failing to find the 2-hop pattern; they are the model linking the *wrong* entity.
"Starting/running your own business" gets resolved to the `WealthStrategy` node ("Do your
own business") instead of the `AssetClass` node ("business ownership"), so it skips the
first hop entirely and then traverses a non-existent direct edge (0 rows). The graph
genuinely supports the chain (verified structurally); the bottleneck is question→entity
linking under paraphrase.

**How I'd cross it.** (1) An explicit entity-linking step — embed question spans against
entity-name embeddings, resolve to node ids, and hand the model resolved anchors instead of
free-text names; (2) guided query decomposition — ask the model for the hop sequence
(strategy?, then goal?) and assemble the Cypher deterministically; (3) a stronger / Cypher-
tuned model. The multi-hop test is marked `xfail(strict=False)` with this finding, so it
flips to green automatically if a future model clears it. Single-hop, the graph structure,
and provenance are all production-solid; multi-hop is a measured, named boundary with a
concrete fix path.

## What is stubbed / out of scope for v1

Guaranteed accuracy under extreme/adversarial noise; streaming / large-scale ingestion;
full ontology coverage; accuracy metrics at scale; auth / multi-tenancy. Architected for,
not built. Multi-hop Q&A on a local ~9B is a measured limitation (above), not stubbed.
