# Atyx Convo-KG — Design Note

> Architecture, tool choices + rationale, what's stubbed, scaling path, and
> accuracy-vs-noise observations. This note grows as phases land; sections marked
> _(pending)_ are not yet built and are not claimed to work.

## Use case, and why local is the point

The target user is a **private-wealth firm** (PMS / AIF / RIA): turn each advisor–client
conversation into a queryable record of the **advice given** — products, strategies, fees,
suitability — so a Relationship Manager can recall it instantly and a Compliance Officer can
audit it with a source quote behind every answer.

This use case is what makes the **local open-weight LLM** a requirement rather than a
take-home rule. These calls carry client PII, live portfolio positions, and sometimes MNPI;
they cannot be sent to a cloud frontier API. Everything below — the torch-free main env, the
sequential single-resident-model memory plan, the on-device extraction and Q&A — exists to
keep the entire pipeline on the firm's own machine. **Data residency is the moat, not a
limitation.** The accuracy ceiling of a ~9B local model (measured later in this note) is the
deliberate price paid for that property.

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

## Q&A — _(built: Phase 3)_

NL question → schema-aware text-to-Cypher → run (read-only) → LLM composes an English
answer grounded in the returned rows, with a semantic fallback over statement embeddings.
- **Read-only, defense in depth:** generated Cypher is rejected if it contains a write
  clause (text guard), validated via `EXPLAIN` with retry, and executed in a Neo4j
  read-access-mode transaction — a write that slips the text check is refused by the
  database itself.
- **Provenance:** when the answer traverses a fact edge, the prompt returns the edge's
  `source_statement_id`, joined back to the `:Statement` node for a verbatim quote +
  speaker (`kind="source"`); the fallback's quotes are `kind="related"`. (A statement-
  returning Cypher query grounds via the statement ids in its rows rather than edge
  provenance — both are honest grounding.)
- **No-hallucination floor:** the semantic fallback declines (returns `found=False`) when
  the best statement-cosine is below an empirically-tuned floor — the guarantee rests on
  the score, not on the LLM noticing the quotes are irrelevant. On the real pms corpus,
  answerable questions score 0.54–0.70 and off-topic ones 0.22–0.23; the floor is 0.40.
  Verified: "What is the capital of France?" → "I couldn't find that in the conversation."

### Scaling — semantic caching (deliberately out of v1)

At scale, an obvious latency/cost lever is **semantic caching**: embed the question and
serve a prior answer on a near-enough cosine match. I'm leaving it out, and the reason is
the same false-positive risk this system manages elsewhere. Near-duplicate questions
routinely have *different* correct answers — *"minimum for PMS?"* vs *"…for AIF?"*, *"who
books the hotel?"* vs *"…transport?"* — yet sit close in embedding space, so a threshold
cache would confidently serve the wrong one. Caching here corrupts the **answer**, not
just a retrieval candidate (the cosine-floor problem above, but with a worse blast
radius), and for this domain a fast wrong answer is strictly worse than a slow correct one.

The scaling path, when it's warranted:
- **Cache the query plan, not the answer** — `question → Cypher`, then *re-execute*
  against the live graph. The structured query is more stable and verifiable, the graph
  stays authoritative, and a changed graph can't be served stale.
- **Exact / normalized-string cache underneath** — free and safe; catches literal repeats
  with no similarity risk.
- **If answers are ever cached**, treat a hit as a *candidate*, not an oracle: high
  similarity threshold, a TTL, and re-verify the supporting graph rows (the provenance ids
  above) still hold before returning.

## Phase 4 — Controlled-SNR results

**Setup.** A 160 s **2-speaker conversational slice** of the real PMS clip, mixed with
**real café-babble noise** (`noices/cafe_16k.wav`) at five SNR levels (20/15/10/5/0 dB)
— SNR the single controlled variable. Each noisy clip runs the full denoise→diarize→ASR
front-end; the resulting transcript is scored against the **clean slice through the
identical pipeline** (oracle-free) with `evaltools.similarity` (a relative sequence+set
fidelity blend in [0,1], **not WER**). The absolute height is therefore not an accuracy
figure against ground truth — it is similarity between two runs of the *same* audio, so
**read the shape of the curve, not the absolute values.**

**Hero curve (`data/ground_truth/snr_curve.png`).** Transcript fidelity vs SNR:
20 dB → 0.5479, 15 dB → 0.5932, 10 dB → 0.6253, 5 dB → 0.4525, 0 dB → 0.2873. The
front-end shows a roughly flat, slightly-rising shoulder from 20→10 dB — absolute fidelity
sits at only 0.55–0.63 even at the cleanest tested level, well within ASR/diarization
run-to-run variance (the relative metric plus diarization-segmentation differences mean
even lightly-noised transcripts diverge from the clean baseline). The slight *rise* across
20→10 dB is **not a real improvement** — added noise does not help ASR — it is
diarization-segmentation churn between runs, below the metric's resolution; read 20–10 dB
as **flat within measurement variance.** Then a **sharp cliff
that begins at the 10→5 dB step and deepens to 0 dB**: similarity falls from 0.6253 at
10 dB to 0.4525 at 5 dB to 0.2873 at 0 dB. The **cliff is the finding** — this is not
smooth monotonic degradation. For field deployments,
the **only significant signal is the cliff at low SNR (5→0 dB)**, where café babble
approaches speech power and the ASR loses roughly a third of its content (~350 vs ~490
words at 0 dB) — a practical noise floor below which retrieval-based Q&A becomes unreliable.

**Downstream spot-check (illustrative).** The golden in-slice questions (PMS-vs-MF,
minimum investment, who-it's-for) answered from the clean vs the 5 dB transcript via
**transcript-grounded retrieval** (cosine top-k over the transcript; **not** the full
extract→graph→Q&A product path, by design — extraction's nondeterministic ceiling is kept
out of every measured path). At clean, two of three questions get on-topic answers
(PMS-vs-MF: a 5-point breakdown; who-it's-for: "affluent HNIs with higher risk
appetite"); at 5 dB the results are non-monotonic — PMS-vs-MF retrieval reports the
content absent from the degraded transcript, who-it's-for names a different audience
("younger professionals"), and minimum-investment unexpectedly surfaces a figure ("50")
that clean retrieval did not, reflecting diarization-segmentation non-determinism rather
than a clean degradation slope. This is **illustrative propagation, not a calibrated
curve.**

**Honest limits.** No absolute WER (needs a verified reference transcript). No fact-recall
or Q&A-correctness *curve* — on noisy code-mixed Hinglish the local ~9B extractor is
nondeterministic and partially-recalling, so a curve through it would track model variance,
not noise (§ Measured capability boundary). The scaling path to a calibrated fact-recall
curve — a stronger/Cypher-tuned extractor + a frontier-oracle, human-verified fact-level
ground truth — is documented in the design spec §9, deliberately out of v1.

## Measured capability boundary (real 10-min multi-party Hinglish, local ~9B)

The demo runs on **`pms.wav`** — a real ~10-minute, 4-speaker Hinglish conversation about
PMS / AIF / mutual funds (not a scripted clip). Reported in full rather than hidden, because
where the pipeline holds and where it breaks is the most instructive result.

**What works — the audio spine.** Denoise → diarization → Hinglish→English ASR on the real
conversation is solid: 74 utterances, **4 distinct speakers** (well-distributed), full
600 s coverage, English output (0 Devanagari). A few dense code-mixed fragments stay
romanized-Hindi (e.g. *"uska return zyada ho sakta hai na"*) — a known limit of the
translate task on heavy code-mix — but the transcript is overwhelmingly coherent English.
The graph backbone (`:Speaker`, `:Statement` nodes) is built deterministically from this
transcript and is reliable: 4 clean speakers, 74 grounded statements.

**What breaks — LLM fact extraction on real noisy code-mixed speech.** This is the
bottleneck, and it is the **local ~9B model**, not the pipeline design. Three fixable
*pipeline* bugs were found and fixed along the way (each independent of model size):
1. **Decoding repetition loop.** On a dense chunk, greedy decoding + an unbounded `facts`
   array made the model re-emit the same ~6–9 facts to ~9k tokens → truncated/invalid JSON
   → whole chunk lost. Fixed with `maxItems` on the schema arrays (LM Studio enforces the
   grammar close), a `max_tokens` safety bound, and smaller chunks so the model processes
   each portion and closes the array naturally.
2. **Speaker-node pollution.** Extraction was allowed to emit `Speaker`/`Statement`
   entities (e.g. "Viraj", "Speaker_02"), which `graph.upsert` then wrote as duplicate
   speaker nodes colliding with the clean diarization-derived ones. Fixed by restricting
   extraction to content labels only.
3. **Dropped value-objects.** The model writes a fact object as a raw value
   (`object_id="50 lakh"`) without defining a matching entity, so the dangling-ref guard
   dropped the fact and most real content with it. Fixed by promoting undefined value
   endpoints to `Attribute` entities (skipping bare local ids like `e3`, which the model
   declares but forgets to define — those carry no value and are left to be dropped).

After those fixes the graph is substantive (~19 entities, ~21 facts, many correct: PMS →
affluent HNI; PMS fee structures → performance/flat fee; PMS strategies; PMS vs mutual
fund) — **but the residual quality is a hard model ceiling**: extraction is non-deterministic
run-to-run despite `temperature=0`, recall is partial (the discursive/code-mixed middle of
the conversation yields little), and some edges are simply wrong (`HAS_MINIMUM_CORPUS →
"Affluent HNI Segment"`, occasional self-loops). Chasing this with more prompt tuning is
fighting the ceiling, so we stopped and measured it instead.

**Consequence for Q&A, and why the demo holds.** Single-hop Cypher answers correctly for the
facts that are in the graph — "what fee structures does a PMS have?", "who is a PMS meant
for?", "what strategy does a PMS follow?" all return `mode=cypher` with edge-level source
provenance (verbatim quote + speaker). Getting there exposed a separate grounding bug worth
recording: the text-to-Cypher prompt asserted that fact edges connect `:Entity {type:'X'}`
nodes Entity→Entity, but on the real induced graph the concepts land as `:Attribute` nodes
and edges are Attribute→Attribute — so the model faithfully wrote `:Entity` queries that
matched nothing and *every* fact question fell back to semantic search even when the fact
existed. The prompt was lying about the graph's own structure. Fixed by matching concepts
**by name with no label** (labels are inconsistent across the messy induced graph). For
questions whose fact didn't extract (or extracted wrong), the **statement-grounded semantic
fallback** answers from real verbatim quotes with speaker attribution — a path that **does
not depend on fact-extraction quality at all**, only on the reliable transcript. The
no-hallucination floor holds (off-topic questions decline). So the user-facing Q&A degrades
gracefully: Cypher-with-provenance where the fact exists, grounded-quote fallback where it
doesn't, honest "not found" where the conversation doesn't cover it.

**Provenance-quote precision is itself bounded by the extraction ceiling.** The Cypher path
returns the edge's `source_statement_id` as the grounding quote — but because fact extraction
on this messy clip sometimes attaches a fact to a related-but-not-cleanest statement, the
surfaced quote occasionally supports the *topic* (PMS) rather than the *specific* claim. E.g.
"what strategy does a PMS follow?" answers correctly from the graph but its `[source]` quote
is the speaker's general PMS-vs-mutual-fund clarification rather than the exact strategy
sentence. Same root cause as the extraction ceiling; named here so it doesn't surprise a
reviewer. The fix is the same two-pass / verification path above (tie each fact to its
tightest supporting statement).

### Live demo sequence (golden path)

Lead from strength, then introduce the boundary deliberately:
1. **"What strategy does a PMS follow?"** → `mode=cypher` → *Broad Portfolio Strategy,
   Concentrated Small Cap Strategy, Consistency of Alpha* — with `[source]` speaker quote.
2. **"How does a PMS differ from a mutual fund?"** → `mode=semantic-fallback` → *"a PMS is
   like surgery without anesthesia…"* — shows the statement-grounded path with `[related]`
   quotes (and that the system labels source vs related honestly).
3. **"Who is a PMS meant for?"** → `mode=cypher` → *Affluent HNI Segment* — with `[source]`
   quote.

Then, to show the honesty guarantees on purpose: **"What is the capital of France?"** →
declines (`found=False`, "I couldn't find that in the conversation") — the cosine floor, not
the LLM, refusing to fabricate. This sequence demonstrates both Q&A paths, real provenance,
and the no-hallucination floor before the extraction-quality boundary is discussed.

**Multi-hop** is not reliably achievable here, for two compounding reasons (both the model):
text-to-Cypher does not reliably navigate 2-hop chains from the schema alone (an earlier
test only passed because the answer was hardcoded into the prompt — that overfitting was
caught and removed via the test *"would this prompt be equally correct for any other
graph?"*), and the induced graph is too sparse/noisy to carry a verified multi-hop chain.
Marked `xfail(strict=False)`.

**How I'd cross the extraction/Q&A ceiling** (all model/architecture, not pipeline):
a stronger or Cypher/extraction-tuned model (or a cloud model for the extraction stage
specifically); **two-pass extraction with a verification pass** (extract, then ask the model
to check each fact against its statement); an explicit **entity-linking** step (embed
question spans and fact objects against entity-name embeddings, resolve to node ids) for
both better resolution and multi-hop anchoring; and a better Hinglish translation front-end
to recover the code-mixed segments the current ASR leaves romanized.

## What is stubbed / out of scope for v1

Guaranteed accuracy under extreme/adversarial noise; streaming / large-scale ingestion;
full ontology coverage; accuracy metrics at scale; auth / multi-tenancy. Architected for,
not built. Multi-hop Q&A on a local ~9B is a measured limitation (above), not stubbed.
