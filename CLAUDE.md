# Atyx Prototype — Conversational Knowledge Graph

> **Context handoff for Claude Code.** Goal: build a working prototype demo by **Mon 29 Jun 2026**.
> This is a founding-engineer take-home for Atyx (private-wealth / capital-markets, AI-first). Reviewer is **Debopam Bhattacherjee** — a systems researcher (ETH PhD, ex-MSR India), measurement-driven. He values honest scoping, controlled experiments, and clear reasoning over polish. Design decisions should be deliberate and defensible.

---

## 1. What we're building (one line)
From recorded **multi-party audio conversations**, extract facts into a **knowledge graph** and let a user ask **natural-language questions** about what was said — running on a **local open-weight LLM**, built to work on **real, noisy audio**.

## 2. Confirmed requirements
- **Input:** audio only — multi-party conversations, noisy / varied real-world environments.
- **Language:** Hinglish (Hindi-English code-mix) in; **all facts and answers in English only**.
- **Speaker attribution:** identify who said what (diarization).
- **Extraction:** facts, entities, relationships from the conversation.
- **Knowledge graph:** extracted facts stored in a queryable graph.
- **Query:** natural-language questions → English answers.
- **Hard constraint:** extraction and Q&A run on a **local open-weight LLM** (no frontier API).

## 3. Debopam's confirmed answers (from email)
1. **Domain:** my choice — "whatever is easier to demo." → *Pick deliberately and justify it.*
2. **Q&A depth:** **single-hop is fine**, but **design the graph so multi-hop is a natural extension later** (store real entities/relationships, not flattened text).
3. **Fact types:** **all** — commitments, decisions, numbers, named entities.

## 4. Scope for Monday
**In scope** — one complete working spine, end to end, on representative noisy audio:
`Audio → Speech enhancement (denoise) → Speaker separation (diarization) → Hinglish→English transcription → LLM fact extraction → Knowledge graph → NL Q&A (single-hop)`

**Out of scope for v1 (architect for, don't build):** guaranteed accuracy under extreme/adversarial noise or every environment, streaming / large-scale ingestion, full ontology coverage, accuracy metrics at scale, auth / multi-tenancy / UI polish.

## 5. Architecture / pipeline stages
1. **Speech enhancement** — suppress background noise before ASR.
2. **Diarization** — segment + label speakers (who said what).
3. **ASR + translation** — transcribe Hinglish, output **English** (transcribe-then-normalize, or Whisper translate task).
4. **Fact extraction** — local LLM reads speaker-attributed chunks, emits **structured JSON** (entities, relations, attributed statements) against the ontology; dedupe/merge entities.
5. **Graph build** — upsert nodes/edges into the graph DB.
6. **Q&A** — NL question → graph query (text-to-Cypher) → run → LLM composes English answer. Keyword/embedding fallback for questions that don't map to a clean query.

> **Memory note:** target machine is an **Apple M4, 24 GB unified memory**. Run stages **sequentially** (enhancement + ASR + diarization first, release, then load the LLM) to stay within memory. Batch, **not real-time**, for v1.

## 6. Proposed stack (candidates — benchmark on-device, document choices in the design note)
- **Local LLM runtime:** Ollama 0.19+ (MLX backend on Apple Silicon) or MLX-LM.
- **LLM (extraction + Q&A):** Qwen 3.5-class instruct, ~7–9B, 4-bit — strong at structured/JSON output, Cypher/tool generation, multilingual. Dense ~9B keeps memory headroom alongside ASR.
- **ASR:** Whisper large-v3 via `mlx-whisper` or `faster-whisper`; **WhisperX** for word alignment + diarization handoff. AI4Bharat / Indic ASR kept as a Hinglish fallback to evaluate.
- **Diarization:** `pyannote.audio` 3.x.
- **Denoise:** DeepFilterNet (or equivalent speech-enhancement model).
- **Graph DB:** Neo4j Community (local, Docker) — Cypher for queries.
- **Glue:** Python pipeline; thin CLI or notebook for the demo (minimal web UI optional).

## 7. Data & noise augmentation
- A **real sample clip** is being used as the base (multi-speaker Hinglish). Whatever the clip's actual content is, **shape the ontology and demo questions around it**.
- Noise is added at **controlled SNR levels** for reproducibility and a degradation curve, via `add_noise.py`:
  ```bash
  python add_noise.py --speech sample.wav --noise cafe.wav --snr 20 10 5 0
  # → noisy/sample_snr20dB.wav ... _snr0dB.wav  (16 kHz mono, ASR-ready)
  ```
  Prefer a **real ambient/babble noise clip** (recorded, or MUSAN/WHAM!) over synthetic.
- **Ground truth:** hand-label the correct facts/answers for the sample clip so Q&A correctness is verifiable in the demo. **Non-negotiable.**
- **Validate the noisy-audio → English-transcript link FIRST** — it's the hardest, accuracy-critical step. Everything downstream is more forgiving.

## 8. Knowledge-graph ontology (template — adapt to the actual clip)
*Example for a group trip-planning conversation; swap to match the clip's domain.*
- **Nodes:** Person, Statement, Topic, Decision, Organization/Vendor, Date, Expense/Number, Task/Commitment
- **Edges:** `SAID` (Person→Statement), `MENTIONS` (Statement→Topic/Org/Person), `DECIDED`/`AGREED` (Person→Decision), `COMMITTED_TO` (Person→Task), `HAS_COST` (Task→Expense), `ON` (Event→Date), `RELATES_TO` (Topic↔Topic)
- **Each Statement** retains: speaker, source clip, English text, timestamp.
- Keep entities as **first-class nodes** (not strings inside text) so single-hop today extends to multi-hop later.

## 9. Demo question set (template — single-hop, build the answer key alongside)
Map each to a graph query and a known-correct answer. Examples (trip domain):
1. Where are they going? 2. Which dates were finalized? 3. What's the budget / per-head cost? 4. Who's booking the hotel? 5. Who's arranging transport? 6. How much has each person paid? 7. What did they decide about the stay?

## 10. Build order / milestones
1. **Audio → clean English transcript** (denoise → diarize → ASR+translate). Validate against noisy audio at multiple SNRs. *(do first)*
2. **Transcript → graph** (define ontology, LLM structured extraction, upsert to Neo4j).
3. **Graph → answers** (text-to-Cypher Q&A, single-hop; one worked multi-hop example for the design note).
4. **Package** (reproducible repo + README, demo walkthrough, short design note).

## 11. Suggested repo structure
```
atyx-convo-kg/
  README.md
  requirements.txt
  data/
    raw/            # original sample clip(s)
    noisy/          # SNR-augmented versions
    ground_truth/   # hand-labelled facts + expected answers
  src/
    enhance.py      # denoise
    diarize_asr.py  # diarization + Hinglish->English transcription
    extract.py      # LLM fact extraction -> structured JSON
    graph.py        # build/query Neo4j
    qa.py           # NL question -> Cypher -> English answer
    pipeline.py     # end-to-end runner (sequential stages)
  scripts/
    add_noise.py    # SNR noise augmentation (already written)
  notebooks/
    demo.ipynb      # walkthrough for the live demo
  design_note.md    # architecture, choices + rationale, what's stubbed, scaling path
```

## 12. Key risks & principles
- **Biggest risk:** Hinglish ASR on noisy, multi-speaker audio. Tune denoise + diarization to feed it; track accuracy vs SNR.
- **Honesty bar:** only claim what actually works; in the design note, clearly mark what's stubbed and how it would scale. (Debopam respects "here's the limit and how I'd fix it" far more than overclaiming.)
- **Single-hop now, multi-hop-ready graph** — don't flatten facts into text blobs.
- **Reproducible** — anyone should be able to run the demo from the README.

## 13. Deliverables (Mon 29 Jun)
- GitHub repo + README (reproducible).
- Live walkthrough of the end-to-end pipeline.
- Short design note: architecture, model/tool choices + rationale, what's stubbed, scaling path, and the accuracy-vs-noise observations.

---

### Decisions log
- Input narrowed to **audio only** (dropped text path).
- Added a **denoise stage** up front (noisy environments are the actual test).
- Q&A scoped to **single-hop**, graph designed for multi-hop later.
- Stack kept **out of the client-facing scope doc**; lives here + in the design note, framed as benchmarked choices.
