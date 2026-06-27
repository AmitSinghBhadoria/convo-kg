# Atyx Convo-KG

From recorded multi-party **Hinglish audio** → speaker-attributed **English transcript** → an
induced-ontology **knowledge graph** (Neo4j) → natural-language **Q&A**, running fact extraction
and Q&A on a **local open-weight LLM**. Built to take arbitrary, noisy real-world audio.

> Status: in development. See `docs/superpowers/specs/` (design) and `docs/superpowers/plans/` (build plans).

## Environment setup

Because the audio ML libraries have **mutually incompatible `torchaudio` requirements**, the heavy
audio stages run in their own pinned virtualenvs, invoked as subprocesses (consistent with the
pipeline's disk-artifact stage isolation). Three environments:

| Env | Purpose | Python | Key pins |
|---|---|---|---|
| `.venv` (main) | orchestration, contracts, LLM client, graph, Q&A, API — **no torch** | 3.12 | pydantic, openai, fastapi, neo4j, soundfile, librosa |
| `.venv-asr` | ASR (mlx-whisper translate) + diarization (pyannote) | 3.12 | **torch 2.2.2 · torchaudio 2.2.2 · transformers 4.40.0 · huggingface_hub 0.25.2 · whisperx 3.3.1 · pyannote.audio 3.3.2 · numpy 1.26.4** |
| `.venv-denoise` | DeepFilterNet speech enhancement | 3.11 | **torch 2.0.1 · torchaudio 2.0.2 · deepfilternet 0.5.6** |

Why pinned: the newest `torchaudio` (2.11) removed `AudioMetaData`, breaking pyannote/WhisperX;
DeepFilterNet conversely needs the *old* `torchaudio.backend`. The pins above are a co-tested set,
verified to import together and load the models. **Do not unpin** without re-verifying.

```bash
# main env
uv venv --python 3.12 && uv sync

# isolated audio stack (ASR + diarization)
uv venv .venv-asr --python 3.12
uv pip install --python .venv-asr -r requirements-asr.txt

# isolated denoise stack (needs the Rust toolchain — deepfilterlib builds from source)
uv venv .venv-denoise --python 3.11
uv pip install --python .venv-denoise -r requirements-denoise.txt
```

### Secrets
Create `.env` (gitignored) with a Hugging Face read token (accept the gated terms on
`pyannote/speaker-diarization-3.1` **and** `pyannote/segmentation-3.0` first):

```
HF_TOKEN=hf_xxxxxxxx
```

### Local LLM
Extraction + Q&A talk to an OpenAI-compatible endpoint (LM Studio default
`http://localhost:1234/v1`, model `qwen/qwen3.5-9b`) — configured in `config.yaml`.

**Disable Reasoning/Thinking for the model in LM Studio.** qwen3.5-9b is a
thinking model; with reasoning on it routes structured JSON to
`reasoning_content` and leaves `content` empty. We need deterministic JSON for
fact extraction and Cypher generation, so turn thinking off (LM Studio →
model settings → Reasoning → off). The client has a `content or
reasoning_content` fallback as a safety net, but thinking-off is the supported
path.

### Graph database (Neo4j)
The knowledge graph is stored in a local **Neo4j 5.26** instance (Neo4j Desktop —
create a local instance named `atyx`, set a password, Start it). Add the
connection details to `.env`:

```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your instance password>
NEO4J_DATABASE=neo4j
```

The default `neo4j` database is all that's needed. Browse the graph at
`http://localhost:7474` (e.g. `MATCH (n) RETURN n LIMIT 100`).

## Running the pipeline

Stages are sequential and pass typed artifacts through `data/work/`. Activate the
main venv first: `source .venv/bin/activate`.

```bash
# Phase 1 — audio -> speaker-attributed English transcript
python -m src.enhance pms               # denoise  -> data/work/pms.clean.wav
python -m src.diarize_asr pms           # diarize + translate -> data/work/pms.transcript.json

# Phase 2 — transcript -> knowledge graph (needs LM Studio + Neo4j running)
python -m src.extract pms               # induce ontology + extract facts -> data/work/pms.facts.json
python -m src.graph pms                 # idempotent upsert into Neo4j

# Phase 3 — Ask Atyx (needs LM Studio + Neo4j running)
python -m src.qa "What did they say about transparency in a PMS?"   # -> Cypher single-hop
python -m src.qa "How does a PMS differ from a mutual fund?"        # -> statement-grounded fallback
python -m src.qa "What is the capital of France?"                   # -> declines (no-hallucination floor)
```

The demo clip is **`pms`** — a real ~10-minute, 4-speaker Hinglish conversation about
PMS / AIF / mutual funds. `<clip>` is a stem under `data/raw/` / `data/work/`. Re-running
`src.graph` is idempotent (MERGE on stable ids). Every fact edge stores a
`source_statement_id` that traces back to the `:Statement` node it came from.

> **On extraction quality:** the audio spine (denoise → diarize → English ASR) and the
> statement-grounded Q&A are solid on the real conversation. LLM **fact extraction** on
> real noisy code-mixed Hinglish is the measured bottleneck (a local ~9B ceiling, not the
> pipeline) — see `design_note.md` § *Measured capability boundary* for the full evidence
> and the fix path.
