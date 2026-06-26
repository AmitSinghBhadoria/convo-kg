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
| `.venv-asr` | ASR + diarization (WhisperX, pyannote, mlx-whisper) | 3.12 | **torch 2.2.2 · torchaudio 2.2.2 · transformers 4.40.0 · huggingface_hub 0.25.2 · whisperx 3.3.1 · pyannote.audio 3.3.2 · numpy 1.26.4** |
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
