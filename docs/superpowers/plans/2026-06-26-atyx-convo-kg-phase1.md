# Atyx Convo-KG — Phase 1: Foundation + Audio→English Transcript Spine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project (uv env, config, typed contracts, runner-agnostic LLM client) and build the hardest stage end-to-end — noisy Hinglish audio → denoise → diarize → English ASR → one speaker-attributed `Transcript`, validated against the reference transcript.

**Architecture:** Sequential pipeline of pure stages communicating through typed artifacts on disk (memory-safe on 24 GB, auditable, resumable). Each stage = `run(clip) -> writes artifact`. Phase 1 delivers the `enhance` and `diarize_asr` stages plus the shared foundation (`config`, `contracts`, `llm`).

**Tech Stack:** **lean main env** (Python 3.12 via uv) = Pydantic v2, OpenAI client → LM Studio, pytest — **no torch**. Heavy audio ML runs in two **isolated, exact-pinned venvs** invoked as subprocess workers: `.venv-asr` (mlx-whisper + WhisperX + pyannote, torch 2.2.2) and `.venv-denoise` (DeepFilterNet, torch 2.0.1). Pins in `requirements-asr.txt` / `requirements-denoise.txt`; see README.

## Global Constraints
- **Python 3.12 pinned via uv** — 3.14 has no ML wheels. Never use system Python.
- **Local open-weight LLM only in product path**; runner-agnostic via config: `base_url=http://localhost:1234/v1`, `model=qwen/qwen3.5-9b`, `embed_model=text-embedding-nomic-embed-text-v2-moe`.
- **24 GB unified memory:** load one heavy model at a time, release before the next. Never concurrent.
- **15-minute clip cap.**
- **Induced ontology** later uses stable labels `:Speaker/:Statement/:Entity/:Claim/:Attribute` + open `type`/`relation` — contracts must keep `label` (fixed) separate from `type` (induced).
- **Every fact is source-grounded**; confidence threshold default `0.6`, precision-biased — both config values.
- **Commit every task** with a descriptive message; push to `origin/main`. Co-author trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **HF_TOKEN** read from `.env` (gitignored) for pyannote.
- **Three-venv isolation (decided during build):** main `.venv` is **torch-free**; audio stages run in pinned isolated venvs — `.venv-asr` (ASR+diarization) and `.venv-denoise` (denoise) — invoked as subprocess workers, because pyannote/WhisperX vs DeepFilterNet have irreconcilable `torchaudio` needs. Pins: `requirements-asr.txt` / `requirements-denoise.txt`. Don't unpin without re-verifying.

## File Structure (Phase 1)

Three environments (see README): **`.venv` (main, torch-free)** · **`.venv-asr`** (pinned ASR+diarization) · **`.venv-denoise`** (pinned denoise). The audio venvs are invoked as subprocess workers.
```
pyproject.toml             # main env (lean: pydantic, openai, fastapi, soundfile, librosa, ...)
requirements-asr.txt       # .venv-asr exact pins (torch2.2.2, whisperx, pyannote, mlx-whisper)
requirements-denoise.txt   # .venv-denoise exact pins (torch2.0.1, deepfilternet)
config.yaml                # llm endpoint, model names, paths, thresholds, snr levels, clip cap
.env                       # HF_TOKEN (gitignored)
src/
  config.py · contracts.py · llm.py · evaltools.py
  asr_merge.py             # pure word->speaker merge (no torch; imports in both envs)
  enhance.py               # orchestrator -> subprocess(.venv-denoise / denoise_worker.py)
  diarize_asr.py           # orchestrator -> subprocess(.venv-asr / asr_worker.py)
scripts/
  prep_clips.py            # build data/raw/{pms,sample2,dev}.wav (16k mono)
  denoise_worker.py        # runs in .venv-denoise (DeepFilterNet)
  asr_worker.py            # runs in .venv-asr (mlx-whisper/whisperx + pyannote + merge)
  asr_worker_final.py      # runs in .venv-asr (WhisperX final pass)
tests/
  test_config.py · test_contracts.py · test_llm.py · test_enhance.py ·
  test_diarize_asr.py · test_evaltools.py
```

---

### Task 1: uv project + dependencies

**Files:**
- Create: `pyproject.toml`, `src/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Produces: an installable env (`uv sync`) with `python>=3.12,<3.13`; importable `src` package.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "atyx-convo-kg"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [   # main env is TORCH-FREE; audio ML lives in isolated .venv-asr/.venv-denoise
  "pydantic>=2.7",
  "pyyaml>=6.0",
  "openai>=1.40",
  "numpy>=1.26",
  "soundfile>=0.12",
  "librosa>=0.10",
  "python-dotenv>=1.0",
  "fastapi>=0.110",
  "uvicorn>=0.29",
]
[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
markers = ["integration: requires LM Studio / models / network"]
addopts = "-m 'not integration'"
```

- [ ] **Step 2: Pin Python and sync**

Run: `uv python install 3.12 && uv sync`
Expected: resolves and installs; creates the lean main `.venv`. The audio stacks are SEPARATE isolated venvs built from `requirements-asr.txt` / `requirements-denoise.txt` (see README) — not part of `uv sync`.

- [ ] **Step 3: Create package markers**

`src/__init__.py` and `tests/__init__.py` — empty files.

- [ ] **Step 4: Verify import**

Run: `uv run python -c "import src; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/__init__.py tests/__init__.py
git commit -m "build: uv project scaffold with pinned ML dependencies (py3.12)"
```

---

### Task 2: Typed config loader

**Files:**
- Create: `config.yaml`, `src/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path="config.yaml") -> Config`; `Config` is a Pydantic model with `.llm.base_url/model/embed_model`, `.paths.raw/work/...`, `.extract.chunk_tokens/overlap/confidence_threshold`, `.eval.snr_levels`, `.limits.max_minutes`. Env vars override (e.g. `LLM_BASE_URL`).

- [ ] **Step 1: Write the failing test** — `tests/test_config.py`

```python
from src.config import load_config

def test_defaults_load():
    c = load_config("config.yaml")
    assert c.llm.model == "qwen/qwen3.5-9b"
    assert c.limits.max_minutes == 15
    assert 0.0 < c.extract.confidence_threshold <= 1.0
    assert c.eval.snr_levels  # non-empty

def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://example/v1")
    c = load_config("config.yaml")
    assert c.llm.base_url == "http://example/v1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: src.config`.

- [ ] **Step 3: Write `config.yaml`**

```yaml
llm:
  base_url: http://localhost:1234/v1
  model: qwen/qwen3.5-9b
  embed_model: text-embedding-nomic-embed-text-v2-moe
paths:
  raw: data/raw
  work: data/work
  noisy: data/noisy
  ground_truth: data/ground_truth
extract:
  chunk_tokens: 1800
  overlap_tokens: 200
  confidence_threshold: 0.6
eval:
  snr_levels: [20, 15, 10, 5, 0]
limits:
  max_minutes: 15
```

- [ ] **Step 4: Write `src/config.py`**

```python
from pathlib import Path
import os, yaml
from pydantic import BaseModel

class LLMCfg(BaseModel):
    base_url: str
    model: str
    embed_model: str

class Paths(BaseModel):
    raw: str; work: str; noisy: str; ground_truth: str

class ExtractCfg(BaseModel):
    chunk_tokens: int; overlap_tokens: int; confidence_threshold: float

class EvalCfg(BaseModel):
    snr_levels: list[int]

class Limits(BaseModel):
    max_minutes: int

class Config(BaseModel):
    llm: LLMCfg; paths: Paths; extract: ExtractCfg; eval: EvalCfg; limits: Limits

def load_config(path: str = "config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text())
    if v := os.getenv("LLM_BASE_URL"): data["llm"]["base_url"] = v
    if v := os.getenv("LLM_MODEL"): data["llm"]["model"] = v
    return Config.model_validate(data)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/config.py tests/test_config.py
git commit -m "feat(config): typed yaml config loader with env override"
```

---

### Task 3: Typed data contracts

**Files:**
- Create: `src/contracts.py`, `tests/test_contracts.py`

**Interfaces:**
- Produces: `Word`, `Utterance`, `Transcript(clip:str, snr:str|None, utterances:list[Utterance])`, `Entity(id,label,type,name,attrs)`, `Fact(subject_id,relation,object_id,statement,speaker,confidence)`, `FactSet(clip,entities,facts)`. All Pydantic v2; JSON round-trip via `model_dump_json()` / `model_validate_json()`.

- [ ] **Step 1: Write the failing test** — `tests/test_contracts.py`

```python
from src.contracts import Word, Utterance, Transcript, Entity, Fact, FactSet

def test_transcript_roundtrip():
    u = Utterance(speaker="SPEAKER_01", text="PMS minimum is 50 lakh", start=8.0, end=11.2,
                  words=[Word(text="PMS", start=8.0, end=8.3, speaker="SPEAKER_01")])
    t = Transcript(clip="dev", snr=None, utterances=[u])
    again = Transcript.model_validate_json(t.model_dump_json())
    assert again.utterances[0].speaker == "SPEAKER_01"
    assert again.utterances[0].words[0].text == "PMS"

def test_fact_requires_grounding():
    f = Fact(subject_id="entity:pms", relation="HAS_MIN", object_id="amount:50l",
             statement="PMS minimum is 50 lakh", speaker="SPEAKER_01", confidence=0.9)
    assert f.statement and f.confidence == 0.9
    fs = FactSet(clip="dev", entities=[Entity(id="entity:pms", label="Entity",
                 type="FinancialProduct", name="PMS", attrs={})], facts=[f])
    assert FactSet.model_validate_json(fs.model_dump_json()).facts[0].relation == "HAS_MIN"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL — `ModuleNotFoundError: src.contracts`.

- [ ] **Step 3: Write `src/contracts.py`**

```python
from pydantic import BaseModel, Field

class Word(BaseModel):
    text: str; start: float; end: float; speaker: str

class Utterance(BaseModel):
    speaker: str; text: str; start: float; end: float
    words: list[Word] = Field(default_factory=list)

class Transcript(BaseModel):
    clip: str
    snr: str | None = None
    utterances: list[Utterance] = Field(default_factory=list)

class Entity(BaseModel):
    id: str               # stable dedupe key, e.g. "entity:pms"
    label: str            # fixed backbone: Speaker|Statement|Entity|Claim|Attribute
    type: str             # induced open vocabulary
    name: str
    attrs: dict = Field(default_factory=dict)

class Fact(BaseModel):
    subject_id: str
    relation: str         # induced open vocabulary
    object_id: str
    statement: str        # MANDATORY source grounding
    speaker: str
    confidence: float

class FactSet(BaseModel):
    clip: str
    entities: list[Entity] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/contracts.py tests/test_contracts.py
git commit -m "feat(contracts): pydantic data contracts for transcript and facts"
```

---

### Task 4: Runner-agnostic LLM client

**Files:**
- Create: `src/llm.py`, `tests/test_llm.py`

**Interfaces:**
- Consumes: `Config.llm` from Task 2.
- Produces: `LLM(cfg).chat_json(system:str, user:str, schema:dict) -> dict` (uses OpenAI `response_format` json_schema; validates+retries once on parse failure) and `LLM(cfg).embed(texts:list[str]) -> list[list[float]]`.

- [ ] **Step 1: Write the failing test** — `tests/test_llm.py`

```python
import pytest
from src.config import load_config
from src.llm import LLM

SCHEMA = {"type":"object","properties":{"answer":{"type":"string"}},
          "required":["answer"],"additionalProperties":False}

@pytest.mark.integration
def test_chat_json_against_lmstudio():
    llm = LLM(load_config("config.yaml").llm)
    out = llm.chat_json("Reply as JSON.", "Say the word 'pong' in the answer field.", SCHEMA)
    assert "answer" in out and isinstance(out["answer"], str)

@pytest.mark.integration
def test_embed_against_lmstudio():
    llm = LLM(load_config("config.yaml").llm)
    vecs = llm.embed(["hello", "world"])
    assert len(vecs) == 2 and len(vecs[0]) > 10
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_llm.py -v -m integration`
Expected: FAIL — `ModuleNotFoundError: src.llm`.

- [ ] **Step 3: Write `src/llm.py`**

```python
import json
from openai import OpenAI
from src.config import LLMCfg

class LLM:
    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key="lm-studio")

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        rf = {"type": "json_schema",
              "json_schema": {"name": "out", "strict": True, "schema": schema}}
        for attempt in range(2):
            r = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format=rf, temperature=0)
            try:
                return json.loads(r.choices[0].message.content)
            except json.JSONDecodeError:
                if attempt == 1: raise
                user = user + "\n\nReturn ONLY valid JSON matching the schema."

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.embeddings.create(model=self.cfg.embed_model, input=texts)
        return [d.embedding for d in r.data]
```

- [ ] **Step 4: Run tests to verify pass (needs LM Studio running — Blocker #3)**

Run: `uv run pytest tests/test_llm.py -v -m integration`
Expected: PASS (2 passed). If LM Studio is off, the test errors with a connection refused — start the server and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat(llm): OpenAI-compatible client with json-schema chat + embeddings"
```

---

### Task 5: Prepare demo clips (16 kHz mono)

**Files:**
- Create: `scripts/prep_clips.py`, `tests/test_prep_clips.py`

**Interfaces:**
- Produces: `data/raw/pms.wav` (full ~10 min), `data/raw/sample2.wav` (~48 s), `data/raw/dev.wav` (~45 s dense slice of PMS) — all 16 kHz mono. Uses ffmpeg via subprocess.

- [ ] **Step 1: Write the failing test** — `tests/test_prep_clips.py`

```python
import soundfile as sf
from pathlib import Path
import pytest

@pytest.mark.integration
def test_clips_are_16k_mono():
    for name in ["pms", "sample2", "dev"]:
        p = Path("data/raw") / f"{name}.wav"
        assert p.exists(), f"missing {p} — run scripts/prep_clips.py"
        info = sf.info(str(p))
        assert info.samplerate == 16000 and info.channels == 1
```

- [ ] **Step 2: Write `scripts/prep_clips.py`**

```python
"""Build 16 kHz mono demo clips from the source mp3s. Idempotent."""
import subprocess, sys
from pathlib import Path

SRC_PMS = "PMS V_s Mutual Fund  Which one is better for you - The BlueFort BluePrint.mp3"
SRC_S2  = "sample2.mp3"
OUT = Path("data/raw"); OUT.mkdir(parents=True, exist_ok=True)

def cut(src, dst, ss=None, t=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if ss: cmd += ["-ss", ss]
    if t:  cmd += ["-t", t]
    cmd += ["-i", src, "-ac", "1", "-ar", "16000", str(dst)]
    subprocess.run(cmd, check=True)
    print("wrote", dst)

if __name__ == "__main__":
    cut(SRC_PMS, OUT/"pms.wav")
    cut(SRC_S2,  OUT/"sample2.wav")
    cut(SRC_PMS, OUT/"dev.wav", ss="00:00:00", t="00:00:45")  # dense opening slice
```

- [ ] **Step 3: Run the script, then the test**

Run: `uv run python scripts/prep_clips.py && uv run pytest tests/test_prep_clips.py -v -m integration`
Expected: three "wrote …" lines; test PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/prep_clips.py tests/test_prep_clips.py
git commit -m "feat(data): prep_clips builds 16k mono pms/sample2/dev wavs"
```

---

### Task 6: `enhance` stage — DeepFilterNet denoise (isolated venv)

> **Architecture note (decided during build):** DeepFilterNet 0.5.6 requires `torchaudio<2.1`, irreconcilable with the ASR stack's torchaudio 2.x. So DeepFilterNet runs in a **separate, pre-built venv** (`.venv-denoise`, Python 3.11, torch 2.0.1) invoked as a **subprocess** — clean dependency isolation, consistent with our disk-artifact stage isolation. The controller has ALREADY created and verified `.venv-denoise` (DeepFilterNet3 loads; `data/raw/dev.wav` → 16 kHz mono confirmed). The README (Phase 6) must document recreating it:
> ```bash
> uv venv .venv-denoise --python 3.11
> uv pip install --python .venv-denoise "torch>=2.0,<2.1" "torchaudio>=2.0,<2.1" deepfilternet soundfile
> ```

**Files:**
- Create: `scripts/denoise_worker.py` (runs in `.venv-denoise`), `src/enhance.py` (main venv), `tests/test_enhance.py`

**Interfaces:**
- Consumes: `data/raw/<clip>.wav`; the pre-built `.venv-denoise`.
- Produces: `run(clip:str) -> Path` writing `data/work/<clip>.clean.wav` (16 kHz mono). CLI: `python -m src.enhance <clip>`. Worker CLI: `python scripts/denoise_worker.py <in.wav> <out.wav>`.

- [ ] **Step 1: Write the denoise worker** — `scripts/denoise_worker.py` (this runs inside `.venv-denoise`, NOT the main env)

```python
"""DeepFilterNet denoise worker — runs in the ISOLATED .venv-denoise (torchaudio<2.1),
which the main env cannot host (DeepFilterNet needs torchaudio 1.x APIs; the ASR stack needs 2.x).
Invoked as a subprocess by src/enhance.py.
Usage: python scripts/denoise_worker.py <in.wav> <out_16k_mono.wav>
"""
import sys, warnings
warnings.filterwarnings("ignore")
import soundfile as sf
import torchaudio
from df.enhance import init_df, enhance, load_audio

def main(src: str, dst: str) -> None:
    model, df_state, _ = init_df()                  # loads DeepFilterNet3 (cached after first run)
    audio, _ = load_audio(src, sr=df_state.sr())    # 48 kHz
    out = enhance(model, df_state, audio)           # torch tensor @ 48 kHz
    if out.dim() == 1:
        out = out.unsqueeze(0)
    out16 = torchaudio.functional.resample(out, df_state.sr(), 16000).squeeze(0).cpu().numpy()
    sf.write(dst, out16, 16000)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Write the tests** — `tests/test_enhance.py` (a fast unit test for the missing-venv guard + an integration test for the real denoise)

```python
import numpy as np, soundfile as sf, pytest
from pathlib import Path

def test_run_raises_when_denoise_venv_missing(monkeypatch, tmp_path):
    import src.enhance as e
    monkeypatch.setattr(e, "DENOISE_PY", tmp_path / "no_such_python")
    with pytest.raises(RuntimeError, match="denoise venv"):
        e.run("dev")

@pytest.mark.integration
def test_enhance_produces_clean_wav():
    from src.enhance import run
    out = run("dev")
    assert Path(out).exists()
    y, sr = sf.read(str(out))
    assert sr == 16000 and y.ndim == 1 and np.isfinite(y).all()
```

- [ ] **Step 3: Run to verify the unit test fails first**

Run: `cd /Users/amit/Personal/Atyx && source .venv/bin/activate && pytest tests/test_enhance.py::test_run_raises_when_denoise_venv_missing -v`
Expected: FAIL — `ModuleNotFoundError: src.enhance` (module not written yet).

- [ ] **Step 4: Write `src/enhance.py`** (main venv — subprocess to the worker)

```python
"""Speech enhancement (denoise) stage — always runs.
DeepFilterNet runs in an isolated venv (.venv-denoise, torchaudio<2.1) because it is
incompatible with the ASR stack's torchaudio 2.x; we invoke it as a subprocess and
capture its output so the main test run stays pristine.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
DENOISE_PY = ROOT / ".venv-denoise" / "bin" / "python"
WORKER = ROOT / "scripts" / "denoise_worker.py"

def run(clip: str) -> Path:
    src = ROOT / "data" / "raw" / f"{clip}.wav"
    dst = WORK / f"{clip}.clean.wav"
    WORK.mkdir(parents=True, exist_ok=True)
    if not Path(DENOISE_PY).exists():
        raise RuntimeError(
            f"denoise venv missing at {DENOISE_PY} — create it (see README 'denoise setup')")
    proc = subprocess.run([str(DENOISE_PY), str(WORKER), str(src), str(dst)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"denoise worker failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    print("wrote", dst)
    return dst

if __name__ == "__main__":
    run(sys.argv[1])
```

- [ ] **Step 5: Run tests to verify pass**

Unit (fast): `cd /Users/amit/Personal/Atyx && source .venv/bin/activate && pytest tests/test_enhance.py -v`
Expected: the guard test PASSES; the integration test is deselected.
Integration (real denoise via subprocess to `.venv-denoise`): `... && pytest tests/test_enhance.py -v -m integration`
Expected: PASS — `data/work/dev.clean.wav` is 16 kHz mono, finite. Output pristine (worker stdout/stderr is captured, not leaked).

- [ ] **Step 6: Commit**

```bash
git add scripts/denoise_worker.py src/enhance.py tests/test_enhance.py
git commit -m "feat(enhance): DeepFilterNet denoise via isolated-venv subprocess -> 16k mono clean wav"
```

---

### Task 7: `diarize_asr` (mlx-whisper translate + pyannote, `.venv-asr` worker)

> **Note (post-build):** this is the **single, live ASR path**. The originally-planned
> WhisperX "final path" (Task 8) was built, evaluated, and **dropped** — see Task 8 for
> why. The shipped `asr_worker.py` is single-path (`run(clip)` / `asr_worker.py <clip>`),
> not the `dev|final` two-mode version sketched in the code block below.

> **Architecture note (decided during build):** the ASR/diarization stack (WhisperX, pyannote, mlx-whisper) is irreconcilable with the main env's deps (torchaudio version), so it lives in the pre-built, exact-pinned `.venv-asr` (`requirements-asr.txt`). The `diarize_asr` stage runs there via a **subprocess worker** — same pattern as denoise. The controller has ALREADY built and verified `.venv-asr` (all import together; pyannote `SpeakerDiarization` loads; mlx-whisper imports). Pure merge logic stays in the main env (unit-testable); the worker imports it.

**Files:**
- Create: `src/asr_merge.py` (pure, main env), `scripts/asr_worker.py` (runs in `.venv-asr`), `src/diarize_asr.py` (main orchestrator), `tests/test_diarize_asr.py`

**Interfaces:**
- Consumes: `data/work/<clip>.clean.wav`; pre-built `.venv-asr`; `HF_TOKEN` (`.env`).
- Produces: `merge_words_to_speakers(words, turns) -> list[dict]` (pure); `run(clip) -> Path` writing `data/work/<clip>.transcript.json` (a `Transcript`). Worker CLI: `python scripts/asr_worker.py <clip>`.

- [ ] **Step 1: pure merge logic + failing unit test**

`src/asr_merge.py` (plain dicts — no pydantic/torch, imports in BOTH envs):
```python
"""Pure word->speaker merge (no torch/pydantic) so it runs in main AND .venv-asr."""
def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def merge_words_to_speakers(words, turns):
    """words: [{'text','start','end'}]; turns: [(speaker,t0,t1)].
    Assign each word the most-overlapping turn's speaker; group consecutive same-speaker
    words into utterances. Returns [{'speaker','text','start','end','words':[...]}]."""
    out = []
    for w in words:
        best, bestov = (turns[0][0] if turns else "SPEAKER_00"), -1.0
        for spk, t0, t1 in turns:
            ov = _overlap(w["start"], w["end"], t0, t1)
            if ov > bestov:
                best, bestov = spk, ov
        wd = {**w, "speaker": best}
        if out and out[-1]["speaker"] == best:
            out[-1]["words"].append(wd); out[-1]["text"] += " " + wd["text"]; out[-1]["end"] = wd["end"]
        else:
            out.append({"speaker": best, "text": wd["text"], "start": wd["start"],
                        "end": wd["end"], "words": [wd]})
    return out
```

`tests/test_diarize_asr.py` (unit):
```python
from src.asr_merge import merge_words_to_speakers

def test_merge_assigns_speaker_by_overlap():
    words = [{"text":"PMS","start":0.1,"end":0.4},
             {"text":"minimum","start":0.5,"end":0.9},
             {"text":"haan","start":2.1,"end":2.4}]
    turns = [("SPEAKER_00",0.0,1.0), ("SPEAKER_01",2.0,3.0)]
    utts = merge_words_to_speakers(words, turns)
    assert [u["speaker"] for u in utts] == ["SPEAKER_00","SPEAKER_01"]
    assert utts[0]["text"] == "PMS minimum" and utts[1]["text"] == "haan"
```

- [ ] **Step 2: run unit test, confirm RED**

`cd /Users/amit/Personal/Atyx && source .venv/bin/activate && pytest tests/test_diarize_asr.py -v`
Expected: FAIL — `ModuleNotFoundError: src.asr_merge`.

- [ ] **Step 3: write the worker** — `scripts/asr_worker.py` (runs in `.venv-asr`, NOT main)

```python
"""ASR + diarization worker — runs in .venv-asr (NOT the main env).
Usage: python scripts/asr_worker.py <clip> <dev|final>
Reads data/work/<clip>.clean.wav, writes data/work/<clip>.transcript.json (Transcript schema).
HF_TOKEN is passed in the environment by the parent (src/diarize_asr.py).
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # import src.asr_merge
sys.path.insert(0, str(HERE))          # import sibling asr_worker_final (Task 8)
from src.asr_merge import merge_words_to_speakers
WORK = Path("data/work")

def _diarize(wav):
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=os.environ["HF_TOKEN"])
    diar = pipe(str(wav))
    return [(spk, float(s.start), float(s.end)) for s, _, spk in diar.itertracks(yield_label=True)]

def _asr_dev(wav):
    import mlx_whisper
    r = mlx_whisper.transcribe(str(wav), path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                               task="translate", word_timestamps=True)
    words = []
    for seg in r["segments"]:
        for w in seg.get("words", []):
            words.append({"text": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
    return words

def run_dev(clip, wav):
    words = _asr_dev(wav)
    turns = _diarize(wav)
    return {"clip": clip, "snr": None, "utterances": merge_words_to_speakers(words, turns)}

def main(clip, mode):
    wav = WORK / f"{clip}.clean.wav"
    if mode == "final":
        from asr_worker_final import run_final
        data = run_final(clip, wav)
    else:
        data = run_dev(clip, wav)
    (WORK / f"{clip}.transcript.json").write_text(json.dumps(data, indent=2))
    print(f"wrote {clip}.transcript.json ({len(data['utterances'])} utterances, {mode})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dev")
```

- [ ] **Step 4: write the orchestrator** — `src/diarize_asr.py` (main env)

```python
"""diarize_asr stage — orchestrates the .venv-asr worker (subprocess) and validates output.
The heavy ASR/diarization runs in .venv-asr; this stays in the (torch-free) main env.
"""
import os, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv
from src.contracts import Transcript

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
ASR_PY = ROOT / ".venv-asr" / "bin" / "python"
WORKER = ROOT / "scripts" / "asr_worker.py"

def run(clip: str, mode: str = "dev") -> Path:
    load_dotenv(ROOT / ".env")
    if not Path(ASR_PY).exists():
        raise RuntimeError(f"audio venv missing at {ASR_PY} — build it (README: requirements-asr.txt)")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN not set (needed for pyannote) — add it to .env")
    proc = subprocess.run([str(ASR_PY), str(WORKER), clip, mode],
                          cwd=str(ROOT), env={**os.environ}, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"asr worker failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}")
    dst = WORK / f"{clip}.transcript.json"
    Transcript.model_validate_json(dst.read_text())     # validate worker output against contract
    print("wrote", dst)
    return dst

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dev")
```

- [ ] **Step 5: add guard unit test + confirm units pass**

Append to `tests/test_diarize_asr.py`:
```python
def test_run_raises_without_asr_venv(monkeypatch, tmp_path):
    import pytest, src.diarize_asr as d
    monkeypatch.setattr(d, "ASR_PY", tmp_path / "no_python")
    with pytest.raises(RuntimeError, match="audio venv"):
        d.run("dev")
```
Run: `... && pytest tests/test_diarize_asr.py -v` → merge + guard PASS (no models needed).

- [ ] **Step 6: integration smoke on dev.wav** (needs `.venv-asr` + HF_TOKEN; downloads mlx whisper-large-v3 ~3GB + pyannote)

`cd /Users/amit/Personal/Atyx && source .venv/bin/activate && python -m src.enhance dev && python -m src.diarize_asr dev`
Expected: writes `data/work/dev.transcript.json` with speaker-tagged English utterances. Eyeball: speakers alternate sensibly; Hindi rendered as English.
If the worker errors (mlx-whisper model/API, pyannote), the orchestrator surfaces stderr — **STOP and report, do not hack.**

- [ ] **Step 7: commit**

```bash
git add src/asr_merge.py scripts/asr_worker.py src/diarize_asr.py tests/test_diarize_asr.py
git commit -m "feat(asr): dev-path diarize_asr via .venv-asr worker (mlx-whisper + pyannote)"
```

---

### Task 8: ~~`diarize_asr` — final path (WhisperX align + diarize)~~ — SUPERSEDED / DROPPED

**Status: built, evaluated, and removed from the live path (2026-06-26).** A WhisperX
final pass (`scripts/asr_worker_final.py`) with `task="translate"` + English wav2vec
forced-alignment was implemented and run on sample2. It failed two hard requirements:

- **Not English-only:** output was 41% Devanagari (translate did not fully engage).
- **~67% of the clip dropped:** only 0–15.9 s of a 47.6 s clip survived.

**Root cause (architectural, not a bug):** WhisperX forced-alignment uses a wav2vec
aligner that requires the transcript to be in the **same language as the audio**. The
translate task produces **English text** for **Hindi audio** — there is no English
audio to align against, so the en aligner drops the segments it cannot align and lets
untranslated Devanagari leak through.

**Decision:** the single live ASR path is **mlx-whisper `task="translate"`** (`asr_worker.py`),
which on the same clip gives **100% coverage, 0% Devanagari**. We rely on **segment-level
grounding** (speaker + segment + quote) — all single-hop Q&A source-tracing needs.
Word-level timing, if ever required, comes from Whisper's own `word_timestamps`, never
wav2vec. The "align in source language, then translate the text" redesign is noted as
future work in `design_note.md §ASR`. `asr_worker_final.py` was deleted; `run(clip)` /
`asr_worker.py <clip>` is the single path. See commit `53770ce`.

### Task 9: Transcript-similarity validation (spine acceptance)

**Files:**
- Create: `src/evaltools.py`, `tests/test_evaltools.py`

**Interfaces:**
- Produces: `normalize(text:str) -> str` and `similarity(hyp:str, ref:str) -> float` (0–1) using token-set + sequence ratio; `transcript_text(t:Transcript) -> str`. This is the *transcript-similarity ceiling* metric reused by the Phase 4 eval harness. **Eval-safe / pure** — no product imports.

- [ ] **Step 1: Write the failing test** — `tests/test_evaltools.py`

```python
from src.evaltools import normalize, similarity

def test_similarity_bounds_and_sense():
    assert similarity("PMS minimum is 50 lakh", "PMS minimum is 50 lakh") > 0.99
    assert similarity("totally different words here", "PMS minimum is 50 lakh") < 0.4
    assert normalize("  PMS,  Minimum!  ") == "pms minimum"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_evaltools.py -v`
Expected: FAIL — `ModuleNotFoundError: src.evaltools`.

- [ ] **Step 3: Write `src/evaltools.py`**

```python
import re
from difflib import SequenceMatcher

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def transcript_text(t) -> str:
    return " ".join(u.text for u in t.utterances)

def similarity(hyp: str, ref: str) -> float:
    h, r = normalize(hyp), normalize(ref)
    seq = SequenceMatcher(None, h, r).ratio()
    hs, rs = set(h.split()), set(r.split())
    jacc = len(hs & rs) / len(hs | rs) if (hs | rs) else 0.0
    return round(0.5 * seq + 0.5 * jacc, 4)
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_evaltools.py -v`
Expected: PASS.

- [ ] **Step 5: Spine acceptance check (manual, integration)**

Run (after Tasks 6–8 produced a transcript):
```bash
uv run python -c "
from src.contracts import Transcript; from src.evaltools import similarity, transcript_text
t = Transcript.model_validate_json(open('data/work/sample2.transcript.json').read())
ref = open('sample2.txt').read()
print('similarity vs reference:', similarity(transcript_text(t), ref))"
```
Expected: a number (~0.5–0.9 on clean audio). **This is the Phase 1 acceptance signal** — the audio→English-transcript link works and is measurable. Record the number; it's the baseline for the SNR curve.

- [ ] **Step 6: Commit**

```bash
git add src/evaltools.py tests/test_evaltools.py
git commit -m "feat(eval): transcript-similarity helper + Phase 1 spine acceptance"
```

---

## Phase 1 Done = 
- `uv run pytest` green (unit tests, no integration marker).
- `data/work/{dev,sample2}.transcript.json` exist with speaker-attributed English.
- A recorded transcript-similarity number vs the reference (the SNR-curve baseline).
- Everything committed and pushed.

**Then:** write the Phase 2 plan (Transcript → induced-ontology graph in Neo4j).
