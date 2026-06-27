# Phase 4 — Controlled-SNR Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/evaluate.py` — a controlled-SNR harness that turns the six sweep transcripts (clean + 5 café-babble SNR levels) into one honest transcript-fidelity hero curve plus an illustrative, transcript-grounded downstream spot-check.

**Architecture:** Pure/IO split mirroring the rest of the codebase. Pure assemblers (`snr_curve`, `downstream_spotcheck`) and a transcript-grounded retriever (`retrieve_answer`, reusing `qa.cosine`/`top_k_statements`/`compose_answer`) are unit-tested with a fake LLM; an IO layer (`emit_results`) writes JSON + a matplotlib PNG; an orchestrator (`evaluate`) reads sweep transcripts already on disk and emits artifacts. The expensive ASR sweep is a separate prep script — it never enters the unit-tested core.

**Tech Stack:** Python 3.12 (main `.venv`, torch-free), Pydantic v2, matplotlib (new), the existing `src/evaltools.py` / `src/qa.py` / `src/llm.py` / `src/add_noise` helpers, pytest.

## Global Constraints

- **Main `.venv` only, torch-free** — activate with `source .venv/bin/activate` (NOT `uv run`). No torch/torchaudio imports in any Phase 4 file.
- **Oracle-free baseline** — the hero curve scores each noisy transcript against the **clean slice run through the identical pipeline** (`{prefix}_clean.transcript.json`). No external/frontier ground truth in Phase 4.
- **All 5 SNR points always** — `[20, 15, 10, 5, 0]` dB. Never reduce the point count.
- **Real café babble noise** — `noices/cafe_16k.wav`. SNR is the only controlled variable.
- **Similarity = relative fidelity, not WER** — `evaltools.similarity` (`0.5·SequenceMatcher.ratio + 0.5·jaccard`, in [0,1]). Every artifact says so.
- **Spot-check is transcript-grounded retrieval, NOT the full extract→graph→Q&A product path** — no fact extraction, no Neo4j mutation in the spot-check. Every artifact carries BOTH labels: "illustrative propagation, not a calibrated curve" AND "transcript-grounded retrieval, not the full product path".
- **Honest PNG** — labelled axes (x: SNR dB, y: similarity [0–1]), y-axis floor fixed at 0 (never auto-scaled to dramatize), the cliff annotated.
- **"2-speaker conversational slice"** — never call the eval slice "multi-party".
- **Commit message footer (verbatim):** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Pytest:** default run skips integration (`addopts = -m 'not integration'`); integration tests run with `pytest -m integration`. Register no new markers.

**Pre-existing on disk (produced by the in-flight sweep, do NOT regenerate):**
`data/work/{pmsslice_clean,pmsslice_snr20,pmsslice_snr15,pmsslice_snr10,pmsslice_snr5,pmsslice_snr0}.transcript.json`.

---

### Task 1: Eval config fields + reproducible prep harness

**Files:**
- Modify: `src/config.py:16-17` (extend `EvalCfg`)
- Modify: `config.yaml:18-19` (add eval fields)
- Create: `scripts/prep_eval_clips.py` (promote `scratch_sweep.py`, config-driven)
- Test: `tests/test_config.py` (add one assertion) — if absent, create it

**Interfaces:**
- Consumes: `src/add_noise.py` — `load_audio(path, sr=16000) -> np.ndarray`, `fit_noise(noise, length, seed=0) -> np.ndarray`, `mix(speech, noise, snr_db) -> np.ndarray`.
- Produces: `EvalCfg` with fields `snr_levels: list[int]`, `source_clip: str`, `slice_start_s: int`, `slice_end_s: int`, `noise_path: str`, `clip_prefix: str`, `degraded_snr: int`, `spotcheck_questions: list[str]`. Clip naming: baseline = `f"{clip_prefix}_clean"`, noisy = `f"{clip_prefix}_snr{snr}"`.

- [ ] **Step 1: Write the failing test**

In `tests/test_config.py` (create if missing) add:

```python
from src.config import load_config

def test_eval_config_carries_slice_noise_and_spotcheck():
    ec = load_config().eval
    assert ec.snr_levels == [20, 15, 10, 5, 0]
    assert ec.clip_prefix == "pmsslice"
    assert ec.degraded_snr in ec.snr_levels
    assert ec.noise_path.endswith("cafe_16k.wav")
    assert ec.slice_end_s > ec.slice_start_s
    assert len(ec.spotcheck_questions) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_config.py::test_eval_config_carries_slice_noise_and_spotcheck -v`
Expected: FAIL — `EvalCfg` has no `clip_prefix` (pydantic ignores unknown / attribute error).

- [ ] **Step 3: Extend `EvalCfg` and `config.yaml`**

`src/config.py` — replace the `EvalCfg` class:

```python
class EvalCfg(BaseModel):
    snr_levels: list[int]
    source_clip: str = "pms"
    slice_start_s: int = 0
    slice_end_s: int = 160
    noise_path: str = "noices/cafe_16k.wav"
    clip_prefix: str = "pmsslice"
    degraded_snr: int = 5
    spotcheck_questions: list[str] = []
```

`config.yaml` — replace the `eval:` block:

```yaml
eval:
  snr_levels: [20, 15, 10, 5, 0]
  source_clip: pms
  slice_start_s: 0
  slice_end_s: 160
  noise_path: noices/cafe_16k.wav
  clip_prefix: pmsslice
  degraded_snr: 5
  spotcheck_questions:
    - "How does a PMS differ from a mutual fund?"
    - "What is the minimum investment for a PMS?"
    - "Who is a PMS suitable for?"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Create the reproducible prep harness**

Create `scripts/prep_eval_clips.py` (config-driven promotion of `scratch_sweep.py`):

```python
"""Reproducible Phase 4 sweep: cut the eval slice, mix real café babble at each
SNR, run the real denoise + ASR pipeline -> six transcripts under data/work/.

Long-pole compute (~20 min on M4). Config-driven (config.yaml eval.*). The
transcripts it writes are consumed by src/evaluate.py — this script is the
documented, reproducible source of those artifacts (it replaces the scratch
scratch_sweep.py used during bring-up).

Run: source .venv/bin/activate && python -m scripts.prep_eval_clips
"""
import subprocess
import sys
from pathlib import Path

import soundfile as sf

import add_noise as an
from src.config import load_config

SR = 16000


def write_and_run(clip: str, audio, raw_dir: Path) -> None:
    sf.write(str(raw_dir / f"{clip}.wav"), audio, SR)
    print(f"[{clip}] enhance...", flush=True)
    subprocess.run([sys.executable, "-m", "src.enhance", clip], check=True)
    print(f"[{clip}] diarize_asr...", flush=True)
    subprocess.run([sys.executable, "-m", "src.diarize_asr", clip], check=True)
    print(f"[{clip}] done", flush=True)


def main() -> None:
    cfg = load_config()
    ec = cfg.eval
    raw = Path(cfg.paths.raw)
    speech = an.load_audio(str(raw / f"{ec.source_clip}.wav"), sr=SR)
    sl = speech[ec.slice_start_s * SR:ec.slice_end_s * SR]
    noise = an.load_audio(ec.noise_path, sr=SR)
    print(f"slice {len(sl) / SR:.0f}s  noise {len(noise) / SR:.0f}s", flush=True)

    write_and_run(f"{ec.clip_prefix}_clean", sl, raw)
    for snr in ec.snr_levels:
        mixed = an.mix(sl, an.fit_noise(noise, len(sl)), snr)
        write_and_run(f"{ec.clip_prefix}_snr{snr}", mixed, raw)
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the sweep artifacts exist and parse (do NOT re-run the 20-min sweep)**

Run:
```bash
source .venv/bin/activate && python -c "
from pathlib import Path
from src.config import load_config
from src.contracts import Transcript
ec = load_config().eval
names = [f'{ec.clip_prefix}_clean'] + [f'{ec.clip_prefix}_snr{s}' for s in ec.snr_levels]
for n in names:
    t = Transcript.model_validate_json(Path(f'data/work/{n}.transcript.json').read_text())
    print(n, len(t.utterances), 'utts')
print('ALL 6 PARSE OK')
"
```
Expected: 6 lines + `ALL 6 PARSE OK`. (If any are missing, run `python -m scripts.prep_eval_clips` — ~20 min — then retry.)

- [ ] **Step 7: Remove the scratch script and commit**

```bash
git rm -f --ignore-unmatch scratch_sweep.py
git add src/config.py config.yaml scripts/prep_eval_clips.py tests/test_config.py
git commit -m "feat(phase4): eval config fields + reproducible prep harness

scripts/prep_eval_clips.py replaces scratch_sweep.py — config-driven slice,
café-babble noise, all 5 SNR levels -> six transcripts. EvalCfg gains the
slice/noise/spot-check fields.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Contracts + `snr_curve` (the hero curve)

**Files:**
- Modify: `src/contracts.py` (append `CurvePoint`, `SpotCheckRow`, `EvalResult`)
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `src/evaltools.py` — `similarity(hyp: str, ref: str) -> float`, `transcript_text(t) -> str`; `src/contracts.py` — `Transcript`, `Utterance`.
- Produces:
  - `CurvePoint(snr: str, similarity: float)`
  - `SpotCheckRow(question: str, clean_answer: str, degraded_answer: str, degraded_snr: int)`
  - `EvalResult(curve: list[CurvePoint], spotcheck: list[SpotCheckRow], meta: dict)`
  - `snr_curve(baseline: Transcript, noisy: dict[str, Transcript]) -> list[CurvePoint]` — one point per `noisy` entry, **same order as the dict**, similarity of each noisy transcript text vs the baseline transcript text.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
from src.contracts import Transcript, Utterance, CurvePoint
from src.evaluate import snr_curve


def _t(text):
    return Transcript(clip="c", utterances=[Utterance(speaker="A", text=text, start=0, end=1)])


def test_snr_curve_scores_each_noisy_against_baseline_preserving_order():
    base = _t("alpha beta gamma delta")
    noisy = {
        "20": _t("alpha beta gamma delta"),   # identical -> 1.0
        "0": _t("zzz qqq"),                    # very different -> low
    }
    pts = snr_curve(base, noisy)
    assert [p.snr for p in pts] == ["20", "0"]          # dict order preserved
    assert all(isinstance(p, CurvePoint) for p in pts)
    assert pts[0].similarity == 1.0                      # self-similarity is exactly 1
    assert pts[1].similarity < pts[0].similarity          # degraded scores strictly lower
    assert 0.0 <= pts[1].similarity <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_evaluate.py::test_snr_curve_scores_each_noisy_against_baseline_preserving_order -v`
Expected: FAIL — `src.evaluate` does not exist / `CurvePoint` not defined.

- [ ] **Step 3: Add contracts**

Append to `src/contracts.py`:

```python
class CurvePoint(BaseModel):
    snr: str                 # SNR level label, e.g. "10"
    similarity: float        # transcript similarity vs clean baseline, [0,1]

class SpotCheckRow(BaseModel):
    question: str
    clean_answer: str
    degraded_answer: str
    degraded_snr: int

class EvalResult(BaseModel):
    curve: list[CurvePoint] = Field(default_factory=list)
    spotcheck: list[SpotCheckRow] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
```

(`Field` and `Any` are already imported at the top of `contracts.py`.)

- [ ] **Step 4: Implement `snr_curve`**

Create `src/evaluate.py`:

```python
"""Phase 4 controlled-SNR evaluation harness.

Pure/IO split: snr_curve + downstream_spotcheck + retrieve_answer are pure /
LLM-injected and unit-tested with a fake; emit_results does IO (JSON + PNG);
evaluate() orchestrates over the sweep transcripts already on disk.

Hero curve = transcript similarity vs SNR, oracle-free (each noisy transcript
scored against the clean slice through the identical pipeline). The downstream
spot-check is transcript-grounded retrieval (NOT the full extract->graph->Q&A
product path) — see docs/superpowers/specs/2026-06-27-atyx-convo-kg-phase4-eval-design.md.
"""
from src.contracts import CurvePoint, Transcript
from src.evaltools import similarity, transcript_text


def snr_curve(baseline: Transcript,
              noisy: dict[str, Transcript]) -> list[CurvePoint]:
    """Score each noisy transcript against the baseline transcript.

    Oracle-free: the baseline is the clean slice through the identical pipeline.
    Order of points follows the insertion order of `noisy`.
    """
    ref = transcript_text(baseline)
    return [CurvePoint(snr=snr, similarity=similarity(transcript_text(t), ref))
            for snr, t in noisy.items()]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/contracts.py src/evaluate.py tests/test_evaluate.py
git commit -m "feat(phase4): eval contracts + snr_curve hero metric

CurvePoint/SpotCheckRow/EvalResult contracts; snr_curve scores each noisy
transcript against the clean-baseline transcript (oracle-free), preserving
SNR order.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Transcript-grounded spot-check (`retrieve_answer` + `downstream_spotcheck`)

**Files:**
- Modify: `src/evaluate.py` (add `retrieve_answer`, `downstream_spotcheck`)
- Test: `tests/test_evaluate.py` (add tests + a `FakeLLM`)

**Interfaces:**
- Consumes: `src/qa.py` — `cosine`, `top_k_statements(question_vec, statements_with_vecs, k)`, `compose_answer(question, rows, provenance, llm) -> str`; `src/contracts.py` — `Provenance(statement_id, speaker, text, kind)`, `Transcript`, `SpotCheckRow`. `llm` must expose `.embed(texts) -> list[list[float]]` and `.chat_json(system, user, schema) -> dict` (the real `LLM`, or a fake in tests).
- Produces:
  - `retrieve_answer(question: str, transcript: Transcript, llm, k: int = 3) -> str` — embeds each utterance, retrieves top-k by cosine, composes an answer from those quotes. No extraction, no graph. Empty transcript → a plain "no content" string.
  - `downstream_spotcheck(questions, clean_answers, degraded_answers, degraded_snr) -> list[SpotCheckRow]` — pure row assembler (zips the three lists). No grading verdict.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluate.py`:

```python
from src.contracts import SpotCheckRow
from src.evaluate import retrieve_answer, downstream_spotcheck


class FakeLLM:
    """Deterministic fake: bag-of-words embeddings + echo-the-quotes compose."""
    VOCAB = ["pms", "mutual", "fund", "minimum", "investment", "younger"]

    def embed(self, texts):
        return [[float(t.lower().count(w)) for w in self.VOCAB] for t in texts]

    def chat_json(self, system, user, schema):
        # Echo the user message so the test can see which quote was retrieved.
        return {"answer": user}


def test_retrieve_answer_grounds_in_nearest_utterance():
    t = Transcript(clip="x", utterances=[
        Utterance(speaker="A", text="The minimum investment for a PMS is fifty lakh", start=0, end=1),
        Utterance(speaker="B", text="We work with a lot of younger investors", start=1, end=2),
    ])
    ans = retrieve_answer("what is the minimum investment for a PMS", t, FakeLLM(), k=1)
    assert "fifty lakh" in ans          # nearest utterance retrieved + composed
    assert "younger" not in ans         # off-topic utterance not retrieved at k=1


def test_retrieve_answer_handles_empty_transcript():
    empty = Transcript(clip="x", utterances=[])
    ans = retrieve_answer("anything", empty, FakeLLM())
    assert isinstance(ans, str) and ans                 # graceful non-empty string, no crash


def test_downstream_spotcheck_zips_rows_without_verdict():
    rows = downstream_spotcheck(
        questions=["q1", "q2"],
        clean_answers=["c1", "c2"],
        degraded_answers=["d1", "d2"],
        degraded_snr=5,
    )
    assert [r.question for r in rows] == ["q1", "q2"]
    assert rows[0].clean_answer == "c1" and rows[0].degraded_answer == "d1"
    assert all(r.degraded_snr == 5 for r in rows)
    assert all(isinstance(r, SpotCheckRow) for r in rows)
    # SpotCheckRow has no verdict/score field — illustrative only
    assert "verdict" not in SpotCheckRow.model_fields and "score" not in SpotCheckRow.model_fields
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_evaluate.py -k "retrieve_answer or downstream_spotcheck" -v`
Expected: FAIL — `retrieve_answer` / `downstream_spotcheck` not defined.

- [ ] **Step 3: Implement both functions**

Add to `src/evaluate.py` (extend the imports line and append the functions):

```python
from src.contracts import CurvePoint, Provenance, SpotCheckRow, Transcript
from src.qa import compose_answer, top_k_statements
```

```python
def retrieve_answer(question: str, transcript: Transcript, llm, k: int = 3) -> str:
    """Answer a question from ONE transcript via cosine retrieval (no graph).

    Reuses the qa semantic-fallback mechanism over the transcript's own
    utterances. This is transcript-grounded retrieval, deliberately NOT the
    extract->graph->Q&A product path — the spot-check isolates transcript->answer.
    """
    utts = transcript.utterances
    if not utts:
        return "No transcript content available to answer from."
    vecs = llm.embed([u.text for u in utts])
    stmts = [{"id": f"u{i}", "speaker": u.speaker, "text": u.text, "vec": v}
             for i, (u, v) in enumerate(zip(utts, vecs))]
    question_vec = llm.embed([question])[0]
    top = top_k_statements(question_vec, stmts, k)
    provenance = [Provenance(statement_id=s["id"], speaker=s["speaker"],
                             text=s["text"], kind="related") for s in top]
    return compose_answer(question, [], provenance, llm)


def downstream_spotcheck(questions: list[str],
                         clean_answers: list[str],
                         degraded_answers: list[str],
                         degraded_snr: int) -> list[SpotCheckRow]:
    """Assemble side-by-side clean/degraded rows. Pure; no grading verdict."""
    return [SpotCheckRow(question=q, clean_answer=ca, degraded_answer=da,
                         degraded_snr=degraded_snr)
            for q, ca, da in zip(questions, clean_answers, degraded_answers)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_evaluate.py -v`
Expected: PASS (all Task 2 + Task 3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat(phase4): transcript-grounded spot-check retrieval

retrieve_answer reuses qa cosine top-k + compose over one transcript's
utterances (no extraction, no graph) so the spot-check isolates
transcript->answer. downstream_spotcheck assembles clean/degraded rows
with no grading verdict.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `emit_results` — JSON + honest PNG

**Files:**
- Modify: `pyproject.toml:5-17` (add `matplotlib>=3.8`)
- Modify: `src/evaluate.py` (add `emit_results`, `_y_limits`, `cliff_index`, `_render_png`)
- Test: `tests/test_evaluate.py` (add tests)

**Interfaces:**
- Consumes: `CurvePoint`, `SpotCheckRow` (Task 2).
- Produces:
  - `cliff_index(curve: list[CurvePoint]) -> int` — index of the point **after** the steepest consecutive similarity drop (0 if <2 points).
  - `_y_limits(curve) -> tuple[float, float]` — always `(0.0, 1.02)` regardless of data (the honest, non-auto-scaled floor).
  - `emit_results(curve, spotcheck, meta, out_json: str, out_png: str) -> tuple[str, str]` — writes the JSON payload and the PNG; returns the two paths.

- [ ] **Step 1: Add matplotlib to the main env**

```bash
source .venv/bin/activate && uv pip install "matplotlib>=3.8" && python -c "import matplotlib; print(matplotlib.__version__)"
```
Then add `"matplotlib>=3.8",` to the `dependencies` list in `pyproject.toml` (after the `neo4j>=5.20` line).
Expected: a version prints (e.g. `3.9.x`).

- [ ] **Step 2: Write the failing test**

Add to `tests/test_evaluate.py`:

```python
import json
from src.contracts import CurvePoint
from src.evaluate import emit_results, cliff_index, _y_limits


def test_y_limits_are_a_fixed_honest_floor_ignoring_data():
    # Even a curve that never drops below 0.9 must still plot from 0 — no
    # auto-scaling to dramatize a small drop.
    c = [CurvePoint(snr="20", similarity=0.92), CurvePoint(snr="0", similarity=0.90)]
    assert _y_limits(c) == (0.0, 1.02)


def test_cliff_index_finds_point_after_steepest_drop():
    c = [CurvePoint(snr="20", similarity=0.95),
         CurvePoint(snr="10", similarity=0.90),
         CurvePoint(snr="5", similarity=0.50)]      # steepest drop is 0.90 -> 0.50
    assert cliff_index(c) == 2


def test_emit_results_writes_json_and_png_with_both_honesty_labels(tmp_path):
    curve = [CurvePoint(snr="20", similarity=0.95), CurvePoint(snr="0", similarity=0.40)]
    sc = [SpotCheckRow(question="q", clean_answer="a", degraded_answer="b", degraded_snr=5)]
    pj, pp = tmp_path / "r.json", tmp_path / "r.png"
    j, p = emit_results(curve, sc, {"noise": "noices/cafe_16k.wav"}, str(pj), str(pp))
    data = json.loads(pj.read_text())
    assert data["curve"][0]["snr"] == "20"
    assert data["spotcheck"][0]["question"] == "q"
    # both required honesty labels present in the machine-readable artifact
    assert "not a calibrated curve" in data["labels"]["spotcheck"]
    assert "not the full product path" in data["labels"]["spotcheck"]
    assert "not WER" in data["labels"]["curve"]
    assert pp.exists() and pp.stat().st_size > 0      # a real PNG was rendered
```

- [ ] **Step 3: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_evaluate.py -k "y_limits or cliff_index or emit_results" -v`
Expected: FAIL — `emit_results` / `cliff_index` / `_y_limits` not defined.

- [ ] **Step 4: Implement the IO layer**

Add to `src/evaluate.py` (add `import json` and `from pathlib import Path` at the top):

```python
CURVE_LABEL = ("transcript fidelity vs SNR — relative sequence+set similarity "
               "(not WER); reference is the clean slice through the identical pipeline")
SPOTCHECK_LABEL = ("illustrative propagation, not a calibrated curve; "
                   "transcript-grounded retrieval, not the full product path")


def cliff_index(curve: list[CurvePoint]) -> int:
    """Index of the point after the steepest consecutive similarity drop."""
    if len(curve) < 2:
        return 0
    drops = [(curve[i].similarity - curve[i + 1].similarity, i + 1)
             for i in range(len(curve) - 1)]
    return max(drops)[1]


def _y_limits(curve: list[CurvePoint]) -> tuple[float, float]:
    """Fixed honest floor at 0 — never auto-scaled to exaggerate small drops."""
    return (0.0, 1.02)


def _render_png(curve: list[CurvePoint], meta: dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [int(p.snr) for p in curve]
    ys = [p.similarity for p in curve]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", color="#b8860b")
    ax.set_xlabel("SNR (dB) — café babble (higher = cleaner)")
    ax.set_ylabel("Transcript similarity vs clean baseline [0–1]")
    ax.set_ylim(*_y_limits(curve))            # honest floor
    ax.invert_xaxis()                          # easy (high SNR) left -> hard (0 dB) right
    ax.set_title("Front-end fidelity vs noise — 2-speaker PMS slice")
    if len(curve) >= 2:
        ci = cliff_index(curve)
        ax.annotate(f"cliff at {xs[ci]} dB",
                    xy=(xs[ci], ys[ci]),
                    xytext=(xs[ci], min(ys[ci] + 0.18, 0.98)),
                    ha="center", arrowprops=dict(arrowstyle="->"))
    ax.text(0.5, -0.16, CURVE_LABEL, transform=ax.transAxes,
            ha="center", fontsize=7, wrap=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def emit_results(curve: list[CurvePoint],
                 spotcheck: list[SpotCheckRow],
                 meta: dict,
                 out_json: str, out_png: str) -> tuple[str, str]:
    """Write the JSON record and the honest PNG. Returns (json_path, png_path)."""
    payload = {
        "meta": meta,
        "labels": {"curve": CURVE_LABEL, "spotcheck": SPOTCHECK_LABEL},
        "curve": [p.model_dump() for p in curve],
        "spotcheck": [r.model_dump() for r in spotcheck],
    }
    Path(out_json).write_text(json.dumps(payload, indent=2))
    _render_png(curve, meta, out_png)
    return out_json, out_png
```

- [ ] **Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_evaluate.py -v`
Expected: PASS (all evaluate tests so far).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/evaluate.py tests/test_evaluate.py
git commit -m "feat(phase4): emit_results — JSON + honest-y-floor PNG

matplotlib PNG with labelled axes, fixed y-floor at 0 (cliff annotated) and
both honesty labels (not-WER; illustrative + transcript-grounded) baked into
the JSON. cliff_index/_y_limits are pure + unit-tested.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `evaluate` orchestrator + CLI + integration test

**Files:**
- Modify: `src/evaluate.py` (add `evaluate`, `__main__`)
- Test: `tests/test_evaluate.py` (add the marked integration test)

**Interfaces:**
- Consumes: `src/config.py` — `load_config() -> Config` (with `.eval`, `.paths`, `.llm`); `src/llm.py` — `LLM(cfg.llm)`; `Transcript.model_validate_json`; all earlier evaluate functions; `EvalResult`.
- Produces: `evaluate(cfg=None) -> EvalResult` — loads the six sweep transcripts from `paths.work`, builds the curve, runs the spot-check (clean vs `degraded_snr`) via the real `LLM`, writes `data/ground_truth/snr_results.json` + `snr_curve.png`, returns the `EvalResult`. Does NOT run ASR.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_evaluate.py`:

```python
import pytest
from pathlib import Path
from src.evaluate import evaluate


@pytest.mark.integration
def test_evaluate_produces_five_point_curve_over_real_sweep_artifacts():
    res = evaluate()
    pts = res.curve
    assert len(pts) == 5                                  # all 5 SNR points, never fewer
    assert all(0.0 <= p.similarity <= 1.0 for p in pts)
    by = {p.snr: p.similarity for p in pts}
    # LOOSE monotonicity ONLY: the clean end (20 dB) must beat the worst end
    # (0 dB) and the general trend is downward. NOT strict point-by-point — a
    # bumpy real curve (e.g. 5 dB slightly above 10 dB from ASR quirks) is valid;
    # the cliff/shape is the finding, not strict monotonicity.
    assert by["20"] >= by["0"]
    assert len(res.spotcheck) >= 2                         # golden questions answered both sides
    assert Path("data/ground_truth/snr_results.json").exists()
    assert Path("data/ground_truth/snr_curve.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_evaluate.py::test_evaluate_produces_five_point_curve_over_real_sweep_artifacts -m integration -v`
Expected: FAIL — `evaluate` not defined.

- [ ] **Step 3: Implement the orchestrator + CLI**

Add to `src/evaluate.py` (extend imports with `from src.config import load_config`, `from src.contracts import EvalResult`, `from src.llm import LLM`):

```python
def evaluate(cfg=None) -> EvalResult:
    """Read the six sweep transcripts, build the hero curve, run the
    transcript-grounded spot-check, emit artifacts. Does not run ASR."""
    cfg = cfg or load_config()
    ec = cfg.eval
    work = Path(cfg.paths.work)

    def _load(name: str) -> Transcript:
        return Transcript.model_validate_json((work / f"{name}.transcript.json").read_text())

    baseline = _load(f"{ec.clip_prefix}_clean")
    noisy = {str(s): _load(f"{ec.clip_prefix}_snr{s}") for s in ec.snr_levels}
    curve = snr_curve(baseline, noisy)

    llm = LLM(cfg.llm)
    degraded = noisy[str(ec.degraded_snr)]
    clean_answers = [retrieve_answer(q, baseline, llm) for q in ec.spotcheck_questions]
    degraded_answers = [retrieve_answer(q, degraded, llm) for q in ec.spotcheck_questions]
    spotcheck = downstream_spotcheck(ec.spotcheck_questions, clean_answers,
                                     degraded_answers, ec.degraded_snr)

    meta = {
        "source_clip": ec.source_clip,
        "slice_s": [ec.slice_start_s, ec.slice_end_s],
        "noise": ec.noise_path,
        "snr_levels": ec.snr_levels,
        "degraded_snr": ec.degraded_snr,
        "speakers": sorted({u.speaker for u in baseline.utterances}),
        "metric": "evaltools.similarity (relative sequence+set fidelity, not WER)",
    }
    gt = Path(cfg.paths.ground_truth)
    gt.mkdir(parents=True, exist_ok=True)
    emit_results(curve, spotcheck, meta,
                 str(gt / "snr_results.json"), str(gt / "snr_curve.png"))
    return EvalResult(curve=curve, spotcheck=spotcheck, meta=meta)


if __name__ == "__main__":
    res = evaluate()
    print(f"speakers={res.meta['speakers']}  noise={res.meta['noise']}")
    for p in res.curve:
        print(f"  SNR {p.snr:>2} dB : similarity {p.similarity:.4f}")
    print(f"spot-check: {len(res.spotcheck)} questions (clean vs {res.meta['degraded_snr']} dB)")
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_evaluate.py::test_evaluate_produces_five_point_curve_over_real_sweep_artifacts -m integration -v`
Expected: PASS (requires LM Studio reachable + the six transcripts on disk).

- [ ] **Step 5: Run the full default (non-integration) suite**

Run: `source .venv/bin/activate && pytest -q`
Expected: PASS, no regressions in the existing suite.

- [ ] **Step 6: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat(phase4): evaluate orchestrator + CLI + integration test

evaluate() reads the six sweep transcripts, builds the 5-point hero curve,
runs the transcript-grounded spot-check (clean vs degraded), emits
snr_results.json + snr_curve.png. Integration test asserts 5 points and
LOOSE monotonicity (clean>=worst, general downward) — not strict.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Generate artifacts + design-note results paragraph + README

**Files:**
- Create: `data/ground_truth/snr_results.json`, `data/ground_truth/snr_curve.png` (generated)
- Modify: `design_note.md` (add a Phase 4 results section)
- Modify: `README.md` (add an `eval` run line)

**Interfaces:**
- Consumes: `evaluate()` (Task 5).

- [ ] **Step 1: Generate the real artifacts**

Run: `source .venv/bin/activate && python -m src.evaluate`
Expected: prints the 5-point curve + spot-check summary; writes `data/ground_truth/snr_results.json` and `snr_curve.png`.

- [ ] **Step 2: Read the actual numbers**

Run: `source .venv/bin/activate && python -c "import json; d=json.load(open('data/ground_truth/snr_results.json')); print([(p['snr'],p['similarity']) for p in d['curve']]); print('speakers', d['meta']['speakers'])"`
Record the five `(snr, similarity)` pairs and the cliff location — they go into the prose below verbatim. (Do not invent numbers; use what the harness produced.)

- [ ] **Step 3: Write the design-note results section**

Append to `design_note.md` a section titled `## Phase 4 — Controlled-SNR results`. Use the ACTUAL numbers from Step 2. Template (fill the bracketed values from the real artifact):

```markdown
## Phase 4 — Controlled-SNR results

**Setup.** A 160 s **2-speaker conversational slice** of the real PMS clip, mixed with
**real café-babble noise** (`noices/cafe_16k.wav`) at five SNR levels (20/15/10/5/0 dB)
— SNR the single controlled variable. Each noisy clip runs the full denoise→diarize→ASR
front-end; the resulting transcript is scored against the **clean slice through the
identical pipeline** (oracle-free) with `evaltools.similarity` (a relative sequence+set
fidelity blend in [0,1], **not WER**).

**Hero curve (`data/ground_truth/snr_curve.png`).** Transcript fidelity vs SNR:
20 dB → [X.XX], 15 dB → [X.XX], 10 dB → [X.XX], 5 dB → [X.XX], 0 dB → [X.XX]. The
front-end holds [up to ~N dB / describe], then degrades — the **cliff at [N] dB** is the
finding. [One sentence on what the shape means for real noisy field audio.]

**Downstream spot-check (illustrative).** The golden in-slice questions
(PMS-vs-MF, minimum investment, who-it's-for) answered from the clean vs the [5] dB
transcript via **transcript-grounded retrieval** (cosine top-k over the transcript;
**not** the full extract→graph→Q&A product path, by design — extraction's
nondeterministic ceiling is kept out of every measured path). [One sentence: at clean
the answers are on-topic; at [5] dB the degraded transcript yields [describe what
happens — vaguer / missing detail].] This is **illustrative propagation, not a
calibrated curve.**

**Honest limits.** No absolute WER (needs a verified reference). No fact-recall or
Q&A-correctness *curve* — on noisy code-mixed Hinglish the local ~9B extractor is
nondeterministic and partially-recalling, so a curve through it would track model
variance, not noise (§ Measured capability boundary). The scaling path to a calibrated
fact-recall curve — a stronger/Cypher-tuned extractor + a frontier-oracle, human-verified
fact-level ground truth — is documented in the design spec §9, deliberately out of v1.
```

- [ ] **Step 4: Add the README eval line**

In `README.md`, near the existing run/demo commands, add:

```markdown
### Evaluation (controlled-SNR)

Reproduce the café-babble SNR sweep and the fidelity curve:

```bash
# 1. Sweep (long, ~20 min): cut the slice, mix all 5 SNR levels, run denoise+ASR
source .venv/bin/activate && python -m scripts.prep_eval_clips
# 2. Build the hero curve + spot-check artifacts (needs LM Studio for the spot-check)
source .venv/bin/activate && python -m src.evaluate
# -> data/ground_truth/snr_results.json + snr_curve.png
```

The curve is **transcript fidelity vs SNR** (relative similarity, not WER); the
downstream Q&A spot-check is **illustrative, transcript-grounded retrieval**, not the
full product path. See `design_note.md § Phase 4`.
```

- [ ] **Step 5: Commit (force-add the generated artifacts)**

```bash
git add -f data/ground_truth/snr_results.json data/ground_truth/snr_curve.png
git add design_note.md README.md
git commit -m "docs(phase4): SNR results artifacts + design-note + README

Generated snr_results.json + snr_curve.png from the real café-babble sweep;
design_note.md Phase 4 results section with the measured curve + cliff and
the honest-limits paragraph; README eval reproduction steps.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage** (against `2026-06-27-atyx-convo-kg-phase4-eval-design.md`):
- §3 controlled single variable / 5 SNR points → Task 1 (config) + Task 5 integration test asserts 5 points. ✓
- §4 oracle-free baseline + `evaltools.similarity` → Task 2 `snr_curve`. ✓
- §5 real café babble + tiling → Task 1 prep harness (uses `add_noise`). ✓
- §6 2-speaker slice naming → Global Constraints + design-note/PNG copy (Tasks 4/6). ✓
- §7 transcript-grounded spot-check + both labels → Task 3 (`retrieve_answer`) + Task 4 (labels in JSON/PNG). ✓
- §8.1 artifacts (JSON + honest PNG) → Task 4 + Task 6. ✓
- §8.2 module shape (`snr_curve`, `retrieve_answer`, `downstream_spotcheck`, `emit_results`, `evaluate`) → Tasks 2–5. ✓ (spec's `downstream_spotcheck(rows)` one-liner is refined here to the builder form `downstream_spotcheck(questions, clean, degraded, snr)` — a real pure assembler.)
- §8.3 pure unit tests w/ fake + one marked integration → Tasks 2–5. ✓
- §9 reconciliation (main spec §12/§18) → already committed before this plan. ✓
- §10 build order → Tasks 1–6 follow it. ✓

**2. Placeholder scan:** the design-note template in Task 6 Step 3 has bracketed `[X.XX]` values — these are **intentional**: real measured numbers are unknowable until the harness runs (Step 1–2), and Step 2 mandates filling them from the artifact, never inventing. Not a plan placeholder. No "TBD"/"handle edge cases"/"similar to" anywhere.

**3. Type consistency:** `CurvePoint(snr:str, similarity:float)`, `SpotCheckRow(question, clean_answer, degraded_answer, degraded_snr)`, `EvalResult(curve, spotcheck, meta)` consistent across Tasks 2/4/5. `retrieve_answer(question, transcript, llm, k=3)` and `downstream_spotcheck(questions, clean_answers, degraded_answers, degraded_snr)` consistent between Task 3 definition and Task 5 use. `emit_results(curve, spotcheck, meta, out_json, out_png)` consistent Task 4↔5. Reused `qa.top_k_statements`/`compose_answer` and `evaltools.similarity`/`transcript_text` signatures verified against source. ✓
