# Atyx — Phase 4: Controlled-SNR Evaluation Harness (`evaluate.py`)

**Date:** 2026-06-27 · **Target demo:** Mon 29 Jun 2026 · **Reviewer:** Debopam Bhattacherjee
**Status:** design approved in brainstorming (2026-06-27); pending final user review before plan.
**Supersedes:** the three-curve / frontier-oracle vision in the main design spec §12 (see §9 reconciliation below).

---

## 1. Goal (one line)
Measure how the audio front-end (denoise → diarize → Hinglish→English ASR) degrades as **controlled background noise** rises, and report it as **one honest, reproducible curve** — transcript fidelity vs SNR — with an *illustrative* look at how that degradation propagates downstream.

## 2. Why this design (the measured reality)
The main spec §12 envisioned **three curves on one SNR axis** — transcript-similarity (ceiling), **fact-recall (hero)**, and Q&A-correctness (outcome) — graded against a frontier-oracle, human-verified, **fact-level** ground truth.

Phase 3 measured a **hard extraction-quality ceiling** on the real `pms` clip: on noisy, code-mixed, discursive Hinglish the local ~9B model's fact extraction is **nondeterministic at temp=0 and only partially recalls** facts (documented in `design_note.md § Measured capability boundary`). A fact-recall *curve* built on top of that would measure **model variance**, not **noise sensitivity** — it would move for reasons unrelated to SNR and mislead the reader about what the experiment controls. Grading honestly means **not drawing a calibrated curve through a noisy estimator.**

So Phase 4 narrows to what the pipeline **can** measure cleanly and defensibly:
- **Hero curve = transcript similarity vs SNR.** This is the *controlled* measurement — one variable (noise), a deterministic ASR front-end, an oracle-free reference. It is exactly the "ceiling" curve of the original three, promoted to hero because it is the one we can stand behind.
- **Downstream is a spot-check, explicitly labelled *illustrative propagation, not a calibrated curve*** — we run the golden Q&A questions at clean vs a degraded SNR and *show* the front-end error flowing through, without claiming a graded accuracy number the estimator can't support.

This is the honesty bar Debopam values: **report the controlled experiment you actually ran, and name the limit of the one you didn't.**

## 3. Methodology — one controlled variable
`add_noise.py` mixes a **real noise clip** into the clean speech slice at a known **SNR** (signal-to-noise ratio, dB), preserving the ratio exactly (RMS-matched, clipping-guarded). **SNR is the single independent variable.** Everything else — the speech slice, the noise source, the denoise model, the ASR model, decoding params — is held fixed across the sweep. Five SNR points trace the **shape** of degradation:

| SNR (dB) | Condition |
|---|---|
| clean | no noise added — baseline reference transcript |
| 20 | light café hum |
| 15 | noticeable |
| 10 | speech ≈ noise territory begins |
| 5 | hard |
| 0 | speech and noise equal power — worst case |

**Hard rule (approved):** keep **all five** SNR points. Never drop to three points to afford a longer clip — five points show the cliff; three hide it.

## 4. The reference — oracle-free, self-contained
The hero curve needs no external ground truth. The **clean slice run through the identical pipeline** produces the **baseline transcript**; each noisy transcript is scored against *that*. This isolates the variable cleanly: we measure **what noise does to the front-end**, with the front-end's own clean output as the yardstick — no second model, no human-labelling oracle, no inter-annotator drift. (A frontier-oracle, human-verified reference would let us measure *absolute* WER; that is the scaling path in §9, deliberately not in Phase 4.)

**Similarity metric** = `src/evaltools.py::similarity(hyp, ref)` — the existing helper:
`round(0.5 · SequenceMatcher.ratio + 0.5 · jaccard(token sets), 4)` over `normalize()`d text. A blend of **sequence** (word order / fluency) and **set** (content-word overlap) similarity, in [0, 1]. It is a **relative** fidelity score, not WER; the curve's *shape and cliff* are the finding, not any single absolute value. This is stated on the artifact.

## 5. Noise source — real café babble (the controlled noise)
The noise is **real recorded café babble**: `noices/cafe_16k.wav` (16 kHz mono, ~2:08, listened-clean). **Not** synthetic pink/white noise.

Rationale: **multi-talker babble is both realistic and the hard case for conversational ASR** — its spectrum overlaps speech and it carries phantom phoneme-like energy that a denoiser cannot cleanly separate, exactly the failure mode a private-wealth client's real recordings hit. Synthetic stationary noise is easier and less representative. Using real babble makes the curve a credible stand-in for field conditions.

`add_noise.py`'s **SNR control is unchanged** — only the noise *source* differs from the original synthetic-pink plan. SNR remains the single controlled variable. The clip is shorter (~128 s) than the 160 s slice; `add_noise.fit_noise` tiles it to length before mixing, so coverage is complete. (Consequence, noted: the last ~32 s of noise repeats the first ~32 s. Harmless for babble — it is stationary-ish ambient texture, not content — and the SNR ratio is preserved exactly across the whole slice.)

**Scope discipline (approved):** the **café-babble 5-point hero curve is THE Phase 4 deliverable.** `eating.wav` / restaurant noise are *optional comparison points only if time allows AFTER* the babble curve is done — do **not** expand the noise battery now and eat the Phase 5 buffer.

## 6. The slice — a 2-speaker conversational slice, golden-fact-bearing
A **160 s slice (`pms.wav[0:160]`)** is the evaluation clip — a **2-speaker conversational slice** (not the full clip's 4-speaker spread). Chosen against three constraints:
1. **Representative 2-speaker conversation** — fresh diarization on the slice labels **2 speakers** (SPEAKER_00/01) in natural back-and-forth (one asks, the expert answers at length), exercising diarization under noise. This window was taken over the originally-considered 360–520 s (4-speaker) window because **only 0–160 s carries the golden facts** the spot-check needs; the front-end fidelity curve is valid on 2 speakers, so the trade favours fact coverage. (The full clip's 4 global speaker labels do not carry into a slice — diarization is re-run per clip and labels are local.) Every artifact calls this a **"2-speaker conversational slice,"** never "multi-party," so the curve is not misrepresented.
2. **Bounded compute** — ~160 s × 6 runs (clean + 5 SNRs) through the real denoise+ASR pipeline fits the Phase 4 time budget.
3. **Carries the golden-Q&A facts (verified, not assumed).** The §6 build-time gate ran against the clean baseline transcript and confirms the window's content: **PMS-vs-Mutual-Fund is strongly present** ("there are quite a few mutual funds which are run as if they are a PMS"; 8× "mutual fund", 16× "PMS"), along with **fees/minimums** (₹ lakh/crore, light-touch regulation), **White Oak** (offers both), and the **audience** ("we're working with a lot of younger investors"). The literal "transparency" and "strategy" statements fall **outside** this window — so the spot-check (§7) uses the facts that **are** in the slice, per the approved contingency, rather than questions the slice cannot answer. The alternative 360–520 s window was rejected because the gate found it carries **zero** golden facts.

## 7. Downstream spot-check — illustrative, not calibrated
After the hero curve, a **bounded** look at propagation: take the **clean** and **one degraded** SNR transcript, run the **in-slice golden Q&A questions** (verified present in §6 — PMS-vs-Mutual-Fund, fees/minimums, who-it's-for) against the graph built from each, and present the answers **side by side** to *show* front-end degradation flowing into the answer.

This is **explicitly labelled on every artifact as "illustrative propagation, not a calibrated curve."** It is one or two concrete before/after examples, not a scored accuracy metric — because (per §2) the extraction estimator is too noisy to calibrate. It demonstrates the *mechanism* (worse transcript → worse/missing answer) honestly, without overclaiming a number.

## 8. Outputs & module design

### 8.1 Artifacts
- **`data/ground_truth/snr_results.json`** — the machine-readable record: per-SNR similarity scores, the slice/noise/metric metadata, and the spot-check Q&A pairs. Single source for the curve and the design-note paragraph; later consumed by the Phase 6 `GET /api/experiment` endpoint.
- **`snr_curve.png`** — the hero figure. **Requirements (approved, non-negotiable):**
  - **Labelled axes:** x = SNR (dB), y = transcript similarity [0–1].
  - **Honest y-axis floor** — do **not** auto-scale to dramatize small drops; the y-range starts at a fixed honest floor (0, or a clearly-marked floor) so the curve's magnitude reads truthfully.
  - **The cliff is annotated** — the SNR where fidelity falls off is called out on the plot.
  - **Downstream spot-check, if shown on the figure, is captioned** "illustrative propagation, not a calibrated curve."
  - **The similarity metric is named** as a relative fidelity blend (not WER).
- **`design_note.md` results paragraph** — the prose finding: the curve shape, where the cliff is, what it means for the front-end, and the explicit limit (no absolute WER; fact-recall not curved — and why).

### 8.2 `src/evaluate.py` — module shape (signatures + contracts; bodies in the plan)
A pure/IO-split design mirroring the rest of the codebase (pure functions unit-tested with a fake transcriber; one integration test over the real sweep artifacts).

```python
# Pure assembly — no IO. Given (snr_label -> transcript) load the curve.
def snr_curve(baseline: Transcript,
              noisy: dict[str, Transcript]) -> list[CurvePoint]
    # CurvePoint = {snr: str, similarity: float}; similarity(noisy_text, baseline_text)
    # using evaltools.similarity. Deterministic, oracle-free.

# Pure — assemble the illustrative spot-check rows from pre-computed answers.
def downstream_spotcheck(rows: list[SpotCheckRow]) -> list[SpotCheckRow]
    # SpotCheckRow = {question, clean_answer, degraded_answer, degraded_snr}
    # No grading verdict — illustrative side-by-side only.

# IO — write JSON + render PNG (matplotlib). Honest y-floor, labelled axes,
# annotated cliff, illustrative caption. Returns the two paths written.
def emit_results(curve: list[CurvePoint],
                 spotcheck: list[SpotCheckRow],
                 meta: dict,
                 out_json: str, out_png: str) -> tuple[str, str]

# Orchestrator (eval entrypoint). Reads the sweep transcripts already on disk
# (data/work/<clip>.transcript.json), builds the curve, runs the spot-check
# Q&A, emits artifacts. Does NOT re-run ASR — the sweep (scratch→prep) owns that.
def evaluate(baseline_clip: str, snr_clips: dict[str, str],
             spotcheck_questions: list[str], degraded_snr: str) -> EvalResult
```

**Contracts (add to `contracts.py`):** `CurvePoint`, `SpotCheckRow`, `EvalResult` (Pydantic) — locked as the Phase 6 `/api/experiment` shape.

**Sweep harness:** the long-pole ASR compute (cut slice → mix each SNR → denoise+ASR) is a **prep step** that produces the six `*.transcript.json` files. In Phase 4 it lives as a small script (`scratch_sweep.py` → promoted to `scripts/prep_eval_clips.py`); `evaluate.py` consumes its disk output. This keeps the expensive, non-deterministic-timing audio stage out of the unit-tested pure core.

### 8.3 Tests
- **Pure unit tests (fake transcriber, no audio, no LLM):** `snr_curve` orders/score-matches points; identical transcripts → similarity 1.0; a degraded transcript scores strictly lower; `downstream_spotcheck` assembles rows without inventing a verdict; `emit_results` writes valid JSON and a non-empty PNG to a tmp path with the honest y-floor (assert axis limits) and required captions.
- **One integration test (marked):** over the **real** sweep artifacts on disk — assert five SNR points present, similarity monotone-ish and within [0,1], the cliff identifiable, artifacts written.

## 9. Reconciliation with the main spec, and the scaling path
**Main spec §12 is updated** (this phase) to record the narrowing: hero = transcript-similarity vs SNR on real café babble; fact-recall and Q&A-correctness **curves deferred** as the scaling path, with the downstream spot-check standing in *illustratively*. The §12 frontier-oracle, human-verified, fact-level ground-truth machinery (`scripts/make_ground_truth.py`) is **not built in Phase 4** — it is the documented path to a calibrated fact-recall curve once a stronger/Cypher-tuned extraction model lifts the ceiling.

**What's honest / what's stubbed (for the design note):**
- **Built & defensible:** controlled single-variable SNR sweep, real-babble noise, oracle-free relative-fidelity hero curve with annotated cliff, reproducible from the README.
- **Illustrative only:** the downstream Q&A propagation (labelled as such on the artifact).
- **Stubbed / scaling path:** absolute WER (needs verified reference), the fact-recall and Q&A-correctness *curves* (need the frontier-oracle ground truth + a higher extraction ceiling), longer/more clips, a battery of noise types.

## 10. Build order (Phase 4)
1. **Sweep prep** (running): cut the 160 s slice, mix all 5 café-babble SNRs, run denoise+ASR → six `*.transcript.json`. *(long pole; in flight)*
2. **Verify** the clean baseline transcript contains the golden-Q&A content (§6 gate).
3. **Contracts** — `CurvePoint`, `SpotCheckRow`, `EvalResult`.
4. **`evaluate.py` pure core** — `snr_curve`, `downstream_spotcheck` + unit tests.
5. **`emit_results`** — JSON + honest PNG + unit tests (axis-limit / caption asserts).
6. **Spot-check** — run golden Q&A on clean vs degraded graphs; assemble rows.
7. **Orchestrator + integration test** over the real sweep artifacts.
8. **Wire-up** — `snr_results.json`, `snr_curve.png`, `design_note.md` results paragraph; promote sweep script to `scripts/prep_eval_clips.py`; main spec §12 note.
