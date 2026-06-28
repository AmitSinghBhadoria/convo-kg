# Atyx Convo-KG — Demo Script

> Live walkthrough for **Mon 29 Jun 2026**. Reviewer: **Debopam Bhattacherjee** —
> systems researcher, measurement-driven, values **honest scoping, controlled
> experiments, and clear reasoning over polish**. The script is built around that:
> lead with verified depth, show real generality live, then name the limits openly.

**Two things to decide before you start:**
1. **Length** — the full arc below runs ~12 min. To do ~6 min, keep Act I + Act III
   and cut Act II (live upload).
2. **Live upload (Act II)** — the plan is to let **Debopam upload his own clip** if he asks.
   It's the headline but the most fragile piece (cold loads, memory, **minutes-long** —
   ~14 min for a ~4-min clip in testing). Run it only if the preflight was clean, and
   **steer him to a short clip**. Act III replay clips are the reliable fallback.

---

## 0. Pre-flight checklist (T-10 min)

- [ ] **Neo4j Desktop** — instance `atyx` is **Started**; password matches `.env`.
- [ ] **LM Studio** — `qwen/qwen3.5-9b` **and** the `nomic-embed` model loaded,
      **Reasoning/Thinking OFF** (model settings → Reasoning → off). This is the
      #1 silent-failure cause: with thinking on, JSON goes to `reasoning_content`
      and extraction/Q&A break.
- [ ] **Memory** — close other heavy apps. Whisper-large-v3 + pyannote + the 9B share
      24 GB; live upload is the tight case.
- [ ] **Start the app** — `./start.sh`. Confirm the three green lines:
      `ok Neo4j reachable`, `ok LM Studio reachable`, `ok graph present (N nodes)`.
      Browser opens at **http://localhost:8000**.
- [ ] **Live upload = his file.** The plan is to let **Debopam upload his own clip** — the
      real generality test. Steer him on two things: **keep it short** (≤ ~2 min is
      watchable; the app hard-caps at **10 min**) and ideally **clean, multi-speaker audio**
      (it diarizes and extracts far better than narrowband phone audio). Timing is real —
      cold model loads + processing ran **~14 min for a 3.9-min clip** in testing, and scales
      with length — so a short clip matters. **Fallback** if he brings nothing:
      `~/Desktop/atyx_live_demo.mp3` (the full call_112, ~3.9 min) is staged.
- [ ] One browser tab, sensible zoom. The app opens on **PMS-advisory.wav** (the hero).

> If `./start.sh` says the graph is empty: `./start.sh --restore` (reloads the verified
> snapshot). If the port is busy: `PORT=8001 ./start.sh`.

---

## 1. Frame it (30 s)

> "Atyx takes **recorded multi-party Hinglish conversations** — noisy, real-world audio —
> and turns them into a **queryable knowledge graph** you can ask natural-language
> questions about. The hard constraint: **extraction and Q&A run entirely on a local,
> open-weight LLM** — no frontier API. Let me show you the verified result first, then
> run the whole pipeline live on an arbitrary clip, then show you exactly where it breaks."

---

## 2. Act I — The verified hero (PMS) · ~4 min

*The depth story. Everything here is verified against hand-labelled ground truth.*

1. **Landing state.** Point out the left rail: the **clip** (PMS-advisory.wav, a real
   ~10-min, multi-speaker Hinglish advisory call) and the **5-stage pipeline**
   (denoise → diarize → ASR Hinglish→EN → extract → graph).
2. **Click "Run replay."** Narrate the stages lighting up. Be honest: *"This replays a
   real processed run at watchable speed — the live version is Act II."*
3. **Transcript** appears: point out **speaker attribution** (two speakers) and that the
   output is **English, translated from Hinglish code-mix**.
4. **Knowledge graph** builds. **Click a node** → its 1-hop neighbourhood highlights.
   *"Entities and relationships are **first-class nodes**, not text blobs — so single-hop
   today extends to multi-hop later."*
5. **Ask Atyx** — click the three presets in order, and read the **◆ source quote** + the
   graph highlight for each:

   | Question | Expected answer |
   |---|---|
   | What strategy does a PMS follow? | Broad Portfolio · Concentrated Small Cap · Consistency of Alpha |
   | How does a PMS differ from a mutual fund? | Grounded answer (the "surgery vs. anaesthesia" analogy) — note: when no clean Cypher maps, it falls back to **statement-grounded** retrieval |
   | Who is a PMS meant for? | The **Affluent HNI** segment |

6. **The no-hallucination floor.** Type **"What is the capital of France?"** → it
   **declines**. *"It only answers what's grounded in the conversation graph — it won't
   make things up."* (This is usually the moment that lands with a researcher.)

---

## 3. Act II — Generality, live · OPTIONAL / RISKY

*The breadth story: prove it works on **arbitrary** audio, end-to-end, in real time —
ideally a clip **he** brings.*

> **Decision point:** only if the preflight was clean and memory is free. **Set the time
> expectation out loud before you start** (step 2).

1. **Hand it to him.** *"Upload anything you like — it runs the full pipeline live."*
   Click **"Upload a conversation to begin"** → choose his file. (Fallback if he has none:
   the staged `~/Desktop/atyx_live_demo.mp3`.)
2. **Set expectations first.** It runs the **real models with cold loads** — in testing a
   3.9-min clip took **~14 min**, and time scales with length (a 10-min clip can be
   20–25 min). So say it plainly and **steer him short:** *"a clip of a minute or two
   finishes while we keep talking."*
3. The **real pipeline** runs: denoise → diarize → ASR → extract, **streamed per-stage**.
   The transcript appears when ASR finishes; facts as extraction completes. Narrate:
   *"it's processing audio it has never seen, on-device, right now."*
4. **Frame it honestly:** *"**facts mode** — automatic extraction, **unverified**, shown
   as-is. Uploaded clips get **no graph or Q&A** — Neo4j Community is single-DB, so
   isolating a per-upload graph means namespacing the verified hero's query path.
   Documented, deferred."* If it's **clean multi-speaker** audio you'll see real
   diarization and richer facts; if it's **phone-quality**, expect **one speaker and sparse
   facts** — the documented ceiling, named not hidden.
5. **If it's slow or fails:** the error is **per-stage and honest** — show it
   (*"that's the failure contract: it tells you which stage broke — no hang, no fake
   success"*), then pivot straight to Act III.

---

## 4. Act III — The honest boundary · ~2–3 min

*The measurement story — Debopam's wheelhouse.*

1. **Clip dropdown → call_103** (911 dispatch — active shooter) → **Run replay**.
   Facts mode: pipeline rail + transcript + extracted facts.
2. Point at the amber note above the transcript: *"**single-channel / phone-quality audio
   — diarization could not separate speakers.** This is a 911 phone call: ~8 kHz,
   compressed, one mixed line. pyannote collapses it to one speaker. I'm **showing you the
   ceiling, not hiding it** — and the live path uses the same diarizer, so it'd hit the
   same wall."*
3. **Switch to the EXPERIMENT tab.** Show the **controlled-SNR curve** — transcript
   **fidelity vs. SNR** over a café-babble sweep. *"The audio spine degrades **gracefully**
   as noise rises; this is the reproducible, controlled experiment behind the honesty."*

---

## 5. Close — what's stubbed, and the scaling path · ~1–2 min

> "To be clear about the limits:
> - **Single-hop Q&A** now — but the graph stores **first-class entities and relations**,
>   so **multi-hop is a natural extension**, not a rewrite.
> - **No graph/Q&A for uploaded clips** — Neo4j Community single-DB; isolating them needs
>   namespacing. Deferred by design; facts mode is the honest surface.
> - **Extraction on noisy code-mixed Hinglish is the measured bottleneck** — a local ~9B
>   ceiling, not a pipeline bug. The fix path (better/larger local model, targeted
>   prompting) is in `design_note.md`.
> - **Provenance is coarse** — it returns a best-effort source statement, not always the
>   single best line. Noted, with the precision fix.
>
> The spine is solid and honest end-to-end; everything stubbed is stubbed **on purpose**,
> with a stated path to close it."

---

## 6. Anticipated questions

| They ask | Your answer |
|---|---|
| Why a local open-weight LLM? | It's the brief's hard constraint — and it proves the approach works without a frontier API (privacy, cost, on-device). |
| Why single-hop only? | Scoped per the brief. The graph is multi-hop-ready (first-class entities/edges) — I can walk a worked multi-hop example in the design note. |
| How accurate is extraction? | Honestly: it's the measured ceiling on noisy Hinglish. See the SNR curve and the capability-boundary section of `design_note.md`. |
| Why one speaker on the 911 clip? | Phone audio (narrowband, single mixed channel) defeats pyannote speaker separation. Named in the UI, not hidden. |
| Is the demo reproducible? | Yes — `./setup.sh` then `./start.sh` from the README; the verified graph restores from a committed snapshot. |

---

## 7. Fallbacks (if something breaks mid-demo)

- **LM Studio down** → Graph + Experiment tabs still work; only Ask Atyx and live Run need
  it. Restart LM Studio (Thinking OFF), reload the page.
- **Graph empty / weird** → `./start.sh --restore` to reset to the verified snapshot.
- **Port busy** → `PORT=8001 ./start.sh`.
- **Live upload flaky** → pivot to Act III replay clips; the per-stage error is itself an
  honest demo of the failure contract.
- **Reload lands on the wrong clip** → reselect **PMS-advisory.wav** from the dropdown
  (active clip is server-global within a session; a fresh `./start.sh` always opens on PMS).
