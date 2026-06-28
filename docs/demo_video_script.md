# Atyx Convo-KG — Demo Video Script (~3 min, email share)

> **Format:** voiceover narration paired with on-screen actions and timestamps.
> **Audience:** Debopam Bhattacherjee (systems researcher) — keep the tone measured and
> honest, not salesy. He rewards "here's what works, here's the limit."
> **Arc:** verified PMS hero (most of the runtime) → a sped-up live-upload time-lapse for
> generality → an honest close.

Total target: **~3:05**. Voiceover ≈ 380 words (comfortable at a calm pace).

---

## The script

| Time | On screen (actions) | Voiceover (read aloud) |
|------|---------------------|------------------------|
| **0:00–0:06** | Title card: **Atyx Convo-KG** — "Conversational knowledge graph on a local LLM." Then cut to the app loaded on the PMS clip. | "This is Atyx Convo-KG." |
| **0:06–0:28** | App on screen (dark console, PMS-advisory.wav selected, pipeline rail on the left, Ask Atyx on the right). | "It turns recorded, multi-party Hinglish conversations — noisy, real-world audio — into a knowledge graph you can ask questions about. The constraint that shaped it: extraction and question-answering run entirely on a local, open-weight model. No frontier API. Let me show you." |
| **0:28–1:02** | Click **Run replay**. The pipeline rail animates through Speech enhancement → Diarization → Transcribe. The transcript panel fills in. | "I'll run our verified clip — a ten-minute private-wealth advisory call, in Hinglish. The pipeline cleans the audio, separates the speakers, then transcribes and translates to English. Here's the result: speaker-attributed, Hindi-English code-mix in, clean English out." |
| **1:02–1:40** | The **Knowledge Graph** renders. Slowly **click one node** → its one-hop neighbourhood highlights. | "From that transcript, a local nine-billion-parameter model pulls out entities and relationships into a Neo4j graph. These aren't text blobs — they're first-class nodes and edges, so the single-hop questions we answer today extend to multi-hop later. Click any node, and you see what it connects to." |
| **1:40–2:02** | In **Ask Atyx**, click preset **"What strategy does a PMS follow?"** Answer appears with the **◆ source quote**. Briefly click the second and third presets. | "Now the questions. 'What strategy does a PMS follow?' — answered, and grounded: every answer carries the exact quote it came from. How it differs from a mutual fund. Who it's for — the affluent HNI segment." |
| **2:02–2:25** | Type **"What is the capital of France?"** → it **declines** ("I couldn't find that in the conversation."). | "And here's the part I care about most. Ask something the conversation never covered — the capital of France — and it declines. It only answers what's grounded in the graph. It will not make things up." |
| **2:25–2:50** | **[SPED-UP TIME-LAPSE ~15×]** Switch to an arbitrary clip, click upload, watch the pipeline stream stages → transcript → extracted facts. On-screen caption: **"live pipeline · sped up ~15×"**. | "That's the verified core. To show it generalises, here's the whole pipeline running live on a clip it has never seen — sped up for time. It processes locally, start to finish, on the same machine." |
| **2:50–3:05** | Cut back to the facts panel, then the graph / title card. | "And I'm honest about the edges: on noisy single-channel phone audio, speaker separation collapses — the demo shows that, it doesn't hide it. The full write-up and a reproducible setup are in the repo. Thanks for watching." |

---

## Production notes

**Before you hit record:**
- Run `./start.sh` fresh so the app opens on the **PMS hero** (graph mode).
- **Warm up LM Studio**: ask one throwaway question and run one replay *before* recording, so the on-camera Q&A responses come back fast (the first call after a cold load is slow). Confirm **Reasoning/Thinking is OFF**.
- Confirm the three preset answers land: *Broad Portfolio / Concentrated Small Cap / Consistency of Alpha* · the mutual-fund contrast · *Affluent HNI segment* · and that "capital of France" **declines**.
- Record at 1080p, browser zoomed so text is legible, cursor visible. One clean take per segment is easier than one long take.

**The live-upload time-lapse (2:25–2:50):**
- Record the real upload run **separately** (it takes minutes end-to-end), then **speed it up ~15×** in your editor to ~20–25 seconds. Add the caption "live pipeline · sped up ~15×" so it's honest.
- Use a **short clip (~60–90 s of audio)** for this segment — less footage to speed up, and the stages still visibly progress. Any conversational clip not already in the picker works (it demonstrates "arbitrary audio").
- Keep the stage rail and the streaming transcript/facts in frame — that progression is the point.

**Editing:**
- Open with the title card; end on the graph or title card.
- Optional lower-third captions for the three Q&A questions and the decline — helps a muted viewer.
- Keep it tight; if you run long, trim the second and third preset questions (2:02 area), not the decline.

**Honesty guardrails (so the video matches the design note):**
- Don't imply uploaded clips get a graph or Q&A — they're facts-only.
- Frame the single-speaker phone-audio result as a known ceiling, not a glitch.
- "Run replay" is a replay of a real processed run — the voiceover already says so for the hero; the live-upload segment is the genuinely-live part.

---

## If you want a 90-second cut later
Keep 0:00–0:06 (hook), 1:02–1:40 (graph), 2:02–2:25 (the decline), 2:50–3:05 (close). Drop the pipeline run and the live-upload time-lapse. That preserves the two moments that land hardest: grounded answers and the refusal.
