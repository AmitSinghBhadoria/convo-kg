# Atyx Convo-KG — Wireframes

ASCII wireframes for all major screens and states. Regions are labelled; box-drawing characters
mark panel borders. Proportions are approximate (not pixel-exact).

**Grid dimensions (actual):**
- Graph mode: `264px | 1fr | 372px` (3 columns)
- Facts / live mode: `264px | 1fr` (2 columns)
- Header bar: 58 px, full width

See also: [./wireflows.md](./wireflows.md) · [./user-stories.md](./user-stories.md) ·
[./sequence-diagrams.md](./sequence-diagrams.md)

---

## WF-1 — Console · Graph mode (pms clip, run complete)

3-column layout. The center column stacks Transcript (top ~37%) and Knowledge Graph (bottom).
The right column is Ask Atyx. Pipeline shows 5 stages.

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  Atyx. · Convo-KG          ┌─────────────────────────┐                          ● live          ║
║                            │  Console  │  Experiment  │                                         ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║ ╔══════════════════════╗  ╔═══════════════════════════════════════╗  ╔══════════════════════════╗ ║
║ ║ ▾ pms                ║  ║ TRANSCRIPT                            ║  ║ ASK ATYX                 ║ ║
║ ║   Private-wealth     ║  ╠═══════════════════════════════════════╣  ║          ● graph-grounded ║ ║
║ ║   advisory           ║  ║ S0 0:03  the portfolio management     ║  ╠══════════════════════════╣ ║
║ ╠══════════════════════╣  ║          service offers a separately  ║  ║                          ║ ║
║ ║ [▶  Run replay      ]║  ║          managed account…             ║  ║  ┌────────────────────┐  ║ ║
║ ╠══════════════════════╣  ║ S1 0:11  yes, minimum ticket size     ║  ║  │ What strategy does │  ║ ║
║ ║ PIPELINE             ║  ║          for PMS is fifty lakhs…      ║  ║  │ a PMS follow?  [→] │  ║ ║
║ ║ ● Speech enhancement ║  ║ S0 0:19  and unlike mutual funds,     ║  ║  └────────────────────┘  ║ ║
║ ║   DeepFilterNet      ║  ║          the investor retains direct  ║  ║  ┌────────────────────┐  ║ ║
║ ║ ● Diarization        ║  ║          ownership of securities…     ║  ║  │ How does a PMS     │  ║ ║
║ ║   pyannote 3.x       ║  ║ S1 0:28  right, and the portfolio     ║  ║  │ differ from a      │  ║ ║
║ ║ ● Transcribe·HI→EN   ║  ║          manager takes discretionary… ║  ║  │ mutual fund?   [→] │  ║ ║
║ ║   Whisper large-v3   ║  ╠═══════════════════════════════════════╣  ║  └────────────────────┘  ║ ║
║ ║ ● Fact extraction    ║  ║ KNOWLEDGE GRAPH  · click node to      ║  ║  ┌────────────────────┐  ║ ║
║ ║   Qwen 9B · 4-bit    ║  ║   explore                             ║  ║  │ Who is a PMS       │  ║ ║
║ ║ ● Graph build        ║  ║                                       ║  ║  │ meant for?     [→] │  ║ ║
║ ║   Neo4j · Cypher     ║  ║   [PMS] ─────FOLLOWS──────▶ [Strategy]║  ║  └────────────────────┘  ║ ║
║ ║                      ║  ║      │                                ║  ╠══════════════════════════╣ ║
║ ║                      ║  ║      DIFFERS_FROM                     ║  ║                          ║ ║
║ ║                      ║  ║      ▼                                ║  ║  Run the pipeline to     ║ ║
║ ║                      ║  ║   [MutualFund]    [Investor]          ║  ║  ground the chat in the  ║ ║
║ ║                      ║  ║        │               │              ║  ║  graph.                  ║ ║
║ ║                      ║  ║        HAS_MIN_TICKET   ELIGIBLE_FOR  ║  ╠══════════════════════════╣ ║
║ ║                      ║  ║        ▼               ▼             ║  ║ ┌──────────────────────┐  ║ ║
║ ║                      ║  ║   [50L Ticket]   [HNI Segment]        ║  ║ │ Ask a question… [↑] │  ║ ║
║ ╚══════════════════════╝  ╚═══════════════════════════════════════╝  ║ └──────────────────────┘  ║ ║
║                                                                      ╚══════════════════════════╝ ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## WF-2 — Console · Facts / live mode (call_100 or uploaded clip)

2-column layout. Graph build stage is absent from the pipeline rail (4 stages only). Knowledge
Graph and Ask Atyx are not rendered. Single-speaker note appears when diarization collapses.

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  Atyx. · Convo-KG          ┌─────────────────────────┐                          ● live          ║
║                            │  Console  │  Experiment  │                                         ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║ ╔══════════════════════╗  ╔══════════════════════════════════════════════════════════════════╗   ║
║ ║ ▾ call_100           ║  ║ TRANSCRIPT                                                      ║   ║
║ ║   911 water rescue   ║  ╠══════════════════════════════════════════════════════════════════╣   ║
║ ╠══════════════════════╣  ║ S0 0:01  fire department, what is your emergency?               ║   ║
║ ║ [▶  Run replay      ]║  ║ S0 0:05  okay we have a water rescue situation at the bridge    ║   ║
║ ╠══════════════════════╣  ║ S0 0:10  we need immediate assistance, send units now           ║   ║
║ ║ PIPELINE             ║  ║ ...                                                             ║   ║
║ ║ ● Speech enhancement ║  ║                                                                 ║   ║
║ ║   DeepFilterNet      ║  ║ ⚠  diarization could not separate speakers for this clip       ║   ║
║ ║ ● Diarization        ║  ╠══════════════════════════════════════════════════════════════════╣   ║
║ ║   pyannote 3.x       ║  ║ EXTRACTED FACTS                                                 ║   ║
║ ║ ● Transcribe·HI→EN   ║  ║ automatic extraction — unverified · this clip has no graph or  ║   ║
║ ║   Whisper large-v3   ║  ║ Q&A                                                             ║   ║
║ ║ ● Fact extraction    ║  ╠══════════════════════════════════════════════════════════════════╣   ║
║ ║   Qwen 9B · 4-bit    ║  ║ • Incident type: water rescue                                   ║   ║
║ ║                      ║  ║ • Location: bridge [address]                                    ║   ║
║ ║ ─ Graph build ─      ║  ║ • Action required: dispatch fire department units               ║   ║
║ ║   (not in this mode) ║  ║ • Urgency: immediate                                            ║   ║
║ ╠══════════════════════╣  ║ ...                                                             ║   ║
║ ║ automatic extraction ║  ║                                                                 ║   ║
║ ║ — unverified         ║  ║                                                                 ║   ║
║ ║ • Incident: water    ║  ║                                                                 ║   ║
║ ║   rescue             ║  ║                                                                 ║   ║
║ ║ • Location: bridge   ║  ║                                                                 ║   ║
║ ╚══════════════════════╝  ╚══════════════════════════════════════════════════════════════════╝   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## WF-3 — Empty / upload state

Shown in the center column before any run when no transcript exists. The left rail clip dropdown
and Run button are still visible. Clicking the upload area or "Run replay" are the two actions
available.

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  Atyx. · Convo-KG          ┌─────────────────────────┐                          ● live          ║
║                            │  Console  │  Experiment  │                                         ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║ ╔══════════════════════╗  ╔══════════════════════════════════════════════════════════════════╗   ║
║ ║ ▾ pms                ║  ║                                                                 ║   ║
║ ║   Private-wealth     ║  ║                                                                 ║   ║
║ ║   advisory           ║  ║                                                                 ║   ║
║ ╠══════════════════════╣  ║                  ┌──────────────────────────────┐               ║   ║
║ ║ [▶  Run replay      ]║  ║                  │                              │               ║   ║
║ ╠══════════════════════╣  ║                  │          ↑                   │               ║   ║
║ ║ PIPELINE             ║  ║                  │                              │               ║   ║
║ ║ ○ Speech enhancement ║  ║                  │  Upload a conversation       │               ║   ║
║ ║   DeepFilterNet      ║  ║                  │  to begin                    │               ║   ║
║ ║ ○ Diarization        ║  ║                  │                              │               ║   ║
║ ║   pyannote 3.x       ║  ║                  │  click to select an audio    │               ║   ║
║ ║ ○ Transcribe·HI→EN   ║  ║                  │  file · or use Run replay    │               ║   ║
║ ║   Whisper large-v3   ║  ║                  │  to load the selected clip   │               ║   ║
║ ║ ○ Fact extraction    ║  ║                  │                              │               ║   ║
║ ║   Qwen 9B · 4-bit    ║  ║                  └──────────────────────────────┘               ║   ║
║ ║ ○ Graph build        ║  ║                                                                 ║   ║
║ ║   Neo4j · Cypher     ║  ║                                                                 ║   ║
║ ║                      ║  ║                                                                 ║   ║
║ ╚══════════════════════╝  ╚══════════════════════════════════════════════════════════════════╝   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

  ○ = pending stage dot   ● = active/complete stage dot
```

---

## WF-4 — Clip dropdown expanded

The picker overlays the left rail when the clip name is clicked. It lists the three registered
clips with their labels and active marker. Clicking outside or pressing Escape closes it.

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  Atyx. · Convo-KG          ┌─────────────────────────┐                          ● live          ║
║                            │  Console  │  Experiment  │                                         ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║ ╔══════════════════════╗                                                                         ║
║ ║ ▴ pms        ← open  ║                                                                         ║
║ ║                      ║  ┌──────────────────────────────────────┐                              ║
║ ║  ┌────────────────┐  ║  │  CLIP PICKER                         │                              ║
║ ║  │ ✓ pms          │  ║  │                                      │                              ║
║ ║  │   Private-wealth│  ║  │  ✓ pms                               │                              ║
║ ║  │   advisory      │  ║  │    Private-wealth advisory · graph   │                              ║
║ ║  │                 │  ║  │    2 speakers · VERIFIED             │                              ║
║ ║  │   call_100      │  ║  │                                      │                              ║
║ ║  │   911 water     │  ║  │    call_100                          │                              ║
║ ║  │   rescue        │  ║  │    911 water rescue · facts          │                              ║
║ ║  │                 │  ║  │    phone audio → 1 speaker           │                              ║
║ ║  │   call_103      │  ║  │                                      │                              ║
║ ║  │   911 active    │  ║  │    call_103                          │                              ║
║ ║  │   shooter       │  ║  │    911 active shooter · facts        │                              ║
║ ║  └────────────────┘  ║  │    phone audio → 1 speaker           │                              ║
║ ║                      ║  └──────────────────────────────────────┘                              ║
║ ╚══════════════════════╝                                                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## WF-5 — Ask-Atyx answer detail

Shown in the right column (graph mode only) after a question is answered. The answer bubble
includes the composed English answer, a ◆ source quote pulled from the matching Statement node,
and the referenced graph nodes are highlighted in the SVG. The answer is HTTP 200 even when
`found: false` — the decline message replaces the answer text.

```
╔══════════════════════════════════════════╗
║ ASK ATYX           ● graph-grounded      ║
╠══════════════════════════════════════════╣
║                                          ║
║  ┌──────────────────────────────────┐    ║
║  │ You                              │    ║
║  │                                  │    ║
║  │  What strategy does a PMS follow?│    ║
║  └──────────────────────────────────┘    ║
║                                          ║
║  ┌──────────────────────────────────┐    ║
║  │ Atyx                             │    ║
║  │                                  │    ║
║  │  A PMS (Portfolio Management     │    ║
║  │  Service) follows a separately   │    ║
║  │  managed account strategy where  │    ║
║  │  the investor retains direct      │    ║
║  │  ownership of each security in   │    ║
║  │  the portfolio.                  │    ║
║  │                                  │    ║
║  │  ◆ "the investor retains direct  │    ║
║  │    ownership of securities in    │    ║
║  │    the portfolio rather than     │    ║
║  │    units in a fund"              │    ║
║  │          — S0, 0:19              │    ║
║  └──────────────────────────────────┘    ║
║                                          ║
║  ┌──────────────────────────────────┐    ║
║  │ How does a PMS differ…      [→]  │    ║
║  └──────────────────────────────────┘    ║
║  ┌──────────────────────────────────┐    ║
║  │ Who is a PMS meant for?     [→]  │    ║
║  └──────────────────────────────────┘    ║
╠══════════════════════════════════════════╣
║ ┌────────────────────────────────────┐   ║
║ │ Ask a question…              [↑]   │   ║
║ └────────────────────────────────────┘   ║
╚══════════════════════════════════════════╝

  Decline state (found: false):
  ┌──────────────────────────────────┐
  │ Atyx                             │
  │                                  │
  │  No answer found in the graph.   │
  └──────────────────────────────────┘
```

---

## WF-6 — Experiment tab

Full-width scrollable page. SNR Degradation Study heading, a line-chart SVG (transcript cosine
similarity vs SNR dB, café-babble sweep), and spotcheck rows comparing clean-clip answers to
degraded-clip answers. If `snr_results.json` is absent, the chart and spotcheck are not rendered
(API 404 — no crash).

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  Atyx. · Convo-KG          ┌─────────────────────────┐                          ● live          ║
║                            │  Console  │  Experiment  │                                         ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                  ║
║   EXPERIMENT · SNR DEGRADATION STUDY                                                            ║
║                                                                                                  ║
║   Where the pipeline breaks under noise                                                          ║
║                                                                                                  ║
║   The same sample clip, degraded with café babble at controlled signal-to-noise ratios.          ║
║   One metric measured: transcript similarity (cosine) vs clean reference.                        ║
║                                                                                                  ║
║  ╔═════════════════════════════════════════════════════════════════════════════════╗              ║
║  ║  ── Transcript similarity                                                       ║              ║
║  ║                                                                                 ║              ║
║  ║  1.00 ┤                                                                         ║              ║
║  ║  0.90 ┤·····╮                                                                   ║              ║
║  ║  0.80 ┤     ╰───╮                                                               ║              ║
║  ║  0.70 ┤         ╰────╮                                                          ║              ║
║  ║  0.60 ┤              ╰───╮                                                      ║              ║
║  ║  0.50 ┤                  ╰────╮                                                 ║              ║
║  ║  0.40 ┤                       ╰────╮                                            ║              ║
║  ║  0.30 ┤                            ╰────────                                    ║              ║
║  ║       └──────┬──────┬──────┬──────┬──────┬────                                 ║              ║
║  ║              20     15     10      5      0  dB                                 ║              ║
║  ║                            SNR (dB) — NOISIER →                                ║              ║
║  ╚═════════════════════════════════════════════════════════════════════════════════╝              ║
║                                                                                                  ║
║   SPOTCHECK  · Q&A answers at clean vs degraded SNR                                             ║
║                                                                                                  ║
║  ┌──────────────────────────────────────────────────────────────────────────────────┐            ║
║  │ Q: What strategy does a PMS follow?                                              │            ║
║  │ clean  → separately managed account; investor owns securities directly           │            ║
║  │ 5 dB   → [degraded transcript answer, possibly truncated or inaccurate]          │            ║
║  └──────────────────────────────────────────────────────────────────────────────────┘            ║
║  ┌──────────────────────────────────────────────────────────────────────────────────┐            ║
║  │ Q: Who is a PMS meant for?                                                       │            ║
║  │ clean  → high-net-worth individuals with minimum 50 lakh ticket size             │            ║
║  │ 0 dB   → [heavily degraded; answer quality visibly drops]                        │            ║
║  └──────────────────────────────────────────────────────────────────────────────────┘            ║
║                                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
```
