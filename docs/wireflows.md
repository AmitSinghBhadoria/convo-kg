# Atyx Convo-KG — Wireflows

Navigation flows for the single-screen local app. Each flow is a Mermaid flowchart. Decision
diamonds mark branching paths; honest negative/edge cases are included throughout.

See also: [./wireframes.md](./wireframes.md) · [./user-stories.md](./user-stories.md) ·
[./sequence-diagrams.md](./sequence-diagrams.md) · [./api-specification.md](./api-specification.md)

---

## Flow 1 — App entry and verified-hero run

The default view when the app loads. The `pms` clip (Private-wealth advisory, graph mode) is
active by default. The Knowledge Graph pre-loads from `/api/graph` on mount, so the graph is
visible even before a Run. Clicking "Run replay" re-executes the 5-stage pipeline over the
committed JSON artifacts and re-populates the transcript and graph.

```mermaid
flowchart TD
    Start([Open localhost:8000]) --> Load[App loads\nConsole tab · pms clip active]
    Load --> GraphMount[GET /api/graph on mount\nKnowledge Graph pre-renders]
    GraphMount --> ReadyIdle[3-column layout\nRun replay button enabled]

    ReadyIdle --> ClickRun{Click Run replay?}

    ClickRun -->|Yes| PostRun[POST /api/run\n→ run_id returned]
    PostRun --> SSE[GET /api/run/id/stream\nSSE stream opens]
    SSE --> Stage1[stage: Speech enhancement\nDeepFilterNet]
    Stage1 --> Stage2[stage: Diarization\npyannote 3.x]
    Stage2 --> Stage3[stage: Transcribe · Hinglish→EN\nWhisper large-v3]
    Stage3 --> TranscriptLines[transcript_line events stream\nSpeaker-attributed EN text]
    TranscriptLines --> Stage4[stage: Fact extraction\nQwen 9B · 4-bit]
    Stage4 --> Stage5[stage: Graph build\nNeo4j · Cypher]
    Stage5 --> DoneEvent{SSE done or error?}

    DoneEvent -->|done| GraphReload[GET /api/graph\nKnowledge Graph reloads]
    GraphReload --> RunDone[Run complete\nButton → ↻ Replay run]
    DoneEvent -->|error| ErrorState[SSE error event\nMessage shown · button resets]

    ClickRun -->|No, just view| GraphAlready[Graph already rendered\nfrom mount fetch]

    RunDone --> NodeClick{Click a graph node?}
    GraphAlready --> NodeClick

    NodeClick -->|Yes| HopHighlight[1-hop neighbourhood highlights\nEdges + neighbours lit · others dim]
    HopHighlight --> AskFlow{Use Ask Atyx?}
    NodeClick -->|No| AskFlow

    AskFlow -->|Preset click| PresetSend[POST /api/ask\nPreset question text]
    AskFlow -->|Typed question| TypedSend[POST /api/ask\nUser-typed question]
    AskFlow -->|Skip| ReadyIdle

    PresetSend --> AnswerCheck{found?}
    TypedSend --> AnswerCheck

    AnswerCheck -->|true · cypher or semantic-fallback| AnswerShown[Answer bubble + ◆ source quote\nReferenced graph nodes highlight]
    AnswerCheck -->|false · cosine below 0.40 floor| Decline[No answer found in the graph.\nNo hallucinated content]

    AnswerShown --> AskFlow
    Decline --> AskFlow
```

---

## Flow 2 — Clip switching

The clip dropdown in the left rail is a click-to-toggle picker showing the three registry clips.
Selecting a clip calls `/api/select_clip`, which returns the clip's mode. The mode drives the
UI layout: `graph` → 3-column with graph and Ask-Atyx; `facts` → 2-column with facts panel
only. The Neo4j graph is only touched for `graph`-mode clips (HERO INVARIANT: `call_100` and
`call_103` never write to Neo4j).

```mermaid
flowchart TD
    ConsoleScreen([Console screen]) --> DropClick[Click clip dropdown\nPicker opens]
    DropClick --> PickerVisible[Picker shows:\npms · call_100 · call_103]

    PickerVisible --> Pick{Pick a clip}

    Pick -->|pms| SelectPMS[POST /api/select_clip id=pms\n→ mode: graph]
    SelectPMS --> GraphLayout[3-column layout activates\nKnowledge Graph + Ask-Atyx visible]
    GraphLayout --> ReloadGraph[GET /api/graph\nPMS snapshot loads into graph panel]
    ReloadGraph --> PMSReady[Run replay · 5-stage pipeline available]

    Pick -->|call_100 or call_103| SelectFacts[POST /api/select_clip id=call_N\n→ mode: facts]
    SelectFacts --> FactsLayout[2-column layout activates\nNo graph · no Ask-Atyx]
    FactsLayout --> FactsReady[Run replay · 4-stage pipeline available\nGraph build stage absent]

    Pick -->|Close without picking| DropClose[Picker closes\nActive clip unchanged]

    PMSReady --> RunReplay1[Click Run replay → Flow 1]
    FactsReady --> RunReplay2[Click Run replay → facts run]
    RunReplay2 --> FactsRun[4-stage SSE stream\nTranscript + Extracted Facts appear]
    FactsRun --> OneSpeakerCheck{Single speaker\ndetected?}
    OneSpeakerCheck -->|Yes - phone audio collapse| HonestNote[Diarization note shown:\ncould not separate speakers]
    OneSpeakerCheck -->|No| FactsDone[Facts mode run complete]
    HonestNote --> FactsDone
```

---

## Flow 3 — Live upload

Any audio file the user uploads is assigned a transient `upload_XXXXXXXXXX` clip ID, switched
into `live` mode, and run immediately. Live mode uses the real pipeline (not replay), produces
facts output only, and never writes to Neo4j. There is no graph or Ask-Atyx for uploaded clips.

```mermaid
flowchart TD
    UploadState([Console · empty upload state]) --> ClickArea[Click upload area\nOS file picker opens]
    ClickArea --> FileSelect{File selected?}

    FileSelect -->|No / cancelled| UploadState

    FileSelect -->|Yes| PostUpload[POST /api/upload\nmultipart audio file]
    PostUpload --> Validate{Validate}

    Validate -->|Not audio mime| Err400A[HTTP 400\nError shown in facts panel]
    Validate -->|Duration > 600 s| Err400B[HTTP 400: exceeds 10-min cap\nError shown in facts panel]
    Validate -->|Valid ≤ 600 s| ClipID[clip_id returned\nfile resampled → 16 kHz mono]

    Err400A --> UploadState
    Err400B --> UploadState

    ClipID --> SelectLive[POST /api/select_clip id=clip_id\n→ mode: live]
    SelectLive --> LiveLayout[2-column layout · live mode\nRun live button — auto-triggered]
    LiveLayout --> AutoRun[POST /api/run auto-starts]
    AutoRun --> LiveSSE[GET /api/run/id/stream\n4-stage SSE: no Graph build]
    LiveSSE --> Enhance[stage: Speech enhancement]
    Enhance --> Diarize[stage: Diarization]
    Diarize --> ASR[stage: Transcribe · Hinglish→EN]
    ASR --> TransLines[transcript_line events\nTranscript panel fills]
    TransLines --> Extract[stage: Fact extraction]
    Extract --> FactEvents[fact events\nExtracted Facts panel fills]
    FactEvents --> LiveDone{done or error?}

    LiveDone -->|done| LiveComplete[Run complete\nTranscript + Facts visible\nNo graph · no Q&A]
    LiveDone -->|error| LiveError[SSE error event\nMessage shown · button resets]

    LiveComplete --> UploadAnother{Upload another?}
    UploadAnother -->|Yes| UploadState
    UploadAnother -->|No| LiveComplete
```

---

## Flow 4 — Experiment tab

The Experiment tab is independent of the clip selection and Q&A flow. It shows the pre-computed
SNR degradation curve (transcript cosine similarity vs. café-babble SNR) and spotcheck rows
comparing clean-clip answers to degraded-clip answers. The data is served from a committed JSON
artifact; if the file is absent (e.g. audio pipeline was skipped), the API returns 404 and the
tab renders an empty state without crashing.

```mermaid
flowchart TD
    AnyScreen([Any screen]) --> TabClick[Click Experiment tab]
    TabClick --> FetchExp[GET /api/experiment]

    FetchExp --> ExpCheck{HTTP status?}

    ExpCheck -->|200 — data present| ExpScreen[SNR Degradation Study screen]
    ExpScreen --> CurveRender[SVG fidelity curve renders\nTranscript similarity cosine vs SNR dB\nCafé-babble sweep]
    CurveRender --> SpotCheck[Spotcheck rows appear\nQuestion · clean answer · degraded answer · SNR]
    SpotCheck --> ReadDone[Analyst reads results]

    ExpCheck -->|404 — snr_results.json absent| ExpEmpty[Experiment screen\nNo curve · no spotcheck\nEmpty state shown · no crash]

    ReadDone --> BackConsole{Switch back?}
    ExpEmpty --> BackConsole
    BackConsole -->|Click Console tab| ConsoleScreen[Console screen restores\nActive clip + transcript unchanged]
    BackConsole -->|Stay| ReadDone
```
