# Atyx — Phase 5: Frontend + FastAPI Surface

**Date:** 2026-06-27 · **Target demo:** Mon 29 Jun 2026 · **Reviewer:** Debopam Bhattacherjee
**Status:** design approved in brainstorming (2026-06-27); pending final user review before plan.
**Supersedes parts of** the main spec §14 (informal endpoint sketches) — see §9 reconciliation.

---

## 1. Goal (one line)
Put a working demo surface on the finished pipeline: a thin FastAPI backend serving the **existing approved dc-app** (the dark-editorial Atyx design), with its mock data swapped for **real** Q&A, graph, SNR-experiment, and a hybrid-replay live run — so the Monday demo shows the real, slightly-messier system, never a clean placeholder.

## 2. The key decision: wire, don't rebuild (path a)
The design artifact `Ui-design/atyx_standalone_src.html` is a **declarative-React app** rendered by a generic do-not-edit runtime (`Ui-design/support.js`). The app has a single **editable controller** — `<script type="text/x-dc" data-dc-script>` → `class Component extends DCLogic` — that holds **all** mock data as class fields (`transcript[]`, `steps[]`, `graphNodes[]`, `graphEdges[]`, `presets[]`, `chart{}`) and all handlers (`ask()`, `startRun`, tab switches).

**We wire FastAPI into that controller and never touch `support.js`.** No pixel rebuild, no Streamlit, no editing compiled JS. This keeps the approved visual identity exactly and confines all change to editable, reviewable code.

## 3. Architecture & file layout
```
src/api.py            # FastAPI app: 4 data endpoints + SSE + static mount. Thin —
                      # adapts existing functions (qa.answer, graph read, file read);
                      # NO business logic.
frontend/
  index.html          # = atyx_standalone_src.html, controller wired (mock -> fetch)
  support.js          # copied VERBATIM from Ui-design/ (do-not-edit runtime)
  assets/vendor/      # react.production.min.js + react-dom.production.min.js (vendored)
  assets/fonts/       # Cormorant Garamond, Inter, JetBrains Mono (self-hosted @font-face)
Ui-design/            # pristine design reference — left untouched
```
`serve` becomes the third entrypoint alongside `run`/`eval`: `uvicorn src.api:app` (a `python -m src.api` shim launches it). React/ReactDOM and fonts are **vendored locally** — the demo must run offline with no CDN that can fail on stage (spec §14 already requires fonts vendored; this extends it to React).

## 4. Backend — `src/api.py` (thin)

| Endpoint | Returns | Source / behaviour |
|---|---|---|
| `GET /` + static mount | the wired frontend | `frontend/` |
| `GET /api/graph` | `{nodes:[{id,label,type,name}], edges:[{from,to,relation}]}` | live read of the **authoritative pre-built `pms` graph** in Neo4j |
| `POST /api/ask` `{question}` | **`QAResult` verbatim** (the locked contract) | `qa.answer(question)` — **live local-LLM every call** |
| `GET /api/experiment` | `snr_results.json` contents (`curve[]`, `spotcheck[]`, `labels`, `meta`) | Phase 4 artifact, read from `data/ground_truth/` |
| `POST /api/run` → `{run_id}` ; `GET /api/run/{run_id}/stream` | SSE | hybrid replay (§6) |

`POST /api/ask` is the real hero: genuine text-to-Cypher + answer composition on the local model, every question. It returns the **`QAResult`** Pydantic model verbatim — `question, answer, mode('cypher'|'semantic-fallback'), found, cypher, rows, provenance[{statement_id,speaker,text,kind}], graph_node_ids, hops`. `found:false` is a normal 200 response (honest no-answer), **never a 500**.

`GET /api/graph` derives nodes/edges from the fact graph (`:Entity`/`:Attribute`/`:Claim` nodes + fact edges; `type` from the node's induced `type`/label for colour). Node `id` is the **stable graph id verbatim** (e.g. `entity:pms`) — see the §7 alignment gate.

## 5. Frontend wiring — the mock→fetch audit (every field, no survivors)

| Controller mock field | Replaced with |
|---|---|
| `transcript[]` | cached `pms` transcript, replayed in the Run timeline |
| `graphNodes[]`, `graphEdges[]` | `GET /api/graph` (authoritative pre-built graph) |
| `presets[]` | kept **only as suggested-prompt buttons**; the canned `main/tail/cypher/nodeIds/source/srcLine` are **deleted** — answers come from `/api/ask` |
| `ask()` canned bot message | `POST /api/ask` → real `QAResult` rendered (answer + provenance quote + cypher toggle + node highlight) |
| `chart{}` (`sim/recall/qa/wer/facts`) | `GET /api/experiment` — **adapted to one curve + spot-check (§8)** |

**Audit guard (constraint):** a test greps `frontend/index.html` for known mock literals — `"12 / 12"`, `"₹50 L"`, `"₹1 Cr"`, `"6%"`, `sim:[0.98`, and each preset answer string — and **fails if any survives**. No hardcoded placeholder can reach the demo. The UI must show the real, slightly-messier results.

## 6. Run tab — SSE hybrid replay, authoritative-graph rule
`startRun`'s `setTimeout` simulation is replaced by an SSE consumer of `/api/run/{id}/stream`. The audio front-end is **replayed** (it is minutes-long and fragile); extraction runs **live** as a visible proof. Events:

- **`stage`** — pipeline-timeline step updates `{index, status, elapsed}`. The denoise/diarize/ASR stages are driven from **cached artifacts** and each is **labelled "replayed" in the UI**.
- **`transcript_line`** `{speaker, t, text}` — the cached transcript revealed in timed order.
- **`fact`** — facts emitted by **live extraction** (real qwen on the cached transcript), streamed into a small **live log** under the extract step.
- **`done`** — signals the graph panel to render the **authoritative pre-built graph** (`/api/graph`), animated in. **`error`** — rendered honestly in the UI.

**Authoritative-graph rule (load-bearing):** the live extraction is **display-only — never upserted to Neo4j**. The authoritative graph is the stable pre-built `pms` graph; **Ask always queries it**, so node highlights stay consistent and the hero never depends on a possibly-messy live run.

**Live-extract honesty framing (decided now, not at demo time):** live extraction on real Hinglish is nondeterministic and sits at the measured extraction ceiling, so the facts in the live log **may visibly differ from / look messier than** the clean pre-built graph the Q&A then queries. The live log is therefore labelled in the UI: **"live extraction — illustrative; Q&A queries the verified pre-built graph."** A mismatch then reads as honest-by-design, not a glitch. A **pure-replay escape hatch** (config flag: replay the cached `facts.json` instead of running live extract) is available for a high-stakes run; default is live.

## 7. Hero alignment gate (explicit, early — not "verify later")
The hero payoff is *ask → the right nodes light up*. That works only if `QAResult.graph_node_ids` and `/api/graph` node `id`s are **byte-identical in format** (e.g. both `entity:pms`, no `Entity:PMS` / `pms` drift).

**This is an explicit build gate at steps 2–3, before proceeding:** an integration test asserts that for a real demo question, every id in `QAResult.graph_node_ids` exists verbatim in the `/api/graph` node-id set. If they differ in format, fix the mapping at the source **before** building the rest. Not deferred.

## 8. Experiment tab — honest adaptation (top-priority wiring)
The three-curve + WER mock predates the Phase 4 narrowing and would display metrics we **never measured** (a WER we didn't compute; recall/Q&A-correctness curves we deliberately didn't draw because of the extraction ceiling). It is **adapted to match the Phase 4 design note exactly**, rendered from `/api/experiment`:

- **One similarity curve** — `curve[].snr` × `curve[].similarity` (real values 0.5479/0.5932/0.6253/0.4525/0.2873), **honest y-floor at 0**, the cliff visible.
- **The illustrative spot-check** — `spotcheck[]` rows (question · clean answer · degraded answer).
- **Verbatim honesty labels** from the artifact: *"transcript fidelity vs SNR — relative similarity, not WER"* and *"illustrative propagation, not a calibrated curve; transcript-grounded retrieval, not the full extract→graph→Q&A product path."*

The `recall`/`qa` polylines and the `facts '12/12'` / `wer '6%'` table are **deleted, not relabelled.** No metric we did not measure appears anywhere in the UI. **This is the single most important piece of the wiring** — a beautiful tab showing fake-precise metrics would undermine the honest-measurement credibility the whole project rests on.

## 9. Reconciliation with main spec §14 (cross-check)
§14's endpoint sketches predate the locked contracts; the differences are **deliberate supersessions of already-approved decisions**, not new divergences:
1. **`/api/ask`** — §14 sketched `{answer, explanation, source, cypher, node_ids, src_line}`; the authoritative shape is **`QAResult`**, which §10.6 already locked as "the Phase 5 frontend contract." `QAResult` governs (`graph_node_ids`, `provenance[]` replace `node_ids`, `source`/`src_line`).
2. **`/api/experiment`** — §14 sketched `{snr, sim, recall, qa, table}` (three curves + WER); superseded by the Phase 4 **one-curve + spot-check** shape, the same narrowing already applied to §12.
3. **SSE / graph** — §14 streamed `graph_node`/`graph_edge` into the rendered graph; under the **authoritative-graph rule** we instead stream `fact` (display log) and reveal the pre-built graph on `done`. Deliberate (§6).
4. **"vanilla JS + D3, rebuild as no-build source"** — superseded by **path (a)**: reuse the approved dc-app and wire its controller (no rebuild). The "no Node toolchain / offline / fonts vendored" intent is preserved and extended (React vendored).

The main spec §14 gets a one-line pointer to this document for the authoritative contracts (consistent with the §12 treatment).

## 10. Error handling & testing
- **Backend:** `/api/ask` always returns a valid `QAResult` (200, `found:false` on no-answer); SSE emits `error` events the UI renders; `/api/experiment` returns a clear 404 if the artifact is absent; `/api/graph` returns empty `{nodes:[],edges:[]}` (not a 500) if the graph is empty. Pytest with FastAPI `TestClient`: `/api/ask` → valid `QAResult` shape; `/api/experiment` → real JSON with `curve`/`spotcheck`/`labels`; `/api/graph` → `nodes`/`edges` present; the **§7 alignment gate** test; marked `integration` where Neo4j/LLM are needed.
- **Frontend:** the **mock-literal grep guard** (a test over `frontend/index.html`); a vendored-asset presence check (no `https://` CDN refs remain). No JS unit framework (YAGNI for a no-build page) — the page is exercised end-to-end via the real endpoints in the demo walkthrough.
- **`/api/ask` loading state (constraint):** the live call takes a few seconds (text-to-Cypher + compose). The UI **must show a visible thinking/loading indicator** while it runs, so a 3–5 s wait reads as "the model is working," not "frozen." Live answer + working indicator is impressive; live answer + frozen UI looks broken.

## 11. Build order (protects the hero)
1. `src/api.py` skeleton + `serve` + static mount; vendor React/ReactDOM + fonts; page loads.
2. **`/api/ask` + Ask-Atyx wired first** — the answer→provenance→node-highlight trace working against the locked `QAResult`, with the loading indicator (§10), before any other polish.
3. **`/api/graph` + graph panel authoritative render + the §7 alignment gate** (byte-identical node ids) — gate passes before proceeding.
4. `/api/experiment` + the Experiment honest adaptation (§8).
5. `/api/run` SSE hybrid replay + Run tab (§6) with the live-extract honesty label + replay escape hatch.
6. Mock-literal audit guard + vendoring finalised; full smoke of the served page.

## 12. Out of scope (architect for, don't build)
Live arbitrary bring-your-own-audio run (documented for later: ≤90 s cap — model-load is fixed cost, ASR/diarization scale with length, so a 1-min clip is watchable-live ~30–60 s but yields a **thin** graph and the extraction ceiling still applies; the depth demo stays the pre-built `pms` graph). Auth, multi-tenancy, upload pipeline, streaming ingestion, mobile layout.
