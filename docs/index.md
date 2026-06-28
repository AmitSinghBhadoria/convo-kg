---
layout: home
hero:
  name: Atyx Convo-KG
  text: Conversation intelligence for private wealth
  tagline: Turn every advisor–client call into a queryable record of the advice given — products, strategies, fees, suitability — on a local open-weight LLM, with nothing leaving the firm.
  actions:
    - theme: brand
      text: Product Overview
      link: ./product-overview
    - theme: alt
      text: Deployment Guide
      link: ./deployment-guide
features:
  - title: Product Overview
    details: Executive summary, key features, technology stack, and honest scope limitations for the Atyx Convo-KG prototype.
    link: ./product-overview
  - title: System Architecture
    details: Three-venv pipeline design, component interactions, cross-venv data flow, and sequential memory management on Apple Silicon.
    link: ./system-architecture
  - title: Entity Relationship
    details: Neo4j graph schema — node labels (Speaker, Statement, Entity, Claim, Attribute), typed relation edges, Pydantic data contracts, and entity resolution rules.
    link: ./entity-relationship
  - title: User Stories
    details: Domain personas (Relationship Manager, Compliance Officer) framing the value, plus prototype interaction roles (Analyst, Operator) with acceptance criteria across the full pipeline.
    link: ./user-stories
  - title: Wireflows
    details: Screen-level user flows for clip selection, pipeline run, live audio upload, Ask-Atyx Q&A, and the controlled-SNR Experiment tab.
    link: ./wireflows
  - title: Wireframes
    details: Annotated UI layout for the Console (2- and 3-column modes) and Experiment screens, including the knowledge-graph panel and chat interface.
    link: ./wireframes
  - title: Sequence Diagrams
    details: Request/response sequences for every REST endpoint and the SSE pipeline-progress stream, from run trigger through transcript and fact events to done/error.
    link: ./sequence-diagrams
  - title: API Specification
    details: Full REST API reference — all eight endpoints, request/response schemas, error model (400/404/SSE error event), and the no-hallucination Q&A contract.
    link: ./api-specification
  - title: Deployment Guide
    details: Prerequisites (uv, Neo4j Desktop, LM Studio), setup.sh / start.sh walkthrough, .env configuration, model loading, and troubleshooting.
    link: ./deployment-guide
---

## Documentation index

### Overview and design

- [Product Overview](./product-overview.md) — executive summary, features, stack table, pipeline diagram, scope and limitations
- [System Architecture](./system-architecture.md) — pipeline stages, three-venv isolation, component diagram, data-flow, memory management

### Data model

- [Entity Relationship](./entity-relationship.md) — Neo4j node labels, relationship types, Pydantic contracts (`Transcript`, `FactSet`, `QAResult`, `EvalResult`)

### User experience

- [User Stories](./user-stories.md) — domain personas (RM, Compliance Officer) + Analyst / Operator interaction roles; acceptance criteria; explicit out-of-scope (no auth, no cloud)
- [Wireflows](./wireflows.md) — user flows: clip picker, run pipeline, live upload, Q&A chat, Experiment tab
- [Wireframes](./wireframes.md) — annotated screen layouts for Console (graph mode, facts/live mode) and Experiment tab

### API and integration

- [Sequence Diagrams](./sequence-diagrams.md) — interaction sequences for `/api/run`, SSE stream, `/api/ask`, `/api/upload`, `/api/experiment`
- [API Specification](./api-specification.md) — full endpoint reference: `GET /api/graph`, `POST /api/ask`, `GET /api/experiment`, `GET /api/clips`, `POST /api/select_clip`, `POST /api/run`, `GET /api/run/{run_id}/stream`, `POST /api/upload`

### Operations

- [Deployment Guide](./deployment-guide.md) — prerequisites, `./setup.sh`, `.env` config, LM Studio model setup, `./start.sh`, Neo4j snapshot restore, troubleshooting
