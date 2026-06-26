# Atyx Convo-KG — Phase 3: Q&A (schema-aware text-to-Cypher + grounded answers)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Plan style:** signatures, data contracts (Pydantic models, JSON schemas), Cypher patterns, and ordered TDD steps with full test assertions. It deliberately does **not** include full function bodies — those are written in the build phase to make the listed tests pass. Snippets appear only where they pin a contract or one tricky function's intended logic.

**Goal:** Answer natural-language questions about a conversation by introspecting the live Neo4j graph, generating read-only Cypher (validated + retried), composing an English answer grounded in the returned rows, and surfacing causal source quotes — with a semantic fallback when Cypher can't resolve.

**Architecture:** A single `src/qa.py` module orchestrates: introspect schema → generate read-only Cypher (LLM) → write-clause reject + `EXPLAIN` (retry ×2) → run in a **read-only transaction** → resolve edge `source_statement_id` → `:Statement` provenance → LLM composes the answer. If Cypher fails after retries OR returns zero rows, a semantic fallback embeds the question against cached `:Statement` texts. Result is a Pydantic `QAResult`. Torch-free; reuses `src/llm.py` and `src/graph.connect()`.

**Tech Stack:** Python 3.12 (torch-free `.venv`), Pydantic v2, `neo4j` driver, local qwen3.5-9b via LM Studio, Neo4j 5.26.

## Global Constraints

- **Main env stays TORCH-FREE.** `qa.py` imports only stdlib + `pydantic` + `neo4j` + `src.*`. No new deps; cosine is hand-rolled pure-Python.
- **Local open-weight LLM only**, via `src/llm.py` (`chat_json` with a one-field JSON schema for Cypher and for the answer; `embed` for the fallback). No frontier API.
- **Read-only, defense in depth** (the one surface where the LLM touches the DB): (1) `is_read_only` rejects any generated Cypher with a write clause BEFORE running; (2) every query runs via the driver's **read access mode** (`session.execute_read`) so a write that slips the text check is refused by Neo4j itself.
- **Provenance is the hero feature, causally edge-linked:** the Cypher-gen prompt must `RETURN r.source_statement_id` when traversing a fact edge; `qa.py` joins those ids to `:Statement` for the verbatim quote + speaker. Each provenance item is tagged `kind="source"` (causal) or `kind="related"` (entity-linked / semantic-fallback). Never label "related" as "source".
- **Semantic fallback** engages on `EXPLAIN`-fail-after-retries **OR** zero rows; `mode='semantic-fallback'`; its provenance is `kind="related"`. Build the Cypher path solid FIRST, then layer the fallback.
- **`QAResult` is the locked Phase 5 frontend contract** (see Task 1).
- **TDD, frequent commits.** Run tests via the activated venv. Integration tests are marked `@pytest.mark.integration` (need LM Studio and/or Neo4j) and deselected by default.

## Confirmed Invariants (verified while finalizing the plan)

- **Multi-hop showcase exists in the real graph** (queried live, not retrofitted): `(business ownership)-[:HAS_STRATEGY]->(Do your own business)-[:ACHIEVES_GOAL]->(get rich before 30)`. The demo multi-hop question — "How does business ownership help you get rich before 30?" — traverses exactly this chain, grounded in `stmt:sample2:0`.
- **`graph.connect()`** returns a `neo4j.Driver`; `:Statement {id,text,speaker,clip,start,end}`, `:Entity {id,name,type}`, `:Speaker {id,name}`; fact edges carry `{confidence, speaker, source_statement_id, statement}`. The DB to use is `os.environ["NEO4J_DATABASE"]` (default `neo4j`).

---

## File Structure

- `src/contracts.py` — **Modify:** add `Provenance` + `QAResult` Pydantic models.
- `src/qa.py` — **Create:** the whole Q&A module (read-only guard, schema introspection, Cypher gen, execution, provenance, answer composition, semantic fallback, `answer()` orchestrator, CLI).
- Tests: `tests/test_qa.py` — **Create.**

---

## Task 1: `Provenance` + `QAResult` contracts

**Files:**
- Modify: `src/contracts.py`
- Test: `tests/test_contracts.py`

**Interfaces — the locked contract:**
```python
class Provenance(BaseModel):
    statement_id: str
    speaker: str
    text: str
    kind: Literal["source", "related"]          # causal edge source vs related context

class QAResult(BaseModel):
    question: str
    answer: str
    mode: Literal["cypher", "semantic-fallback"]
    found: bool
    cypher: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    graph_node_ids: list[str] = Field(default_factory=list)   # nodes behind the answer (node-highlight UI)
    hops: Literal["single", "multi"] = "single"
```

- [ ] **Step 1: Write the failing test.** Append to `tests/test_contracts.py`:
```python
def test_qaresult_and_provenance_roundtrip_and_literals():
    import pytest
    from pydantic import ValidationError
    from src.contracts import QAResult, Provenance
    p = Provenance(statement_id="stmt:sample2:0", speaker="SPEAKER_00", text="...", kind="source")
    r = QAResult(question="q", answer="a", mode="cypher", found=True,
                 cypher="MATCH (n) RETURN n", provenance=[p], graph_node_ids=["entity:pms"], hops="multi")
    r2 = QAResult.model_validate_json(r.model_dump_json())
    assert r2.provenance[0].kind == "source" and r2.hops == "multi" and r2.found is True
    assert QAResult(question="q", answer="a", mode="semantic-fallback", found=False).rows == []  # defaults
    with pytest.raises(ValidationError):
        Provenance(statement_id="x", speaker="s", text="t", kind="bogus")  # kind is a closed Literal
    with pytest.raises(ValidationError):
        QAResult(question="q", answer="a", mode="sql", found=True)         # mode is a closed Literal
```

- [ ] **Step 2: Run, confirm RED.** `source .venv/bin/activate && pytest tests/test_contracts.py -k qaresult -v` → FAIL (no `QAResult`).

- [ ] **Step 3: Implement.** Add `Provenance` and `QAResult` to `src/contracts.py` exactly as the contract above (reuse the existing `Literal`/`Any`/`Field` imports).

- [ ] **Step 4: Run, confirm GREEN + full suite.** `pytest tests/test_contracts.py -v && pytest -q` → green.

- [ ] **Step 5: Commit.**
```bash
git add src/contracts.py tests/test_contracts.py
git commit -m "feat(qa): QAResult + Provenance contracts (locked frontend shape)"
```

---

## Task 2: Read-only write-clause guard (`is_read_only`)

**Files:**
- Create: `src/qa.py` (this task starts the module)
- Test: `tests/test_qa.py`

**Interfaces:**
- Produces: `is_read_only(cypher: str) -> bool` — the FIRST read-only gate.
- **Intended logic:** case-insensitive word-boundary scan for write clauses `CREATE, MERGE, DELETE, SET, REMOVE, DROP, DETACH, FOREACH, LOAD` (and `CALL` write procedures — for v1, reject any `CALL` that is not `db.labels/db.relationshipTypes/db.schema`). Return `False` if any present, else `True`. Word boundaries so `n.createdAt` / a property named `created` do NOT trip it. **Conservative is correct:** over-rejecting a read query that contains a write *keyword inside a string literal* is acceptable (safe direction) — the read-only transaction (Task 5) is the true enforcement; this gate is the cheap first line.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_qa.py`:
```python
from src.qa import is_read_only

def test_read_queries_pass():
    assert is_read_only("MATCH (n) RETURN n")
    assert is_read_only("MATCH (a)-[r]->(b) RETURN a, r.source_statement_id, b LIMIT 5")
    assert is_read_only("MATCH (n) WHERE n.createdAt > 0 RETURN n.created")  # word-boundary: not a write

def test_write_queries_rejected():
    for q in ["CREATE (n)", "MATCH (n) DELETE n", "MATCH (n) DETACH DELETE n",
              "MATCH (n) SET n.x = 1", "MERGE (n:X {id:1})", "MATCH (n) REMOVE n.x",
              "DROP INDEX foo", "create (n)", "match (n) set n.x=1"]:
        assert not is_read_only(q), q
```

- [ ] **Step 2: Run, confirm RED.** `pytest tests/test_qa.py -v` → FAIL (`No module named 'src.qa'`).

- [ ] **Step 3: Implement** `is_read_only` in a new `src/qa.py` per the intended logic.

- [ ] **Step 4: Run, confirm GREEN.** `pytest tests/test_qa.py -v` → PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/qa.py tests/test_qa.py
git commit -m "feat(qa): read-only write-clause guard (first gate)"
```

---

## Task 3: Schema introspection (`format_schema` pure + `introspect_schema` live)

**Files:**
- Modify: `src/qa.py`
- Test: `tests/test_qa.py`

**Interfaces:**
- `format_schema(labels: list[str], rel_types: list[str], entity_types: list[str]) -> str` — pure; produces a compact schema description for the prompt, including the known property model (`:Entity {id,name,type}`, `:Statement {id,text,speaker,clip,start,end}`, `:Speaker {id,name}`, fact edges carry `source_statement_id`).
- `introspect_schema(driver, database) -> str` — live; queries `CALL db.labels()`, `CALL db.relationshipTypes()`, and `MATCH (e:Entity) RETURN DISTINCT e.type`, then returns `format_schema(...)`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_qa.py`:
```python
from src.qa import format_schema

def test_format_schema_lists_labels_rels_types_and_property_model():
    s = format_schema(labels=["Entity", "Statement", "Speaker"],
                      rel_types=["ACHIEVES_GOAL", "REQUIRES_INVESTMENT"],
                      entity_types=["FinancialGoal", "WealthStrategy"])
    assert "ACHIEVES_GOAL" in s and "FinancialGoal" in s and "Statement" in s
    assert "source_statement_id" in s                      # the prompt must know edges carry grounding
```

- [ ] **Step 2: Run, confirm RED.** `pytest tests/test_qa.py -k format_schema -v` → FAIL.

- [ ] **Step 3: Implement** `format_schema` (pure) and `introspect_schema` (live; reuses the same Neo4j read pattern) in `src/qa.py`.

- [ ] **Step 4: Run, confirm GREEN.** `pytest tests/test_qa.py -k format_schema -v` → PASS.

- [ ] **Step 5: Integration smoke (marked).** Append:
```python
import pytest

@pytest.mark.integration
def test_introspect_schema_against_live_graph():
    import os
    from src.graph import connect
    from src.qa import introspect_schema
    drv = connect()
    try:
        s = introspect_schema(drv, os.environ.get("NEO4J_DATABASE", "neo4j"))
    finally:
        drv.close()
    assert "Entity" in s and "Statement" in s               # labels present from the live graph
```
Run (Neo4j up): `pytest tests/test_qa.py -m integration -o addopts="" -k introspect -v` → PASS.

- [ ] **Step 6: Commit.**
```bash
git add src/qa.py tests/test_qa.py
git commit -m "feat(qa): live Neo4j schema introspection for the Cypher prompt"
```

---

## Task 4: Cypher generation (`build_cypher_prompt` pure + `generate_cypher` live)

**Files:**
- Modify: `src/qa.py`
- Test: `tests/test_qa.py`

**Interfaces:**
- `CYPHER_SCHEMA = {"type":"object","properties":{"cypher":{"type":"string"}},"required":["cypher"],"additionalProperties":False}`
- `build_cypher_prompt(question, schema_text, error=None) -> tuple[str,str]` — pure (system, user). System rules: produce ONE **read-only** Cypher query (MATCH/WHERE/RETURN/ORDER BY/LIMIT only — NO CREATE/MERGE/DELETE/SET/REMOVE); when the answer traverses a fact relationship, **`RETURN` that relationship's `source_statement_id`** (alias `provenance`); also RETURN the relevant node `id`s; prefer single-hop, use multi-hop only if the question needs it. If `error` is given, include the previous error and instruct a fix.
- `generate_cypher(question, schema_text, llm, error=None) -> str` — calls `llm.chat_json(*build_cypher_prompt(...), CYPHER_SCHEMA)` and returns the `cypher` field, stripped.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_qa.py`:
```python
from src.qa import build_cypher_prompt, generate_cypher

def test_cypher_prompt_carries_schema_and_readonly_and_provenance_rules():
    system, user = build_cypher_prompt("what is the goal?", "SCHEMA-TEXT-HERE")
    assert "SCHEMA-TEXT-HERE" in user
    low = (system + user).lower()
    assert "read-only" in low or "read only" in low
    assert "source_statement_id" in (system + user)         # provenance rule present
    assert "create" in low and "delete" in low              # forbids writes explicitly

def test_cypher_prompt_includes_prior_error_on_retry():
    _, user = build_cypher_prompt("q", "S", error="SyntaxError: unexpected FOO")
    assert "SyntaxError: unexpected FOO" in user

def test_generate_cypher_returns_cypher_field():
    class FakeLLM:
        def chat_json(self, system, user, schema): return {"cypher": "MATCH (n) RETURN n"}
    assert generate_cypher("q", "S", FakeLLM()) == "MATCH (n) RETURN n"
```

- [ ] **Step 2: Run, confirm RED.** `pytest tests/test_qa.py -k cypher -v` → FAIL.

- [ ] **Step 3: Implement** `CYPHER_SCHEMA`, `build_cypher_prompt`, `generate_cypher` per the interfaces.

- [ ] **Step 4: Run, confirm GREEN.** `pytest tests/test_qa.py -k cypher -v` → PASS.

- [ ] **Step 5: Integration smoke (marked).** Append:
```python
@pytest.mark.integration
def test_generate_cypher_is_readonly_against_live_schema():
    import os
    from src.graph import connect
    from src.config import load_config
    from src.llm import LLM
    from src.qa import introspect_schema, generate_cypher, is_read_only
    drv = connect(); db = os.environ.get("NEO4J_DATABASE", "neo4j")
    try:
        schema = introspect_schema(drv, db)
        cy = generate_cypher("What strategies help get rich before 30?", schema, LLM(load_config().llm))
    finally:
        drv.close()
    assert is_read_only(cy), f"generated non-read-only Cypher:\n{cy}"
    assert "RETURN" in cy.upper()                            # produced an actual query
```
Run: `pytest tests/test_qa.py -m integration -o addopts="" -k generate_cypher -v`. (Runnability via `EXPLAIN` is exercised in Task 6's acceptance, once `explain_ok` exists.)

- [ ] **Step 6: Commit.**
```bash
git add src/qa.py tests/test_qa.py
git commit -m "feat(qa): schema-aware, read-only, provenance-returning Cypher generation"
```

---

## Task 5: Execution + provenance (`explain_ok`, `run_read` read-backstop, provenance resolution)

**Files:**
- Modify: `src/qa.py`
- Test: `tests/test_qa.py`

**Interfaces:**
- `explain_ok(driver, database, cypher) -> tuple[bool, str|None]` — runs `EXPLAIN <cypher>` in a read transaction; `(True, None)` if it plans, else `(False, <error message>)`.
- `run_read(driver, database, cypher) -> list[dict]` — runs `cypher` via **`session.execute_read`** (read access mode — the hard backstop; Neo4j refuses writes here) and returns rows as dicts.
- `extract_node_ids(rows) -> list[str]` — pure; collect distinct string `id` values from node-shaped values in the rows (dedup, order-preserving).
- `extract_provenance_ids(rows) -> list[str]` — pure; collect distinct `source_statement_id` values present in the rows (under the `provenance` alias or any `*.source_statement_id`), ignoring `None`.
- `resolve_provenance(driver, database, statement_ids) -> list[Provenance]` — live; for each id `MATCH (s:Statement {id:$id}) RETURN s.text, s.speaker`, build `Provenance(..., kind="source")`; skip ids with no match.

- [ ] **Step 1: Write the failing tests (pure helpers).** Append to `tests/test_qa.py`:
```python
from src.qa import extract_node_ids, extract_provenance_ids

def test_extract_provenance_ids_collects_distinct_nonnull():
    rows = [{"provenance": "stmt:sample2:0", "x": 1},
            {"provenance": "stmt:sample2:0"},          # dup
            {"provenance": None},                       # ignored
            {"r.source_statement_id": "stmt:sample2:1"}]
    assert extract_provenance_ids(rows) == ["stmt:sample2:0", "stmt:sample2:1"]

def test_extract_node_ids_collects_distinct():
    rows = [{"a": {"id": "entity:pms", "name": "PMS"}, "b": {"id": "entity:aif"}},
            {"a": {"id": "entity:pms"}}]
    assert extract_node_ids(rows) == ["entity:pms", "entity:aif"]
```

- [ ] **Step 2: Run, confirm RED.** `pytest tests/test_qa.py -k extract_ -v` → FAIL.

- [ ] **Step 3: Implement** `explain_ok`, `run_read`, `extract_node_ids`, `extract_provenance_ids`, `resolve_provenance`. `run_read` MUST use `session.execute_read` (not `session.run`) so write access is impossible.

- [ ] **Step 4: Run, confirm GREEN (pure).** `pytest tests/test_qa.py -k extract_ -v` → PASS.

- [ ] **Step 5: Integration tests (marked) — incl. the read-only backstop.** Append:
```python
@pytest.mark.integration
def test_run_read_executes_and_resolves_provenance():
    import os
    from src.graph import connect
    from src.qa import run_read, resolve_provenance, extract_provenance_ids
    drv = connect(); db = os.environ.get("NEO4J_DATABASE", "neo4j")
    try:
        rows = run_read(drv, db, "MATCH (a:Entity)-[r]->(b:Entity) "
                                 "RETURN a.id AS a, r.source_statement_id AS provenance LIMIT 5")
        assert rows
        prov = resolve_provenance(drv, db, extract_provenance_ids(rows))
        assert prov and all(p.kind == "source" and p.text for p in prov)
    finally:
        drv.close()

@pytest.mark.integration
def test_run_read_refuses_writes_at_db_level():
    import os, pytest
    from src.graph import connect
    from src.qa import run_read
    drv = connect(); db = os.environ.get("NEO4J_DATABASE", "neo4j")
    try:
        with pytest.raises(Exception):                       # Neo4j read tx refuses the write
            run_read(drv, db, "CREATE (z:ZzzQaTest {id:'x'}) RETURN z")
    finally:
        drv.close()
```
Run (Neo4j up): `pytest tests/test_qa.py -m integration -o addopts="" -k "run_read or resolve" -v` → PASS (the write is refused).

- [ ] **Step 6: Commit.**
```bash
git add src/qa.py tests/test_qa.py
git commit -m "feat(qa): read-only execution backstop + edge-level provenance resolution"
```

---

## Task 6: Cypher-path orchestrator (`compose_answer`, `infer_hops`, `answer`)

**Files:**
- Modify: `src/qa.py`
- Test: `tests/test_qa.py`

**Interfaces:**
- `ANSWER_SCHEMA = {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":False}`
- `compose_answer(question, rows, provenance, llm) -> str` — `llm.chat_json` with a system prompt that says: answer in English using ONLY these rows + quoted source statements; if they don't contain the answer, say so plainly.
- `infer_hops(cypher) -> Literal["single","multi"]` — pure; `"multi"` if the query traverses ≥2 relationship patterns (count `-[` occurrences ≥ 2), else `"single"`.
- `MAX_RETRIES = 2`
- `answer(question, llm=None, driver=None, database=None) -> QAResult` — the Cypher-path orchestrator (semantic fallback is wired in Task 7). **Intended logic:** introspect schema; loop up to `1 + MAX_RETRIES`: `generate_cypher(question, schema, llm, error)`, reject if not `is_read_only` (set error, continue), `explain_ok` (on fail set error, continue), else break with a valid cypher. If a valid cypher: `rows = run_read(...)`; if rows non-empty → `prov = resolve_provenance(extract_provenance_ids(rows))`, `node_ids = extract_node_ids(rows)`, `ans = compose_answer(...)`, return `QAResult(mode="cypher", found=True, cypher=cypher, rows=rows, provenance=prov, graph_node_ids=node_ids, hops=infer_hops(cypher))`. If no valid cypher after retries, OR rows empty → return a `found=False` placeholder for now (Task 7 replaces this branch with the semantic fallback). Owns its driver lifecycle if `driver is None` (connect + close).

- [ ] **Step 1: Write the failing tests (pure).** Append to `tests/test_qa.py`:
```python
from src.qa import infer_hops

def test_infer_hops_counts_relationship_patterns():
    assert infer_hops("MATCH (a)-[r]->(b) RETURN a") == "single"
    assert infer_hops("MATCH (a)-[r1]->(b)-[r2]->(c) RETURN a") == "multi"
    assert infer_hops("MATCH (n) RETURN n") == "single"
```

- [ ] **Step 2: Run, confirm RED.** `pytest tests/test_qa.py -k infer_hops -v` → FAIL.

- [ ] **Step 3: Implement** `ANSWER_SCHEMA`, `compose_answer`, `infer_hops`, and the `answer()` Cypher-path orchestrator.

- [ ] **Step 4: Run, confirm GREEN (pure).** `pytest tests/test_qa.py -k infer_hops -v` → PASS.

- [ ] **Step 5: Integration acceptance (marked) — the hero path + verified multi-hop.** Append:
```python
@pytest.mark.integration
def test_answer_single_hop_is_grounded_with_source_provenance():
    from src.qa import answer
    r = answer("What strategies help you get rich before 30?")
    assert r.found and r.mode == "cypher"
    assert any(p.kind == "source" for p in r.provenance)
    assert all(p.statement_id.startswith("stmt:sample2:") for p in r.provenance)

@pytest.mark.integration
def test_answer_multi_hop_business_ownership_chain():
    from src.qa import answer
    r = answer("How does business ownership help you get rich before 30?")
    assert r.found and r.mode == "cypher" and r.hops == "multi"
    assert any(p.kind == "source" for p in r.provenance)
```
Run (LM Studio + Neo4j up): `pytest tests/test_qa.py -m integration -o addopts="" -k "answer_single or answer_multi" -v`.
**If the model can't produce runnable Cypher for these graph-supported questions after retries, STOP and report — do not hack the test to pass.** (Spot-check that the surfaced quote supports the answer.)

- [ ] **Step 6: Commit.**
```bash
git add src/qa.py tests/test_qa.py
git commit -m "feat(qa): Cypher-path answer orchestrator with grounded provenance + hop tagging"
```

---

## Task 7: Semantic fallback + wire-in

**Files:**
- Modify: `src/qa.py`
- Test: `tests/test_qa.py`

**Interfaces:**
- `FALLBACK_TOP_K = 3`
- `cosine(u, v) -> float` — pure (hand-rolled; reuse the Task-3-style math, zero-vector safe).
- `top_k_statements(question_vec, statements_with_vecs, k) -> list[dict]` — pure; rank `[{id,speaker,text,vec}]` by cosine to `question_vec`, return top-k (without the vec).
- `_statement_embeddings(driver, database, llm) -> list[dict]` — live; load `:Statement {id,speaker,text}`, embed texts via `llm.embed`, **cache** in a module-level dict keyed by statement id (reuse across calls). Returns `[{id,speaker,text,vec}]`.
- `semantic_fallback(question, driver, database, llm) -> tuple[str, list[Provenance], list[str]]` — embed the question, `top_k_statements`, `compose_answer`-style answer from the quotes; provenance `kind="related"`; node ids `[]`.
- `answer(...)` — **wire it in:** replace the Task-6 `found=False` branch (no valid cypher after retries, OR zero rows) with the fallback → `QAResult(mode="semantic-fallback", found=True, cypher=<last cypher or None>, provenance=<related>, ...)`. If even the fallback finds nothing relevant (no statements), return `found=False`.

- [ ] **Step 1: Write the failing tests (pure).** Append to `tests/test_qa.py`:
```python
from src.qa import cosine, top_k_statements

def test_cosine_basic_and_zero_safe():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0          # zero-vector safe, no div-by-zero

def test_top_k_statements_ranks_by_cosine():
    stmts = [{"id": "s0", "speaker": "A", "text": "alpha", "vec": [1.0, 0.0]},
             {"id": "s1", "speaker": "B", "text": "beta",  "vec": [0.0, 1.0]},
             {"id": "s2", "speaker": "C", "text": "gamma", "vec": [0.9, 0.1]}]
    top = top_k_statements([1.0, 0.0], stmts, k=2)
    assert [s["id"] for s in top] == ["s0", "s2"]          # closest two, in order
    assert "vec" not in top[0]                              # vec stripped from output
```

- [ ] **Step 2: Run, confirm RED.** `pytest tests/test_qa.py -k "cosine or top_k" -v` → FAIL.

- [ ] **Step 3: Implement** `cosine`, `top_k_statements`, `_statement_embeddings` (cached), `semantic_fallback`, and wire the fallback into `answer()`.

- [ ] **Step 4: Run, confirm GREEN (pure) + full suite.** `pytest tests/test_qa.py -k "cosine or top_k" -v && pytest -q` → green.

- [ ] **Step 5: Integration acceptance (marked) — fallback rescues zero-row, honest no-answer otherwise.** Append:
```python
@pytest.mark.integration
def test_answer_falls_back_semantically_for_offscript_question():
    from src.qa import answer
    # phrased so text-to-Cypher likely returns zero rows, but the statement text covers it
    r = answer("What does the speaker say about investing early and compounding?")
    assert r.found
    if r.mode == "semantic-fallback":
        assert all(p.kind == "related" for p in r.provenance) and r.provenance

@pytest.mark.integration
def test_answer_reports_not_found_for_unanswerable():
    from src.qa import answer
    r = answer("What is the capital of France?")               # nothing in this graph/statements
    assert r.found is False
```
Run (LM Studio + Neo4j up): `pytest tests/test_qa.py -m integration -o addopts="" -k "fall_back or not_found" -v`.
**If the fallback misclassifies (claims found for the France question, or labels related as source), STOP and report — do not hack.**

- [ ] **Step 6: Commit.**
```bash
git add src/qa.py tests/test_qa.py
git commit -m "feat(qa): semantic fallback (cached statement embeddings) wired into answer()"
```

---

## Notes for the executor
- **CLI + README:** add a `if __name__ == "__main__":` to `qa.py` (`python -m src.qa "<question>"` prints the answer + provenance) and a short "Phase 3 / Ask" line to the README. Fold into the Task 6 or 7 commit — keep docs current.
- **Build order is load-bearing:** Tasks 1–6 (the Cypher path with real provenance) must be solid and answering the demo questions BEFORE Task 7 (fallback). Do not build the fallback in parallel.
- **Don't hack:** if the local model can't produce runnable Cypher for graph-supported questions, or the fallback misclassifies found/kind, STOP and report rather than weakening a test (project rule).
