# Atyx Convo-KG — Phase 2: Transcript → Knowledge Graph

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Plan style:** this plan gives signatures, data/Cypher contracts, and the ordered TDD steps with their test assertions. It deliberately does **not** include full function bodies — those are written in the build phase to make the listed tests pass. Snippets appear only where they pin a contract (a Pydantic model, a JSON schema, a Cypher MERGE pattern, or one tricky function's intended logic).

**Goal:** Turn a `data/work/<clip>.transcript.json` into a populated, browsable, traceable Neo4j knowledge graph: induce a per-clip ontology, extract source-grounded facts with a local LLM, consolidate entities/relations, and idempotently upsert into Neo4j.

**Architecture:** Pure, unit-testable core (chunking, identity/resolution) + LLM-backed stages (ontology proposal, extraction) + a Neo4j writer. Stages communicate via disk artifacts (`<clip>.facts.json`) like Phase 1. The main env stays torch-free; all LLM calls go through the existing `src/llm.py` OpenAI-compatible client; Neo4j via the official driver with creds from `.env`.

**Tech Stack:** Python 3.12 (torch-free main `.venv`), Pydantic v2, `tiktoken` (token ruler), `neo4j` driver, local qwen3.5-9b via LM Studio, Neo4j 5.26 Desktop.

## Global Constraints

- **Main env stays TORCH-FREE.** New deps: `tiktoken`, `neo4j` (both pure-Python). No torch/transformers in `src/` or main env.
- **Local open-weight LLM only**, via `src/llm.py` (`chat_json`/`embed`) against the config endpoint; runner-agnostic. No frontier API in the product path.
- **Source-grounding is mandatory:** every `Fact` carries a `statement` (cited span) AND a `statement_id` linking to its `:Statement` node. Facts that cannot be grounded are dropped.
- **Confidence threshold (`config.yaml` `extract.confidence_threshold`, default 0.6, precision-biased):** drop facts below it.
- **Entity resolution is precision-biased:** exact normalized-name match primary; embedding fallback only for non-matching names, gated at **cosine ≥ 0.85 AND same backbone `label` + induced `type`**. When uncertain, DO NOT merge. Stable ID = `slug(canonical_name)`.
- **Relation canonicalization:** near-synonyms must collapse — `"min investment"` and `"minimum investment amount"` map to the SAME relation type.
- **Neo4j writes:** parameterized Cypher only; relationship/label strings interpolated ONLY after validation against a closed allowlist / safe charset. All writes idempotent `MERGE` on stable IDs.
- **Chunking:** speaker turn is the HARD boundary; `extract.chunk_tokens` (1800) is a SOFT target; an oversized single turn is kept intact; the previous chunk's last turn is included as READ-ONLY context (never extracted from).
- **TDD, frequent commits.** Tests run via the activated venv: `source .venv/bin/activate && pytest ...`. Integration tests are marked `@pytest.mark.integration` (require LM Studio and/or a running Neo4j) and are deselected by default.

---

## File Structure

- `src/chunking.py` — token counting (`tiktoken`) + pure turn-boundary chunker. **Create.**
- `src/resolve.py` — identity + resolution: name normalization, slugs, stable IDs, relation canonicalization, `EntityResolver`. **Create.**
- `src/ontology.py` — per-clip ontology proposal pass + base fallback. **Create.**
- `src/extract.py` — chunk → per-chunk LLM extraction → consolidate → `FactSet` → `<clip>.facts.json`. **Create.**
- `src/graph.py` — Neo4j connection + idempotent upsert (schema mapping). **Create.**
- `src/contracts.py` — **Modify:** add `Fact.statement_id`.
- `pyproject.toml` — **Modify:** add `tiktoken`, `neo4j`.
- Tests: `tests/test_chunking.py`, `tests/test_resolve.py`, `tests/test_ontology.py`, `tests/test_extract.py`, `tests/test_graph.py`. **Create.**

---

## Task 1: Phase 2 dependencies + `Fact.statement_id`

**Files:**
- Modify: `pyproject.toml` (add `tiktoken`, `neo4j`)
- Modify: `src/contracts.py` (add `statement_id` to `Fact`)
- Test: `tests/test_contracts.py` (add a round-trip assertion)

**Interfaces:**
- Produces: `Fact.statement_id: str = ""` (grounding anchor → `:Statement.id`). Default `""` keeps Phase 1 facts valid.

- [ ] **Step 1: Add dependencies.** In `pyproject.toml` `[project] dependencies`, add `"tiktoken>=0.7"` and `"neo4j>=5.20"` (keep the existing torch-free set).

- [ ] **Step 2: Install + confirm torch-free.**
Run: `cd /Users/amit/Personal/Atyx && source .venv/bin/activate && uv pip install "tiktoken>=0.7" "neo4j>=5.20" && python -c "import tiktoken, neo4j; print('ok', neo4j.__version__)" && python -c "import torch" 2>&1 | head -1`
Expected: `ok <version>` then `ModuleNotFoundError: No module named 'torch'` (main env is torch-free — that error is REQUIRED).

- [ ] **Step 3: Write the failing test.** Append to `tests/test_contracts.py`:
```python
def test_fact_carries_statement_id_default_and_roundtrip():
    from src.contracts import Fact
    f = Fact(subject_id="entity:pms", relation="HAS_MINIMUM_INVESTMENT",
             object_id="attribute:50-lakh", statement="PMS needs 50 lakh minimum",
             speaker="SPEAKER_00", confidence=0.9)
    assert f.statement_id == ""                      # default, Phase 1 back-compat
    f2 = Fact.model_validate_json(
        f.model_copy(update={"statement_id": "stmt:pms:3"}).model_dump_json())
    assert f2.statement_id == "stmt:pms:3"
```

- [ ] **Step 4: Run it, confirm RED.**
Run: `source .venv/bin/activate && pytest tests/test_contracts.py::test_fact_carries_statement_id_default_and_roundtrip -v`
Expected: FAIL — `Fact` has no `statement_id` attribute.

- [ ] **Step 5: Implement.** Add a `statement_id: str = ""` field to `class Fact` (after `statement`), with a comment noting it links to the `:Statement` node and is set during extraction.

- [ ] **Step 6: Run it, confirm GREEN + full suite.**
Run: `source .venv/bin/activate && pytest tests/test_contracts.py -v && pytest -q`
Expected: new test PASSES; whole suite stays green.

- [ ] **Step 7: Commit.**
```bash
git add pyproject.toml src/contracts.py tests/test_contracts.py
git commit -m "feat(phase2): add tiktoken+neo4j deps; Fact.statement_id grounding anchor"
```

---

## Task 2: Chunking (`src/chunking.py`)

**Files:**
- Create: `src/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `src.contracts.Utterance`.
- Produces:
  - `count_tokens(text: str) -> int` — `tiktoken` `cl100k_base` (a length *ruler* only).
  - **Contract — the `Chunk` model:**
    ```python
    class Chunk(BaseModel):
        index: int
        utterances: list[Utterance]          # turns to extract from
        indices: list[int]                   # their global utterance indices (stable statement ids)
        context: Utterance | None = None      # previous chunk's last turn, read-only overlap
        context_index: int | None = None
    ```
  - `chunk_transcript(utterances, target_tokens, count=count_tokens) -> list[Chunk]`.
- **Intended logic (build phase):** accumulate whole turns; when adding the next turn would exceed `target_tokens` and the current chunk is non-empty, cut BEFORE that turn (never mid-turn); a single turn larger than the target lands alone as an oversized chunk; each chunk after the first carries the previous chunk's last turn as `context`. `count` is injectable so unit tests stay pure.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_chunking.py`:
```python
from src.contracts import Utterance
from src.chunking import chunk_transcript, count_tokens

def U(text, spk="S0"):
    return Utterance(speaker=spk, text=text, start=0.0, end=1.0)

def words(text):                 # pure fake counter: 1 token per word (no tiktoken needed)
    return len(text.split())

def test_splits_at_next_turn_boundary_never_midturn():
    utts = [U("a a a a a"), U("b b b b b"), U("c c c c c")]      # 5 "tokens" each
    chunks = chunk_transcript(utts, target_tokens=8, count=words)
    assert [c.index for c in chunks] == [0, 1]
    assert [u.text for u in chunks[0].utterances] == ["a a a a a"]       # +5 -> 10>8 -> cut
    assert [u.text for u in chunks[1].utterances] == ["b b b b b", "c c c c c"]
    assert chunks[0].indices == [0] and chunks[1].indices == [1, 2]

def test_oversized_single_turn_kept_intact():
    utts = [U("x " * 20), U("y y")]                             # first turn = 20 > target
    chunks = chunk_transcript(utts, target_tokens=8, count=words)
    assert [u.text for u in chunks[0].utterances] == ["x " * 20]        # intact, alone
    assert chunks[1].utterances[0].text == "y y"

def test_context_is_previous_chunks_last_turn():
    utts = [U("a a a a a"), U("b b b b b"), U("c c c c c")]
    chunks = chunk_transcript(utts, target_tokens=8, count=words)
    assert chunks[0].context is None and chunks[0].context_index is None
    assert chunks[1].context.text == "a a a a a"
    assert chunks[1].context_index == 0

def test_empty_input():
    assert chunk_transcript([], target_tokens=8, count=words) == []

def test_count_tokens_real_tiktoken_is_positive():
    assert count_tokens("hello world") > 0
```

- [ ] **Step 2: Run, confirm RED.**
Run: `source .venv/bin/activate && pytest tests/test_chunking.py -v` → FAIL (`No module named 'src.chunking'`).

- [ ] **Step 3: Implement** `src/chunking.py` — `count_tokens` (module-level `cl100k_base` encoder), the `Chunk` model above, and `chunk_transcript` per the intended logic. Make the tests pass.

- [ ] **Step 4: Run, confirm GREEN.**
Run: `source .venv/bin/activate && pytest tests/test_chunking.py -v` → all 5 PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/chunking.py tests/test_chunking.py
git commit -m "feat(extract): turn-boundary chunker with tiktoken ruler + read-only overlap"
```

---

## Task 3: Identity + resolution (`src/resolve.py`)

**Files:**
- Create: `src/resolve.py`
- Test: `tests/test_resolve.py`

**Interfaces:**
- Consumes: `src.contracts.Entity`.
- Produces:
  - `normalize_name(s)->str` (lower, strip, collapse whitespace), `slugify(s)->str`, `entity_id(label,name)->str` (= `f"{label.lower()}:{slugify(name)}"`), `statement_id(clip,idx)->str` (= `f"stmt:{clip}:{idx}"`).
  - `canonical_relation(relation, vocab=None)->str` — lexical canonicalization → valid Neo4j rel charset. **Contract for the acceptance case:** expand abbreviations (`min→minimum`, etc.), drop low-content words (`amount`, `value`, `the`, …), join `_` + UPPER, strip to `[A-Z0-9_]` — so `"min investment"` and `"minimum investment amount"` both yield `"MINIMUM_INVESTMENT"`. If `vocab` is given and the result is in it, return it.
  - `safe_rel_type(rel)->str` — returns `rel` if it matches `^[A-Z][A-Z0-9_]*$`, else raises `ValueError`.
  - `class EntityResolver(embed_fn, threshold=0.85)` with `resolve(entities)->tuple[list[Entity], dict[str,str]]` (representative entities + old_id→new_id map).
- **Intended logic for `EntityResolver.resolve` (build phase):** iterate entities; (1) **exact normalized-name** match → reuse representative; (2) else **embedding fallback** — only against existing reps with the **same `label` AND `type`**, pick the best with cosine ≥ `threshold`; (3) else create a new representative with `id = entity_id(label, name)`. Record every input id → chosen rep id in the map. `embed_fn(list[str])->list[vec]` is injectable so unit tests run without the LLM. Cosine is a small pure helper.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_resolve.py`:
```python
import pytest
from src.contracts import Entity
from src.resolve import (normalize_name, slugify, entity_id, statement_id,
                         canonical_relation, safe_rel_type, EntityResolver)

def test_normalize_and_ids():
    assert normalize_name("  White   Oak  ") == "white oak"
    assert slugify("White Oak Capital!") == "white-oak-capital"
    assert entity_id("Entity", "White Oak") == "entity:white-oak"
    assert statement_id("pms", 3) == "stmt:pms:3"

def test_relation_near_synonyms_collapse():            # mandatory acceptance case
    assert canonical_relation("min investment") == canonical_relation("minimum investment amount")
    assert canonical_relation("min investment") == "MINIMUM_INVESTMENT"

def test_relation_safe_charset():
    assert safe_rel_type("HAS_VALUE") == "HAS_VALUE"
    with pytest.raises(ValueError):
        safe_rel_type("DROP TABLE; --")
    assert safe_rel_type(canonical_relation("decides (on) the! plan")) == "DECIDES_PLAN"

def E(eid, name, typ, label="Entity"):
    return Entity(id=eid, label=label, type=typ, name=name)

def test_exact_name_merge():
    res = EntityResolver(embed_fn=lambda xs: [[0.0]] * len(xs))   # embed unused here
    reps, idmap = res.resolve([E("a", "PMS", "Instrument"), E("b", "pms", "Instrument")])
    assert len(reps) == 1
    assert idmap["a"] == idmap["b"] == reps[0].id == "entity:pms"

def test_embedding_fallback_merges_paraphrase_same_type():
    vecs = {"White Oak": [1.0, 0.0], "White Oak Capital": [0.99, 0.01]}   # cosine ~1.0
    res = EntityResolver(embed_fn=lambda xs: [vecs[x] for x in xs], threshold=0.85)
    reps, idmap = res.resolve([E("a", "White Oak", "Organization"),
                               E("b", "White Oak Capital", "Organization")])
    assert len(reps) == 1
    assert idmap["a"] == idmap["b"]

def test_same_type_guard_blocks_cross_type_merge():
    # identical embeddings for everything: only the type guard can prevent the merge
    res = EntityResolver(embed_fn=lambda xs: [[1.0, 0.0] for _ in xs], threshold=0.85)
    reps, idmap = res.resolve([E("a", "Alpha", "Organization"),
                               E("b", "Alpha2", "Metric")])      # different name + different type
    assert len(reps) == 2                                        # cross-type never merges

def test_distinct_entities_below_threshold_not_merged():        # protects PMS vs AIF
    vecs = {"PMS": [1.0, 0.0], "AIF": [0.0, 1.0]}               # cosine 0.0 < 0.85
    res = EntityResolver(embed_fn=lambda xs: [vecs[x] for x in xs], threshold=0.85)
    reps, idmap = res.resolve([E("a", "PMS", "Instrument"), E("b", "AIF", "Instrument")])
    assert len(reps) == 2                                        # same type but dissimilar -> separate
```

- [ ] **Step 2: Run, confirm RED.**
Run: `source .venv/bin/activate && pytest tests/test_resolve.py -v` → FAIL (`No module named 'src.resolve'`).

- [ ] **Step 3: Implement** `src/resolve.py` per the interfaces + intended logic. Define the `_ABBREV` and low-content word sets so the near-synonym acceptance test passes. Make all tests pass.

- [ ] **Step 4: Run, confirm GREEN.**
Run: `source .venv/bin/activate && pytest tests/test_resolve.py -v` → all PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/resolve.py tests/test_resolve.py
git commit -m "feat(extract): identity + precision-biased entity/relation resolution"
```

---

## Task 4: Ontology proposal pass (`src/ontology.py`)

**Files:**
- Create: `src/ontology.py`
- Test: `tests/test_ontology.py`

**Interfaces:**
- Consumes: `src.llm.LLM` (only `.chat_json`), `src.resolve.canonical_relation`.
- Produces:
  - **Contract — the model + schema + fallback:**
    ```python
    class Ontology(BaseModel):
        entity_types: list[str]
        relations: list[str]

    BASE_ONTOLOGY = Ontology(
        entity_types=["Person", "Organization", "Instrument", "Metric", "Decision", "Task", "Topic", "Date"],
        relations=["MENTIONS", "HAS_ATTRIBUTE", "HAS_VALUE", "DECIDED", "COMMITTED_TO", "COMPARES_WITH", "RELATES_TO"],
    )

    ONTOLOGY_SCHEMA = {           # strict json_schema passed to chat_json
        "type": "object",
        "properties": {
            "entity_types": {"type": "array", "items": {"type": "string"}},
            "relations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["entity_types", "relations"],
        "additionalProperties": False,
    }
    ```
  - `propose_ontology(transcript_text:str, llm)->Ontology`.
- **Intended logic (build phase):** call `llm.chat_json(system, user, ONTOLOGY_SCHEMA)` with a system prompt asking for a SMALL canonical vocabulary; validate into `Ontology`; if it errors OR `entity_types`/`relations` is empty → return `BASE_ONTOLOGY`; otherwise set `relations = sorted({canonical_relation(r) for r in relations})` (dedup + canonicalize) and return.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_ontology.py`:
```python
from src.ontology import propose_ontology, Ontology, BASE_ONTOLOGY

class FakeLLM:
    def __init__(self, payload): self.payload = payload
    def chat_json(self, system, user, schema): return self.payload

def test_uses_llm_proposal_and_canonicalizes_relations():
    llm = FakeLLM({"entity_types": ["Instrument", "Organization"],
                   "relations": ["min investment", "minimum investment amount"]})
    ont = propose_ontology("some transcript text", llm)
    assert ont.entity_types == ["Instrument", "Organization"]
    assert ont.relations == ["MINIMUM_INVESTMENT"]          # near-synonyms collapsed + deduped

def test_falls_back_on_empty_proposal():
    assert propose_ontology("t", FakeLLM({"entity_types": [], "relations": []})) == BASE_ONTOLOGY

def test_falls_back_on_llm_error():
    class Boom:
        def chat_json(self, *a, **k): raise RuntimeError("no server")
    assert propose_ontology("t", Boom()) == BASE_ONTOLOGY
```

- [ ] **Step 2: Run, confirm RED.**
Run: `source .venv/bin/activate && pytest tests/test_ontology.py -v` → FAIL (`No module named 'src.ontology'`).

- [ ] **Step 3: Implement** `src/ontology.py` per the contract + intended logic.

- [ ] **Step 4: Run, confirm GREEN.**
Run: `source .venv/bin/activate && pytest tests/test_ontology.py -v` → 3 PASS.

- [ ] **Step 5: Add an integration smoke test (marked).** Append:
```python
import pytest

@pytest.mark.integration
def test_propose_ontology_against_lmstudio():
    from src.config import load_config
    from src.llm import LLM
    ont = propose_ontology("Speaker A: PMS needs 50 lakh. Speaker B: AIF needs 1 crore.",
                           LLM(load_config().llm))
    assert ont.entity_types and ont.relations
```
Run (only if LM Studio is up): `source .venv/bin/activate && pytest tests/test_ontology.py -m integration -o addopts="" -v` → PASS (non-empty vocab; falls back to `BASE_ONTOLOGY` if the server errors).

- [ ] **Step 6: Commit.**
```bash
git add src/ontology.py tests/test_ontology.py
git commit -m "feat(extract): per-clip ontology proposal pass with base fallback"
```

---

## Task 5: Extraction (`src/extract.py`)

**Files:**
- Create: `src/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: `Transcript`/`Entity`/`Fact`/`FactSet` (contracts), `chunk_transcript`/`Chunk` (chunking), `EntityResolver`/`canonical_relation`/`statement_id` (resolve), `Ontology`/`propose_ontology` (ontology), `LLM`, `load_config`.
- Produces:
  - **Contract — `EXTRACT_SCHEMA`** (strict `json_schema` for one chunk's output): an object with `entities[]` (each `{id, label∈{Speaker,Statement,Entity,Claim,Attribute}, type, name}`) and `facts[]` (each `{subject_id, relation, object_id, statement, statement_id, speaker, confidence}`), `additionalProperties:false`, all fields required.
  - `build_prompt(chunk, ontology, clip)->tuple[str,str]` (system, user).
  - `consolidate(raw_entities, raw_facts, vocab, threshold, resolver, clip)->FactSet`.
  - `extract(clip, cfg=None, llm=None)->FactSet` — writes `data/work/<clip>.facts.json`; CLI `python -m src.extract <clip>`.
- **Intended logic (build phase):**
  - `build_prompt`: render each extractable turn as a line `"[{statement_id(clip, idx)}] {speaker}: {text}"`; if `chunk.context` is set, prepend its line tagged `(CONTEXT-ONLY — do not extract, do not cite)`. System prompt biases toward `ontology.entity_types`/`relations` (or "induce sensible …" when `ontology is None`) and states the rules: every fact sets `statement_id` to the tag of its source line; use the entity ids you define; only extract stated facts with a `confidence` in [0,1]; never extract from or cite CONTEXT-ONLY lines.
  - `_extract_chunk(chunk, ontology, clip, llm)`: call `llm.chat_json(system, user, EXTRACT_SCHEMA)`; on any error, log to stderr and return `([], [])` (partial graph > crash — reported, not silent); namespace returned ids with a `f"c{chunk.index}:"` prefix (entities and the subject/object refs) so cross-chunk ids never collide.
  - `consolidate`: `reps, id_map = resolver.resolve(raw_entities)`; for each fact: drop if `confidence < threshold`; remap `subject_id`/`object_id` through `id_map`; drop if either id is not among the reps (ungrounded/dangling); `relation = canonical_relation(relation, vocab)`; dedup on `(subject_id, relation, object_id)`. Return `FactSet(clip, reps, facts)`.
  - `extract`: load transcript → `propose_ontology(full_text, llm)` → `chunk_transcript(utterances, cfg.extract.chunk_tokens)` → `_extract_chunk` per chunk → `consolidate(..., EntityResolver(embed_fn=llm.embed), set(ontology.relations), cfg.extract.confidence_threshold, ..., clip)` → write `<clip>.facts.json`.

- [ ] **Step 1: Write the failing tests (pure consolidation + prompt).** Create `tests/test_extract.py`:
```python
from src.contracts import Entity, Fact, Utterance
from src.chunking import Chunk
from src.resolve import EntityResolver
from src.extract import consolidate, build_prompt

def _resolver():
    return EntityResolver(embed_fn=lambda xs: [[0.0]] * len(xs))   # exact-name only

def test_consolidate_merges_dedups_and_filters_confidence():
    ents = [Entity(id="c0:e1", label="Entity", type="Instrument", name="PMS"),
            Entity(id="c1:e1", label="Entity", type="Instrument", name="pms"),
            Entity(id="c0:e2", label="Attribute", type="Money", name="50 lakh")]
    facts = [
        Fact(subject_id="c0:e1", relation="min investment", object_id="c0:e2",
             statement="PMS 50 lakh", speaker="S0", confidence=0.9, statement_id="stmt:pms:0"),
        Fact(subject_id="c1:e1", relation="minimum investment amount", object_id="c0:e2",
             statement="pms 50 lakh again", speaker="S0", confidence=0.95, statement_id="stmt:pms:2"),
        Fact(subject_id="c0:e1", relation="HAS_VALUE", object_id="c0:e2",
             statement="weak", speaker="S0", confidence=0.3, statement_id="stmt:pms:0"),  # < 0.6
    ]
    fs = consolidate(ents, facts, vocab=None, threshold=0.6, resolver=_resolver(), clip="pms")
    assert len(fs.entities) == 2                          # PMS (two names merged) + 50 lakh
    assert [f.relation for f in fs.facts] == ["MINIMUM_INVESTMENT"]   # dup merged; low-conf dropped
    assert fs.facts[0].subject_id == "entity:pms" and fs.facts[0].object_id == "attribute:50-lakh"

def test_consolidate_drops_facts_with_unknown_entity_refs():
    ents = [Entity(id="c0:e1", label="Entity", type="Instrument", name="PMS")]
    facts = [Fact(subject_id="c0:e1", relation="HAS_VALUE", object_id="c0:UNKNOWN",
                  statement="x", speaker="S0", confidence=0.9)]
    fs = consolidate(ents, facts, vocab=None, threshold=0.6, resolver=_resolver(), clip="pms")
    assert fs.facts == []                                 # dangling object ref -> dropped

def test_prompt_marks_overlap_context_only_and_tags_statement_ids():
    ch = Chunk(index=1,
               utterances=[Utterance(speaker="S1", text="AIF needs 1 crore", start=0, end=1)],
               indices=[2],
               context=Utterance(speaker="S0", text="PMS needs 50 lakh", start=0, end=1),
               context_index=1)
    system, user = build_prompt(ch, ontology=None, clip="pms")
    assert "stmt:pms:2" in user                           # extractable turn tagged with its id
    assert "CONTEXT-ONLY" in user and "stmt:pms:1" in user # overlap turn present but marked
    assert "do not extract" in user.lower() and "do not cite" in user.lower()
```

- [ ] **Step 2: Run, confirm RED.**
Run: `source .venv/bin/activate && pytest tests/test_extract.py -v` → FAIL (`No module named 'src.extract'`).

- [ ] **Step 3: Implement** `src/extract.py` per the interfaces + intended logic. Make the 3 unit tests pass.

- [ ] **Step 4: Run, confirm GREEN.**
Run: `source .venv/bin/activate && pytest tests/test_extract.py -v` → 3 PASS.

- [ ] **Step 5: Add the integration test (marked) — needs LM Studio.** Append:
```python
import pytest

@pytest.mark.integration
def test_extract_sample2_against_lmstudio():
    from src.extract import extract
    fs = extract("sample2")                               # needs data/work/sample2.transcript.json
    assert fs.entities, "expected at least one entity"
    assert all(f.statement_id for f in fs.facts), "every fact must be grounded with a statement_id"
```
Run (LM Studio up): `source .venv/bin/activate && pytest tests/test_extract.py -m integration -o addopts="" -v` → writes `data/work/sample2.facts.json`; entities non-empty; every fact grounded.
**If the local LLM's output is unusable or the schema fights the model, STOP and report — do not hack.**

- [ ] **Step 6: Commit.**
```bash
git add src/extract.py tests/test_extract.py
git commit -m "feat(extract): chunk->LLM extract->consolidate into grounded FactSet"
```

---

## Task 6: Graph build (`src/graph.py`)

**Files:**
- Create: `src/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `FactSet`/`Transcript` (contracts), `safe_rel_type`/`slugify`/`statement_id` (resolve), `neo4j.GraphDatabase`, `dotenv`.
- Produces:
  - `BACKBONE_LABELS = {"Speaker","Statement","Entity","Claim","Attribute"}`.
  - `flatten_props(d)->dict` — keep `str|int|float|bool|None`; `json.dumps` everything else (Neo4j can't store nested maps/lists).
  - `_entity_merge_cypher(label)->str` — raises `ValueError` unless `label in BACKBONE_LABELS`; returns the MERGE string with the label backtick-interpolated.
  - `connect()->Driver` — reads `NEO4J_URI/USERNAME/PASSWORD` from `.env`; raises a clear "is the 'atyx' instance running?" error if unset/unreachable.
  - `upsert(factset, transcript, driver, database="neo4j")->dict` — returns `{"statements","entities","facts"}` counts.
  - `run(clip)->dict` — load `<clip>.facts.json` + `<clip>.transcript.json`, connect, upsert (db from `NEO4J_DATABASE`, default `neo4j`), print counts; CLI `python -m src.graph <clip>`.
- **Contract — Cypher MERGE patterns (idempotent; labels/rel-types validated before interpolation, all values parameterized):**
  ```cypher
  // statement + speaker (one per transcript utterance)
  MERGE (s:Statement {id:$id})
    SET s.text=$text, s.speaker=$spk, s.clip=$clip, s.start=$start, s.end=$end
  MERGE (p:Speaker {id:$pid}) SET p.name=$spk
  MERGE (p)-[:SAID]->(s)

  // entity (label from BACKBONE_LABELS only; $attrs pre-flattened)
  MERGE (n:`<label>` {id:$id}) SET n.name=$name, n.type=$type, n += $attrs

  // fact edge (rel from safe_rel_type(); grounding on the edge)
  MATCH (a {id:$sid}), (b {id:$oid})
  MERGE (a)-[r:`<REL>`]->(b)
    SET r.confidence=$conf, r.speaker=$spk, r.source_statement_id=$ssid, r.statement=$stmt
  ```
- **Intended logic (build phase):** `upsert` opens one session, writes all `:Statement`/`:Speaker` (per transcript utterance, id via `statement_id(clip, i)`, speaker id `f"speaker:{slugify(speaker)}"`), then entities, then fact edges; each via `session.execute_write`. Relationship/label strings come only from `safe_rel_type` / the `BACKBONE_LABELS` guard, so interpolation is safe.

- [ ] **Step 1: Write the failing tests (pure helpers + guards).** Create `tests/test_graph.py`:
```python
import pytest
from src.graph import flatten_props, BACKBONE_LABELS, _entity_merge_cypher

def test_flatten_props_stringifies_nonprimitives():
    out = flatten_props({"a": 1, "b": "x", "c": [1, 2], "d": {"k": 1}, "e": 1.5, "f": True})
    assert out["a"] == 1 and out["b"] == "x" and out["e"] == 1.5 and out["f"] is True
    assert out["c"] == "[1, 2]" and out["d"] == '{"k": 1}'

def test_entity_merge_cypher_rejects_unknown_label():
    with pytest.raises(ValueError):
        _entity_merge_cypher("Person")                   # not in the backbone allowlist
    assert "`Entity`" in _entity_merge_cypher("Entity")
    assert BACKBONE_LABELS == {"Speaker", "Statement", "Entity", "Claim", "Attribute"}
```

- [ ] **Step 2: Run, confirm RED.**
Run: `source .venv/bin/activate && pytest tests/test_graph.py -v` → FAIL (`No module named 'src.graph'`).

- [ ] **Step 3: Implement** `src/graph.py` per the interfaces + Cypher patterns + intended logic. Make the 2 unit tests pass. (The live upsert is exercised by Task 7's integration test.)

- [ ] **Step 4: Run, confirm GREEN.**
Run: `source .venv/bin/activate && pytest tests/test_graph.py -v` → 2 PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/graph.py tests/test_graph.py
git commit -m "feat(graph): idempotent Neo4j upsert with grounded schema mapping"
```

---

## Task 7: End-to-end Phase 2 acceptance (extract → graph on sample2)

**Files:**
- Test: `tests/test_graph.py` (append a marked integration acceptance test)

**Interfaces:**
- Consumes: `src.extract.extract`, `src.graph` (`connect`/`upsert`).
- **What it verifies:** a real `transcript → FactSet → Neo4j` run is non-empty, every fact edge's `source_statement_id` resolves to a real `:Statement` node (traceability), and a second upsert does not change node counts (idempotency).

- [ ] **Step 1: Write the acceptance integration test.** Append to `tests/test_graph.py`:
```python
@pytest.mark.integration
def test_phase2_end_to_end_sample2():
    """Requires LM Studio + a running Neo4j 'atyx' instance + sample2.transcript.json."""
    import os
    from src.extract import extract
    from src.graph import connect, upsert
    from src.contracts import Transcript
    fs = extract("sample2")
    assert fs.entities
    t = Transcript.model_validate_json(open("data/work/sample2.transcript.json").read_text())
    drv = connect()
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    try:
        counts = upsert(fs, t, drv, database=db)
        assert counts["statements"] == len(t.utterances)
        with drv.session(database=db) as s:              # traceability
            for f in fs.facts:
                rec = s.run("MATCH (st:Statement {id:$id}) RETURN st.text AS text",
                            id=f.statement_id).single()
                assert rec is not None, f"ungrounded edge: {f.statement_id}"
            before = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        upsert(fs, t, drv, database=db)                  # idempotency: re-run
        with drv.session(database=db) as s:
            after = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        assert after == before, "re-upsert created duplicate nodes (not idempotent)"
    finally:
        drv.close()
```

- [ ] **Step 2: Run the acceptance test (LM Studio + Neo4j up).**
Run: `source .venv/bin/activate && pytest tests/test_graph.py::test_phase2_end_to_end_sample2 -m integration -o addopts="" -v`
Expected: PASS — facts grounded to real `:Statement` nodes; re-upsert idempotent. Eyeball in Neo4j Browser (`localhost:7474`): `MATCH (n) RETURN n LIMIT 100`.
**If extraction or upsert fails or the graph is empty/garbled, STOP and report — do not hack.**

- [ ] **Step 3: Verify the full unit suite stays green.**
Run: `source .venv/bin/activate && pytest -q` → all unit tests pass (integration deselected by default).

- [ ] **Step 4: Commit.**
```bash
git add tests/test_graph.py
git commit -m "test(graph): end-to-end Phase 2 acceptance — grounded, idempotent graph on sample2"
```

---

## Notes for the executor
- **README:** after Task 6/7, add a short "Phase 2 / Neo4j" section (instance setup recap + `NEO4J_*` `.env` keys + `python -m src.extract <clip>` / `python -m src.graph <clip>`). Fold into the Task 6 commit or a follow-up docs commit — keep docs current.
- **Don't hack:** if the local LLM's structured output fights the extraction schema, or Neo4j behaves unexpectedly, STOP and report rather than working around it (project rule).
