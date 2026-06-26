"""
qa.py — Natural-language Q&A over the Neo4j knowledge graph.

Phase 3, Task 2: read-only write-clause guard (first gate).
The database-level read-only transaction (Task 5) is the true enforcement;
this is the cheap first line that rejects obviously mutating Cypher before
it is ever sent to the driver.
"""

import re

# Write clauses that must never appear in generated Cypher.
_WRITE_CLAUSES = [
    "CREATE",
    "MERGE",
    "DELETE",
    "SET",
    "REMOVE",
    "DROP",
    "DETACH",
    "FOREACH",
    "LOAD",
]

# Read-only CALL targets explicitly permitted (db introspection).
_ALLOWED_CALL_PROCS = re.compile(
    r"\bCALL\s+db\.(labels|relationshipTypes|schema)\b",
    re.IGNORECASE,
)

# Word-boundary pattern for each write clause.
_WRITE_RE = re.compile(
    r"\b(?:" + "|".join(_WRITE_CLAUSES) + r")\b",
    re.IGNORECASE,
)

# Any CALL that is NOT one of the allowed introspection procs.
_CALL_RE = re.compile(r"\bCALL\b", re.IGNORECASE)


def is_read_only(cypher: str) -> bool:
    """Return True iff *cypher* contains no write clauses.

    Uses word-boundary matching so property names like ``n.createdAt`` or
    ``n.created`` do not trigger the guard.  Conservative: a write keyword
    inside a string literal will still cause rejection — the database-level
    read-only transaction (Task 5) is the authoritative enforcement layer.
    """
    # Reject any recognised write clause.
    if _WRITE_RE.search(cypher):
        return False

    # Reject CALL unless it is one of the allowed read introspection procs.
    for m in _CALL_RE.finditer(cypher):
        # Check whether this CALL is followed by an allowed proc name.
        # Use the string from the match position onward for the allowed check.
        tail = cypher[m.start():]
        if not _ALLOWED_CALL_PROCS.match(tail):
            return False

    return True


# ---------------------------------------------------------------------------
# Schema introspection — Task 3
# ---------------------------------------------------------------------------

def format_schema(
    labels: list[str],
    rel_types: list[str],
    entity_types: list[str],
) -> str:
    """Return a compact schema description for use in LLM Cypher prompts.

    Pure function (no DB access).  Includes the live-discovered labels,
    relationship types, and entity sub-types, plus the fixed property model
    for the Atyx knowledge graph so the LLM knows the shape of every node and
    that fact edges carry ``source_statement_id`` for attribution grounding.
    """
    labels_str = ", ".join(f":{lbl}" for lbl in labels) if labels else "(none)"
    rels_str = ", ".join(rel_types) if rel_types else "(none)"
    types_str = ", ".join(entity_types) if entity_types else "(none)"

    return (
        "=== Graph Schema ===\n"
        f"Node labels: {labels_str}\n"
        f"Relationship types: {rels_str}\n"
        f"Entity sub-types (e.type): {types_str}\n"
        "\n"
        "Property model:\n"
        "  :Entity        {id, name, type}\n"
        "  :Statement     {id, text, speaker, clip, start, end}\n"
        "  :Speaker       {id, name}\n"
        "  fact edges     carry source_statement_id (links each relationship to its source Statement)\n"
        "==================="
    )


def introspect_schema(driver, database: str) -> str:
    """Query the live Neo4j graph and return a formatted schema string.

    Runs three read queries against *database*:
    - ``CALL db.labels()`` — all node labels.
    - ``CALL db.relationshipTypes()`` — all relationship type names.
    - ``MATCH (e:Entity) RETURN DISTINCT e.type AS type`` — Entity sub-types
      (null/empty values are filtered out).

    Returns ``format_schema(...)`` with the collected results.
    """
    with driver.session(database=database) as session:
        labels = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label")]
        rel_types = [r["relationshipType"] for r in session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )]
        entity_types = [
            r["type"]
            for r in session.run(
                "MATCH (e:Entity) RETURN DISTINCT e.type AS type"
            )
            if r["type"]
        ]

    return format_schema(labels=labels, rel_types=rel_types, entity_types=entity_types)


# ---------------------------------------------------------------------------
# Cypher generation — Task 4
# ---------------------------------------------------------------------------

# JSON schema that enforces a structured LLM response containing only a
# "cypher" string.  Passed directly to chat_json as the response format.
CYPHER_SCHEMA: dict = {
    "type": "object",
    "properties": {"cypher": {"type": "string"}},
    "required": ["cypher"],
    "additionalProperties": False,
}


def build_cypher_prompt(
    question: str,
    schema_text: str,
    error: str | None = None,
) -> tuple[str, str]:
    """Return a (system, user) prompt pair for read-only Cypher generation.

    *question* — the natural-language question to answer.
    *schema_text* — the formatted graph schema (from ``format_schema``).
    *error* — if provided (retry path), the previous error text is included
               in the user message with an instruction to fix it.
    """
    system = (
        "You are an expert Neo4j Cypher query generator.\n"
        "\n"
        "RULES — you MUST follow all of them:\n"
        "1. Produce exactly ONE read-only Cypher query.\n"
        "2. Use ONLY these clauses: MATCH, WHERE, RETURN, ORDER BY, LIMIT.\n"
        "   Absolutely FORBIDDEN: CREATE, MERGE, DELETE, SET, REMOVE, DROP,\n"
        "   DETACH, FOREACH, LOAD, CALL (unless db.labels / db.relationshipTypes).\n"
        "3. When the answer requires traversing a fact relationship, bind the\n"
        "   relationship to a variable and RETURN its source_statement_id aliased\n"
        "   as provenance, e.g.:\n"
        "     MATCH (a)-[r:SOME_REL]->(b) RETURN r.source_statement_id AS provenance, a.id, b.id\n"
        "   Do NOT alias a node property (such as a Statement's id) as provenance.\n"
        "4. Also RETURN the relevant node id(s) so individual nodes can be fetched.\n"
        "5. Prefer single-hop queries; use multi-hop only if the question\n"
        "   strictly requires it.\n"
        "6. Output ONLY valid Cypher inside the JSON field 'cypher'.\n"
        "   Do NOT include explanations or markdown fences.\n"
        "7. Use ONLY the node labels and relationship types that appear in the\n"
        "   schema below. Do NOT invent labels or relationship types that are\n"
        "   not listed there.\n"
    )

    user_parts = [
        "Graph schema:\n",
        schema_text,
        "\n\nQuestion: " + question,
    ]

    if error:
        user_parts.append(
            f"\n\nThe previous query produced this error:\n{error}\n"
            "Please fix the Cypher so it no longer causes that error."
        )

    user = "".join(user_parts)
    return system, user


def generate_cypher(
    question: str,
    schema_text: str,
    llm,
    error: str | None = None,
) -> str:
    """Ask the local LLM to generate a read-only Cypher query.

    Uses ``build_cypher_prompt`` to form the prompt and ``CYPHER_SCHEMA`` to
    enforce a structured JSON response.  Returns the ``cypher`` field,
    stripped of leading/trailing whitespace.
    """
    system, user = build_cypher_prompt(question, schema_text, error)
    result = llm.chat_json(system, user, CYPHER_SCHEMA)
    return result["cypher"].strip()


# ---------------------------------------------------------------------------
# Task 5: execution backstop + provenance resolution
# ---------------------------------------------------------------------------

from src.contracts import Provenance


def explain_ok(driver, database: str, cypher: str) -> tuple[bool, str | None]:
    """Plan *cypher* via ``EXPLAIN`` inside a read transaction.

    Returns ``(True, None)`` if the query plans without error, or
    ``(False, <error message>)`` if the database raises during planning.
    The EXPLAIN is run inside a read-access-mode transaction so no mutations
    can occur even if the query itself contains write clauses.
    """
    explain_cypher = f"EXPLAIN {cypher}"

    def _run(tx):
        list(tx.run(explain_cypher))

    try:
        with driver.session(database=database) as session:
            session.execute_read(_run)
        return True, None
    except Exception as exc:
        return False, str(exc)


def run_read(driver, database: str, cypher: str) -> list[dict]:
    """Execute *cypher* in a read-access-mode transaction and return rows.

    Uses ``session.execute_read`` — Neo4j enforces read-only access at the
    database level, so any write clause will raise without reaching the graph.
    This is the hard backstop that complements the text-level ``is_read_only``
    guard applied by the orchestrator upstream.

    Each neo4j Record is converted to a plain ``dict`` via ``record.data()``.
    """
    def _run(tx):
        return [record.data() for record in tx.run(cypher)]

    with driver.session(database=database) as session:
        return session.execute_read(_run)


def extract_node_ids(rows: list[dict]) -> list[str]:
    """Return distinct node ``id`` values from *rows*, order-preserving.

    A value is "node-shaped" if it is a ``dict`` containing a string ``"id"``
    key.  Plain scalars and dicts without an ``"id"`` key are ignored.
    """
    seen: set[str] = set()
    result: list[str] = []
    for row in rows:
        for value in row.values():
            if isinstance(value, dict) and "id" in value and isinstance(value["id"], str):
                node_id = value["id"]
                if node_id not in seen:
                    seen.add(node_id)
                    result.append(node_id)
    return result


def extract_provenance_ids(rows: list[dict]) -> list[str]:
    """Return distinct non-None statement ids from *rows*, order-preserving.

    Collects ids found under the key ``"provenance"`` or any key ending with
    ``"source_statement_id"`` (e.g. ``"r.source_statement_id"``).  ``None``
    values are skipped.
    """
    seen: set[str] = set()
    result: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key == "provenance" or key.endswith("source_statement_id"):
                if value is not None:
                    sid = str(value)
                    if sid not in seen:
                        seen.add(sid)
                        result.append(sid)
    return result


def resolve_provenance(
    driver, database: str, statement_ids: list[str]
) -> list[Provenance]:
    """Fetch :Statement nodes for *statement_ids* and return Provenance records.

    For each id, runs a parameterised ``MATCH`` — the id is passed as a query
    parameter, never string-interpolated.  Ids that match no :Statement node
    are silently skipped.
    """
    provenance: list[Provenance] = []
    with driver.session(database=database) as session:
        for sid in statement_ids:
            records = session.execute_read(
                lambda tx, _id=sid: list(
                    tx.run(
                        "MATCH (s:Statement {id: $id}) RETURN s.text AS text, s.speaker AS speaker",
                        id=_id,
                    )
                )
            )
            for record in records:
                provenance.append(
                    Provenance(
                        statement_id=sid,
                        speaker=record["speaker"],
                        text=record["text"],
                        kind="source",
                    )
                )
    return provenance


# ---------------------------------------------------------------------------
# Task 6: Cypher-path orchestrator (compose_answer, infer_hops, answer)
# ---------------------------------------------------------------------------

import os
from typing import Literal

from src.contracts import QAResult

ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

MAX_RETRIES = 2


def compose_answer(
    question: str,
    rows: list[dict],
    provenance: list[Provenance],
    llm,
) -> str:
    """Compose a grounded English answer from graph rows and provenance quotes.

    Instructs the LLM to use ONLY the provided rows and source statements;
    if they do not contain the answer, say so plainly (do not invent).
    """
    system = (
        "You are a factual question-answering assistant.\n"
        "Answer in English using ONLY the provided graph rows and the quoted "
        "source statements below.\n"
        "If they do not contain the answer, say so plainly — do not invent anything."
    )
    prov_lines = "\n".join(
        f'  [{p.statement_id}] {p.speaker}: "{p.text}"'
        for p in provenance
    )
    user = (
        f"Question: {question}\n\n"
        f"Graph rows:\n{rows}\n\n"
        f"Source statements:\n{prov_lines if prov_lines else '(none)'}"
    )
    result = llm.chat_json(system, user, ANSWER_SCHEMA)
    return result["answer"].strip()


def infer_hops(cypher: str) -> Literal["single", "multi"]:
    """Return 'multi' if cypher traverses >=2 relationship patterns, else 'single'.

    Pure function. Counts occurrences of '-[' as a proxy for relationship
    patterns in the query.
    """
    return "multi" if cypher.count("-[") >= 2 else "single"


def answer(
    question: str,
    llm=None,
    driver=None,
    database: str | None = None,
) -> QAResult:
    """Orchestrate the full Cypher Q&A path.

    Sequence: introspect schema → retry loop (1 + MAX_RETRIES) →
    generate_cypher → is_read_only guard → explain_ok check →
    run_read → resolve provenance → compose_answer → QAResult.

    If no valid Cypher is produced after all retries, OR if the query
    returns no rows, returns a found=False placeholder (Task 7 replaces
    this branch with the semantic fallback).

    Owns the driver lifecycle when driver is None (connect + close).
    """
    from src.config import load_config
    from src.llm import LLM

    if llm is None:
        llm = LLM(load_config().llm)
    if database is None:
        database = os.environ.get("NEO4J_DATABASE", "neo4j")

    own_driver = driver is None
    if own_driver:
        from src.graph import connect
        driver = connect()

    try:
        schema = introspect_schema(driver, database)

        # Augment schema with live relationship direction examples so the LLM
        # knows which node labels each relationship type connects and in which
        # direction.  This prevents the model from inventing wrong directions
        # (e.g. Statement→Entity instead of Entity→Entity for fact edges).
        # Collect known node labels for label-guard below.
        _known_labels: set[str] = set()
        try:
            _label_rows = run_read(
                driver, database,
                "CALL db.labels() YIELD label RETURN label"
            )
            _known_labels = {r["label"] for r in _label_rows}
        except Exception:
            pass

        try:
            _rel_samples = run_read(
                driver, database,
                "MATCH (a)-[r]->(b) "
                "RETURN labels(a)[0] AS fl, a.type AS ft, "
                "       type(r) AS rt, "
                "       labels(b)[0] AS tl, b.type AS tt "
                "LIMIT 30"
            )
            _seen: set[tuple] = set()
            _dir_lines: list[str] = []
            for _row in _rel_samples:
                _key = (
                    _row.get("fl"), _row.get("ft") or "",
                    _row.get("rt"),
                    _row.get("tl"), _row.get("tt") or "",
                )
                if _key not in _seen:
                    _seen.add(_key)
                    _from = (
                        f":{_row['fl']}{{type:'{_row['ft']}'}}"
                        if _row.get("ft") else f":{_row['fl']}"
                    )
                    _to = (
                        f":{_row['tl']}{{type:'{_row['tt']}'}}"
                        if _row.get("tt") else f":{_row['tl']}"
                    )
                    _dir_lines.append(f"  ({_from})-[:{_row['rt']}]->({_to})")
            if _dir_lines:
                schema += (
                    "\n\nRelationship directions (sampled from live graph — "
                    "use ONLY these directions):\n"
                    + "\n".join(_dir_lines)
                )

            # Append entity name samples so the model can reference actual values.
            _ent_samples = run_read(
                driver, database,
                "MATCH (e:Entity) "
                "RETURN e.type AS type, e.name AS name "
                "ORDER BY e.type, e.name "
                "LIMIT 30"
            )
            if _ent_samples:
                _ent_by_type: dict[str, list[str]] = {}
                for _e in _ent_samples:
                    _ent_by_type.setdefault(_e["type"], []).append(_e["name"])
                _ent_lines = [
                    f"  {_t}: {_names}"
                    for _t, _names in _ent_by_type.items()
                ]
                schema += (
                    "\n\nEntity names in the graph (use these to filter by name):\n"
                    + "\n".join(_ent_lines)
                )

            # Critical clarification: provenance is a STRING property on fact
            # edges (not a link to/from Statement nodes). Fact edges connect
            # Entity nodes only.
            _labels_str = ", ".join(sorted(_known_labels)) if _known_labels else "Entity, Statement, Speaker"
            schema += (
                "\n\nCRITICAL SCHEMA RULES — violating these causes empty results:\n"
                f"1. The ONLY valid Neo4j node labels are: {_labels_str}.\n"
                "   WealthStrategy, FinancialGoal, AssetClass, MonetaryAmount, TimePeriod "
                "are NOT labels — they are VALUES of the 'type' property on :Entity nodes.\n"
                "   CORRECT: (a:Entity {type:'WealthStrategy'})  "
                "WRONG: (a:WealthStrategy)\n"
                "2. Fact edges (ACHIEVES_GOAL, HAS_STRATEGY, etc.) connect "
                "Entity nodes to Entity nodes — NOT to Statement or Speaker nodes.\n"
                "3. source_statement_id is a STRING PROPERTY on the fact edge. "
                "Return it as: RETURN r.source_statement_id AS provenance\n"
                "4. Correct single-hop example: "
                "MATCH (a:Entity {type:'WealthStrategy'})-[r:ACHIEVES_GOAL]->"
                "(b:Entity {type:'FinancialGoal'}) "
                "RETURN r.source_statement_id AS provenance, a.id, a.name\n"
                "5. Correct multi-hop example: "
                "MATCH (a:Entity)-[r1:HAS_STRATEGY]->(b:Entity {type:'WealthStrategy'})"
                "-[r2:ACHIEVES_GOAL]->(c:Entity {type:'FinancialGoal'}) "
                "RETURN r1.source_statement_id AS provenance, a.id, a.name, b.id, b.name"
            )
        except Exception:
            pass  # Best-effort; fallback to base schema

        last_cypher: str | None = None
        error: str | None = None
        valid_cypher: str | None = None

        for _ in range(1 + MAX_RETRIES):
            cypher = generate_cypher(question, schema, llm, error)
            last_cypher = cypher
            if not is_read_only(cypher):
                error = f"Query contains a disallowed write clause: {cypher!r}"
                continue
            ok, err = explain_ok(driver, database, cypher)
            if not ok:
                error = err
                continue
            # Label guard: explain_ok passes even for non-existent labels
            # (they just return 0 rows).  Catch them here so the retry loop
            # can feed back a corrective error message.
            if _known_labels:
                _used = set(re.findall(r'\(\w*:(\w+)', cypher))
                _bad = _used - _known_labels
                if _bad:
                    error = (
                        f"Query uses undefined node labels: {sorted(_bad)}. "
                        f"The ONLY valid labels are {sorted(_known_labels)}. "
                        "Entity sub-types (WealthStrategy, FinancialGoal, etc.) "
                        "are NOT labels — use :Entity {type:'...'} syntax instead."
                    )
                    continue
            # Valid Cypher found
            valid_cypher = cypher
            break

        if valid_cypher is None:
            return QAResult(
                question=question,
                answer="Could not find an answer: failed to produce a valid Cypher query.",
                mode="cypher",
                found=False,
                cypher=last_cypher,
            )

        rows = run_read(driver, database, valid_cypher)

        if not rows:
            return QAResult(
                question=question,
                answer="Could not find an answer: the query returned no results.",
                mode="cypher",
                found=False,
                cypher=valid_cypher,
            )

        prov_ids = extract_provenance_ids(rows)
        prov = resolve_provenance(driver, database, prov_ids)
        node_ids = extract_node_ids(rows)
        ans = compose_answer(question, rows, prov, llm)

        return QAResult(
            question=question,
            answer=ans,
            mode="cypher",
            found=True,
            cypher=valid_cypher,
            rows=rows,
            provenance=prov,
            graph_node_ids=node_ids,
            hops=infer_hops(valid_cypher),
        )
    finally:
        if own_driver:
            driver.close()
