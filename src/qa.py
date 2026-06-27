"""
qa.py — Natural-language Q&A over the Neo4j knowledge graph.

Pipeline: introspect + augment schema → generate read-only Cypher → guard
(is_read_only text reject → EXPLAIN → label guard) → execute in a read-access
transaction → resolve edge-level provenance → compose a grounded English answer;
a semantic fallback over statement embeddings answers (with a cosine floor that
declines off-topic questions) when the Cypher path returns nothing.
Read-only is enforced in depth: the text guard is the cheap first line, the
database-level read-access transaction (run_read / explain_ok) is the true backstop.
"""

import os
import re
from typing import Literal

from src.contracts import Provenance, QAResult

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


def augment_schema(
    driver,
    database: str,
    base_schema: str,
) -> tuple[str, set[str]]:
    """Augment the base schema text with live graph data for better Cypher generation.

    Adds three sections to *base_schema*:
    1. Live relationship direction examples (sampled from the graph) so the LLM
       knows which node types each relationship connects and in which direction.
    2. Entity name samples grouped by live sub-type so the model can reference
       actual entity names when filtering.
    3. CRITICAL SCHEMA RULES — generic, data-driven rules teaching the model
       that Entity sub-types are *type property values*, not Neo4j labels.
       The sub-type list is built from the live graph (not hardcoded), and the
       worked-example queries use schematic ``<PLACEHOLDER>`` names (not demo
       relationship types or entity names).

    Each section has its own ``try/except`` so a failure in one does not
    silently suppress the others (especially the CRITICAL RULES block).

    Returns:
        (augmented_schema_text, known_labels_set)
        The known-labels set is returned so the caller can reuse it for the
        label guard without issuing a second ``db.labels()`` query.
    """
    schema = base_schema
    known_labels: set[str] = set()
    _ent_by_type: dict[str, list[str]] = {}

    # ── Step 1: known labels (single query — reused for CRITICAL RULES + returned) ──
    try:
        _label_rows = run_read(
            driver, database,
            "CALL db.labels() YIELD label RETURN label",
        )
        known_labels = {r["label"] for r in _label_rows}
    except Exception:
        pass

    # ── Step 2: relationship direction samples ──
    try:
        _rel_samples = run_read(
            driver, database,
            "MATCH (a)-[r]->(b) "
            "RETURN labels(a)[0] AS fl, a.type AS ft, "
            "       type(r) AS rt, "
            "       labels(b)[0] AS tl, b.type AS tt "
            "LIMIT 30",
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
    except Exception:
        pass

    # ── Step 3: entity name samples by type ──
    try:
        _ent_samples = run_read(
            driver, database,
            "MATCH (e:Entity) "
            "RETURN e.type AS type, e.name AS name "
            "ORDER BY e.type, e.name "
            "LIMIT 30",
        )
        for _e in _ent_samples:
            _ent_by_type.setdefault(_e["type"], []).append(_e["name"])
        if _ent_by_type:
            _ent_lines = [
                f"  {_t}: {_names}"
                for _t, _names in _ent_by_type.items()
            ]
            schema += (
                "\n\nEntity names in the graph (use these to filter by name):\n"
                + "\n".join(_ent_lines)
            )
    except Exception:
        pass

    # ── Step 4: CRITICAL SCHEMA RULES (always append; data-driven sub-type list) ──
    # Build entity sub-types from live samples; fall back to a generic note.
    _entity_subtypes_str = (
        ", ".join(sorted(_ent_by_type))
        if _ent_by_type
        else "Entity sub-types (see 'Entity sub-types' in schema above)"
    )
    _labels_str = (
        ", ".join(sorted(known_labels))
        if known_labels
        else "Entity, Statement, Speaker"
    )
    schema += (
        "\n\nCRITICAL SCHEMA RULES — violating these causes empty results:\n"
        f"1. The ONLY valid Neo4j node labels are: {_labels_str}. Concept and value\n"
        "   nodes are labelled :Entity OR :Attribute, and their kind (e.g. "
        f"{_entity_subtypes_str})\n"
        "   is a VALUE of the 'type' property, NOT a label.\n"
        "2. MATCH a concept BY NAME with NO label — labels are inconsistent (the same\n"
        "   concept may be :Entity or :Attribute), so a label filter often misses it:\n"
        "   CORRECT: MATCH (a {name:'<NAME>'})-[r:<REL_TYPE>]->(b)\n"
        "   WRONG:   MATCH (a:Entity {type:'<SUB_TYPE>'})...   (misses :Attribute nodes)\n"
        "   WRONG:   MATCH (a:Speaker {name:'<NAME>'})...      (concepts are NOT speakers)\n"
        "3. Fact edges connect two content nodes (Entity/Attribute) DIRECTLY — they do\n"
        "   NOT pass through :Statement or :Speaker nodes.\n"
        "4. source_statement_id is a STRING PROPERTY on the fact edge:\n"
        "   RETURN r.source_statement_id AS provenance\n"
        "5. Single-hop pattern (match the subject by name; return the object names):\n"
        "   MATCH (a {name:'<NAME>'})-[r:<REL_TYPE>]->(b)\n"
        "   RETURN r.source_statement_id AS provenance, b.name\n"
        "6. Multi-hop pattern (chain by name, no labels):\n"
        "   MATCH (a {name:'<NAME>'})-[r1:<REL1>]->(b)-[r2:<REL2>]->(c)\n"
        "   RETURN r1.source_statement_id AS provenance, b.name, c.name\n"
        "7. Use the exact entity names and relationship types listed in the schema above."
    )

    return schema, known_labels


# ---------------------------------------------------------------------------
# Task 7: Semantic fallback — cosine, top_k_statements, fallback_is_confident,
#          _statement_embeddings (cached), semantic_fallback
# ---------------------------------------------------------------------------

FALLBACK_TOP_K: int = 3
FALLBACK_MIN_COSINE: float = 0.40  # empirically tuned on the real 74-statement pms corpus:
# answerable questions score 0.54–0.70 (PMS min 0.63, transparency 0.63, corpus 0.54, when-to-consider 0.70);
# unanswerable score 0.22–0.23 (capital of France 0.23, weather 0.23). Floor 0.40 sits between with margin,
# so off-topic questions decline on the SCORE (found=False), not on the LLM noticing. Re-tune if the corpus changes.

# Module-level embedding cache: statement_id -> {id, speaker, text, vec}
_EMBEDDING_CACHE: dict[str, dict] = {}


def cosine(u: list[float], v: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Zero-vector safe: returns 0.0 if either vector has zero norm (no div-by-zero).
    """
    dot = sum(a * b for a, b in zip(u, v))
    norm_u = sum(a * a for a in u) ** 0.5
    norm_v = sum(b * b for b in v) ** 0.5
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (norm_u * norm_v)


def top_k_statements(
    question_vec: list[float],
    statements_with_vecs: list[dict],
    k: int,
) -> list[dict]:
    """Rank statements by cosine similarity to the question vector.

    Args:
        question_vec: Embedding of the question.
        statements_with_vecs: List of dicts with keys {id, speaker, text, vec}.
        k: Number of top results to return.

    Returns:
        Top-k dicts with keys {id, speaker, text, score} — vec stripped, cosine
        score added — sorted descending by score.
    """
    scored = []
    for stmt in statements_with_vecs:
        score = cosine(question_vec, stmt["vec"])
        scored.append({
            "id": stmt["id"],
            "speaker": stmt["speaker"],
            "text": stmt["text"],
            "score": score,
        })
    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored[:k]


def fallback_is_confident(
    top: list[dict],
    floor: float = FALLBACK_MIN_COSINE,
) -> bool:
    """Return True iff the best top-k result clears the cosine floor.

    The no-hallucination guarantee rests on this score — never on LLM judgment.
    """
    return bool(top) and top[0]["score"] >= floor


def _statement_embeddings(driver, database: str, llm) -> list[dict]:
    """Load all :Statement {id,speaker,text} nodes and embed their texts.

    Results are cached in the module-level _EMBEDDING_CACHE keyed by
    statement id — repeated answer() calls reuse the cache without
    re-embedding.

    Returns a list of {id, speaker, text, vec} dicts.
    """
    global _EMBEDDING_CACHE

    # Load all statement nodes
    with driver.session(database=database) as session:
        records = session.execute_read(
            lambda tx: list(
                tx.run(
                    "MATCH (s:Statement) RETURN s.id AS id, s.speaker AS speaker, s.text AS text"
                )
            )
        )

    # Identify which statements need embedding (not yet cached)
    uncached = [
        {"id": r["id"], "speaker": r["speaker"], "text": r["text"]}
        for r in records
        if r["id"] not in _EMBEDDING_CACHE
    ]

    if uncached:
        texts = [s["text"] for s in uncached]
        vecs = llm.embed(texts)
        for stmt, vec in zip(uncached, vecs):
            _EMBEDDING_CACHE[stmt["id"]] = {
                "id": stmt["id"],
                "speaker": stmt["speaker"],
                "text": stmt["text"],
                "vec": vec,
            }

    # Return all statements (cached + freshly embedded), preserving record order
    all_ids = {r["id"] for r in records}
    return [v for k, v in _EMBEDDING_CACHE.items() if k in all_ids]


def semantic_fallback(
    question: str,
    driver,
    database: str,
    llm,
) -> tuple[bool, str, list["Provenance"], list[str]]:
    """Attempt a cosine-similarity answer from cached :Statement embeddings.

    The no-hallucination guarantee rests on the cosine floor
    (FALLBACK_MIN_COSINE): if the best match score is below it, decline
    and return (False, "", [], []).  Never claim found=True on a weak match.

    Returns:
        (found, answer_text, provenance, graph_node_ids)
        found=False means decline (score too low or no statements).
    """
    stmts = _statement_embeddings(driver, database, llm)
    if not stmts:
        return (False, "", [], [])

    question_vec = llm.embed([question])[0]
    top = top_k_statements(question_vec, stmts, FALLBACK_TOP_K)

    if not fallback_is_confident(top):
        return (False, "", [], [])

    # Build Provenance with kind="related" (semantically similar, not causal source)
    provenance = [
        Provenance(
            statement_id=s["id"],
            speaker=s["speaker"],
            text=s["text"],
            kind="related",
        )
        for s in top
    ]

    # compose_answer expects rows — pass an empty list since we have no Cypher rows
    ans = compose_answer(question, [], provenance, llm)
    return (True, ans, provenance, [])


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
    returns no rows, falls through to the semantic fallback over statement
    embeddings (which itself declines with found=False when the best match is
    below the cosine floor).

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
        schema, _known_labels = augment_schema(
            driver, database, introspect_schema(driver, database)
        )

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
                        "Entity sub-types are NOT labels — "
                        "use :Entity {type:'<SUB_TYPE>'} syntax instead."
                    )
                    continue
            # Valid Cypher found
            valid_cypher = cypher
            break

        if valid_cypher is None:
            # Cypher path exhausted — try semantic fallback
            fb_found, fb_ans, fb_prov, _ = semantic_fallback(question, driver, database, llm)
            if fb_found:
                return QAResult(
                    question=question,
                    answer=fb_ans,
                    mode="semantic-fallback",
                    found=True,
                    cypher=last_cypher,
                    provenance=fb_prov,
                )
            return QAResult(
                question=question,
                answer="I couldn't find that in the conversation.",
                mode="semantic-fallback",
                found=False,
                cypher=last_cypher,
            )

        rows = run_read(driver, database, valid_cypher)

        if not rows:
            # Cypher returned no rows — try semantic fallback
            fb_found, fb_ans, fb_prov, _ = semantic_fallback(question, driver, database, llm)
            if fb_found:
                return QAResult(
                    question=question,
                    answer=fb_ans,
                    mode="semantic-fallback",
                    found=True,
                    cypher=valid_cypher,
                    provenance=fb_prov,
                )
            return QAResult(
                question=question,
                answer="I couldn't find that in the conversation.",
                mode="semantic-fallback",
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


# ---------------------------------------------------------------------------
# CLI entry point: python -m src.qa "<question>"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.qa \"<question>\"")
        sys.exit(1)

    question = sys.argv[1]
    result = answer(question)

    print(f"\nQ: {question}")
    print(f"A: {result.answer}")
    print(f"\nMode: {result.mode}  |  Found: {result.found}  |  Hops: {result.hops}")

    if result.provenance:
        print("\nProvenance:")
        for p in result.provenance:
            print(f"  [{p.kind}] {p.speaker}: \"{p.text}\"")
    else:
        print("\nProvenance: (none)")
