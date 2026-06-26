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
        "3. When the answer requires traversing a fact relationship (any edge in\n"
        "   the graph), RETURN that relationship's source_statement_id aliased\n"
        "   as provenance so the caller can trace the answer back to its source.\n"
        "4. Also RETURN the relevant node id(s) so individual nodes can be fetched.\n"
        "5. Prefer single-hop queries; use multi-hop only if the question\n"
        "   strictly requires it.\n"
        "6. Output ONLY valid Cypher inside the JSON field 'cypher'.\n"
        "   Do NOT include explanations or markdown fences.\n"
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
