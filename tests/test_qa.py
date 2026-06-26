from src.qa import is_read_only


def test_read_queries_pass():
    assert is_read_only("MATCH (n) RETURN n")
    assert is_read_only("MATCH (a)-[r]->(b) RETURN a, r.source_statement_id, b LIMIT 5")
    assert is_read_only("MATCH (n) WHERE n.createdAt > 0 RETURN n.created")  # word-boundary: not a write


def test_write_queries_rejected():
    for q in ["CREATE (n)", "MATCH (n) DELETE n", "MATCH (n) DETACH DELETE n",
              "MATCH (n) SET n.x = 1", "MERGE (n:X {id:1})", "MATCH (n) REMOVE n.x",
              "DROP INDEX foo", "FOREACH (x IN [1] | SET x.y = 1)", "LOAD CSV FROM 'f' AS r CREATE (n)",
              "create (n)", "match (n) set n.x=1"]:
        assert not is_read_only(q), q


def test_call_allowlist():
    # read introspection procs pass
    assert is_read_only("CALL db.labels()")
    assert is_read_only("CALL db.relationshipTypes()")
    # write / non-allowlisted procs rejected
    assert not is_read_only("CALL apoc.create.node(['X'], {}) YIELD node RETURN node")
    assert not is_read_only("MATCH (n) CALL apoc.refactor.cloneNodes([n]) YIELD output RETURN output")


from src.qa import format_schema


def test_format_schema_lists_labels_rels_types_and_property_model():
    s = format_schema(labels=["Entity", "Statement", "Speaker"],
                      rel_types=["ACHIEVES_GOAL", "REQUIRES_INVESTMENT"],
                      entity_types=["FinancialGoal", "WealthStrategy"])
    assert "ACHIEVES_GOAL" in s and "FinancialGoal" in s and "Statement" in s
    assert "source_statement_id" in s                      # the prompt must know edges carry grounding


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


# ---------------------------------------------------------------------------
# Task 4: Cypher generation
# ---------------------------------------------------------------------------

from src.qa import build_cypher_prompt, generate_cypher, CYPHER_SCHEMA


def test_cypher_prompt_carries_schema_and_readonly_and_provenance_rules():
    system, user = build_cypher_prompt("what is the goal?", "SCHEMA-TEXT-HERE")
    assert "SCHEMA-TEXT-HERE" in user
    low = (system + user).lower()
    assert "read-only" in low or "read only" in low
    assert "source_statement_id" in (system + user)         # provenance rule present
    assert "create" in low and "delete" in low              # forbids writes explicitly
    assert "merge" in low and "set" in low and "remove" in low  # full forbidden list


def test_cypher_prompt_includes_prior_error_on_retry():
    _, user = build_cypher_prompt("q", "S", error="SyntaxError: unexpected FOO")
    assert "SyntaxError: unexpected FOO" in user


def test_generate_cypher_returns_cypher_field():
    class FakeLLM:
        def chat_json(self, system, user, schema):
            assert schema == CYPHER_SCHEMA, f"Expected CYPHER_SCHEMA, got {schema!r}"
            return {"cypher": "MATCH (n) RETURN n"}
    assert generate_cypher("q", "S", FakeLLM()) == "MATCH (n) RETURN n"


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
