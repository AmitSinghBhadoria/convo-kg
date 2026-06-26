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
