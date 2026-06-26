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


@pytest.mark.integration
def test_phase2_end_to_end_sample2():
    """Requires LM Studio + a running Neo4j 'atyx' instance + sample2.transcript.json."""
    import os
    from src.extract import extract
    from src.graph import connect, upsert
    from src.contracts import Transcript
    fs = extract("sample2")
    assert fs.entities
    t = Transcript.model_validate_json(open("data/work/sample2.transcript.json").read())
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


@pytest.mark.integration
def test_upsert_empty_factset_does_not_crash():
    """Zero usable facts must still produce a valid (statements-only) graph, no crash."""
    import os
    from src.graph import connect, upsert
    from src.contracts import FactSet, Transcript, Utterance
    t = Transcript(clip="emptytest", utterances=[Utterance(speaker="S0", text="hello", start=0, end=1)])
    fs = FactSet(clip="emptytest", entities=[], facts=[])
    drv = connect()
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    try:
        assert upsert(fs, t, drv, database=db) == {"statements": 1, "entities": 0, "facts": 0}
    finally:
        drv.close()
