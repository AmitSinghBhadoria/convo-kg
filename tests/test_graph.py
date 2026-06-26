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
