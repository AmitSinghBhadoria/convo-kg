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
    # "the" dropped; "on" survives (preposition kept); punctuation stripped
    assert safe_rel_type(canonical_relation("decides (on) the! plan")) == "DECIDES_ON_PLAN"

def test_safe_rel_type_rejects_trailing_newline():        # defense-in-depth backstop must be airtight
    with pytest.raises(ValueError):
        safe_rel_type("SAID\n")                           # $ would match before \n; \Z must not

def test_canonical_relation_does_not_passthrough_trailing_newline():
    assert canonical_relation("SAID\n") == "SAID"         # \n must fail the idempotency guard, then strip

def test_base_ontology_relations_roundtrip_clean():       # canonical vocab must be a fixed point
    from src.ontology import BASE_ONTOLOGY
    for rel in BASE_ONTOLOGY.relations:
        assert canonical_relation(rel) == rel, f"{rel} must canonicalize to itself"

def test_relations_that_must_not_collapse_stay_distinct():
    assert canonical_relation("min investment") == "MINIMUM_INVESTMENT"
    assert canonical_relation("max investment") == "MAXIMUM_INVESTMENT"
    assert canonical_relation("min investment") != canonical_relation("max investment")

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
