from src.contracts import Entity, Fact, Utterance
from src.chunking import Chunk
from src.resolve import EntityResolver
from src.extract import consolidate, build_prompt, namespace


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


def test_confidence_threshold_is_inclusive_at_boundary():     # confidence == threshold is KEPT (drop is strict <)
    ents = [Entity(id="c0:e1", label="Entity", type="Instrument", name="PMS"),
            Entity(id="c0:e2", label="Attribute", type="Money", name="50 lakh"),
            Entity(id="c0:e3", label="Attribute", type="Money", name="1 crore")]
    at = Fact(subject_id="c0:e1", relation="HAS_VALUE", object_id="c0:e2",
              statement="x", speaker="S0", confidence=0.6, statement_id="stmt:pms:0")    # == threshold
    below = Fact(subject_id="c0:e1", relation="HAS_VALUE", object_id="c0:e3",
                 statement="y", speaker="S0", confidence=0.599, statement_id="stmt:pms:1")  # < threshold
    fs = consolidate(ents, [at, below], vocab=None, threshold=0.6, resolver=_resolver(), clip="pms")
    assert [f.object_id for f in fs.facts] == ["attribute:50-lakh"]   # 0.6 kept, 0.599 dropped


def test_consolidate_empty_inputs_yield_valid_empty_factset():
    fs = consolidate([], [], vocab=None, threshold=0.6, resolver=_resolver(), clip="pms")
    assert fs.clip == "pms" and fs.entities == [] and fs.facts == []


def test_namespacing_keeps_reused_local_ids_separate():
    # chunk 0 used local id "e1" for PMS; chunk 1 reused "e1" for AIF — must NOT merge
    e0, f0 = namespace([Entity(id="e1", label="Entity", type="Instrument", name="PMS")],
                       [Fact(subject_id="e1", relation="HAS_MIN", object_id="e1",
                             statement="PMS", speaker="S0", confidence=0.9)], 0)
    e1, f1 = namespace([Entity(id="e1", label="Entity", type="Instrument", name="AIF")],
                       [Fact(subject_id="e1", relation="HAS_MIN", object_id="e1",
                             statement="AIF", speaker="S0", confidence=0.9)], 1)
    assert e0[0].id == "c0:e1" and e1[0].id == "c1:e1"                 # same local id -> distinct ids
    assert f0[0].subject_id == "c0:e1" and f1[0].subject_id == "c1:e1"
    fs = consolidate(e0 + e1, f0 + f1, vocab=None, threshold=0.6, resolver=_resolver(), clip="pms")
    assert {e.id for e in fs.entities} == {"entity:pms", "entity:aif"} # stayed separate


import pytest

@pytest.mark.integration
def test_extract_sample2_against_lmstudio():
    from src.extract import extract
    fs = extract("sample2")                               # needs data/work/sample2.transcript.json
    assert fs.entities, "expected at least one entity"
    assert all(f.statement_id for f in fs.facts), "every fact must be grounded with a statement_id"
