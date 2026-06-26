import pytest
from pydantic import ValidationError
from src.contracts import Word, Utterance, Transcript, Entity, Fact, FactSet

def test_transcript_roundtrip():
    u = Utterance(speaker="SPEAKER_01", text="PMS minimum is 50 lakh", start=8.0, end=11.2,
                  words=[Word(text="PMS", start=8.0, end=8.3, speaker="SPEAKER_01")])
    t = Transcript(clip="dev", snr=None, utterances=[u])
    again = Transcript.model_validate_json(t.model_dump_json())
    assert again.utterances[0].speaker == "SPEAKER_01"
    assert again.utterances[0].words[0].text == "PMS"

def test_entity_label_must_be_backbone():
    with pytest.raises(ValidationError):
        Entity(id="x", label="Foobar", type="Whatever", name="X", attrs={})

def test_fact_statement_is_mandatory():
    with pytest.raises(ValidationError):
        Fact(subject_id="a", relation="R", object_id="b", speaker="S1", confidence=0.9)

def test_fact_requires_grounding():
    f = Fact(subject_id="entity:pms", relation="HAS_MIN", object_id="amount:50l",
             statement="PMS minimum is 50 lakh", speaker="SPEAKER_01", confidence=0.9)
    assert f.statement and f.confidence == 0.9
    fs = FactSet(clip="dev", entities=[Entity(id="entity:pms", label="Entity",
                 type="FinancialProduct", name="PMS", attrs={})], facts=[f])
    assert FactSet.model_validate_json(fs.model_dump_json()).facts[0].relation == "HAS_MIN"

def test_fact_carries_statement_id_default_and_roundtrip():
    from src.contracts import Fact
    f = Fact(subject_id="entity:pms", relation="HAS_MINIMUM_INVESTMENT",
             object_id="attribute:50-lakh", statement="PMS needs 50 lakh minimum",
             speaker="SPEAKER_00", confidence=0.9)
    assert f.statement_id == ""                      # default, Phase 1 back-compat
    f2 = Fact.model_validate_json(
        f.model_copy(update={"statement_id": "stmt:pms:3"}).model_dump_json())
    assert f2.statement_id == "stmt:pms:3"
