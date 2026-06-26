from src.ontology import propose_ontology, Ontology, BASE_ONTOLOGY

class FakeLLM:
    def __init__(self, payload): self.payload = payload
    def chat_json(self, system, user, schema): return self.payload

def test_uses_llm_proposal_and_canonicalizes_relations():
    llm = FakeLLM({"entity_types": ["Instrument", "Organization"],
                   "relations": ["min investment", "minimum investment amount"]})
    ont = propose_ontology("some transcript text", llm)
    assert ont.entity_types == ["Instrument", "Organization"]
    assert ont.relations == ["MINIMUM_INVESTMENT"]          # near-synonyms collapsed + deduped

def test_falls_back_on_empty_proposal():
    assert propose_ontology("t", FakeLLM({"entity_types": [], "relations": []})) == BASE_ONTOLOGY

def test_falls_back_on_llm_error():
    class Boom:
        def chat_json(self, *a, **k): raise RuntimeError("no server")
    assert propose_ontology("t", Boom()) == BASE_ONTOLOGY

import pytest

@pytest.mark.integration
def test_propose_ontology_against_lmstudio():
    from src.config import load_config
    from src.llm import LLM
    ont = propose_ontology("Speaker A: PMS needs 50 lakh. Speaker B: AIF needs 1 crore.",
                           LLM(load_config().llm))
    assert ont.entity_types and ont.relations
