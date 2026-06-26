import pytest
from src.config import load_config
from src.llm import LLM

SCHEMA = {"type":"object","properties":{"answer":{"type":"string"}},
          "required":["answer"],"additionalProperties":False}

@pytest.mark.integration
def test_chat_json_against_lmstudio():
    llm = LLM(load_config("config.yaml").llm)
    out = llm.chat_json("Reply as JSON.", "Say the word 'pong' in the answer field.", SCHEMA)
    assert "answer" in out and isinstance(out["answer"], str)

@pytest.mark.integration
def test_embed_against_lmstudio():
    llm = LLM(load_config("config.yaml").llm)
    vecs = llm.embed(["hello", "world"])
    assert len(vecs) == 2 and len(vecs[0]) > 10
