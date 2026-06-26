from src.config import load_config

def test_defaults_load():
    c = load_config("config.yaml")
    assert c.llm.model == "qwen/qwen3.5-9b"
    assert c.limits.max_minutes == 15
    assert 0.0 < c.extract.confidence_threshold <= 1.0
    assert c.eval.snr_levels  # non-empty

def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://example/v1")
    c = load_config("config.yaml")
    assert c.llm.base_url == "http://example/v1"
