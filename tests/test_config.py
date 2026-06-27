from src.config import load_config

def test_eval_config_carries_slice_noise_and_spotcheck():
    ec = load_config().eval
    assert ec.snr_levels == [20, 15, 10, 5, 0]
    assert ec.clip_prefix == "pmsslice"
    assert ec.degraded_snr in ec.snr_levels
    assert ec.noise_path.endswith("cafe_16k.wav")
    assert ec.slice_end_s > ec.slice_start_s
    assert len(ec.spotcheck_questions) >= 2

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
