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

def test_clip_registry_parses_with_modes():
    cfg = load_config()
    clips = cfg.demo.clips
    assert len(clips) >= 1
    pms = next(c for c in clips if c.id == "pms")
    assert pms.mode == "graph"
    assert pms.label and pms.domain
    for c in clips:
        assert c.mode in {"graph", "facts", "live"}, c.mode

def test_uploads_path_present():
    assert load_config().paths.uploads  # non-empty path string

import json, pathlib
def test_graph_clips_have_committed_snapshots():
    # Every graph-mode clip must ship a committed, non-empty snapshot — that is
    # what restore_snapshot() loads when the clip is selected (single Neo4j DB).
    cfg = load_config()
    gt = pathlib.Path(cfg.paths.ground_truth)
    graph_clips = [c for c in cfg.demo.clips if c.mode == "graph"]
    assert graph_clips, "expected at least one graph clip in the registry"
    for c in graph_clips:
        snap = gt / f"{c.id}_graph_snapshot.json"
        assert snap.exists(), f"missing snapshot for graph clip {c.id}: {snap}"
        data = json.loads(snap.read_text())
        assert data.get("nodes"), f"empty snapshot for graph clip {c.id}"

def test_facts_clips_have_committed_artifacts():
    # Every facts-mode clip ships committed transcript + facts artifacts (the
    # replay clips serve these without re-running the audio pipeline).
    cfg = load_config()
    work = pathlib.Path(cfg.paths.work)
    facts_clips = [c for c in cfg.demo.clips if c.mode == "facts"]
    assert facts_clips, "expected at least one facts clip in the registry"
    for c in facts_clips:
        assert (work / f"{c.id}.transcript.json").exists(), f"missing transcript for {c.id}"
        facts = json.loads((work / f"{c.id}.facts.json").read_text())
        assert len(facts.get("facts", [])) >= 1, f"empty extraction for {c.id}"
