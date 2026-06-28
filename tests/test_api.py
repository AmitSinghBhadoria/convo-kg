import io
import pytest
from fastapi.testclient import TestClient
import src.api as api
from src.api import app

client = TestClient(app)


# ── Upload tests ─────────────────────────────────────────────────────────────

def test_upload_rejects_too_long(monkeypatch):
    monkeypatch.setattr(api.audioprep, "probe_duration", lambda p: 601.0)
    r = client.post("/api/upload", files={"file": ("x.wav", io.BytesIO(b"RIFFxxxx"), "audio/wav")})
    assert r.status_code == 400 and "10 min" in r.json()["detail"]

def test_upload_rejects_non_audio(monkeypatch):
    def boom(p): raise ValueError("not audio")
    monkeypatch.setattr(api.audioprep, "probe_duration", boom)
    r = client.post("/api/upload", files={"file": ("x.txt", io.BytesIO(b"hello"), "text/plain")})
    assert r.status_code == 400

def test_upload_reencode_failure_returns_400(monkeypatch):
    monkeypatch.setattr(api.audioprep, "probe_duration", lambda p: 30.0)
    def boom(s, d): raise RuntimeError("ffmpeg exited with code 1")
    monkeypatch.setattr(api.audioprep, "to_16k_mono", boom)
    r = client.post("/api/upload", files={"file": ("x.wav", io.BytesIO(b"RIFFxxxx"), "audio/wav")})
    assert r.status_code == 400
    assert "re-encode" in r.json()["detail"]

def test_upload_accepts_valid(monkeypatch, tmp_path):
    monkeypatch.setattr(api.audioprep, "probe_duration", lambda p: 42.0)
    written = {}
    monkeypatch.setattr(api.audioprep, "to_16k_mono", lambda s, d: written.update(dst=d))
    r = client.post("/api/upload", files={"file": ("x.wav", io.BytesIO(b"RIFFxxxx"), "audio/wav")})
    assert r.status_code == 200
    cid = r.json()["clip_id"]
    assert cid and written["dst"].endswith(f"{cid}.wav")


# ── Existing tests ────────────────────────────────────────────────────────────

def test_root_serves_frontend():
    r = client.get("/")
    assert r.status_code == 200
    assert "x-dc" in r.text or "support.js" in r.text          # the dc-app is served


def test_experiment_returns_phase4_artifact_shape():
    r = client.get("/api/experiment")
    assert r.status_code == 200
    body = r.json()
    assert "curve" in body and "spotcheck" in body and "labels" in body
    assert len(body["curve"]) == 5                             # the 5 SNR points


@pytest.mark.integration
def test_ask_returns_qaresult_shape():
    r = client.post("/api/ask", json={"question": "What did they say about transparency in a PMS?"})
    assert r.status_code == 200
    qa = r.json()
    for k in ("question", "answer", "mode", "found", "cypher", "rows",
              "provenance", "graph_node_ids", "hops"):
        assert k in qa                                        # full QAResult contract
    assert isinstance(qa["graph_node_ids"], list)


@pytest.mark.integration
def test_run_stream_emits_stage_and_done():
    run_id = client.post("/api/run").json()["run_id"]
    with client.stream("GET", f"/api/run/{run_id}/stream?replay=1") as s:
        body = "".join(chunk for chunk in s.iter_text())
    assert "event: stage" in body and "event: done" in body     # pipeline narrated + completed
    assert "event: transcript_line" in body                      # transcript replayed


# ── Task 4: clips / select_clip / live-mode dispatch ────────────────────────

def test_default_clip_is_verified_graph_hero():
    # the demo must open on the verified PMS hero (graph mode) on a fresh start,
    # never a facts clip. _ACTIVE_CLIP initializes from CFG.demo.clip.
    assert api.CFG.demo.clip == "pms"
    assert api._clip_mode("pms") == "graph"

def test_clips_lists_registry():
    r = client.get("/api/clips")
    assert r.status_code == 200
    body = r.json()
    assert "active" in body and any(c["id"] == "pms" for c in body["clips"])
    assert all({"id","label","mode"} <= set(c) for c in body["clips"])

def test_select_facts_clip_no_neo4j_write(monkeypatch):
    # selecting a non-graph clip must not call connect()/restore
    monkeypatch.setattr(api, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("neo4j touched")))
    r = client.post("/api/select_clip", json={"id": "upload_0123456789"})
    assert r.status_code == 200 and r.json()["mode"] == "live"

def test_select_clip_rejects_invalid_ids():
    # path-traversal and spoofed upload ids never reach the filesystem
    for bad in ("upload_../../etc/passwd", "../../etc/passwd",
                "pms/../call_100", "upload_xx", "a" * 65):
        r = client.post("/api/select_clip", json={"id": bad})
        assert r.status_code == 400, bad

def test_run_live_mode_drives_orchestrator(monkeypatch):
    monkeypatch.setattr(api, "_ACTIVE_CLIP", "upload_0123456789")
    monkeypatch.setattr(api, "_CLIP_MODE", {"upload_0123456789": "live"})
    def fake_live(clip):
        yield {"event": "stage", "data": {"index": 0, "label": "Speech enhancement", "sub": "x", "status": "active"}}
        yield {"event": "done", "data": {"clip": clip}}
    monkeypatch.setattr(api.pipeline, "run_live", fake_live)
    rid = client.post("/api/run").json()["run_id"]
    body = client.get(f"/api/run/{rid}/stream").text
    assert "event: stage" in body and "event: done" in body


@pytest.mark.integration
def test_alignment_gate_golden_question_node_ids_match_graph():
    # The hero payoff (ask -> nodes light up) requires QAResult.graph_node_ids to be
    # byte-identical to /api/graph node ids. Use a GOLDEN demo question that resolves
    # via Cypher with provenance (NOT a synthetic one) — a synthetic pass while a
    # golden question's ids drift would be a false green.
    graph_ids = {n["id"] for n in client.get("/api/graph").json()["nodes"]}
    qa = client.post("/api/ask",
                     json={"question": "What is the fee structure of a PMS?"}).json()
    assert qa["mode"] == "cypher", f"golden question fell back ({qa['mode']}) — cypher/node-id regression"
    assert qa["graph_node_ids"], "graph_node_ids empty — node-id extraction regressed"
    missing = [i for i in qa["graph_node_ids"] if i not in graph_ids]
    assert not missing, f"node-id format drift — not in /api/graph: {missing}"
