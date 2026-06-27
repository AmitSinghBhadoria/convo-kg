import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


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
