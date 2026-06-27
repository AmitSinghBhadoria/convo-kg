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
