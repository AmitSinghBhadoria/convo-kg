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
