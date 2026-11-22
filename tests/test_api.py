import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

SAMPLE = json.loads((Path(__file__).resolve().parent.parent / "scripts" / "sample_request.json").read_text())


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained["model_dir"] / "model.joblib"))
    monkeypatch.setenv("METADATA_PATH", str(trained["model_dir"] / "metadata.json"))
    import api.main
    importlib.reload(api.main)  # pick up env vars
    with TestClient(api.main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["cancel_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}


def test_predict_partial_payload_uses_defaults(client):
    r = client.post("/predict", json={"lead_time": 10})
    assert r.status_code == 200


def test_predict_rejects_bad_input(client):
    assert client.post("/predict", json={"lead_time": -5}).status_code == 422
    assert client.post("/predict", json={"not_a_field": 1}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"bookings": [SAMPLE, {"lead_time": 5}]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2
