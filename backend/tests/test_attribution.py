"""Attribution / licensing endpoint."""

from fastapi.testclient import TestClient

from app.main import app


def test_attribution_endpoint_lists_owner_and_models():
    client = TestClient(app)
    resp = client.get("/attribution")
    assert resp.status_code == 200
    data = resp.json()
    assert data["project"] == "THR3SHR"
    assert data["owner"] == "Dinamush"
    assert data["licenses"]["software"] == "MIT"
    assert data["licenses"]["creative_works"] == "CC-BY-4.0"
    assert data["models_not_redistributed"] is True
    settings = {m["setting"] for m in data["models"]}
    assert "wd_swinv2_v3" in settings
    assert "ml_danbooru" in settings
    assert any(m.get("credit") == "SmilingWolf" for m in data["models"])
    assert any(m.get("credit") == "deepghs" for m in data["models"])
