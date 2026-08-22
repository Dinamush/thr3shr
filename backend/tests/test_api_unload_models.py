"""Unload GPU-resident tagger sessions via POST /api/models/unload."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.storage import execute


def test_unload_models_clears_loaded_list():
    with TestClient(app) as client:
        resp = client.post("/api/models/unload")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["unloaded"], list)
        assert "message" in body
        providers = client.get("/api/providers").json()
        assert providers["loaded_models"] == []


def test_unload_models_rejects_while_run_is_active():
    execute(
        """
        INSERT INTO runs (
            created_at, root_repo, categories_root, confidence_threshold,
            status, total_images, processed_images, failed_images
        ) VALUES ('2026-08-16T00:00:00+00:00', 'x', 'y', 0.45, 'running', 10, 1, 0)
        """
    )
    with TestClient(app) as client:
        resp = client.post("/api/models/unload")
        assert resp.status_code == 409
        assert "running" in resp.json()["detail"].lower()
