"""Only one run may be pending or running at a time."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import FolderMapping
from app.storage import execute, fetch_one


def _seed_active_run() -> int:
    return execute(
        """
        INSERT INTO runs (
            root_repo, categories_root, confidence_threshold,
            status, total_images, processed_images, failed_images
        ) VALUES ('/inbox/a', '/cats', 0.45, 'running', 10, 1, 0)
        """
    )


def _start_body(monkeypatch, tmp_path: Path) -> dict:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda *_a, **_k: [
            FolderMapping(
                folder_name="1girl", normalized_name="1girl", matched_tag="1girl", matched=True
            )
        ],
    )
    monkeypatch.setattr(
        "app.api._execute_run",
        lambda run_id, **_k: execute(
            "UPDATE runs SET status = 'completed' WHERE id = ?", (run_id,)
        ),
    )
    return {
        "root_repo": str(root),
        "categories_root": str(cats),
        "confidence_threshold": 0.6,
        "run_mode": "classify",
    }


def test_start_is_refused_while_another_run_is_active(monkeypatch, tmp_path: Path):
    body = _start_body(monkeypatch, tmp_path)

    with TestClient(app) as client:
        active_id = _seed_active_run()
        refused = client.post("/api/runs/start", json=body)
        assert refused.status_code == 409
        detail = refused.json()["detail"]
        assert f"run {active_id}" in detail
        assert "cancel" in detail.lower()
        assert [r["run_id"] for r in client.get("/api/runs").json()] == [active_id]

        execute("UPDATE runs SET status = 'completed' WHERE id = ?", (active_id,))
        allowed = client.post("/api/runs/start", json=body)
        assert allowed.status_code == 200
        new_id = allowed.json()["run_id"]
        assert new_id != active_id


def test_startup_fails_runs_orphaned_by_a_previous_backend(monkeypatch, tmp_path: Path):
    body = _start_body(monkeypatch, tmp_path)
    orphan_id = _seed_active_run()

    with TestClient(app) as client:
        orphan = fetch_one(
            "SELECT status, finished_at, last_error FROM runs WHERE id = ?", (orphan_id,)
        )
        assert orphan["status"] == "failed"
        assert orphan["finished_at"]
        assert "interrupted" in orphan["last_error"].lower()

        allowed = client.post("/api/runs/start", json=body)
        assert allowed.status_code == 200
        assert allowed.json()["run_id"] != orphan_id
