"""Reject must leave files in place and not invent migrate destinations."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.storage import execute, fetch_one, to_json


def _seed(root: Path, cats: Path) -> tuple[int, int, Path]:
    src = root / "skip_me.jpg"
    src.write_bytes(b"skip")
    run_id = execute(
        """
        INSERT INTO runs (
            root_repo, categories_root, confidence_threshold, status,
            total_images, processed_images, failed_images, cancel_requested, tagger_model
        ) VALUES (?, ?, 0.6, 'completed', 1, 1, 0, 0, 'wd_swinv2_v3')
        """,
        (str(root), str(cats)),
    )
    item_id = execute(
        """
        INSERT INTO items (
            run_id, file_path, relative_path, primary_tag, primary_score, secondary_json,
            full_scores_json, suggested_destination, final_tag, final_destination,
            status, needs_review, review_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            str(src),
            src.name,
            None,
            None,
            to_json([{"tag": "loli", "score": 0.4}]),
            to_json({}),
            None,
            None,
            None,
            "proposed",
            1,
            "Below threshold",
        ),
    )
    return run_id, item_id, src


def test_reject_clears_destination_and_keeps_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    _run_id, item_id, src = _seed(root, cats)

    with TestClient(app) as client:
        resp = client.patch(f"/api/items/{item_id}", json={"status": "rejected"})
        resp.raise_for_status()
        body = resp.json()
        assert body["status"] == "rejected"
        assert body["needs_review"] is False
        assert body["final_tag"] is None
        assert body["final_destination"] is None
        assert "Rejected" in (body["review_reason"] or "")
        assert src.exists()

        row = fetch_one("SELECT status, final_destination FROM items WHERE id = ?", (item_id,))
        assert row["status"] == "rejected"
        assert row["final_destination"] is None

        mig = client.post(
            f"/api/runs/{_run_id}/migrate",
            json={"mode": "move", "create_missing_folders": True},
        )
        mig.raise_for_status()
        assert mig.json()["total_candidates"] == 0
        assert src.exists()


def test_batch_reject_does_not_assign_folder(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    run_id, item_id, src = _seed(root, cats)

    with TestClient(app) as client:
        resp = client.post(
            f"/api/runs/{run_id}/batch",
            json={"item_ids": [item_id], "status": "rejected"},
        )
        resp.raise_for_status()
        assert resp.json()["updated"] == 1
        row = fetch_one(
            "SELECT status, final_tag, final_destination FROM items WHERE id = ?",
            (item_id,),
        )
        assert row["status"] == "rejected"
        assert row["final_tag"] is None
        assert row["final_destination"] is None
        assert src.exists()
