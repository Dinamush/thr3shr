"""Review table data and item-state guards: global tops, previews, migrated lock."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.storage import execute, fetch_all, fetch_one, to_json


def _seed_run(root: Path, cats: Path, items: list[dict]) -> int:
    run_id = execute(
        """
        INSERT INTO runs (
            root_repo, categories_root, confidence_threshold, status,
            total_images, processed_images, failed_images, cancel_requested, tagger_model
        ) VALUES (?, ?, 0.6, 'completed', ?, ?, 0, 0, 'wd_swinv2_v3')
        """,
        (str(root), str(cats), len(items), len(items)),
    )
    for item in items:
        execute(
            """
            INSERT INTO items (
                run_id, file_path, relative_path, primary_tag, primary_score, secondary_json,
                full_scores_json, suggested_destination, final_tag, final_destination,
                status, needs_review, review_reason, migrated_to
            ) VALUES (?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, 0, NULL, ?)
            """,
            (
                run_id,
                item["file_path"],
                Path(item["file_path"]).name,
                item.get("primary_tag"),
                item.get("primary_score"),
                to_json(item.get("scores") or {}),
                item.get("final_destination"),
                item.get("primary_tag"),
                item.get("final_destination"),
                item.get("status", "approved"),
                item.get("migrated_to"),
            ),
        )
    return run_id


def test_items_list_backfills_global_tops_for_legacy_rows(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    image = root / "a.jpg"
    image.write_bytes(b"x")
    run_id = _seed_run(
        root,
        cats,
        [
            {
                "file_path": str(image),
                "primary_tag": "scenery",
                "primary_score": 0.7,
                "final_destination": str(cats / "scenery"),
                "scores": {"sky": 0.95, "no_humans": 0.9, "scenery": 0.7, "cloud": 0.4},
            }
        ],
    )

    with TestClient(app) as client:
        first = client.get(f"/api/runs/{run_id}/items")
        first.raise_for_status()
        tops = [t["tag"] for t in first.json()[0]["global_top_tags"]]
        assert tops == ["sky", "no_humans", "scenery", "cloud"]

        # Backfilled once; later list reads come from the side table.
        stored = fetch_all("SELECT item_id FROM item_global_tops WHERE run_id = ?", (run_id,))
        assert len(stored) == 1
        again = client.get(f"/api/runs/{run_id}/items")
        assert [t["tag"] for t in again.json()[0]["global_top_tags"]] == tops


def test_items_index_on_run_id_exists() -> None:
    names = {row["name"] for row in fetch_all("PRAGMA index_list(items)")}
    assert "idx_items_run_id" in names


def test_migrated_item_cannot_be_reapproved_or_retagged(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    (cats / "scenery").mkdir(parents=True)
    source = root / "a.jpg"
    source.write_bytes(b"aaa")
    copied = cats / "scenery" / "a.jpg"
    copied.write_bytes(b"aaa")
    run_id = _seed_run(
        root,
        cats,
        [
            {
                "file_path": str(source),
                "primary_tag": "scenery",
                "primary_score": 0.9,
                "final_destination": str(cats / "scenery"),
                "status": "migrated",
                "migrated_to": str(copied),
            }
        ],
    )
    item_id = fetch_one("SELECT id FROM items WHERE run_id = ?", (run_id,))["id"]

    with TestClient(app) as client:
        approve = client.patch(f"/api/items/{item_id}", json={"status": "approved"})
        assert approve.status_code == 409
        retag = client.patch(f"/api/items/{item_id}", json={"final_tag": "sky"})
        assert retag.status_code == 409
        batch = client.post(
            f"/api/runs/{run_id}/batch", json={"item_ids": [item_id], "status": "approved"}
        )
        assert batch.json() == {"updated": 0, "skipped": 1}

        # A second copy-mode migrate must not duplicate the already-copied file.
        migrate = client.post(f"/api/runs/{run_id}/migrate", json={"mode": "copy"})
        migrate.raise_for_status()
        assert migrate.json()["total_candidates"] == 0

    assert sorted(p.name for p in (cats / "scenery").iterdir()) == ["a.jpg"]
    row = fetch_one("SELECT status, final_tag FROM items WHERE id = ?", (item_id,))
    assert row == {"status": "migrated", "final_tag": "scenery"}


def test_client_cannot_mark_item_migrated_without_migrating(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "a.jpg"
    source.write_bytes(b"aaa")
    run_id = _seed_run(
        root,
        tmp_path / "cats",
        [{"file_path": str(source), "primary_tag": "scenery", "status": "approved"}],
    )
    item_id = fetch_one("SELECT id FROM items WHERE run_id = ?", (run_id,))["id"]

    with TestClient(app) as client:
        single = client.patch(f"/api/items/{item_id}", json={"status": "migrated"})
        assert single.status_code == 400
        batch = client.post(
            f"/api/runs/{run_id}/batch", json={"item_ids": [item_id], "status": "migrated"}
        )
        assert batch.status_code == 400

    assert fetch_one("SELECT status FROM items WHERE id = ?", (item_id,))["status"] == "approved"


def test_preview_serves_jpeg_still_for_extensionless_and_tiff_images(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    extensionless = root / "no_extension"
    Image.new("RGB", (32, 24), (10, 120, 200)).save(extensionless, format="PNG")
    tiff = root / "scan.tiff"
    Image.new("RGB", (32, 24), (200, 120, 10)).save(tiff, format="TIFF")
    not_image = root / "notes"
    not_image.write_text("plain text")
    run_id = _seed_run(
        root,
        tmp_path / "cats",
        [{"file_path": str(p), "status": "proposed"} for p in (extensionless, tiff, not_image)],
    )
    ids = [row["id"] for row in fetch_all("SELECT id FROM items WHERE run_id = ? ORDER BY id", (run_id,))]

    with TestClient(app) as client:
        for item_id in ids[:2]:
            resp = client.get(f"/api/items/{item_id}/preview")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/jpeg"
            assert resp.content[:2] == b"\xff\xd8"
        assert client.get(f"/api/items/{ids[2]}/preview").status_code == 415


def test_media_still_preview_handles_non_ascii_file_names(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    gif = root / "夕焼け.gif"
    frames = [Image.new("RGB", (32, 32), (i * 60, 90, 140)) for i in range(3)]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=80, loop=0)
    run_id = _seed_run(root, tmp_path / "cats", [{"file_path": str(gif), "status": "proposed"}])
    item_id = fetch_one("SELECT id FROM items WHERE run_id = ?", (run_id,))["id"]

    with TestClient(app) as client:
        resp = client.get(f"/api/items/{item_id}/preview")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert "utf-8''" in resp.headers["content-disposition"]


def test_settings_reject_out_of_range_confidence_threshold() -> None:
    with TestClient(app) as client:
        current = client.get("/api/settings").json()
        resp = client.put("/api/settings", json={**current, "confidence_threshold": 6})
        assert resp.status_code == 422
        assert client.get("/api/settings").json()["confidence_threshold"] == current[
            "confidence_threshold"
        ]


def test_settings_keep_zero_wd_general_threshold() -> None:
    with TestClient(app) as client:
        current = client.get("/api/settings").json()
        saved = client.put("/api/settings", json={**current, "wd_general_threshold": 0.0})
        saved.raise_for_status()
        assert client.get("/api/settings").json()["wd_general_threshold"] == 0.0
