"""Post-approval migrate path: API + filesystem behavior."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.storage import execute, fetch_one, to_json


def _seed_completed_run(
    root: Path,
    cats: Path,
    items: list[dict],
    *,
    status: str = "completed",
) -> int:
    run_id = execute(
        """
        INSERT INTO runs (
            root_repo, categories_root, confidence_threshold, status,
            total_images, processed_images, failed_images, cancel_requested, tagger_model
        ) VALUES (?, ?, 0.6, ?, ?, ?, 0, 0, 'wd_swinv2_v3')
        """,
        (str(root), str(cats), status, len(items), len(items)),
    )
    for item in items:
        execute(
            """
            INSERT INTO items (
                run_id, file_path, relative_path, primary_tag, primary_score, secondary_json,
                full_scores_json, suggested_destination, final_tag, final_destination,
                status, needs_review, review_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                item["file_path"],
                item.get("relative_path") or Path(item["file_path"]).name,
                item.get("primary_tag"),
                item.get("primary_score"),
                to_json(item.get("secondary") or []),
                to_json(item.get("scores") or {}),
                item.get("suggested_destination"),
                item.get("final_tag", item.get("primary_tag")),
                item.get("final_destination"),
                item.get("status", "approved"),
                1 if item.get("needs_review") else 0,
                item.get("review_reason"),
            ),
        )
    return run_id


def test_migrate_copy_approved_only_and_skips_non_approved(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    a = root / "a.jpg"
    b = root / "b.jpg"
    c = root / "c.jpg"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    c.write_bytes(b"ccc")
    dest_loli = cats / "loli"
    dest_shota = cats / "shota"

    run_id = _seed_completed_run(
        root,
        cats,
        [
            {
                "file_path": str(a),
                "primary_tag": "loli",
                "primary_score": 0.9,
                "final_destination": str(dest_loli),
                "status": "approved",
            },
            {
                "file_path": str(b),
                "primary_tag": "shota",
                "primary_score": 0.85,
                "final_destination": str(dest_shota),
                "status": "proposed",
            },
            {
                "file_path": str(c),
                "primary_tag": "loli",
                "primary_score": 0.7,
                "final_destination": str(dest_loli),
                "status": "rejected",
            },
        ],
    )

    with TestClient(app) as client:
        resp = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "copy", "create_missing_folders": True},
        )
        resp.raise_for_status()
        payload = resp.json()
        assert payload["total_candidates"] == 1
        assert payload["migrated_count"] == 1
        assert payload["failed_count"] == 0
        assert (dest_loli / "a.jpg").is_file()
        assert (dest_loli / "a.jpg").read_bytes() == b"aaa"
        assert a.exists()  # copy keeps source
        assert not (dest_shota / "b.jpg").exists()
        assert not (dest_loli / "c.jpg").exists()

        item = fetch_one("SELECT status, migrated_to FROM items WHERE file_path = ?", (str(a),))
        assert item["status"] == "migrated"
        assert item["migrated_to"].endswith("a.jpg")


def test_migrate_move_removes_source(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats" / "NTR"
    root.mkdir()
    cats.mkdir(parents=True)
    src = root / "ntr.jpg"
    src.write_bytes(b"ntr-bytes")
    run_id = _seed_completed_run(
        root,
        cats.parent,
        [
            {
                "file_path": str(src),
                "primary_tag": "NTR",
                "primary_score": 0.8,
                "final_destination": str(cats),
                "status": "approved",
            }
        ],
    )

    with TestClient(app) as client:
        resp = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "move", "create_missing_folders": False},
        )
        resp.raise_for_status()
        payload = resp.json()
        assert payload["migrated_count"] == 1
        assert not src.exists()
        assert (cats / "ntr.jpg").read_bytes() == b"ntr-bytes"


def test_migrate_collision_suffix(tmp_path: Path) -> None:
    root = tmp_path / "root"
    dest = tmp_path / "cats" / "furry"
    root.mkdir()
    dest.mkdir(parents=True)
    src = root / "dup.jpg"
    src.write_bytes(b"new")
    (dest / "dup.jpg").write_bytes(b"old")
    run_id = _seed_completed_run(
        root,
        dest.parent,
        [
            {
                "file_path": str(src),
                "primary_tag": "furry",
                "final_destination": str(dest),
                "status": "approved",
            }
        ],
    )

    with TestClient(app) as client:
        resp = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "copy", "create_missing_folders": True},
        )
        resp.raise_for_status()
        assert resp.json()["migrated_count"] == 1
        assert (dest / "dup.jpg").read_bytes() == b"old"
        assert (dest / "dup_1.jpg").read_bytes() == b"new"


def test_migrate_creates_taxonomy_folder(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    src = root / "poke.webp"
    src.write_bytes(b"poke")
    dest = cats / "Pokemon"
    run_id = _seed_completed_run(
        root,
        cats,
        [
            {
                "file_path": str(src),
                "primary_tag": "Pokemon",
                "final_destination": str(dest),
                "status": "approved",
            }
        ],
    )

    with TestClient(app) as client:
        resp = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "copy", "create_missing_folders": True},
        )
        resp.raise_for_status()
        assert resp.json()["migrated_count"] == 1
        assert dest.is_dir()
        assert (dest / "poke.webp").is_file()


def test_migrate_fails_missing_source_and_missing_destination(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    missing = root / "gone.jpg"
    present = root / "ok.jpg"
    present.write_bytes(b"ok")
    run_id = _seed_completed_run(
        root,
        cats,
        [
            {
                "file_path": str(missing),
                "primary_tag": "loli",
                "final_destination": str(cats / "loli"),
                "status": "approved",
            },
            {
                "file_path": str(present),
                "primary_tag": "loli",
                "final_destination": None,
                "status": "approved",
            },
        ],
    )

    with TestClient(app) as client:
        resp = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "copy", "create_missing_folders": True},
        )
        resp.raise_for_status()
        payload = resp.json()
        assert payload["total_candidates"] == 2
        # Missing source fails; missing final_destination is repaired from
        # primary_tag via _resolve_item_assignment and can succeed.
        assert payload["failed_count"] == 1
        assert payload["migrated_count"] == 1
        errors = {r["error"] for r in payload["results"]}
        assert any("does not exist" in (e or "") for e in errors)
        assert any(r.get("success") for r in payload["results"])


def test_migrate_create_missing_folders_false_fails_when_absent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    src = root / "x.jpg"
    src.write_bytes(b"x")
    dest = cats / "incest"  # not created
    run_id = _seed_completed_run(
        root,
        cats,
        [
            {
                "file_path": str(src),
                "primary_tag": "incest",
                "final_destination": str(dest),
                "status": "approved",
            }
        ],
    )

    with TestClient(app) as client:
        resp = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "copy", "create_missing_folders": False},
        )
        resp.raise_for_status()
        payload = resp.json()
        assert payload["migrated_count"] == 0
        assert payload["failed_count"] == 1
        assert not dest.exists()
        assert "does not exist" in (payload["results"][0].get("error") or "").lower()


def test_migrate_rejects_running_run(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    run_id = _seed_completed_run(root, cats, [], status="running")
    with TestClient(app) as client:
        resp = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "copy", "create_missing_folders": True},
        )
        assert resp.status_code == 409


def test_migrate_idempotent_second_pass(tmp_path: Path) -> None:
    root = tmp_path / "root"
    dest = tmp_path / "cats" / "fellatio"
    root.mkdir()
    dest.mkdir(parents=True)
    src = root / "f.jpg"
    src.write_bytes(b"f")
    run_id = _seed_completed_run(
        root,
        dest.parent,
        [
            {
                "file_path": str(src),
                "primary_tag": "fellatio",
                "final_destination": str(dest),
                "status": "approved",
            }
        ],
    )
    with TestClient(app) as client:
        first = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "copy", "create_missing_folders": True},
        )
        first.raise_for_status()
        assert first.json()["migrated_count"] == 1
        second = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "copy", "create_missing_folders": True},
        )
        second.raise_for_status()
        assert second.json()["total_candidates"] == 0
        assert second.json()["migrated_count"] == 0


def test_approve_then_migrate_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    src = root / "review_me.jpg"
    src.write_bytes(b"rev")
    dest = cats / "nakadashi"
    run_id = _seed_completed_run(
        root,
        cats,
        [
            {
                "file_path": str(src),
                "primary_tag": "nakadashi",
                "primary_score": 0.55,
                "suggested_destination": str(dest),
                "final_tag": "nakadashi",
                "final_destination": str(dest),
                "status": "proposed",
                "needs_review": True,
                "review_reason": "Below threshold",
            }
        ],
    )
    item = fetch_one("SELECT id FROM items WHERE run_id = ?", (run_id,))
    with TestClient(app) as client:
        patch = client.patch(f"/api/items/{item['id']}", json={"status": "approved"})
        patch.raise_for_status()
        assert patch.json()["status"] == "approved"
        mig = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "copy", "create_missing_folders": True},
        )
        mig.raise_for_status()
        assert mig.json()["migrated_count"] == 1
        assert (dest / "review_me.jpg").is_file()


def test_approve_secondary_only_then_migrate(tmp_path: Path) -> None:
    """Below-threshold items clear primary_tag but keep secondary — approve must still migrate."""
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    src = root / "weak.jpg"
    src.write_bytes(b"weak")
    run_id = _seed_completed_run(
        root,
        cats,
        [
            {
                "file_path": str(src),
                "primary_tag": None,
                "primary_score": None,
                "secondary": [{"tag": "loli", "score": 0.45}],
                "suggested_destination": None,
                "final_tag": None,
                "final_destination": None,
                "status": "proposed",
                "needs_review": True,
                "review_reason": "Below threshold",
            }
        ],
    )
    item = fetch_one("SELECT id FROM items WHERE run_id = ?", (run_id,))
    with TestClient(app) as client:
        patch = client.patch(f"/api/items/{item['id']}", json={"status": "approved"})
        patch.raise_for_status()
        body = patch.json()
        assert body["status"] == "approved"
        assert body["final_tag"] == "loli"
        assert body["final_destination"].endswith("loli")
        mig = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "move", "create_missing_folders": True},
        )
        mig.raise_for_status()
        assert mig.json()["migrated_count"] == 1
        assert mig.json()["failed_count"] == 0
        assert not src.exists()
        assert (cats / "loli" / "weak.jpg").is_file()

def test_migrate_reconciles_already_at_destination(tmp_path: Path) -> None:
    """Source gone but file already in destination → mark migrated."""
    root = tmp_path / "root"
    dest = tmp_path / "cats" / "real_life"
    root.mkdir()
    dest.mkdir(parents=True)
    src = root / "photo.jpg"
    # File already at destination; source path is stale.
    (dest / "photo.jpg").write_bytes(b"already-there")
    run_id = _seed_completed_run(
        root,
        dest.parent,
        [
            {
                "file_path": str(src),
                "primary_tag": "real_life",
                "primary_score": 0.9,
                "final_destination": str(dest),
                "status": "approved",
            }
        ],
    )
    with TestClient(app) as client:
        resp = client.post(
            f"/api/runs/{run_id}/migrate",
            json={"mode": "move", "create_missing_folders": True},
        )
        resp.raise_for_status()
        payload = resp.json()
        assert payload["migrated_count"] == 1
        assert payload["failed_count"] == 0
        item = fetch_one("SELECT status, migrated_to FROM items WHERE run_id = ?", (run_id,))
        assert item["status"] == "migrated"
        assert item["migrated_to"].endswith("photo.jpg")
