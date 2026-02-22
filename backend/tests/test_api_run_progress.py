import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import FolderMapping, ScanStats
from app.services import ScanOutput
from app.storage import execute


def _wait_for_status(client: TestClient, run_id: int, terminal: set[str], timeout_s: float = 5.0):
    start = time.time()
    last = None
    while time.time() - start < timeout_s:
        resp = client.get(f"/api/runs/{run_id}/status")
        resp.raise_for_status()
        data = resp.json()
        last = data
        if data["status"] in terminal:
            return data
        time.sleep(0.05)
    return last


def test_run_progress_reaches_completed(monkeypatch, tmp_path: Path):
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    file_path = root / "a.jpg"
    file_path.write_text("fake", encoding="utf-8")

    def fake_scan_images(_root):
        return ScanOutput(
            image_paths=[file_path],
            stats=ScanStats(
                total_files=1,
                eligible_images=1,
                ignored_unsupported=0,
                ignored_gif=0,
                failed_to_read=0,
            ),
        )

    monkeypatch.setattr("app.api.scan_images", fake_scan_images)
    monkeypatch.setattr("app.api.extract_scores", lambda _p: {"1girl": 0.91})
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda _root, _tags, _selected: [
            FolderMapping(
                folder_name="1girl", normalized_name="1girl", matched_tag="1girl", matched=True
            )
        ],
    )

    with TestClient(app) as client:
        start_resp = client.post(
            "/api/runs/start",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "confidence_threshold": 0.6,
                "selected_folders": ["1girl"],
            },
        )
        start_resp.raise_for_status()
        run_id = start_resp.json()["run_id"]
        final = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"})
        assert final is not None
        assert final["status"] == "completed"
        assert final["processed_images"] >= 1
        assert final["has_items"] is True
        items_resp = client.get(f"/api/runs/{run_id}/items")
        items_resp.raise_for_status()
        items = items_resp.json()
        assert len(items) == 1
        assert items[0]["needs_review"] is False
        assert items[0]["status"] == "approved"


def test_item_preview_returns_image(monkeypatch, tmp_path: Path):
    root = tmp_path / "root_preview"
    cats = tmp_path / "cats_preview"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    file_path = root / "preview.jpg"
    file_path.write_bytes(b"fake-image-bytes")

    def fake_scan_images(_root):
        return ScanOutput(
            image_paths=[file_path],
            stats=ScanStats(
                total_files=1,
                eligible_images=1,
                ignored_unsupported=0,
                ignored_gif=0,
                failed_to_read=0,
            ),
        )

    monkeypatch.setattr("app.api.scan_images", fake_scan_images)
    monkeypatch.setattr("app.api.extract_scores", lambda _p: {"1girl": 0.92})
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda _root, _tags, _selected: [
            FolderMapping(
                folder_name="1girl", normalized_name="1girl", matched_tag="1girl", matched=True
            )
        ],
    )

    with TestClient(app) as client:
        start_resp = client.post(
            "/api/runs/start",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "confidence_threshold": 0.6,
                "selected_folders": ["1girl"],
            },
        )
        start_resp.raise_for_status()
        run_id = start_resp.json()["run_id"]
        final = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"})
        assert final is not None
        assert final["status"] == "completed"

        items_resp = client.get(f"/api/runs/{run_id}/items")
        items_resp.raise_for_status()
        item_id = items_resp.json()[0]["id"]
        preview_resp = client.get(f"/api/items/{item_id}/preview")
        preview_resp.raise_for_status()
        assert preview_resp.headers["content-type"].startswith("image/jpeg")
        assert preview_resp.content == b"fake-image-bytes"


def test_run_cancel_sets_cancelled(monkeypatch, tmp_path: Path):
    root = tmp_path / "root_cancel"
    cats = tmp_path / "cats_cancel"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    file_paths = []
    for idx in range(20):
        p = root / f"{idx}.jpg"
        p.write_text("fake", encoding="utf-8")
        file_paths.append(p)

    def fake_scan_images(_root):
        return ScanOutput(
            image_paths=file_paths,
            stats=ScanStats(
                total_files=len(file_paths),
                eligible_images=len(file_paths),
                ignored_unsupported=0,
                ignored_gif=0,
                failed_to_read=0,
            ),
        )

    def slow_scores(_p):
        time.sleep(0.03)
        return {"1girl": 0.88}

    monkeypatch.setattr("app.api.scan_images", fake_scan_images)
    monkeypatch.setattr("app.api.extract_scores", slow_scores)
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda _root, _tags, _selected: [
            FolderMapping(
                folder_name="1girl", normalized_name="1girl", matched_tag="1girl", matched=True
            )
        ],
    )

    with TestClient(app) as client:
        start_resp = client.post(
            "/api/runs/start",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "confidence_threshold": 0.6,
                "selected_folders": ["1girl"],
            },
        )
        start_resp.raise_for_status()
        run_id = start_resp.json()["run_id"]
        cancel_resp = client.post(f"/api/runs/{run_id}/cancel")
        cancel_resp.raise_for_status()
        final = _wait_for_status(client, run_id, {"cancelled", "completed", "failed"}, timeout_s=6.0)
        assert final is not None
        assert final["cancel_requested"] is True
        assert final["status"] in {"cancelled", "completed"}

    # Keep test environment clean of inserted rows.
    execute("DELETE FROM items WHERE run_id = ?", (run_id,))
    execute("DELETE FROM runs WHERE id = ?", (run_id,))
