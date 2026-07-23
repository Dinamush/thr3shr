import time
import os
import random
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import _classify_from_scores
from app.main import app
from app.schemas import FolderMapping, ScanStats
from app.services import ScanOutput
from app.storage import execute


def test_providers_endpoint_shape():
    with TestClient(app) as client:
        resp = client.get("/api/providers")
        resp.raise_for_status()
        payload = resp.json()
        assert "available_providers" in payload
        assert "likely_device" in payload
        assert "cuda_available" in payload
        assert "forced_cpu" in payload
        assert "tagger_model" in payload
        assert "note" in payload


def test_settings_round_trip_includes_tagger_model(tmp_path: Path):
    with TestClient(app) as client:
        payload = {
            "root_repo": str(tmp_path / "root"),
            "categories_root": str(tmp_path / "cats"),
            "confidence_threshold": 0.55,
            "default_migrate_mode": "copy",
            "scan_recursive": True,
            "experimental_media_enabled": False,
            "selected_tags": [],
            "max_inference_workers": 2,
            "inference_batch_size": 1,
            "force_cpu_inference": False,
            "tagger_model": "wd_eva02_large",
            "wd_general_threshold": 0.4,
        }
        put_resp = client.put("/api/settings", json=payload)
        put_resp.raise_for_status()
        saved = put_resp.json()
        assert saved["tagger_model"] == "wd_eva02_large"
        assert saved["wd_general_threshold"] == 0.4
        get_resp = client.get("/api/settings")
        get_resp.raise_for_status()
        loaded = get_resp.json()
        assert loaded["tagger_model"] == "wd_eva02_large"
        assert loaded["wd_general_threshold"] == 0.4
        assert loaded["max_inference_workers"] == 2


def test_start_run_missing_root_returns_400(tmp_path: Path):
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    with TestClient(app) as client:
        client.put(
            "/api/settings",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "confidence_threshold": 0.6,
                "default_migrate_mode": "copy",
                "scan_recursive": True,
                "experimental_media_enabled": False,
                "selected_tags": ["1girl"],
                "max_inference_workers": 2,
                "inference_batch_size": 1,
                "force_cpu_inference": False,
                "tagger_model": "wd_swinv2_v3",
                "wd_general_threshold": 0.35,
            },
        ).raise_for_status()
        resp = client.post(
            "/api/runs/start",
            json={
                "root_repo": str(tmp_path / "missing_root"),
                "categories_root": str(cats),
                "selected_folders": ["1girl"],
            },
        )
        assert resp.status_code == 400
        assert "root_repo" in resp.json()["detail"]
        # Missing categories_root is auto-created when selected tags are provided.
        resp_ok = client.post(
            "/api/runs/start",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "selected_folders": ["1girl"],
            },
        )
        assert resp_ok.status_code == 200
        assert cats.is_dir()


def test_below_threshold_null_primary_and_global_top_tags(monkeypatch, tmp_path: Path):
    root = tmp_path / "root_thresh"
    cats = tmp_path / "cats_thresh"
    root.mkdir()
    cats.mkdir()
    (cats / "loli").mkdir()
    file_path = root / "weak.jpg"
    file_path.write_text("fake", encoding="utf-8")

    def fake_scan_images(_root, **kwargs):
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
    monkeypatch.setattr(
        "app.api.extract_scores",
        lambda *_a, **_k: {
            "1girl": 0.97,
            "solo": 0.9,
            "loli": 0.27,
        },
    )
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"loli", "1girl", "solo"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda _root, _tags, _selected: [
            FolderMapping(
                folder_name="loli", normalized_name="loli", matched_tag="loli", matched=True
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
                "selected_folders": ["loli"],
            },
        )
        start_resp.raise_for_status()
        run_id = start_resp.json()["run_id"]
        final = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"})
        assert final is not None
        assert final["status"] == "completed"
        assert final.get("tagger_model")
        items_resp = client.get(f"/api/runs/{run_id}/items")
        items_resp.raise_for_status()
        items = items_resp.json()
        assert len(items) == 1
        item = items[0]
        assert item["primary_tag"] is None
        assert item["needs_review"] is True
        assert item["secondary_suggestions"]
        assert item["secondary_suggestions"][0]["tag"] == "loli"
        tops = [t["tag"] for t in item["global_top_tags"]]
        assert tops[:2] == ["1girl", "solo"]


def test_classify_noise_floor_clears_weak_primary(tmp_path: Path):
    result = _classify_from_scores(
        tmp_path / "x.jpg",
        {"loli": 0.2, "1girl": 0.95},
        {"loli"},
        confidence_threshold=0.6,
    )
    assert result.primary_tag is None
    assert result.needs_review is True
    assert "noise floor" in (result.reason or "").lower()
    assert result.secondary[0]["tag"] == "loli"


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

    def fake_scan_images(_root, **kwargs):
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
    monkeypatch.setattr("app.api.extract_scores", lambda *_a, **_k: {"1girl": 0.91})
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

    def fake_scan_images(_root, **kwargs):
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
    monkeypatch.setattr("app.api.extract_scores", lambda *_a, **_k: {"1girl": 0.92})
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


def test_selected_tag_wins_when_global_top_not_selected(monkeypatch, tmp_path: Path):
    root = tmp_path / "root_selected"
    cats = tmp_path / "cats_selected"
    root.mkdir()
    cats.mkdir()
    (cats / "monster_girl").mkdir()
    (cats / "slime_girl").mkdir()
    file_path = root / "s.png"
    file_path.write_text("fake", encoding="utf-8")

    def fake_scan_images(_root, **kwargs):
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
    monkeypatch.setattr(
        "app.api.extract_scores",
        lambda *_a, **_k: {"1girl": 0.99, "monster_girl": 0.85, "slime_girl": 0.82},
    )
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl", "monster_girl", "slime_girl"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda _root, _tags, _selected: [
            FolderMapping(
                folder_name="monster_girl",
                normalized_name="monster_girl",
                matched_tag="monster_girl",
                matched=True,
            ),
            FolderMapping(
                folder_name="slime_girl",
                normalized_name="slime_girl",
                matched_tag="slime_girl",
                matched=True,
            ),
        ],
    )

    with TestClient(app) as client:
        start_resp = client.post(
            "/api/runs/start",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "confidence_threshold": 0.8,
                "selected_folders": ["monster_girl", "slime_girl"],
            },
        )
        start_resp.raise_for_status()
        run_id = start_resp.json()["run_id"]
        final = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"})
        assert final is not None
        assert final["status"] == "completed"
        items_resp = client.get(f"/api/runs/{run_id}/items")
        items_resp.raise_for_status()
        item = items_resp.json()[0]
        assert item["primary_tag"] == "monster_girl"
        assert item["status"] == "approved"
        assert item["needs_review"] is False


def test_item_scores_debug_endpoint(monkeypatch, tmp_path: Path):
    root = tmp_path / "root_scores"
    cats = tmp_path / "cats_scores"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    file_path = root / "z.jpg"
    file_path.write_text("fake", encoding="utf-8")

    def fake_scan_images(_root, **kwargs):
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
    monkeypatch.setattr("app.api.extract_scores", lambda *_a, **_k: {"1girl": 0.92, "solo": 0.88})
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl", "solo"})
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
        debug_resp = client.get(f"/api/items/{item_id}/scores")
        debug_resp.raise_for_status()
        payload = debug_resp.json()
        assert payload["item_id"] == item_id
        assert payload["full_scores"]["1girl"] == 0.92
        assert payload["full_scores"]["solo"] == 0.88


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

    def fake_scan_images(_root, **kwargs):
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

    def slow_scores(_p, **_kwargs):
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
        if final["status"] == "cancelled":
            items_resp = client.get(f"/api/runs/{run_id}/items")
            items_resp.raise_for_status()
            assert items_resp.json() == []
            assert final["processed_images"] == 0
            assert final["total_images"] == 0

    # Keep test environment clean of inserted rows.
    execute("DELETE FROM items WHERE run_id = ?", (run_id,))
    execute("DELETE FROM runs WHERE id = ?", (run_id,))


def test_seeded_queue_shuffle_is_deterministic(monkeypatch, tmp_path: Path):
    root = tmp_path / "root_seed"
    cats = tmp_path / "cats_seed"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    file_paths = []
    for idx in range(6):
        p = root / f"{idx}.jpg"
        p.write_text("fake", encoding="utf-8")
        file_paths.append(p)

    observed_order: list[str] = []

    def fake_scan_images(_root, **kwargs):
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

    def record_scores(path: Path, **_kwargs):
        observed_order.append(path.name)
        return {"1girl": 0.88}

    monkeypatch.setattr("app.api.scan_images", fake_scan_images)
    monkeypatch.setattr("app.api.extract_scores", record_scores)
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda _root, _tags, _selected: [
            FolderMapping(
                folder_name="1girl", normalized_name="1girl", matched_tag="1girl", matched=True
            )
        ],
    )

    prev_shuffle = os.environ.get("QUEUE_SHUFFLE_ENABLED")
    prev_seed = os.environ.get("QUEUE_SHUFFLE_SEED")
    prev_workers = os.environ.get("MAX_INFERENCE_WORKERS")
    prev_mode = os.environ.get("INFERENCE_MODE")
    os.environ["QUEUE_SHUFFLE_ENABLED"] = "true"
    os.environ["QUEUE_SHUFFLE_SEED"] = "1337"
    os.environ["MAX_INFERENCE_WORKERS"] = "1"
    os.environ["INFERENCE_MODE"] = "single"
    try:
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
            final = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"}, timeout_s=8.0)
            assert final is not None
            assert final["status"] == "completed"
            expected = [p.name for p in file_paths]
            random.Random(1337).shuffle(expected)
            assert observed_order == expected
            execute("DELETE FROM items WHERE run_id = ?", (run_id,))
            execute("DELETE FROM runs WHERE id = ?", (run_id,))
    finally:
        if prev_shuffle is None:
            os.environ.pop("QUEUE_SHUFFLE_ENABLED", None)
        else:
            os.environ["QUEUE_SHUFFLE_ENABLED"] = prev_shuffle
        if prev_seed is None:
            os.environ.pop("QUEUE_SHUFFLE_SEED", None)
        else:
            os.environ["QUEUE_SHUFFLE_SEED"] = prev_seed
        if prev_workers is None:
            os.environ.pop("MAX_INFERENCE_WORKERS", None)
        else:
            os.environ["MAX_INFERENCE_WORKERS"] = prev_workers
        if prev_mode is None:
            os.environ.pop("INFERENCE_MODE", None)
        else:
            os.environ["INFERENCE_MODE"] = prev_mode


def test_batch_mode_falls_back_to_single(monkeypatch, tmp_path: Path):
    root = tmp_path / "root_batch_fallback"
    cats = tmp_path / "cats_batch_fallback"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    file_paths = []
    for idx in range(8):
        p = root / f"{idx}.jpg"
        p.write_text("fake", encoding="utf-8")
        file_paths.append(p)

    def fake_scan_images(_root, **kwargs):
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

    def fail_batch(_paths, **_kwargs):
        raise RuntimeError("synthetic batch failure")

    monkeypatch.setattr("app.api.scan_images", fake_scan_images)
    monkeypatch.setattr("app.api.extract_scores_batch", fail_batch)
    monkeypatch.setattr("app.api.extract_scores", lambda *_a, **_k: {"1girl": 0.9})
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda _root, _tags, _selected: [
            FolderMapping(
                folder_name="1girl", normalized_name="1girl", matched_tag="1girl", matched=True
            )
        ],
    )

    prev_mode = os.environ.get("INFERENCE_MODE")
    prev_batch = os.environ.get("INFERENCE_BATCH_SIZE")
    os.environ["INFERENCE_MODE"] = "batch"
    os.environ["INFERENCE_BATCH_SIZE"] = "4"
    try:
        with TestClient(app) as client:
            settings = client.get("/api/settings").json()
            settings["inference_batch_size"] = 4
            settings["max_inference_workers"] = 2
            client.put("/api/settings", json=settings).raise_for_status()
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
            final = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"}, timeout_s=8.0)
            assert final is not None
            assert final["status"] == "completed"
            assert final["failed_images"] == 0
            assert final["processed_images"] == len(file_paths)
            assert final["inference_mode"] in {"single", "batch_fallback", "single_fallback"}
            execute("DELETE FROM items WHERE run_id = ?", (run_id,))
            execute("DELETE FROM runs WHERE id = ?", (run_id,))
    finally:
        if prev_mode is None:
            os.environ.pop("INFERENCE_MODE", None)
        else:
            os.environ["INFERENCE_MODE"] = prev_mode
        if prev_batch is None:
            os.environ.pop("INFERENCE_BATCH_SIZE", None)
        else:
            os.environ["INFERENCE_BATCH_SIZE"] = prev_batch
