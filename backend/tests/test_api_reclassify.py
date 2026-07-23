import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import FolderMapping, ScanStats
from app.services import ScanOutput
from app.storage import execute, fetch_one


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


def _seed_settings(client: TestClient, root: Path, cats: Path, tags: list[str]) -> None:
    client.put(
        "/api/settings",
        json={
            "root_repo": str(root),
            "categories_root": str(cats),
            "confidence_threshold": 0.6,
            "default_migrate_mode": "copy",
            "scan_recursive": True,
            "experimental_media_enabled": False,
            "selected_tags": tags,
            "max_inference_workers": 1,
            "inference_batch_size": 1,
            "force_cpu_inference": False,
            "tagger_model": "wd_swinv2_v3",
            "wd_general_threshold": 0.35,
        },
    ).raise_for_status()


def _patch_inference(monkeypatch, image_paths: list[Path], score_fn) -> None:
    def fake_scan_images(_root, **kwargs):
        return ScanOutput(
            image_paths=image_paths,
            stats=ScanStats(
                total_files=len(image_paths),
                eligible_images=len(image_paths),
                ignored_unsupported=0,
                ignored_gif=0,
                failed_to_read=0,
            ),
        )

    monkeypatch.setattr("app.api.scan_images", fake_scan_images)
    monkeypatch.setattr("app.api.extract_scores", score_fn)
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl", "loli", "solo"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda _root, _tags, _selected: [
            FolderMapping(
                folder_name=name,
                normalized_name=name,
                matched_tag=name,
                matched=True,
            )
            for name in (_selected or ["1girl"])
        ],
    )


def test_reclassify_promotes_needs_review_item(monkeypatch, tmp_path: Path):
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    weak = root / "weak.jpg"
    weak.write_text("fake", encoding="utf-8")

    scores = {"1girl": 0.4}

    def score_fn(*_a, **_k):
        return dict(scores)

    _patch_inference(monkeypatch, [weak], score_fn)

    with TestClient(app) as client:
        _seed_settings(client, root, cats, ["1girl"])
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

        items = client.get(f"/api/runs/{run_id}/items").json()
        assert len(items) == 1
        assert items[0]["needs_review"] is True
        assert items[0]["status"] == "proposed"
        item_id = items[0]["id"]

        scores["1girl"] = 0.95
        re_resp = client.post(
            f"/api/runs/{run_id}/reclassify",
            json={"tagger_model": "wd_eva02_large"},
        )
        re_resp.raise_for_status()
        body = re_resp.json()
        assert body["eligible_count"] == 1
        assert body["tagger_model"] == "wd_eva02_large"

        after = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"})
        assert after is not None
        assert after["status"] == "completed"

        updated = client.get(f"/api/runs/{run_id}/items").json()[0]
        assert updated["id"] == item_id
        assert updated["needs_review"] is False
        assert updated["status"] == "approved"
        assert updated["primary_tag"] == "1girl"
        assert updated["primary_score"] >= 0.9


def test_reclassify_skips_approved_and_rejected(monkeypatch, tmp_path: Path):
    root = tmp_path / "root2"
    cats = tmp_path / "cats2"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    a = root / "a.jpg"
    b = root / "b.jpg"
    c = root / "c.jpg"
    for path in (a, b, c):
        path.write_text("fake", encoding="utf-8")

    def score_fn(path, *_a, **_k):
        name = Path(path).name if not isinstance(path, Path) else path.name
        if name == "a.jpg":
            return {"1girl": 0.95}
        return {"1girl": 0.3}

    _patch_inference(monkeypatch, [a, b, c], score_fn)

    with TestClient(app) as client:
        _seed_settings(client, root, cats, ["1girl"])
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

        items = {row["file_path"]: row for row in client.get(f"/api/runs/{run_id}/items").json()}
        approved = next(v for k, v in items.items() if k.endswith("a.jpg"))
        needs_b = next(v for k, v in items.items() if k.endswith("b.jpg"))
        needs_c = next(v for k, v in items.items() if k.endswith("c.jpg"))
        assert approved["status"] == "approved"
        assert needs_b["needs_review"] is True

        client.patch(f"/api/items/{needs_c['id']}", json={"status": "rejected"}).raise_for_status()

        call_paths: list[str] = []

        def tracking_scores(path, *_a, **_k):
            call_paths.append(str(path))
            return {"1girl": 0.96}

        monkeypatch.setattr("app.api.extract_scores", tracking_scores)

        re_resp = client.post(
            f"/api/runs/{run_id}/reclassify",
            json={"tagger_model": "wd_eva02_large"},
        )
        re_resp.raise_for_status()
        assert re_resp.json()["eligible_count"] == 1

        after = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"})
        assert after is not None
        assert after["status"] == "completed"

        assert len(call_paths) == 1
        assert call_paths[0].endswith("b.jpg")

        refreshed = {row["id"]: row for row in client.get(f"/api/runs/{run_id}/items").json()}
        assert refreshed[approved["id"]]["status"] == "approved"
        assert refreshed[needs_c["id"]]["status"] == "rejected"
        assert refreshed[needs_b["id"]]["status"] == "approved"
        assert refreshed[needs_b["id"]]["needs_review"] is False


def test_reclassify_empty_eligible_returns_400(monkeypatch, tmp_path: Path):
    root = tmp_path / "root3"
    cats = tmp_path / "cats3"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    strong = root / "strong.jpg"
    strong.write_text("fake", encoding="utf-8")

    _patch_inference(monkeypatch, [strong], lambda *_a, **_k: {"1girl": 0.97})

    with TestClient(app) as client:
        _seed_settings(client, root, cats, ["1girl"])
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

        re_resp = client.post(
            f"/api/runs/{run_id}/reclassify",
            json={"tagger_model": "wd_eva02_large"},
        )
        assert re_resp.status_code == 400
        assert "eligible" in re_resp.json()["detail"].lower()


def test_reclassify_cancel_restores_completed(monkeypatch, tmp_path: Path):
    root = tmp_path / "root_cancel"
    cats = tmp_path / "cats_cancel"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    paths = []
    for idx in range(4):
        p = root / f"{idx}.jpg"
        p.write_text("fake", encoding="utf-8")
        paths.append(p)

    gate = {"block": False, "release": False}

    def score_fn(*_a, **_k):
        if gate["block"]:
            for _ in range(400):
                if gate["release"]:
                    break
                time.sleep(0.01)
        return {"1girl": 0.2}

    _patch_inference(monkeypatch, paths, score_fn)

    with TestClient(app) as client:
        _seed_settings(client, root, cats, ["1girl"])
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

        gate["block"] = True
        gate["release"] = False
        re_resp = client.post(
            f"/api/runs/{run_id}/reclassify",
            json={"tagger_model": "wd_eva02_large"},
        )
        re_resp.raise_for_status()
        assert re_resp.json()["status"] == "running"

        # Wait until worker has claimed running, then cancel.
        for _ in range(40):
            st = client.get(f"/api/runs/{run_id}/status").json()
            if st["status"] == "running":
                break
            time.sleep(0.05)
        cancel_resp = client.post(f"/api/runs/{run_id}/cancel")
        cancel_resp.raise_for_status()
        gate["release"] = True

        after = _wait_for_status(client, run_id, {"completed", "failed"}, timeout_s=8.0)
        assert after is not None
        assert after["status"] == "completed"
        assert after.get("cancel_requested") is False

        # Must be able to start another reclassify after cancel.
        gate["block"] = False
        re2 = client.post(
            f"/api/runs/{run_id}/reclassify",
            json={"tagger_model": "wd_eva02_large"},
        )
        assert re2.status_code == 200
        done = _wait_for_status(client, run_id, {"completed", "failed"}, timeout_s=8.0)
        assert done is not None
        assert done["status"] == "completed"


def test_reclassify_allows_cancelled_run(monkeypatch, tmp_path: Path):
    root = tmp_path / "root_cancelled"
    cats = tmp_path / "cats_cancelled"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    weak = root / "weak.jpg"
    weak.write_text("fake", encoding="utf-8")

    scores = {"1girl": 0.3}

    def score_fn(*_a, **_k):
        return dict(scores)

    _patch_inference(monkeypatch, [weak], score_fn)

    with TestClient(app) as client:
        _seed_settings(client, root, cats, ["1girl"])
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

        execute(
            "UPDATE runs SET status = 'cancelled', cancel_requested = 0 WHERE id = ?",
            (run_id,),
        )
        scores["1girl"] = 0.95
        re_resp = client.post(
            f"/api/runs/{run_id}/reclassify",
            json={"tagger_model": "wd_eva02_large"},
        )
        assert re_resp.status_code == 200
        after = _wait_for_status(client, run_id, {"completed", "failed"}, timeout_s=8.0)
        assert after is not None
        assert after["status"] == "completed"
        item = client.get(f"/api/runs/{run_id}/items").json()[0]
        assert item["status"] == "approved"
        assert item["needs_review"] is False


def test_reclassify_rejects_when_running(monkeypatch, tmp_path: Path):
    root = tmp_path / "root4"
    cats = tmp_path / "cats4"
    root.mkdir()
    cats.mkdir()
    (cats / "1girl").mkdir()
    weak = root / "weak.jpg"
    weak.write_text("fake", encoding="utf-8")

    _patch_inference(monkeypatch, [weak], lambda *_a, **_k: {"1girl": 0.3})

    with TestClient(app) as client:
        _seed_settings(client, root, cats, ["1girl"])
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

        execute(
            "UPDATE runs SET status = 'running', cancel_requested = 0 WHERE id = ?",
            (run_id,),
        )
        row = fetch_one("SELECT status FROM runs WHERE id = ?", (run_id,))
        assert row["status"] == "running"

        re_resp = client.post(
            f"/api/runs/{run_id}/reclassify",
            json={"tagger_model": "ml_danbooru"},
        )
        assert re_resp.status_code == 400
        assert "running" in re_resp.json()["detail"].lower()
