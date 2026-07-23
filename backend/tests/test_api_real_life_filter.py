"""Real-life filter run mode: keep only confident real_life hits."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import FolderMapping, ScanStats
from app.services import ScanOutput


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


def test_real_life_filter_keeps_only_hits(monkeypatch, tmp_path: Path):
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    photo = root / "photo.jpg"
    anime = root / "anime.jpg"
    gif = root / "clip.gif"
    photo.write_text("fake", encoding="utf-8")
    anime.write_text("fake", encoding="utf-8")
    gif.write_text("fake", encoding="utf-8")

    def fake_scan_images(_root, **kwargs):
        assert kwargs.get("experimental_media_enabled") is True
        return ScanOutput(
            image_paths=[photo, anime, gif],
            stats=ScanStats(
                total_files=3,
                eligible_images=3,
                ignored_unsupported=0,
                ignored_gif=0,
                failed_to_read=0,
            ),
        )

    def fake_scores(path, *_a, **_k):
        name = Path(path).name
        if name == "photo.jpg":
            return {"photorealistic": 0.92, "realistic": 0.88, "1girl": 0.4}
        if name == "clip.gif":
            return {"photorealistic": 0.85, "realistic": 0.8}
        return {"1girl": 0.95, "solo": 0.9, "anime_style": 0.7}

    from app.style_detectors import BUCKET_ANIME, BUCKET_PHOTO, StylePrediction

    def fake_style(path, **_k):
        name = Path(path).name
        if name in {"photo.jpg", "clip.gif"}:
            return StylePrediction(
                detector_id="t",
                method="t",
                label="real",
                bucket=BUCKET_PHOTO,
                confidence=0.9,
                scores={"real": 0.9, "anime": 0.1},
                detail={},
            )
        return StylePrediction(
            detector_id="t",
            method="t",
            label="anime",
            bucket=BUCKET_ANIME,
            confidence=0.95,
            scores={"real": 0.05, "anime": 0.95},
            detail={},
        )

    monkeypatch.setattr("app.api.scan_images", fake_scan_images)
    monkeypatch.setattr("app.api.extract_scores", fake_scores)
    monkeypatch.setattr("app.api.extract_scores_with_experimental_media", fake_scores)
    monkeypatch.setattr("app.api.extract_scores_batch", lambda paths, *_a, **_k: [fake_scores(p) for p in paths])
    monkeypatch.setattr("app.api.load_known_tags", lambda _p: {"1girl", "solo"})
    monkeypatch.setattr(
        "app.api.discover_tag_folders",
        lambda _root, _tags, selected: [
            FolderMapping(
                folder_name=name,
                normalized_name=name,
                matched_tag=name,
                matched=True,
            )
            for name in (selected or ["real_life"])
        ],
    )
    monkeypatch.setattr("app.api.probe_execution_providers", lambda: {"likely_device": "cpu"})
    monkeypatch.setattr("app.style_detectors.detect_production_style", fake_style)

    with TestClient(app) as client:
        start_resp = client.post(
            "/api/runs/start",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "confidence_threshold": 0.6,
                "run_mode": "real_life_filter",
            },
        )
        start_resp.raise_for_status()
        body = start_resp.json()
        assert "real_life" in (body.get("message") or "").lower()
        run_id = body["run_id"]
        final = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"})
        assert final is not None
        assert final["status"] == "completed"
        assert final["processed_images"] == 3

        items_resp = client.get(f"/api/runs/{run_id}/items")
        items_resp.raise_for_status()
        items = items_resp.json()
        assert len(items) == 2
        paths = {Path(i["file_path"]).name for i in items}
        assert paths == {"photo.jpg", "clip.gif"}
        assert all(i["status"] == "approved" for i in items)
        assert all(i["primary_tag"] == "real_life" for i in items)
        assert all(i["needs_review"] is False for i in items)
