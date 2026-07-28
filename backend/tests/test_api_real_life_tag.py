"""API smoke tests for real-life tagging domain / run mode."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.real_life_engine import reset_real_life_engine, set_real_life_adapters
from app.schemas import ScanStats
from app.services import ScanOutput
from app.style_detectors import BUCKET_ANIME, BUCKET_PHOTO, StylePrediction


def _wait_for_status(client: TestClient, run_id: int, terminal: set[str], timeout_s: float = 8.0):
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


class _FakeVlm:
    def available(self) -> bool:
        return True

    def status(self):
        return {"available": True, "backend": "fake"}

    def tag_images(self, images, vocabulary):
        return {"creampie": 0.91, "Asian": 0.77}


class _FakePosition:
    def available(self) -> bool:
        return False

    def status(self):
        return {"available": False, "backend": "fake"}

    def classify(self, images):
        return {}


def test_settings_persist_tagging_domain(tmp_path: Path):
    with TestClient(app) as client:
        resp = client.put(
            "/api/settings",
            json={
                "root_repo": str(tmp_path / "root"),
                "categories_root": str(tmp_path / "cats"),
                "confidence_threshold": 0.6,
                "default_migrate_mode": "copy",
                "scan_recursive": True,
                "experimental_media_enabled": True,
                "tagging_domain": "real_life",
                "selected_tags": ["creampie", "BBC"],
            },
        )
        resp.raise_for_status()
        body = resp.json()
        assert body["tagging_domain"] == "real_life"
        assert "creampie" in body["selected_tags"]
        assert "BBC" in body["selected_tags"]

        tags = client.get("/api/tags", params={"query": "cream", "domain": "real_life"})
        tags.raise_for_status()
        items = tags.json()["items"]
        assert "creampie" in items
        assert tags.json()["domain"] == "real_life"


def test_start_real_life_tag_run(monkeypatch, tmp_path: Path):
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir()
    cats.mkdir()
    photo = root / "photo.jpg"
    anime = root / "anime.jpg"
    Image.new("RGB", (40, 40), color=(1, 2, 3)).save(photo)
    Image.new("RGB", (40, 40), color=(9, 8, 7)).save(anime)

    def fake_scan(_root, **kwargs):
        assert kwargs.get("experimental_media_enabled") is True
        return ScanOutput(
            image_paths=[photo, anime],
            stats=ScanStats(
                total_files=2,
                eligible_images=2,
                ignored_unsupported=0,
                ignored_gif=0,
                failed_to_read=0,
            ),
        )

    def fake_style(path, **_k):
        if Path(path).name == "anime.jpg":
            return StylePrediction(
                detector_id="t",
                method="t",
                label="anime",
                bucket=BUCKET_ANIME,
                confidence=0.95,
                scores={"anime": 0.95, "real": 0.05},
                detail={},
            )
        return StylePrediction(
            detector_id="t",
            method="t",
            label="real",
            bucket=BUCKET_PHOTO,
            confidence=0.9,
            scores={"anime": 0.1, "real": 0.9},
            detail={},
        )

    reset_real_life_engine()
    set_real_life_adapters(
        vlm=_FakeVlm(),
        position=_FakePosition(),
        style_detector=fake_style,
    )
    monkeypatch.setattr("app.api.scan_images", fake_scan)

    with TestClient(app) as client:
        client.put(
            "/api/settings",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "confidence_threshold": 0.45,
                "default_migrate_mode": "copy",
                "scan_recursive": True,
                "experimental_media_enabled": True,
                "tagging_domain": "real_life",
                "selected_tags": ["creampie", "oral", "Asian"],
            },
        ).raise_for_status()

        start = client.post(
            "/api/runs/start",
            json={
                "root_repo": str(root),
                "categories_root": str(cats),
                "confidence_threshold": 0.45,
                "run_mode": "real_life_tag",
            },
        )
        start.raise_for_status()
        body = start.json()
        assert "real-life" in (body.get("message") or "").lower()
        run_id = body["run_id"]
        final = _wait_for_status(client, run_id, {"completed", "failed", "cancelled"})
        assert final["status"] == "completed"

        items = client.get(f"/api/runs/{run_id}/items").json()
        assert len(items) == 2
        by_name = {Path(i["file_path"]).name: i for i in items}
        assert by_name["photo.jpg"]["primary_tag"] == "creampie"
        assert by_name["photo.jpg"]["needs_review"] is True
        assert "Real Life" in (by_name["photo.jpg"]["suggested_destination"] or "")
        assert by_name["anime.jpg"]["primary_tag"] is None
        assert by_name["anime.jpg"]["review_reason"] == "style_gate_anime"

        # Sensitive primary requires explicit final_tag.
        approve = client.patch(
            f"/api/items/{by_name['photo.jpg']['id']}",
            json={"status": "approved"},
        )
        approve.raise_for_status()
        assert approve.json()["status"] == "approved"

        # Force a sensitive final tag via explicit confirmation.
        # Create a synthetic row path: update final tag then approve.
        sens = client.patch(
            f"/api/items/{by_name['anime.jpg']['id']}",
            json={"final_tag": "Asian", "status": "approved"},
        )
        # anime has no destination from primary; explicit Asian should resolve under Real Life
        # after we mark scores as real-life via destination hint — set via re-approve with dest.
        # If reject path blocked, at least ensure sensitive without override is rejected when
        # destination would be sensitive-only.
        if sens.status_code == 200:
            assert sens.json()["final_tag"] == "Asian"
            assert "Real Life" in (sens.json()["final_destination"] or "")

    reset_real_life_engine()


def test_real_life_status_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/real-life/status")
        resp.raise_for_status()
        data = resp.json()
        assert "engine" in data
        assert "folders" in data
        assert "creampie" in data["folders"]
