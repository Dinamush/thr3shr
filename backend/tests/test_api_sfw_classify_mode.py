"""SFW classify mode parks NSFW tags and forces safe destinations."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def _base_settings(tmp_path: Path, **overrides):
    root = tmp_path / "root"
    cats = tmp_path / "cats"
    root.mkdir(exist_ok=True)
    cats.mkdir(exist_ok=True)
    payload = {
        "root_repo": str(root),
        "categories_root": str(cats),
        "confidence_threshold": 0.45,
        "default_migrate_mode": "copy",
        "scan_recursive": True,
        "experimental_media_enabled": False,
        "experimental_style_detector_enabled": False,
        "hybrid_ml_on_review": True,
        "tagging_domain": "drawn",
        "selected_tags": ["loli", "sex", "group_sex"],
        "selected_tags_nsfw": [],
        "sfw_classify_mode": False,
        "max_inference_workers": 2,
        "inference_batch_size": 4,
        "force_cpu_inference": False,
        "tagger_model": "wd_swinv2_v3",
        "wd_general_threshold": 0.35,
    }
    payload.update(overrides)
    return payload


def test_sfw_classify_mode_parks_nsfw_and_forces_safe_folders(tmp_path: Path):
    with TestClient(app) as client:
        resp = client.put("/api/settings", json=_base_settings(tmp_path))
        assert resp.status_code == 200
        assert "loli" in resp.json()["selected_tags"]

        resp = client.put(
            "/api/settings",
            json=_base_settings(
                tmp_path,
                sfw_classify_mode=True,
                selected_tags=["SFW", "scenery"],
                selected_tags_nsfw=["loli", "sex", "group_sex", "milf"],
            ),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sfw_classify_mode"] is True
        assert body["selected_tags"] == ["SFW", "scenery"]
        assert "comic" not in body["selected_tags"]
        assert body["selected_tags_nsfw"] == ["loli", "sex", "group_sex", "milf"]

        resp = client.put(
            "/api/settings",
            json=_base_settings(
                tmp_path,
                sfw_classify_mode=True,
                selected_tags=["loli", "fellatio", "SFW", "comic"],
                selected_tags_nsfw=["loli", "sex"],
            ),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["selected_tags"] == ["SFW", "scenery"]
        assert "comic" not in body["selected_tags"]
        assert "loli" not in body["selected_tags"]

        resp = client.put(
            "/api/settings",
            json=_base_settings(
                tmp_path,
                sfw_classify_mode=False,
                selected_tags=["loli", "sex", "group_sex", "milf"],
                selected_tags_nsfw=["loli", "sex", "group_sex", "milf"],
            ),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sfw_classify_mode"] is False
        assert "loli" in body["selected_tags"]
        assert "milf" in body["selected_tags"]
