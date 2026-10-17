"""List and grid thumbnails: GET /api/items/{id}/preview?thumb=1."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.storage import execute


@pytest.fixture(autouse=True)
def _tmp_preview_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "preview_cache"
    cache.mkdir()
    monkeypatch.setattr("app.services._preview_cache_dir", lambda: cache)
    return cache


def _item_for(path: Path) -> int:
    run_id = execute(
        """
        INSERT INTO runs (root_repo, categories_root, confidence_threshold, status, total_images, processed_images)
        VALUES (?, ?, 0.6, 'completed', 1, 1)
        """,
        (str(path.parent), str(path.parent / "cats")),
    )
    return execute(
        """
        INSERT INTO items (run_id, file_path, relative_path, secondary_json, status, needs_review)
        VALUES (?, ?, ?, '[]', 'proposed', 1)
        """,
        (run_id, str(path), path.name),
    )


def _thumb(client: TestClient, item_id: int) -> Image.Image:
    resp = client.get(f"/api/items/{item_id}/preview", params={"thumb": 1})
    resp.raise_for_status()
    assert resp.headers["content-type"] == "image/webp"
    assert "inline" in resp.headers["content-disposition"]
    return Image.open(io.BytesIO(resp.content))


def test_thumb_shrinks_large_jpeg_to_short_edge(tmp_path: Path) -> None:
    src = tmp_path / "big.jpg"
    Image.new("RGB", (3000, 4000), (200, 40, 40)).save(src, quality=90)
    with TestClient(app) as client:
        thumb = _thumb(client, _item_for(src))
    assert thumb.size == (384, 512)
    assert thumb.mode == "RGB"
    r, g, b = thumb.getpixel((192, 256))
    assert abs(r - 200) < 8 and g < 60 and b < 60


def test_thumb_keeps_small_images_and_transparency(tmp_path: Path) -> None:
    src = tmp_path / "icon.png"
    Image.new("RGBA", (120, 80), (0, 0, 0, 0)).save(src)
    with TestClient(app) as client:
        thumb = _thumb(client, _item_for(src))
    assert thumb.size == (120, 80)
    assert thumb.mode == "RGBA"
    assert thumb.getpixel((10, 10))[3] == 0


def test_thumb_applies_exif_orientation(tmp_path: Path) -> None:
    src = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    Image.new("RGB", (800, 400), (10, 120, 10)).save(src, exif=exif)
    with TestClient(app) as client:
        thumb = _thumb(client, _item_for(src))
    assert thumb.size[0] < thumb.size[1]


def test_thumb_is_cached_and_original_still_served(tmp_path: Path, _tmp_preview_cache: Path) -> None:
    src = tmp_path / "photo.jpg"
    Image.new("RGB", (1200, 900), (30, 30, 200)).save(src)
    with TestClient(app) as client:
        item_id = _item_for(src)
        first = client.get(f"/api/items/{item_id}/preview", params={"thumb": 1}).content
        assert len(list(_tmp_preview_cache.iterdir())) == 1
        assert client.get(f"/api/items/{item_id}/preview", params={"thumb": 1}).content == first
        original = client.get(f"/api/items/{item_id}/preview")
    assert original.headers["content-type"] == "image/jpeg"
    assert original.content == src.read_bytes()


def test_thumb_falls_back_to_original_when_undecodable(tmp_path: Path) -> None:
    src = tmp_path / "broken.png"
    src.write_bytes(b"not really a png")
    with TestClient(app) as client:
        resp = client.get(f"/api/items/{_item_for(src)}/preview", params={"thumb": 1})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == b"not really a png"
