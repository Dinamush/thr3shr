"""FILTER_MEDIA_SAMPLE_MAX must only apply to real_life filter / hybrid WD path."""

from __future__ import annotations

from pathlib import Path

from app.api import _extract_scores_for_hybrid, _infer_one_image
from app.style_detectors import FILTER_MEDIA_SAMPLE_MAX


def test_normal_infer_does_not_cap_media_sample_count(monkeypatch, tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    seen: dict[str, object] = {}

    def fake_media_extract(path, enabled, **kwargs):
        seen["kwargs"] = dict(kwargs)
        return {"1girl": 0.9}

    monkeypatch.setattr("app.api.is_experimental_media", lambda _p: True)
    monkeypatch.setattr("app.api.extract_scores_with_experimental_media", fake_media_extract)
    monkeypatch.setattr(
        "app.api._classify_from_scores",
        lambda *a, **k: type("R", (), {"image_path": media})(),
    )

    _infer_one_image(
        media,
        {"loli"},
        0.6,
        experimental_media_enabled=True,
        hybrid_real_life=False,
    )
    assert "sample_count" not in seen["kwargs"]


def test_hybrid_extract_caps_media_sample_count(monkeypatch, tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    seen: dict[str, object] = {}

    def fake_media_extract(path, enabled, **kwargs):
        seen["sample_count"] = kwargs.get("sample_count", "MISSING")
        return {"1girl": 0.9}

    monkeypatch.setattr("app.api.is_experimental_media", lambda _p: True)
    monkeypatch.setattr("app.api.extract_scores_with_experimental_media", fake_media_extract)

    scores = _extract_scores_for_hybrid(
        media,
        experimental_media_enabled=True,
        tagger_model="wd_swinv2_v3",
        wd_general_threshold=0.35,
    )
    assert scores["1girl"] == 0.9
    assert seen["sample_count"] == FILTER_MEDIA_SAMPLE_MAX
