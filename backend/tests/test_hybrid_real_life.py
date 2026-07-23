"""Hybrid WD realism + style-detector blending for real_life filter."""

from __future__ import annotations

from pathlib import Path

from app.api import (
    _ImageInferenceResult,
    _apply_hybrid_real_life_filter,
    _infer_hybrid_one_image,
)
from app.style_detectors import (
    BUCKET_ANIME,
    BUCKET_PHOTO,
    StylePrediction,
    blend_real_life_scores,
    is_style_anime_early_reject,
    wd_realism_score,
)


def test_wd_realism_score_max():
    assert wd_realism_score({"realistic": 0.22, "photorealistic": 0.18, "1girl": 0.9}) == 0.22
    assert wd_realism_score({"1girl": 0.9}) == 0.0


def test_blend_accepts_moderate_wd_with_style_photo():
    style = StylePrediction(
        detector_id="t",
        method="t",
        label="real",
        bucket=BUCKET_PHOTO,
        confidence=0.9,
        scores={"real": 0.9, "anime": 0.1},
        detail={},
    )
    ok, hybrid, detail = blend_real_life_scores({"realistic": 0.22}, style)
    assert ok is True
    assert hybrid >= 0.32
    assert detail["wd_realism"] == 0.22


def test_blend_rejects_anime_veto():
    style = StylePrediction(
        detector_id="t",
        method="t",
        label="anime",
        bucket=BUCKET_ANIME,
        confidence=0.95,
        scores={"real": 0.05, "anime": 0.95},
        detail={},
    )
    ok, _hybrid, _detail = blend_real_life_scores({"realistic": 0.25}, style)
    assert ok is False


def test_blend_rejects_wd_noise_without_style():
    style = StylePrediction(
        detector_id="t",
        method="t",
        label="anime",
        bucket=BUCKET_ANIME,
        confidence=0.7,
        scores={"real": 0.3, "anime": 0.7},
        detail={},
    )
    ok, _hybrid, _detail = blend_real_life_scores({"realistic": 0.2}, style)
    assert ok is False


def test_blend_rejects_mid_style_without_wd():
    style = StylePrediction(
        detector_id="t",
        method="t",
        label="real",
        bucket=BUCKET_PHOTO,
        confidence=0.59,
        scores={"real": 0.59, "anime": 0.41},
        detail={},
    )
    ok, _hybrid, _detail = blend_real_life_scores({}, style)
    assert ok is False


def test_hybrid_filter_routes_hit(monkeypatch, tmp_path: Path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        "app.style_detectors.detect_production_style",
        lambda _p, **_k: StylePrediction(
            detector_id="t",
            method="t",
            label="real",
            bucket=BUCKET_PHOTO,
            confidence=0.88,
            scores={"real": 0.88, "anime": 0.12},
            detail={},
        ),
    )
    base = _ImageInferenceResult(
        image_path=path,
        scores={"realistic": 0.21, "1girl": 0.4},
        primary_tag=None,
        primary_score=None,
        secondary=[],
        needs_review=True,
        reason="below",
        inference_failed=False,
    )
    out = _apply_hybrid_real_life_filter(base)
    assert out.primary_tag == "real_life"
    assert out.needs_review is False
    assert out.primary_score is not None and out.primary_score >= 0.32


def test_style_anime_early_reject_helper():
    anime = StylePrediction(
        detector_id="t",
        method="t",
        label="anime",
        bucket=BUCKET_ANIME,
        confidence=0.96,
        scores={"real": 0.04, "anime": 0.96},
        detail={},
    )
    photo = StylePrediction(
        detector_id="t",
        method="t",
        label="real",
        bucket=BUCKET_PHOTO,
        confidence=0.9,
        scores={"real": 0.9, "anime": 0.1},
        detail={},
    )
    assert is_style_anime_early_reject(anime) is True
    assert is_style_anime_early_reject(photo) is False


def test_infer_hybrid_skips_wd_on_strong_anime(monkeypatch, tmp_path: Path):
    path = tmp_path / "anime.jpg"
    path.write_bytes(b"x")
    called = {"wd": 0}
    monkeypatch.setattr(
        "app.style_detectors.detect_production_style",
        lambda _p, **_k: StylePrediction(
            detector_id="t",
            method="t",
            label="anime",
            bucket=BUCKET_ANIME,
            confidence=0.97,
            scores={"real": 0.03, "anime": 0.97},
            detail={},
        ),
    )

    def boom(*_a, **_k):
        called["wd"] += 1
        raise AssertionError("WD should be skipped on style early reject")

    monkeypatch.setattr("app.api.extract_scores", boom)
    monkeypatch.setattr("app.api.extract_scores_with_experimental_media", boom)
    out = _infer_hybrid_one_image(
        path,
        {"real_life"},
        0.6,
        experimental_media_enabled=True,
        tagger_model="wd_swinv2_v3",
        wd_general_threshold=0.35,
    )
    assert called["wd"] == 0
    assert out.primary_tag is None
    assert "style_early_reject" in (out.reason or "")
