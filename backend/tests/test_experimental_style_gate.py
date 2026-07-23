from pathlib import Path

from app.api import _ImageInferenceResult, _apply_experimental_style_gate
from app.style_detectors import StylePrediction, BUCKET_PHOTO, BUCKET_ANIME, BUCKET_UNCERTAIN


def test_style_gate_routes_real_to_real_life(monkeypatch, tmp_path: Path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"fake")

    monkeypatch.setattr(
        "app.style_detectors.detect_production_style",
        lambda _p: StylePrediction(
            detector_id="imgutils_caformer_band",
            method="test",
            label="real",
            bucket=BUCKET_PHOTO,
            confidence=0.97,
            scores={"real": 0.97, "anime": 0.03},
            detail={},
        ),
    )
    base = _ImageInferenceResult(
        image_path=path,
        scores={"1girl": 0.9},
        primary_tag="1girl",
        primary_score=0.9,
        secondary=[],
        needs_review=False,
        reason=None,
        inference_failed=False,
    )
    out = _apply_experimental_style_gate(base, {"1girl", "real_life"})
    assert out.primary_tag == "real_life"
    assert out.primary_score == 0.97
    assert out.needs_review is False
    assert out.secondary[0]["tag"] == "1girl"


def test_style_gate_uncertain_needs_review(monkeypatch, tmp_path: Path):
    path = tmp_path / "edge.jpg"
    path.write_bytes(b"fake")
    monkeypatch.setattr(
        "app.style_detectors.detect_production_style",
        lambda _p: StylePrediction(
            detector_id="imgutils_caformer_band",
            method="test",
            label="uncertain",
            bucket=BUCKET_UNCERTAIN,
            confidence=0.6,
            scores={"real": 0.55, "anime": 0.45},
            detail={},
        ),
    )
    base = _ImageInferenceResult(
        image_path=path,
        scores={"1girl": 0.8},
        primary_tag="1girl",
        primary_score=0.8,
        secondary=[],
        needs_review=False,
        reason=None,
        inference_failed=False,
    )
    out = _apply_experimental_style_gate(base, {"1girl", "real_life"})
    assert out.primary_tag is None
    assert out.needs_review is True
    assert "uncertain" in (out.reason or "").lower()


def test_style_gate_anime_keeps_taxonomy(monkeypatch, tmp_path: Path):
    path = tmp_path / "anime.jpg"
    path.write_bytes(b"fake")
    monkeypatch.setattr(
        "app.style_detectors.detect_production_style",
        lambda _p: StylePrediction(
            detector_id="imgutils_caformer_band",
            method="test",
            label="anime",
            bucket=BUCKET_ANIME,
            confidence=0.99,
            scores={"real": 0.01, "anime": 0.99},
            detail={},
        ),
    )
    base = _ImageInferenceResult(
        image_path=path,
        scores={"1girl": 0.95},
        primary_tag="1girl",
        primary_score=0.95,
        secondary=[],
        needs_review=False,
        reason=None,
        inference_failed=False,
    )
    out = _apply_experimental_style_gate(base, {"1girl", "real_life"})
    assert out.primary_tag == "1girl"
    assert out.needs_review is False
