from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_sfw_fixture(path: Path, size: tuple[int, int] = (512, 640)) -> Path:
    """Solid + soft blob — deterministic, non-explicit test image."""
    img = Image.new("RGB", size, (210, 190, 230))
    for x in range(180, 330):
        for y in range(140, 360):
            img.putpixel((x, y), (240, 210, 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def test_wd_preprocess_matches_imgutils(tmp_path: Path) -> None:
    from imgutils.tagging.wd14 import _prepare_image_for_tagging

    from app.inference_engine import preprocess_wd14

    image_path = _make_sfw_fixture(tmp_path / "wd.png")
    expected = _prepare_image_for_tagging(str(image_path), 448)
    actual = preprocess_wd14(image_path, 448)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_ml_preprocess_matches_imgutils(tmp_path: Path) -> None:
    from imgutils.data import load_image
    from imgutils.tagging.mldanbooru import _resize_align, _to_tensor

    from app.inference_engine import preprocess_mldanbooru

    image_path = _make_sfw_fixture(tmp_path / "ml.png", size=(640, 480))
    pil = load_image(str(image_path), mode="RGB")
    expected = _to_tensor(_resize_align(pil, 448, True))[None, ...]
    actual = preprocess_mldanbooru(image_path, size=448, keep_ratio=True)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_force_cpu_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import inference_engine as eng

    monkeypatch.setenv("FORCE_CPU_INFERENCE", "true")
    providers = eng.ort_providers()
    assert providers == ["CPUExecutionProvider"]

    monkeypatch.setenv("FORCE_CPU_INFERENCE", "false")
    providers = eng.ort_providers()
    assert providers[0] in {"CUDAExecutionProvider", "CPUExecutionProvider"}
    assert "CPUExecutionProvider" in providers


@pytest.mark.live_onnx
def test_engine_scores_close_to_imgutils_wd(tmp_path: Path) -> None:
    from imgutils.tagging import get_wd14_tags

    from app.inference_engine import InferenceEngine
    from app.services import _normalize_score_tags, _parse_wd14_raw

    image_path = _make_sfw_fixture(tmp_path / "live_wd.png")
    raw = get_wd14_tags(
        str(image_path),
        model_name="SwinV2_v3",
        general_threshold=0.35,
        no_underline=False,
        drop_overlap=False,
        fmt="general",
    )
    expected = _parse_wd14_raw(raw)

    engine = InferenceEngine()
    engine.warm("wd_swinv2_v3")
    actual = engine.score_one(
        image_path,
        tagger_model="wd_swinv2_v3",
        wd_general_threshold=0.35,
    )

    assert set(actual) == set(expected)
    for tag, score in expected.items():
        # GPU/ORT nondeterminism can exceed 1e-4; tagging decisions use ~1e-3.
        assert abs(actual[tag] - score) < 1e-3, tag


@pytest.mark.live_onnx
def test_engine_scores_close_to_imgutils_ml(tmp_path: Path) -> None:
    from imgutils.tagging import get_mldanbooru_tags

    from app.inference_engine import InferenceEngine
    from app.services import _normalize_score_tags, _parse_mldanbooru_raw

    image_path = _make_sfw_fixture(tmp_path / "live_ml.png")
    raw = get_mldanbooru_tags(
        str(image_path),
        threshold=0.0,
        size=448,
        keep_ratio=True,
        drop_overlap=False,
        use_real_name=False,
    )
    expected = _normalize_score_tags(_parse_mldanbooru_raw(raw))

    engine = InferenceEngine()
    engine.warm("ml_danbooru")
    actual = engine.score_one(
        image_path,
        tagger_model="ml_danbooru",
        wd_general_threshold=0.35,
    )

    assert set(actual) == set(expected)
    for tag, score in expected.items():
        assert abs(actual[tag] - score) < 1e-3, tag


@pytest.mark.live_onnx
def test_wd_batch_matches_single(tmp_path: Path) -> None:
    from app.inference_engine import InferenceEngine

    paths = [
        _make_sfw_fixture(tmp_path / "a.png", (448, 448)),
        _make_sfw_fixture(tmp_path / "b.png", (512, 384)),
        _make_sfw_fixture(tmp_path / "c.png", (600, 600)),
    ]
    engine = InferenceEngine()
    engine.warm("wd_swinv2_v3")
    singles = [
        engine.score_one(p, tagger_model="wd_swinv2_v3", wd_general_threshold=0.35)
        for p in paths
    ]
    batched = engine.score_many(
        paths,
        tagger_model="wd_swinv2_v3",
        wd_general_threshold=0.35,
        batch_size=3,
    )
    assert len(batched) == 3
    for single, batch in zip(singles, batched):
        assert set(single) == set(batch)
        for tag, score in single.items():
            # Batched CUDA runs can differ slightly from N=1 (cuDNN algorithms).
            assert abs(batch[tag] - score) < 1e-3, tag


def test_engine_clear_drops_cached_sessions() -> None:
    from app.inference_engine import InferenceEngine

    engine = InferenceEngine()
    engine._sessions["wd_swinv2_v3"] = object()
    engine._sessions["ml_danbooru"] = object()
    assert engine.loaded_models() == ["ml_danbooru", "wd_swinv2_v3"]
    assert engine.clear() == ["ml_danbooru", "wd_swinv2_v3"]
    assert engine.loaded_models() == []


def _reference_wd_general_scores(tag_names, general_idx, pred, threshold, raw_general):
    """The per-tag postprocessing the engine used before precomputing labels."""
    from app.inference_engine import (
        WD_REALISM_ALWAYS_TAGS,
        WD_REALISM_FLOOR,
        _normalize_scores,
    )

    labels = list(zip(tag_names, pred.astype(float)))
    general = {}
    for i in general_idx:
        name, score = labels[i]
        value = float(score)
        if raw_general or value > threshold:
            general[name] = value
        elif name in WD_REALISM_ALWAYS_TAGS and value >= WD_REALISM_FLOOR:
            general[name] = value
    return _normalize_scores(general)


@pytest.mark.parametrize("raw_general", [False, True])
def test_wd_general_scores_match_reference(raw_general: bool) -> None:
    from app.inference_engine import InferenceEngine

    # Rating and character tags, names that normalize to the same key or to
    # nothing, and the realism tags kept below the threshold.
    tag_names = ["general", "sensitive", "1girl", "solo", "long hair", "long_hair",
                 "long-hair", ":)", "^^^", "realistic", "photorealistic", "hatsune_miku",
                 "sky", "Sky", "outdoors"]
    rating_idx, char_idx = [0, 1], [11]
    general_idx = [i for i in range(len(tag_names)) if i not in rating_idx + char_idx]
    engine = InferenceEngine()
    engine._wd_labels["SwinV2_v3"] = (tag_names, rating_idx, general_idx, char_idx)

    rng = np.random.default_rng(0)
    rows = [rng.random(len(tag_names), dtype=np.float32) for _ in range(200)]
    # Values on the threshold and realism floor in float32 and float64 terms.
    edge = np.full(len(tag_names), 0.35, dtype=np.float32)
    edge[9] = np.float32(0.18)
    edge[10] = np.nextafter(np.float32(0.18), np.float32(0))
    rows += [edge, np.nextafter(edge, np.float32(1)), np.zeros(len(tag_names), np.float32)]

    for pred in rows:
        expected = _reference_wd_general_scores(tag_names, general_idx, pred, 0.35, raw_general)
        actual = engine._wd_general_scores("SwinV2_v3", pred, 0.35, raw_general)
        assert actual == expected
        assert list(actual) == list(expected)
