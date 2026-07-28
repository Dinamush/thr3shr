"""Unit tests for real-life engine frame pooling and classification."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from PIL import Image

from app.real_life_engine import (
    DEFAULT_VLM_FILENAME,
    LlamaCppVlmAdapter,
    RealLifeEngine,
    build_contact_sheet,
    pool_frame_scores,
    reset_real_life_engine,
    set_real_life_adapters,
)
from app.style_detectors import BUCKET_ANIME, BUCKET_PHOTO, StylePrediction


class _FakeVlm:
    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def available(self) -> bool:
        return True

    def status(self) -> dict:
        return {"available": True, "backend": "fake"}

    def tag_images(self, images, vocabulary):
        assert images
        return dict(self._scores)


class _FakePosition:
    def available(self) -> bool:
        return True

    def status(self) -> dict:
        return {"available": True, "backend": "fake"}

    def classify(self, images):
        return {"blowjob": 0.4}


def test_pool_frame_scores_damps_single_frame_spikes():
    pooled = pool_frame_scores(
        [
            {"creampie": 0.9, "oral": 0.2},
            {"oral": 0.8},
            {"oral": 0.7},
        ]
    )
    assert pooled["oral"] >= 0.7
    # creampie only on one of three frames → damped
    assert pooled["creampie"] < 0.9


def test_build_contact_sheet():
    frames = [Image.new("RGB", (64, 64), color=(i * 40, 0, 0)) for i in range(4)]
    sheet = build_contact_sheet(frames, tile=32, cols=2)
    assert sheet.size == (64, 64)


def test_classify_anime_gated(tmp_path: Path):
    reset_real_life_engine()
    photo = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(photo)

    def style(_path, **_k):
        return StylePrediction(
            detector_id="t",
            method="t",
            label="anime",
            bucket=BUCKET_ANIME,
            confidence=0.95,
            scores={"anime": 0.95, "real": 0.05},
            detail={},
        )

    engine = set_real_life_adapters(
        vlm=_FakeVlm({"creampie": 0.99}),
        position=_FakePosition(),
        style_detector=style,
    )
    result = engine.classify_path(photo, categories_root=tmp_path)
    assert result.needs_review
    assert result.reason == "style_gate_anime"
    assert result.primary_tag is None
    reset_real_life_engine()


def test_classify_photo_produces_real_life_destination(tmp_path: Path):
    reset_real_life_engine()
    photo = tmp_path / "b.jpg"
    Image.new("RGB", (48, 48), color=(200, 100, 50)).save(photo)

    def style(_path, **_k):
        return StylePrediction(
            detector_id="t",
            method="t",
            label="real",
            bucket=BUCKET_PHOTO,
            confidence=0.9,
            scores={"anime": 0.1, "real": 0.9},
            detail={},
        )

    engine = set_real_life_adapters(
        vlm=_FakeVlm({"creampie": 0.92, "Asian": 0.8}),
        position=_FakePosition(),
        style_detector=style,
    )
    result = engine.classify_path(photo, categories_root=tmp_path)
    assert result.primary_tag == "creampie"
    assert result.needs_review  # calibration policy
    assert result.suggested_destination is not None
    assert "Real Life" in Path(result.suggested_destination).parts
    assert any(s["tag"] == "Asian" for s in result.secondary)
    assert result.scores
    assert all(str(k).startswith("rl:") for k in result.scores)
    reset_real_life_engine()


def test_engine_status_reports_fallback_without_vlm():
    reset_real_life_engine()
    engine = RealLifeEngine()
    status = engine.status()
    assert "vlm" in status
    assert "message" in status
    reset_real_life_engine()


def test_engine_status_probes_vlm_before_reporting_its_status():
    reset_real_life_engine()
    class LazyVlm:
        ready = False

        def available(self):
            self.ready = True
            return True

        def status(self):
            return {"available": self.ready}

        def tag_images(self, images, vocabulary):
            return {}

    engine = RealLifeEngine(vlm=LazyVlm(), position=_FakePosition())

    status = engine.status()

    assert status["vlm"]["available"] is True
    assert status["fallback"] is False
    reset_real_life_engine()


def test_vlm_loads_vision_projection_with_model(monkeypatch):
    downloaded: list[str] = []
    handler_paths: list[str | None] = []
    llama_kwargs: dict = {}

    def fake_download(repo_id, filename):
        assert repo_id == "example/model"
        downloaded.append(filename)
        return f"C:/models/{filename}"

    class FakeLlama:
        def __init__(self, **kwargs):
            llama_kwargs.update(kwargs)

    class FakeHandler:
        def __init__(self, *, clip_model_path):
            handler_paths.append(clip_model_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_download),
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp.llama_chat_format",
        types.SimpleNamespace(Qwen25VLChatHandler=FakeHandler),
    )

    adapter = LlamaCppVlmAdapter(repo_id="example/model")
    adapter._ensure()

    mmproj_filename = "mmproj-thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-f16.gguf"
    assert downloaded == [DEFAULT_VLM_FILENAME, mmproj_filename]
    assert handler_paths == [f"C:/models/{mmproj_filename}"]
    assert llama_kwargs["chat_handler"] is not None


def test_prepare_vlm_images_downscales_and_caps_frames():
    from app.real_life_engine import prepare_vlm_images

    frames = [Image.new("RGB", (1920, 1080), color=(i * 20, 0, 0)) for i in range(6)]
    prepared = prepare_vlm_images(frames, max_frames=2, max_side=512)
    assert len(prepared) == 2
    assert all(max(img.size) <= 512 for img in prepared)


def test_extract_message_content_supports_list_parts():
    from app.real_life_engine import extract_message_content

    assert extract_message_content("{\"tags\":[]}") == "{\"tags\":[]}"
    assert (
        extract_message_content(
            [{"type": "text", "text": "{\"primary\":\"oral\"}"}, {"type": "text", "text": " extra"}]
        )
        == "{\"primary\":\"oral\"} extra"
    )


def test_position_adapter_reads_label_names_from_config(monkeypatch, tmp_path: Path):
    from app.real_life_engine import TimmPositionAdapter

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"label_names": ["blowjob", "hardcore", "handjob"], "num_classes": 3}),
        encoding="utf-8",
    )

    def fake_download(repo_id, filename):
        assert filename == "config.json"
        return str(config_path)

    class FakeModel:
        num_classes = 3

        def eval(self):
            return self

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_download),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "timm",
        types.SimpleNamespace(create_model=lambda *a, **k: FakeModel()),
    )

    adapter = TimmPositionAdapter(repo_id="porntech/sex-position")
    assert adapter.available()
    assert adapter.status()["labels"] == ["blowjob", "hardcore", "handjob"]
