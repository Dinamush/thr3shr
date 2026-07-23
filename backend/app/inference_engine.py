from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

TAGGER_MODEL_ML = "ml_danbooru"
TAGGER_MODEL_WD_SWINV2 = "wd_swinv2_v3"
TAGGER_MODEL_WD_EVA02 = "wd_eva02_large"

WD_MODEL_NAMES = {
    TAGGER_MODEL_WD_SWINV2: "SwinV2_v3",
    TAGGER_MODEL_WD_EVA02: "EVA02_Large",
}

WD_HF_REPOS = {
    "SwinV2_v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "EVA02_Large": "SmilingWolf/wd-eva02-large-tagger-v3",
}

# Prefer deepghs mirror (same files imgutils uses) when available.
WD_DEEPGHS_PREFIX = {
    "SwinV2_v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "EVA02_Large": "SmilingWolf/wd-eva02-large-tagger-v3",
}

# Always surface these for real_life routing even when below wd_general_threshold.
# Probe: anime max realistic often ≪0.05; real photos commonly 0.18–0.99.
WD_REALISM_ALWAYS_TAGS = frozenset({"realistic", "photorealistic"})
# Keep weak-but-useful photo signal (≈0.2+) for hybrid real_life filter blends.
WD_REALISM_FLOOR = 0.18


def _force_cpu() -> bool:
    return os.getenv("FORCE_CPU_INFERENCE", "").strip().lower() in {"1", "true", "yes", "on"}


def ort_providers() -> list[str]:
    from .providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

    ensure_nvidia_dll_search_path()
    preload_onnx_runtime_dlls()
    import onnxruntime as ort

    if _force_cpu():
        return ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _load_rgb(image: Path | Image.Image | str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    from imgutils.data import load_image

    return load_image(str(image), mode="RGB")


def preprocess_wd14(image: Path | Image.Image | str, target_size: int) -> np.ndarray:
    """White-pad square, bicubic resize, BGR float32 NHWC — matches imgutils WD14."""
    if isinstance(image, Image.Image):
        pil = image.convert("RGBA") if image.mode == "RGBA" else image.convert("RGB")
        # Match imgutils load_image path for non-path inputs as closely as practical.
        from imgutils.data import load_image

        pil = load_image(pil, force_background=None, mode=None)
    else:
        from imgutils.data import load_image

        pil = load_image(str(image), force_background=None, mode=None)

    image_shape = pil.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    try:
        padded.paste(pil, (pad_left, pad_top), mask=pil)
    except ValueError:
        padded.paste(pil, (pad_left, pad_top))

    if max_dim != target_size:
        padded = padded.resize((target_size, target_size), Image.BICUBIC)

    array = np.asarray(padded, dtype=np.float32)
    array = array[:, :, ::-1]
    return np.expand_dims(array, axis=0)


def _resize_align_ml(image: Image.Image, size: int, keep_ratio: bool = True, align: int = 4) -> Image.Image:
    if not keep_ratio:
        target_size = (size, size)
    else:
        min_edge = min(image.size)
        target_size = (
            int(image.size[0] / min_edge * size),
            int(image.size[1] / min_edge * size),
        )
    target_size = (
        (target_size[0] // align) * align,
        (target_size[1] // align) * align,
    )
    return image.resize(target_size, resample=Image.BILINEAR)


def _to_tensor_ml(image: Image.Image) -> np.ndarray:
    img = np.array(image, dtype=np.uint8, copy=True)
    img = img.reshape((image.size[1], image.size[0], len(image.getbands())))
    img = img.transpose((2, 0, 1))
    return img.astype(np.float32) / 255.0


def preprocess_mldanbooru(
    image: Path | Image.Image | str,
    *,
    size: int = 448,
    keep_ratio: bool = True,
) -> np.ndarray:
    pil = _load_rgb(image)
    tensor = _to_tensor_ml(_resize_align_ml(pil, size, keep_ratio))
    return tensor.reshape(1, *tensor.shape)


def _normalize_tag(tag: str) -> str:
    text = tag.strip().lower()
    parts: list[str] = []
    for ch in text:
        if ch.isalnum():
            parts.append(ch)
        elif ch in {" ", "-", ".", "/", "_"}:
            parts.append("_")
    return "".join(parts).strip("_")


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for tag, score in scores.items():
        key = _normalize_tag(str(tag))
        if not key:
            continue
        current = normalized.get(key)
        if current is None or float(score) > current:
            normalized[key] = float(score)
    return normalized


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class InferenceEngine:
    """Owned ORT sessions with preprocess outside the run lock."""

    def __init__(self) -> None:
        self._run_lock = threading.Lock()
        self._meta_lock = threading.Lock()
        self._sessions: dict[str, Any] = {}
        self._wd_labels: dict[str, tuple[list[str], list[int], list[int], list[int]]] = {}
        self._ml_labels: list[str] | None = None
        self._wd_target_size: dict[str, int] = {}

    def warm(self, tagger_model: str) -> None:
        from .providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

        ensure_nvidia_dll_search_path()
        preload_onnx_runtime_dlls()
        session = self._get_session(tagger_model)
        if tagger_model == TAGGER_MODEL_ML:
            feed = np.zeros((1, 3, 448, 448), dtype=np.float32)
            with self._run_lock:
                session.run(["output"], {"input": feed})
            self._get_ml_labels()
            logger.info("inference_engine_warmed model=%s", tagger_model)
            return

        wd_name = WD_MODEL_NAMES[tagger_model]
        target = self._wd_target_size[wd_name]
        feed = np.zeros((1, target, target, 3), dtype=np.float32)
        outputs = session.get_outputs()
        input_name = session.get_inputs()[0].name
        with self._run_lock:
            session.run([outputs[0].name, outputs[1].name], {input_name: feed})
        self._get_wd_labels(wd_name)
        logger.info("inference_engine_warmed model=%s", tagger_model)

    def score_one(
        self,
        image: Path | Image.Image | str,
        *,
        tagger_model: str,
        wd_general_threshold: float = 0.35,
    ) -> dict[str, float]:
        return self.score_many(
            [image],
            tagger_model=tagger_model,
            wd_general_threshold=wd_general_threshold,
            batch_size=1,
        )[0]

    def score_many(
        self,
        images: list[Path | Image.Image | str],
        *,
        tagger_model: str,
        wd_general_threshold: float = 0.35,
        batch_size: int = 1,
    ) -> list[dict[str, float]]:
        if not images:
            return []
        if tagger_model == TAGGER_MODEL_ML:
            return [
                self._score_ml_one(image)
                for image in images
            ]
        return self._score_wd_many(
            images,
            tagger_model=tagger_model,
            general_threshold=wd_general_threshold,
            batch_size=max(1, int(batch_size)),
        )

    def _score_ml_one(self, image: Path | Image.Image | str) -> dict[str, float]:
        tensor = preprocess_mldanbooru(image, size=448, keep_ratio=True)
        session = self._get_session(TAGGER_MODEL_ML)
        with self._run_lock:
            native_output, = session.run(["output"], {"input": tensor})
        probs = _sigmoid(native_output).reshape(-1)
        labels = self._get_ml_labels()
        scores = {
            labels[i]: float(probs[i])
            for i in range(min(len(labels), len(probs)))
        }
        return _normalize_scores(scores)

    def _score_wd_many(
        self,
        images: list[Path | Image.Image | str],
        *,
        tagger_model: str,
        general_threshold: float,
        batch_size: int,
    ) -> list[dict[str, float]]:
        wd_name = WD_MODEL_NAMES.get(tagger_model)
        if wd_name is None:
            raise ValueError(f"Unsupported tagger_model: {tagger_model}")

        session = self._get_session(tagger_model)
        target = self._wd_target_size[wd_name]
        # Preprocess outside the lock so CPU decode overlaps other workers' GPU wait.
        tensors = [preprocess_wd14(image, target) for image in images]

        preds_list: list[np.ndarray] = []
        input_name = session.get_inputs()[0].name
        out0 = session.get_outputs()[0].name
        out1 = session.get_outputs()[1].name

        for start in range(0, len(tensors), batch_size):
            chunk = tensors[start : start + batch_size]
            if len(chunk) == 1:
                feed = chunk[0]
            else:
                feed = np.concatenate(chunk, axis=0)
            try:
                with self._run_lock:
                    preds, _embeddings = session.run([out0, out1], {input_name: feed})
            except Exception:
                if len(chunk) == 1:
                    raise
                logger.warning(
                    "wd_batch_run_failed size=%d; falling back to serial",
                    len(chunk),
                )
                preds_rows = []
                for single in chunk:
                    with self._run_lock:
                        pred, _emb = session.run([out0, out1], {input_name: single})
                    preds_rows.append(pred[0])
                preds = np.stack(preds_rows, axis=0)
            for row in preds:
                preds_list.append(row)

        tag_names, _rating_idx, general_idx, _char_idx = self._get_wd_labels(wd_name)
        results: list[dict[str, float]] = []
        for pred in preds_list:
            labels = list(zip(tag_names, pred.astype(float)))
            general: dict[str, float] = {}
            for i in general_idx:
                name, score = labels[i]
                value = float(score)
                if value > general_threshold:
                    general[name] = value
                elif name in WD_REALISM_ALWAYS_TAGS and value >= WD_REALISM_FLOOR:
                    # Keep weak-but-discriminative realism for accidental photos
                    # (moon/sky etc.) without lowering the global general threshold.
                    general[name] = value
            results.append(_normalize_scores(general))
        return results

    def _get_session(self, tagger_model: str) -> Any:
        with self._meta_lock:
            cached = self._sessions.get(tagger_model)
            if cached is not None:
                return cached

            import onnxruntime as ort
            from onnxruntime import GraphOptimizationLevel, SessionOptions

            from .providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

            ensure_nvidia_dll_search_path()
            preload_onnx_runtime_dlls()

            options = SessionOptions()
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ort_providers()
            if providers == ["CPUExecutionProvider"]:
                options.intra_op_num_threads = os.cpu_count() or 4

            if tagger_model == TAGGER_MODEL_ML:
                model_path = hf_hub_download(
                    "deepghs/ml-danbooru-onnx",
                    "ml_caformer_m36_dec-5-97527.onnx",
                )
            else:
                wd_name = WD_MODEL_NAMES[tagger_model]
                prefix = WD_DEEPGHS_PREFIX[wd_name]
                model_path = hf_hub_download(
                    "deepghs/wd14_tagger_with_embeddings",
                    f"{prefix}/model.onnx",
                )

            session = ort.InferenceSession(model_path, options, providers=providers)
            logger.info(
                "inference_session_created model=%s providers=%s active=%s",
                tagger_model,
                providers,
                session.get_providers(),
            )
            if tagger_model != TAGGER_MODEL_ML:
                wd_name = WD_MODEL_NAMES[tagger_model]
                _, target_size, _, _ = session.get_inputs()[0].shape
                self._wd_target_size[wd_name] = int(target_size)

            self._sessions[tagger_model] = session
            return session

    def _get_wd_labels(
        self, wd_name: str
    ) -> tuple[list[str], list[int], list[int], list[int]]:
        with self._meta_lock:
            cached = self._wd_labels.get(wd_name)
            if cached is not None:
                return cached
            repo = WD_HF_REPOS[wd_name]
            path = hf_hub_download(repo, "selected_tags.csv")
            df = pd.read_csv(path)
            tag_names = df["name"].tolist()
            rating_indexes = list(np.where(df["category"] == 9)[0])
            general_indexes = list(np.where(df["category"] == 0)[0])
            character_indexes = list(np.where(df["category"] == 4)[0])
            packed = (tag_names, rating_indexes, general_indexes, character_indexes)
            self._wd_labels[wd_name] = packed
            return packed

    def _get_ml_labels(self) -> list[str]:
        with self._meta_lock:
            if self._ml_labels is not None:
                return self._ml_labels
            path = hf_hub_download(
                "deepghs/imgutils-models",
                "mldanbooru/mldanbooru_tags.csv",
            )
            df = pd.read_csv(path)
            self._ml_labels = df["name"].tolist()
            return self._ml_labels

    def clear(self) -> None:
        with self._meta_lock:
            self._sessions.clear()
            self._wd_labels.clear()
            self._ml_labels = None
            self._wd_target_size.clear()


def get_engine() -> InferenceEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = InferenceEngine()
        return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            _ENGINE.clear()
        _ENGINE = None


_ENGINE: InferenceEngine | None = None
_ENGINE_LOCK = threading.Lock()
