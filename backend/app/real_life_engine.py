"""Local real-life adult tagging engine (VLM + optional position classifier).

Optional heavy deps:
  - llama-cpp-python  (adult-tuned Qwen2.5-VL GGUF)
  - torch + timm      (porntech/sex-position)

When optional deps are missing, the engine still runs: style gate + closed-vocab
fusion work, but tag scores stay empty/low and every item needs review. Tests
inject mock adapters via ``set_real_life_adapters``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image

from .media_sampling import sample_gif_frames, sample_video_frames
from .real_life_taxonomy import (
    fuse_real_life_scores,
    get_real_life_taxonomy,
    parse_closed_vocabulary_tags,
    real_life_destination,
)
from .services import VIDEO_EXTENSIONS
from .style_detectors import (
    BUCKET_ANIME,
    BUCKET_PHOTO,
    BUCKET_UNCERTAIN,
    StylePrediction,
    detect_production_style,
)

logger = logging.getLogger(__name__)

DEFAULT_VLM_REPO = os.environ.get(
    "REAL_LIFE_VLM_REPO",
    "bartowski/thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-GGUF",
)
DEFAULT_VLM_FILENAME = os.environ.get(
    "REAL_LIFE_VLM_FILENAME",
    "thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-Q4_K_M.gguf",
)
DEFAULT_VLM_MMPROJ_FILENAME = os.environ.get(
    "REAL_LIFE_VLM_MMPROJ_FILENAME",
    "mmproj-thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-f16.gguf",
)
DEFAULT_POSITION_REPO = "porntech/sex-position"
MIN_RECOMMENDED_RAM_GB = 12
REAL_LIFE_SAMPLE_MAX = 8

_ENGINE_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
_ENGINE: "RealLifeEngine | None" = None


class VlmAdapter(Protocol):
    def available(self) -> bool: ...

    def status(self) -> dict[str, Any]: ...

    def tag_images(self, images: list[Image.Image], vocabulary: list[str]) -> dict[str, float]: ...


class PositionAdapter(Protocol):
    def available(self) -> bool: ...

    def status(self) -> dict[str, Any]: ...

    def classify(self, images: list[Image.Image]) -> dict[str, float]: ...


@dataclass
class RealLifeClassification:
    path: Path
    scores: dict[str, float]
    primary_tag: str | None
    primary_score: float | None
    secondary: list[dict[str, float]]
    needs_review: bool
    reason: str | None
    inference_failed: bool = False
    style_bucket: str = BUCKET_UNCERTAIN
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_destination: str | None = None


@dataclass
class _StubVlm:
    reason: str = "vlm_unavailable"

    def available(self) -> bool:
        return False

    def status(self) -> dict[str, Any]:
        return {"available": False, "backend": "stub", "reason": self.reason}

    def tag_images(self, images: list[Image.Image], vocabulary: list[str]) -> dict[str, float]:
        return {}


@dataclass
class _StubPosition:
    reason: str = "position_classifier_unavailable"

    def available(self) -> bool:
        return False

    def status(self) -> dict[str, Any]:
        return {"available": False, "backend": "stub", "reason": self.reason}

    def classify(self, images: list[Image.Image]) -> dict[str, float]:
        return {}


# Sex-position label → taxonomy folder aliases (porntech/sex-position class names vary).
_POSITION_ALIASES: dict[str, str] = {
    "missionary": "vaginal",
    "doggy": "vaginal",
    "doggy_style": "vaginal",
    "cowgirl": "vaginal",
    "reverse_cowgirl": "vaginal",
    "hardcore": "vaginal",
    "blowjob": "blowjob",
    "oral": "oral",
    "pussy-licking": "oral",
    "pussy_licking": "oral",
    "anal": "anal",
    "handjob": "handjob",
    "titjob": "handjob",
    "masturbation": "masturbation",
    "fingering": "masturbation",
    "solo": "masturbation",
    "cumshot": "cumshot",
    "facial": "facial",
    "creampie": "creampie",
}

# Keep VLM vision tokens bounded on 8GB GPUs (full HD × 4 frames blows n_ctx).
DEFAULT_VLM_MAX_FRAMES = 2
DEFAULT_VLM_MAX_SIDE = 512
DEFAULT_VLM_N_CTX = 8192
DEFAULT_MEDIA_SAMPLE_COUNT = 4


def prepare_vlm_images(
    images: list[Image.Image],
    *,
    max_frames: int = DEFAULT_VLM_MAX_FRAMES,
    max_side: int = DEFAULT_VLM_MAX_SIDE,
) -> list[Image.Image]:
    """Cap frame count and downscale so multimodal prompts fit n_ctx."""
    if not images:
        return []
    max_frames = max(1, int(max_frames))
    max_side = max(64, int(max_side))
    if len(images) <= max_frames:
        selected = list(images)
    else:
        # Evenly sample across the clip instead of only the opening frames.
        step = (len(images) - 1) / float(max_frames - 1)
        idxs = sorted({int(round(i * step)) for i in range(max_frames)})
        selected = [images[i] for i in idxs]
    prepared: list[Image.Image] = []
    for image in selected:
        rgb = image.convert("RGB")
        w, h = rgb.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            rgb = rgb.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        prepared.append(rgb)
    return prepared


def extract_message_content(content: Any) -> str:
    """Normalize chat completion content that may be a string or part list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text is None and isinstance(part.get("content"), str):
                    text = part["content"]
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


class LlamaCppVlmAdapter:
    """Adult-tuned Qwen2.5-VL GGUF via llama-cpp-python (optional)."""

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_VLM_REPO,
        filename: str = DEFAULT_VLM_FILENAME,
        mmproj_filename: str = DEFAULT_VLM_MMPROJ_FILENAME,
        n_ctx: int = DEFAULT_VLM_N_CTX,
        max_frames: int = DEFAULT_VLM_MAX_FRAMES,
        max_side: int = DEFAULT_VLM_MAX_SIDE,
    ) -> None:
        self.repo_id = repo_id
        self.filename = filename
        self.mmproj_filename = mmproj_filename
        self.n_ctx = n_ctx
        self.max_frames = max_frames
        self.max_side = max_side
        self._llm = None
        self._error: str | None = None
        self._model_path: str | None = None
        self._mmproj_path: str | None = None
        self.last_raw_text: str | None = None

    def available(self) -> bool:
        try:
            self._ensure()
            return self._llm is not None
        except Exception as err:  # noqa: BLE001
            self._error = str(err)
            return False

    def status(self) -> dict[str, Any]:
        return {
            "available": self._llm is not None,
            "backend": "llama_cpp",
            "repo_id": self.repo_id,
            "filename": self.filename,
            "model_path": self._model_path,
            "mmproj_filename": self.mmproj_filename,
            "mmproj_path": self._mmproj_path,
            "reason": self._error,
        }

    def _ensure(self) -> None:
        if self._llm is not None or self._error:
            return
        try:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        except Exception as err:  # noqa: BLE001
            self._error = f"llama-cpp-python unavailable: {err}"
            logger.warning("real_life_vlm_import_failed err=%s", err)
            return
        try:
            path = hf_hub_download(self.repo_id, self.filename)
            mmproj_path = hf_hub_download(self.repo_id, self.mmproj_filename)
            self._model_path = path
            self._mmproj_path = mmproj_path
            chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
            n_gpu_layers = -1 if os.environ.get("FORCE_CPU_INFERENCE", "").lower() not in {
                "1",
                "true",
                "yes",
            } else 0
            self._llm = Llama(
                model_path=path,
                chat_handler=chat_handler,
                n_ctx=self.n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as err:  # noqa: BLE001
            self._error = str(err)
            logger.exception("real_life_vlm_load_failed")

    def tag_images(self, images: list[Image.Image], vocabulary: list[str]) -> dict[str, float]:
        self._ensure()
        self.last_raw_text = None
        if self._llm is None or not images:
            return {}
        prepared = prepare_vlm_images(
            images, max_frames=self.max_frames, max_side=self.max_side
        )
        if not prepared:
            return {}
        vocab = ", ".join(sorted(set(vocabulary)))
        prompt = (
            "You are an adult content tagger for real-life photos/videos. "
            "Respond with ONLY JSON matching "
            '{"tags":[{"tag":"<label>","score":0.0-1.0}],"primary":"<label>"}. '
            f"Use only these labels: {vocab}. "
            "Prefer visible acts. Demographic labels (BBC, Ebony, Asian) only when clear. "
            "Contextual labels (cuck, hotwife) only with strong evidence."
        )
        try:
            import base64
            from io import BytesIO

            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image in prepared:
                buf = BytesIO()
                image.save(buf, format="JPEG", quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
            result = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_tokens=512,
            )
            raw_content = result["choices"][0]["message"]["content"]
            text = extract_message_content(raw_content)
            self.last_raw_text = text
            parsed = parse_closed_vocabulary_tags(_extract_json_object(text))
            if not parsed:
                logger.warning(
                    "real_life_vlm_empty_parse frames=%d chars=%d preview=%r",
                    len(prepared),
                    len(text or ""),
                    (text or "")[:240],
                )
            return parsed
        except Exception as err:  # noqa: BLE001
            logger.warning("real_life_vlm_infer_failed err=%s", err)
            self.last_raw_text = f"__error__:{err}"
            return {}

    def close(self) -> None:
        llm = self._llm
        self._llm = None
        if llm is None:
            return
        closer = getattr(llm, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                logger.exception("real_life_vlm_close_failed")
        del llm


class TimmPositionAdapter:
    """Optional porntech/sex-position corroboration via torch/timm."""

    def __init__(self, repo_id: str = DEFAULT_POSITION_REPO) -> None:
        self.repo_id = repo_id
        self._model = None
        self._labels: list[str] = []
        self._error: str | None = None

    def available(self) -> bool:
        try:
            self._ensure()
            return self._model is not None
        except Exception as err:  # noqa: BLE001
            self._error = str(err)
            return False

    def status(self) -> dict[str, Any]:
        return {
            "available": self._model is not None,
            "backend": "timm",
            "repo_id": self.repo_id,
            "labels": list(self._labels),
            "reason": self._error,
        }

    def _ensure(self) -> None:
        if self._model is not None or self._error:
            return
        try:
            import timm
            import torch
            from huggingface_hub import hf_hub_download
        except Exception as err:  # noqa: BLE001
            self._error = f"torch/timm unavailable: {err}"
            return
        try:
            # Prefer a config/labels file if present; otherwise use timm create_model hub.
            labels_path = None
            for name in ("labels.txt", "classes.txt", "config.json"):
                try:
                    labels_path = hf_hub_download(self.repo_id, name)
                    break
                except Exception:  # noqa: BLE001
                    continue
            if labels_path and labels_path.endswith(".txt"):
                self._labels = [
                    line.strip()
                    for line in Path(labels_path).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            elif labels_path and labels_path.endswith(".json"):
                raw = json.loads(Path(labels_path).read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    if isinstance(raw.get("label_names"), list):
                        self._labels = [str(x) for x in raw["label_names"] if str(x).strip()]
                    elif "id2label" in raw:
                        self._labels = [
                            raw["id2label"][str(i)] for i in range(len(raw["id2label"]))
                        ]
                elif isinstance(raw, list):
                    self._labels = [str(x) for x in raw]
            self._model = timm.create_model(f"hf_hub:{self.repo_id}", pretrained=True)
            self._model.eval()
            self._torch = torch
            if not self._labels:
                # Fall back to num_classes placeholder labels so status is honest.
                num_classes = int(getattr(self._model, "num_classes", 0) or 0)
                if num_classes > 0:
                    self._labels = [str(i) for i in range(num_classes)]
                    logger.warning(
                        "real_life_position_labels_missing repo=%s num_classes=%d",
                        self.repo_id,
                        num_classes,
                    )
        except Exception as err:  # noqa: BLE001
            self._error = str(err)
            logger.warning("real_life_position_load_failed err=%s", err)

    def classify(self, images: list[Image.Image]) -> dict[str, float]:
        self._ensure()
        if self._model is None or not images:
            return {}
        torch = self._torch
        scores: dict[str, float] = {}
        try:
            data_config = {}
            try:
                from timm.data import resolve_data_config
                from timm.data.transforms_factory import create_transform

                data_config = resolve_data_config({}, model=self._model)
                transform = create_transform(**data_config)
            except Exception:  # noqa: BLE001
                transform = None
            for image in images[:REAL_LIFE_SAMPLE_MAX]:
                rgb = image.convert("RGB")
                if transform is not None:
                    tensor = transform(rgb).unsqueeze(0)
                else:
                    resized = rgb.resize((224, 224))
                    import numpy as np

                    arr = np.asarray(resized).astype("float32") / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    logits = self._model(tensor)
                    probs = torch.softmax(logits, dim=-1)[0].tolist()
                for idx, prob in enumerate(probs):
                    label = self._labels[idx] if idx < len(self._labels) else str(idx)
                    key = label.lower().replace(" ", "_")
                    mapped = _POSITION_ALIASES.get(key) or _POSITION_ALIASES.get(
                        label.lower()
                    ) or label
                    scores[mapped] = max(scores.get(mapped, 0.0), float(prob))
        except Exception as err:  # noqa: BLE001
            logger.warning("real_life_position_infer_failed err=%s", err)
        return scores

    def close(self) -> None:
        self._model = None


def _extract_json_object(text: str) -> Any:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def build_contact_sheet(images: list[Image.Image], *, tile: int = 256, cols: int = 3) -> Image.Image:
    if not images:
        return Image.new("RGB", (tile, tile), color=(0, 0, 0))
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile), color=(0, 0, 0))
    for idx, image in enumerate(images):
        thumb = image.convert("RGB").copy()
        thumb.thumbnail((tile, tile))
        x = (idx % cols) * tile + (tile - thumb.width) // 2
        y = (idx // cols) * tile + (tile - thumb.height) // 2
        sheet.paste(thumb, (x, y))
    return sheet


def load_media_frames(path: Path, *, sample_count: int | None = None) -> list[Image.Image]:
    suffix = path.suffix.lower()
    count = sample_count or REAL_LIFE_SAMPLE_MAX
    if suffix == ".gif":
        return sample_gif_frames(path, sample_count=count)
    if suffix in VIDEO_EXTENSIONS:
        return sample_video_frames(path, sample_count=count)
    with Image.open(path) as img:
        return [img.convert("RGB")]


def check_system_capability() -> dict[str, Any]:
    """Best-effort RAM/VRAM advisory for the heavy VLM path."""
    info: dict[str, Any] = {
        "ok": True,
        "warnings": [],
        "recommended_ram_gb": MIN_RECOMMENDED_RAM_GB,
    }
    try:
        import psutil  # type: ignore

        total_gb = psutil.virtual_memory().total / (1024**3)
        info["ram_gb"] = round(total_gb, 1)
        if total_gb < MIN_RECOMMENDED_RAM_GB:
            info["ok"] = False
            info["warnings"].append(
                f"System RAM {total_gb:.1f} GB is below recommended "
                f"{MIN_RECOMMENDED_RAM_GB} GB for the Qwen2.5-VL GGUF tagger."
            )
    except Exception:  # noqa: BLE001
        info["warnings"].append("Could not probe system RAM (psutil missing).")
    return info


class RealLifeEngine:
    def __init__(
        self,
        *,
        vlm: VlmAdapter | None = None,
        position: PositionAdapter | None = None,
        style_detector: Callable[..., StylePrediction] | None = None,
    ) -> None:
        self.vlm = vlm or LlamaCppVlmAdapter()
        self.position = position or TimmPositionAdapter()
        self.style_detector = style_detector or detect_production_style

    def status(self) -> dict[str, Any]:
        capability = check_system_capability()
        vlm_available = self.vlm.available()
        self.position.available()
        return {
            "vlm": self.vlm.status(),
            "position": self.position.status(),
            "capability": capability,
            "fallback": not vlm_available,
            "message": (
                "Adult VLM unavailable — style gate + review-only path active. "
                "Install llama-cpp-python and download the GGUF to enable rich tags."
                if not vlm_available
                else "Adult VLM ready."
            ),
        }

    def classify_path(
        self,
        path: Path,
        *,
        categories_root: Path | None = None,
        selected_folders: set[str] | None = None,
        confidence_threshold: float = 0.45,
    ) -> RealLifeClassification:
        tax = get_real_life_taxonomy()
        evidence: dict[str, Any] = {"sources": []}
        try:
            style = self.style_detector(path, uncertain_threshold=None)
        except Exception as err:  # noqa: BLE001
            logger.warning("real_life_style_failed path=%s err=%s", path, err)
            style = StylePrediction(
                detector_id="error",
                method="error",
                label="uncertain",
                bucket=BUCKET_UNCERTAIN,
                confidence=0.0,
                scores={},
                detail={"error": str(err)},
            )
        evidence["style"] = {
            "bucket": style.bucket,
            "label": style.label,
            "confidence": style.confidence,
            "scores": dict(style.scores or {}),
        }

        if style.bucket == BUCKET_ANIME:
            return RealLifeClassification(
                path=path,
                scores={},
                primary_tag=None,
                primary_score=None,
                secondary=[],
                needs_review=True,
                reason="style_gate_anime",
                style_bucket=style.bucket,
                evidence=evidence,
            )

        try:
            frames = load_media_frames(path, sample_count=DEFAULT_MEDIA_SAMPLE_COUNT)
        except Exception as err:  # noqa: BLE001
            logger.warning("real_life_frame_load_failed path=%s err=%s", path, err)
            return RealLifeClassification(
                path=path,
                scores={},
                primary_tag=None,
                primary_score=None,
                secondary=[],
                needs_review=True,
                reason=f"frame_load_failed:{err}",
                inference_failed=True,
                style_bucket=style.bucket,
                evidence=evidence,
            )

        if not frames:
            return RealLifeClassification(
                path=path,
                scores={},
                primary_tag=None,
                primary_score=None,
                secondary=[],
                needs_review=True,
                reason="no_frames",
                inference_failed=True,
                style_bucket=style.bucket,
                evidence=evidence,
            )

        evidence["frame_count"] = len(frames)
        vocab = sorted(tax.closed_vocabulary)
        with _RUN_LOCK:
            vlm_scores = self.vlm.tag_images(frames, vocab)
            raw_preview = getattr(self.vlm, "last_raw_text", None)
            if isinstance(raw_preview, str) and raw_preview:
                evidence["vlm_raw_preview"] = raw_preview[:500]
            position_scores = self.position.classify(frames)

        evidence["vlm_scores"] = dict(vlm_scores)
        evidence["position_scores"] = dict(position_scores)
        evidence["sources"].append("vlm" if vlm_scores else "vlm_empty")
        if position_scores:
            evidence["sources"].append("position")

        merged: dict[str, float] = {}
        for key, value in vlm_scores.items():
            merged[key] = max(merged.get(key, 0.0), float(value))
        # Position classifier is corroboration only — mild boost, never sole winner unless strong.
        for key, value in position_scores.items():
            boost = min(0.85, float(value) * 0.9)
            merged[key] = max(merged.get(key, 0.0), boost)

        # Aggregate conservatively: keep max across frames already done by adapters;
        # require style photo signal when accepting.
        if style.bucket == BUCKET_UNCERTAIN:
            for key in list(merged):
                merged[key] = float(merged[key]) * 0.85

        primary, primary_score, secondary, folder_scores, flags = fuse_real_life_scores(
            merged,
            selected_folders=selected_folders,
            tax=tax,
        )
        evidence["folder_scores"] = folder_scores
        evidence["review_flags"] = flags

        reasons = list(flags)
        if style.bucket == BUCKET_UNCERTAIN:
            reasons.append("style_uncertain")
        if not self.vlm.available():
            reasons.append("vlm_unavailable_fallback")
        if not vlm_scores:
            reasons.append("vlm_empty")
        if primary_score is not None and primary_score < confidence_threshold:
            reasons.append("below_confidence_threshold")
            primary = None
            primary_score = None

        # Calibration policy: always review.
        needs_review = True
        dest = None
        if primary and categories_root is not None:
            dest = str(real_life_destination(categories_root, primary))

        # Persist model evidence under namespaced keys so anime taxonomy never sees them.
        namespaced = {f"rl:{k}": float(v) for k, v in folder_scores.items()}
        namespaced["rl:__style_photo__"] = float((style.scores or {}).get("real") or 0.0)
        namespaced["rl:__style_anime__"] = float((style.scores or {}).get("anime") or 0.0)
        namespaced["rl:__vlm_tag_count__"] = float(len(vlm_scores))
        namespaced["rl:__position_tag_count__"] = float(len(position_scores))
        if not vlm_scores:
            namespaced["rl:__vlm_empty__"] = 1.0

        return RealLifeClassification(
            path=path,
            scores=namespaced,
            primary_tag=primary,
            primary_score=primary_score,
            secondary=secondary,
            needs_review=needs_review,
            reason=";".join(dict.fromkeys(reasons)) or "needs_review",
            style_bucket=style.bucket,
            evidence=evidence,
            suggested_destination=dest,
        )


def get_real_life_engine() -> RealLifeEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = RealLifeEngine()
        return _ENGINE


def reset_real_life_engine() -> list[str]:
    """Drop VLM / position adapters. Returns ids that were resident."""
    global _ENGINE
    with _ENGINE_LOCK:
        engine = _ENGINE
        _ENGINE = None
    dropped: list[str] = []
    if engine is None:
        return dropped
    vlm = getattr(engine, "vlm", None)
    position = getattr(engine, "position", None)
    if vlm is not None and getattr(vlm, "_llm", None) is not None:
        dropped.append("real_life_vlm")
    if position is not None and getattr(position, "_model", None) is not None:
        dropped.append("real_life_position")
    for adapter in (vlm, position):
        closer = getattr(adapter, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                logger.exception("real_life_adapter_close_failed")
    return dropped


def set_real_life_adapters(
    *,
    vlm: VlmAdapter | None = None,
    position: PositionAdapter | None = None,
    style_detector: Callable[..., StylePrediction] | None = None,
) -> RealLifeEngine:
    """Test helper: inject mock adapters and replace the singleton."""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = RealLifeEngine(vlm=vlm, position=position, style_detector=style_detector)
        return _ENGINE


def pool_frame_scores(per_frame: list[dict[str, float]]) -> dict[str, float]:
    """Max-pool then lightly damp tags that appear on only one frame."""
    if not per_frame:
        return {}
    totals: dict[str, list[float]] = {}
    for scores in per_frame:
        for key, value in scores.items():
            totals.setdefault(str(key), []).append(float(value))
    pooled: dict[str, float] = {}
    n = len(per_frame)
    for key, values in totals.items():
        peak = max(values)
        support = len(values) / max(1, n)
        pooled[key] = peak if support >= 0.34 else peak * 0.7
    return pooled
