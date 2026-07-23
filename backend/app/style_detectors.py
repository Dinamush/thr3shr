"""Real vs anime style detectors (debug compare + experimental classify gate)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

logger = logging.getLogger(__name__)

# Map detector labels onto realism_eval buckets (photo = real people).
BUCKET_PHOTO = "photo"
BUCKET_ANIME = "anime"
BUCKET_UNCERTAIN = "uncertain"

DEFAULT_PRODUCTION_STYLE_MODEL = "caformer_s36_v1.4"
DEFAULT_UNCERTAIN_THRESHOLD = 0.85
REAL_LIFE_FOLDER = "real_life"

# Hybrid WD realism + style-detector blend (real_life filter mode).
HYBRID_WD_WEIGHT = 0.45
HYBRID_STYLE_WEIGHT = 0.55
# Tuned for WD SwinV2 photo signal ≈0.2+ blended with CAFormer real score.
HYBRID_ACCEPT_THRESHOLD = 0.32
# Strong anime veto even if WD realism is moderately elevated.
HYBRID_ANIME_VETO = 0.75
# Style-first cascade: skip WD entirely when CAFormer is clearly anime.
STYLE_EARLY_REJECT_ANIME = 0.85
STYLE_EARLY_REJECT_REAL_MAX = 0.25
# Cap multi-frame WD sampling in real_life filter mode (speed).
FILTER_MEDIA_SAMPLE_MAX = 8

ImageInput = Path | Image.Image | str


@dataclass(frozen=True)
class StylePrediction:
    detector_id: str
    method: str
    label: str
    bucket: str
    confidence: float
    scores: dict[str, float]
    detail: dict[str, object]


DetectorFn = Callable[[Path], StylePrediction]


def load_style_probe_image(path: Path) -> ImageInput:
    """Return a still suitable for style classification (GIF/video → first frame)."""
    from .services import VIDEO_EXTENSIONS, sample_gif_frames, sample_video_frames

    suffix = path.suffix.lower()
    try:
        if suffix == ".gif":
            frames = sample_gif_frames(path, sample_count=1)
            if frames:
                return frames[0]
        if suffix in VIDEO_EXTENSIONS:
            frames = sample_video_frames(path, sample_count=1)
            if frames:
                return frames[0]
    except Exception:
        logger.exception("style_probe_frame_failed path=%s", path)
    return path


def _imgutils_predict(
    image: ImageInput,
    *,
    detector_id: str,
    model_name: str,
    uncertain_threshold: float | None = None,
) -> StylePrediction:
    from imgutils.validate import anime_real_score

    scores = {str(k): float(v) for k, v in anime_real_score(image, model_name).items()}
    real_s = float(scores.get("real", 0.0))
    anime_s = float(scores.get("anime", 0.0))
    if real_s >= anime_s:
        label = "real"
        confidence = real_s
        bucket = BUCKET_PHOTO
    else:
        label = "anime"
        confidence = anime_s
        bucket = BUCKET_ANIME

    if uncertain_threshold is not None and confidence < float(uncertain_threshold):
        label = "uncertain"
        bucket = BUCKET_UNCERTAIN

    return StylePrediction(
        detector_id=detector_id,
        method=f"imgutils.anime_real:{model_name}",
        label=label,
        bucket=bucket,
        confidence=confidence,
        scores=scores,
        detail={"model_name": model_name, "uncertain_threshold": uncertain_threshold},
    )


def detect_production_style(
    path: Path,
    *,
    model_name: str = DEFAULT_PRODUCTION_STYLE_MODEL,
    uncertain_threshold: float | None = DEFAULT_UNCERTAIN_THRESHOLD,
) -> StylePrediction:
    """Experimental classify-path detector (CAFormer + uncertain band)."""
    probe = load_style_probe_image(path)
    return _imgutils_predict(
        probe,
        detector_id="imgutils_caformer_band",
        model_name=model_name,
        uncertain_threshold=uncertain_threshold,
    )


def wd_realism_score(scores: dict[str, float] | None) -> float:
    """Max of WD realism tags used for hybrid blending."""
    if not scores:
        return 0.0
    best = 0.0
    for key, value in scores.items():
        name = str(key).lower().replace(" ", "_")
        if name in {"realistic", "photorealistic"}:
            best = max(best, float(value))
    return best


def is_style_anime_early_reject(
    style: StylePrediction,
    *,
    anime_min: float = STYLE_EARLY_REJECT_ANIME,
    real_max: float = STYLE_EARLY_REJECT_REAL_MAX,
) -> bool:
    """True when style detector alone is enough to reject (skip WD)."""
    style_real = float(style.scores.get("real", 0.0))
    style_anime = float(style.scores.get("anime", 0.0))
    return style_anime >= float(anime_min) and style_real < float(real_max)


def blend_real_life_scores(
    wd_scores: dict[str, float] | None,
    style: StylePrediction,
    *,
    wd_weight: float = HYBRID_WD_WEIGHT,
    style_weight: float = HYBRID_STYLE_WEIGHT,
    accept_threshold: float = HYBRID_ACCEPT_THRESHOLD,
    anime_veto: float = HYBRID_ANIME_VETO,
) -> tuple[bool, float, dict[str, float]]:
    """Blend WD realism with style-detector real score.

    Returns (is_real_life, hybrid_score, detail_scores).
    """
    wd = wd_realism_score(wd_scores)
    style_real = float(style.scores.get("real", 0.0))
    style_anime = float(style.scores.get("anime", 0.0))
    total_w = max(1e-6, float(wd_weight) + float(style_weight))
    hybrid = (float(wd_weight) * wd + float(style_weight) * style_real) / total_w
    detail = {
        "wd_realism": wd,
        "style_real": style_real,
        "style_anime": style_anime,
        "hybrid": hybrid,
    }
    if style_anime >= float(anime_veto) and style_real < style_anime:
        return False, hybrid, detail
    # Require style not clearly anime-leaning unless WD is already strong.
    if style_anime > style_real and wd < 0.35:
        return False, hybrid, detail
    # Mid hybrid needs WD corroboration so photoreal-anime edges without
    # realistic/photorealistic tags do not sneak in on style alone.
    accepted = (
        hybrid >= float(accept_threshold)
        and style_real >= (style_anime * 0.85)
        and wd >= 0.15
    )
    # Strong style photo can carry a weak WD signal (still prefer some WD).
    if style_real >= 0.85 and style_real > style_anime and wd >= 0.10:
        accepted = True
        hybrid = max(hybrid, style_real)
        detail["hybrid"] = hybrid
    # Very strong style with near-zero WD still counts (clear photo portraits).
    elif style_real >= 0.92 and style_real > style_anime:
        accepted = True
        hybrid = max(hybrid, style_real)
        detail["hybrid"] = hybrid
    return accepted, hybrid, detail


def predict_wd_taxonomy(
    path: Path,
    *,
    tagger_model: str,
    wd_general_threshold: float,
    selected: set[str],
) -> StylePrediction:
    from .realism_eval import predict_real_life
    from .services import extract_scores

    scores = extract_scores(
        path,
        tagger_model=tagger_model,
        wd_general_threshold=float(wd_general_threshold),
    )
    is_rl, folder, folder_score, evidence = predict_real_life(scores, selected=selected)
    bucket = BUCKET_PHOTO if is_rl else BUCKET_ANIME
    label = "real" if is_rl else "anime"
    confidence = float(folder_score or 0.0)
    top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    return StylePrediction(
        detector_id="wd_taxonomy",
        method=f"wd_taxonomy:{tagger_model}",
        label=label,
        bucket=bucket,
        confidence=confidence,
        scores={tag: float(evidence.get(tag, 0.0)) for tag in evidence},
        detail={
            "primary_folder": folder,
            "primary_score": folder_score,
            "global_top_tags": [{"tag": t, "score": float(s)} for t, s in top],
        },
    )


def list_style_detectors() -> list[dict[str, object]]:
    return [
        {
            "id": "wd_taxonomy",
            "label": "WD taxonomy real_life",
            "method": "tagger + taxonomy priority 0",
            "needs_tagger": True,
        },
        {
            "id": "imgutils_mobilenet",
            "label": "imgutils MobileNetV3",
            "method": "deepghs/anime_real_cls mobilenetv3_v1.4_dist",
            "needs_tagger": False,
        },
        {
            "id": "imgutils_caformer",
            "label": "imgutils CAFormer-S36",
            "method": "deepghs/anime_real_cls caformer_s36_v1.4",
            "needs_tagger": False,
        },
        {
            "id": "imgutils_caformer_band",
            "label": "imgutils CAFormer + uncertain band",
            "method": "caformer_s36_v1.4 with low-confidence → uncertain",
            "needs_tagger": False,
        },
    ]


DEFAULT_STYLE_DETECTOR_IDS = [
    "wd_taxonomy",
    "imgutils_mobilenet",
    "imgutils_caformer",
    "imgutils_caformer_band",
]


def build_detector(
    detector_id: str,
    *,
    tagger_model: str,
    wd_general_threshold: float,
    selected: set[str],
    uncertain_threshold: float = 0.85,
) -> DetectorFn:
    if detector_id == "wd_taxonomy":
        def _wd(path: Path) -> StylePrediction:
            return predict_wd_taxonomy(
                path,
                tagger_model=tagger_model,
                wd_general_threshold=wd_general_threshold,
                selected=selected,
            )

        return _wd

    if detector_id == "imgutils_mobilenet":
        def _mobile(path: Path) -> StylePrediction:
            return _imgutils_predict(
                path,
                detector_id=detector_id,
                model_name="mobilenetv3_v1.4_dist",
            )

        return _mobile

    if detector_id == "imgutils_caformer":
        def _caf(path: Path) -> StylePrediction:
            return _imgutils_predict(
                path,
                detector_id=detector_id,
                model_name="caformer_s36_v1.4",
            )

        return _caf

    if detector_id == "imgutils_caformer_band":
        def _band(path: Path) -> StylePrediction:
            return _imgutils_predict(
                path,
                detector_id=detector_id,
                model_name="caformer_s36_v1.4",
                uncertain_threshold=uncertain_threshold,
            )

        return _band

    raise ValueError(f"Unknown style detector: {detector_id}")
