from __future__ import annotations

import os
import logging
import random
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from .schemas import (
    AppSettings,
    BatchUpdateRequest,
    ClassifiedItem,
    MigrateRequest,
    MigrateResponse,
    ReclassifyRequest,
    ReclassifyResponse,
    RealismDebugEvalRequest,
    RealismDebugEvalResponse,
    StyleDebugEvalRequest,
    TagFpEvalRequest,
    TagRecallEvalRequest,
    RunStatusResponse,
    SaveSettingsRequest,
    SfwDebugEvalRequest,
    SfwDebugEvalResponse,
    StartRunRequest,
    StartRunResponse,
    UpdateItemRequest,
)
from .services import (
    VIDEO_EXTENSIONS,
    destination_path,
    discover_tag_folders,
    extract_scores,
    extract_scores_batch,
    extract_scores_with_experimental_media,
    global_top_tags,
    is_experimental_media,
    load_known_tags,
    media_preview_still_jpeg,
    migrate_file,
    normalize_tag_name,
    resolve_settings,
    sanitize_folder_name,
    categories_exclude_dirs,
    scan_images,
)
from .taxonomy import (
    bucket_role_for_folder,
    choose_best_destination,
    resolve_taxonomy_folder,
    taxonomy_folder_names,
)
from .hybrid_ml import merge_ml_allowlist_scores, should_run_hybrid_ml
from .providers import probe_execution_providers
from .storage import execute, fetch_all, fetch_one, from_json, to_json

router = APIRouter(prefix="/api")
REPO_ROOT = Path(__file__).resolve().parents[2]
TAGS_CSV = REPO_ROOT / "tags.csv"
logger = logging.getLogger(__name__)
SUPPORTED_PREVIEW_SUFFIXES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}
_RUN_TELEMETRY: dict[int, dict[str, object]] = {}
_RUN_TELEMETRY_LOCK = threading.Lock()


@dataclass
class _ImageInferenceResult:
    image_path: Path
    scores: dict[str, float]
    primary_tag: str | None
    primary_score: float | None
    secondary: list[dict[str, float]]
    needs_review: bool
    reason: str | None
    inference_failed: bool


def _assignment_noise_floor(confidence_threshold: float) -> float:
    return max(0.15, float(confidence_threshold) * 0.5)


def _maybe_hybrid_ml_rescue(
    result: _ImageInferenceResult,
    matched_tags: set[str],
    confidence_threshold: float,
    experimental_media_enabled: bool,
    tagger_model: str,
    wd_general_threshold: float,
    experimental_style_detector_enabled: bool,
    hybrid_ml_on_review: bool,
) -> _ImageInferenceResult:
    """On WD needs_review, merge allowlisted ML scores and re-route."""
    if not should_run_hybrid_ml(
        enabled=hybrid_ml_on_review,
        tagger_model=tagger_model,
        needs_review=result.needs_review,
        inference_failed=result.inference_failed,
    ):
        return result
    try:
        if experimental_media_enabled and is_experimental_media(result.image_path):
            ml_scores = extract_scores_with_experimental_media(
                result.image_path,
                experimental_media_enabled,
                tagger_model="ml_danbooru",
                wd_general_threshold=wd_general_threshold,
            )
        else:
            ml_scores = extract_scores(
                result.image_path,
                tagger_model="ml_danbooru",
                wd_general_threshold=wd_general_threshold,
            )
        merged = merge_ml_allowlist_scores(result.scores, ml_scores)
        rescued = _classify_from_scores(
            result.image_path,
            merged,
            matched_tags,
            confidence_threshold,
            experimental_style_detector_enabled=experimental_style_detector_enabled,
            hybrid_real_life=False,
        )
        note = "Hybrid ML allowlist rescue."
        if rescued.reason:
            rescued.reason = f"{rescued.reason} {note}"
        elif rescued.needs_review != result.needs_review or rescued.primary_tag != result.primary_tag:
            rescued.reason = note
        return rescued
    except Exception:
        logger.exception("hybrid_ml_rescue_failed image=%s", result.image_path)
        return result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_max_inference_workers(settings_workers: int | None = None) -> int:
    if settings_workers is not None:
        return max(1, min(int(settings_workers), 16))
    raw = os.getenv("MAX_INFERENCE_WORKERS", "2").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 2
    return max(1, min(value, 16))


def _get_inference_mode() -> str:
    raw = os.getenv("INFERENCE_MODE", "batch").strip().lower()
    return "single" if raw == "single" else "batch"


def _get_inference_batch_size(settings_batch: int | None = None) -> int:
    # WD true-batches inside InferenceEngine; default stays 1 unless settings raise it.
    if settings_batch is not None:
        return max(1, min(int(settings_batch), 64))
    raw = os.getenv("INFERENCE_BATCH_SIZE", "1").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, min(value, 64))


def _get_queue_shuffle_enabled() -> bool:
    raw = os.getenv("QUEUE_SHUFFLE_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _get_queue_shuffle_seed(run_id: int) -> int:
    raw = os.getenv("QUEUE_SHUFFLE_SEED", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return int(run_id)


def _set_run_telemetry(run_id: int, **kwargs) -> None:
    with _RUN_TELEMETRY_LOCK:
        telemetry = _RUN_TELEMETRY.get(run_id, {})
        telemetry.update(kwargs)
        _RUN_TELEMETRY[run_id] = telemetry


def _get_run_telemetry(run_id: int) -> dict[str, object]:
    with _RUN_TELEMETRY_LOCK:
        return dict(_RUN_TELEMETRY.get(run_id, {}))


def _is_provider_related_error(err: Exception) -> bool:
    msg = str(err).lower()
    keywords = ("cuda", "cudnn", "executionprovider", "provider", "onnxruntime", "gpu")
    return any(k in msg for k in keywords)


def _classify_from_scores(
    image_path: Path,
    scores: dict[str, float],
    matched_tags: set[str],
    confidence_threshold: float,
    experimental_style_detector_enabled: bool = False,
    hybrid_real_life: bool = False,
) -> _ImageInferenceResult:
    primary_tag, primary_score, secondary = choose_best_destination(scores, matched_tags)
    needs_review = False
    reason = None
    if not scores:
        return _ImageInferenceResult(
            image_path=image_path,
            scores=scores,
            primary_tag=None,
            primary_score=None,
            secondary=[],
            needs_review=True,
            reason="Inference returned no tag scores for this image.",
            inference_failed=True,
        )
    if primary_tag is None:
        needs_review = True
        reason = "No matching tags found among selected tags."
    elif primary_score is not None:
        noise_floor = _assignment_noise_floor(confidence_threshold)
        if primary_score < noise_floor:
            needs_review = True
            reason = (
                f"Below noise floor ({primary_score:.3f} < {noise_floor:.3f}); "
                "no reliable selected-tag match."
            )
            secondary = [{"tag": primary_tag, "score": float(primary_score)}, *secondary][:4]
            primary_tag = None
            primary_score = None
        elif primary_score < confidence_threshold:
            needs_review = True
            reason = f"Below threshold ({primary_score:.3f} < {confidence_threshold:.3f})."
            # Character folders keep a mid-confidence primary for faster review approve.
            # Act/theme/other weak winners stay suggestion-only (cleared from primary).
            if bucket_role_for_folder(primary_tag) != "character":
                secondary = [
                    {"tag": primary_tag, "score": float(primary_score)},
                    *secondary,
                ][:4]
                primary_tag = None
                primary_score = None
    result = _ImageInferenceResult(
        image_path=image_path,
        scores=scores,
        primary_tag=primary_tag,
        primary_score=primary_score,
        secondary=secondary,
        needs_review=needs_review,
        reason=reason,
        inference_failed=False,
    )
    if hybrid_real_life:
        return _apply_hybrid_real_life_filter(result)
    if experimental_style_detector_enabled:
        return _apply_experimental_style_gate(result, matched_tags)
    return result


def _hybrid_reject(
    image_path: Path,
    scores: dict[str, float] | None,
    reason: str,
) -> _ImageInferenceResult:
    return _ImageInferenceResult(
        image_path=image_path,
        scores=scores or {},
        primary_tag=None,
        primary_score=None,
        secondary=[],
        needs_review=False,
        reason=reason,
        inference_failed=False,
    )


def _apply_hybrid_real_life_filter(
    result: _ImageInferenceResult,
    style=None,
) -> _ImageInferenceResult:
    """Blend WD realism tags with experimental real-vs-anime detector scores."""
    from .style_detectors import (
        REAL_LIFE_FOLDER,
        blend_real_life_scores,
        detect_production_style,
        is_style_anime_early_reject,
    )

    if result.inference_failed:
        return result
    if style is None:
        try:
            # Raw scores (no uncertain band) so weak-but-useful style signal can blend.
            style = detect_production_style(result.image_path, uncertain_threshold=None)
        except Exception:
            logger.exception("hybrid_style_detector_failed path=%s", result.image_path)
            return _ImageInferenceResult(
                image_path=result.image_path,
                scores=result.scores,
                primary_tag=None,
                primary_score=None,
                secondary=result.secondary,
                needs_review=True,
                reason="Hybrid style detector failed; skipped for real_life filter.",
                inference_failed=False,
            )

    if is_style_anime_early_reject(style) and not result.scores:
        # Style-first path already rejected; keep a clear reason.
        return _hybrid_reject(
            result.image_path,
            {},
            (
                f"style_early_reject "
                f"(style_real={float(style.scores.get('real', 0.0)):.3f}, "
                f"style_anime={float(style.scores.get('anime', 0.0)):.3f})"
            ),
        )

    accepted, hybrid, detail = blend_real_life_scores(result.scores, style)
    note = (
        f"hybrid={hybrid:.3f} "
        f"(wd={detail['wd_realism']:.3f}, style_real={detail['style_real']:.3f}, "
        f"style_anime={detail['style_anime']:.3f})"
    )
    if not accepted:
        return _hybrid_reject(result.image_path, result.scores, f"Not real_life ({note})")
    secondary = list(result.secondary or [])
    if result.primary_tag and result.primary_tag != REAL_LIFE_FOLDER:
        secondary = [
            {"tag": result.primary_tag, "score": float(result.primary_score or 0.0)},
            *secondary,
        ][:4]
    return _ImageInferenceResult(
        image_path=result.image_path,
        scores=result.scores,
        primary_tag=REAL_LIFE_FOLDER,
        primary_score=float(hybrid),
        secondary=secondary,
        needs_review=False,
        reason=note,
        inference_failed=False,
    )


def _extract_scores_for_hybrid(
    image_path: Path,
    *,
    experimental_media_enabled: bool,
    tagger_model: str,
    wd_general_threshold: float,
) -> dict[str, float]:
    from .style_detectors import FILTER_MEDIA_SAMPLE_MAX

    if experimental_media_enabled and is_experimental_media(image_path):
        return extract_scores_with_experimental_media(
            image_path,
            experimental_media_enabled,
            tagger_model=tagger_model,
            wd_general_threshold=wd_general_threshold,
            sample_count=FILTER_MEDIA_SAMPLE_MAX,
        )
    return extract_scores(
        image_path,
        tagger_model=tagger_model,
        wd_general_threshold=wd_general_threshold,
    )


def _infer_hybrid_one_image(
    image_path: Path,
    matched_tags: set[str],
    confidence_threshold: float,
    experimental_media_enabled: bool,
    tagger_model: str,
    wd_general_threshold: float,
) -> _ImageInferenceResult:
    """Style-first hybrid: skip WD when CAFormer is strongly anime."""
    from .style_detectors import detect_production_style, is_style_anime_early_reject

    try:
        style = detect_production_style(image_path, uncertain_threshold=None)
    except Exception:
        logger.exception("hybrid_style_detector_failed path=%s", image_path)
        return _ImageInferenceResult(
            image_path=image_path,
            scores={},
            primary_tag=None,
            primary_score=None,
            secondary=[],
            needs_review=True,
            reason="Hybrid style detector failed; skipped for real_life filter.",
            inference_failed=False,
        )

    if is_style_anime_early_reject(style):
        return _hybrid_reject(
            image_path,
            {},
            (
                f"style_early_reject "
                f"(style_real={float(style.scores.get('real', 0.0)):.3f}, "
                f"style_anime={float(style.scores.get('anime', 0.0)):.3f})"
            ),
        )

    try:
        scores = _extract_scores_for_hybrid(
            image_path,
            experimental_media_enabled=experimental_media_enabled,
            tagger_model=tagger_model,
            wd_general_threshold=wd_general_threshold,
        )
    except Exception as err:
        if _is_provider_related_error(err):
            logger.exception("inference_provider_failure image=%s", image_path)
        else:
            logger.exception("inference_failed image=%s", image_path)
        return _ImageInferenceResult(
            image_path=image_path,
            scores={},
            primary_tag=None,
            primary_score=None,
            secondary=[],
            needs_review=True,
            reason="Inference failed for this image; requires manual review.",
            inference_failed=True,
        )

    base = _classify_from_scores(
        image_path,
        scores,
        matched_tags,
        confidence_threshold,
        hybrid_real_life=False,
    )
    return _apply_hybrid_real_life_filter(base, style=style)


def _infer_hybrid_batch(
    image_paths: list[Path],
    matched_tags: set[str],
    confidence_threshold: float,
    experimental_media_enabled: bool,
    tagger_model: str,
    wd_general_threshold: float,
) -> list[_ImageInferenceResult]:
    """Style-first batch: early-reject anime, WD-batch still survivors only."""
    from .style_detectors import detect_production_style, is_style_anime_early_reject

    results: list[_ImageInferenceResult | None] = [None] * len(image_paths)
    survivors: list[tuple[int, Path, object]] = []

    for idx, path in enumerate(image_paths):
        try:
            style = detect_production_style(path, uncertain_threshold=None)
        except Exception:
            logger.exception("hybrid_style_detector_failed path=%s", path)
            results[idx] = _ImageInferenceResult(
                image_path=path,
                scores={},
                primary_tag=None,
                primary_score=None,
                secondary=[],
                needs_review=True,
                reason="Hybrid style detector failed; skipped for real_life filter.",
                inference_failed=False,
            )
            continue
        if is_style_anime_early_reject(style):
            results[idx] = _hybrid_reject(
                path,
                {},
                (
                    f"style_early_reject "
                    f"(style_real={float(style.scores.get('real', 0.0)):.3f}, "
                    f"style_anime={float(style.scores.get('anime', 0.0)):.3f})"
                ),
            )
            continue
        survivors.append((idx, path, style))

    stills = [(i, p, s) for i, p, s in survivors if not is_experimental_media(p)]
    media = [(i, p, s) for i, p, s in survivors if is_experimental_media(p)]

    if stills:
        still_paths = [p for _i, p, _s in stills]
        try:
            scores_list = extract_scores_batch(
                still_paths,
                tagger_model=tagger_model,
                wd_general_threshold=wd_general_threshold,
            )
            if len(scores_list) != len(still_paths):
                raise RuntimeError("Batch inference result count mismatch")
            for (idx, path, style), scores in zip(stills, scores_list):
                base = _classify_from_scores(
                    path,
                    scores,
                    matched_tags,
                    confidence_threshold,
                    hybrid_real_life=False,
                )
                results[idx] = _apply_hybrid_real_life_filter(base, style=style)
        except Exception:
            logger.exception("hybrid_still_batch_failed n=%d", len(stills))
            for idx, path, style in stills:
                results[idx] = _infer_hybrid_one_image(
                    path,
                    matched_tags,
                    confidence_threshold,
                    experimental_media_enabled,
                    tagger_model,
                    wd_general_threshold,
                )

    for idx, path, style in media:
        try:
            scores = _extract_scores_for_hybrid(
                path,
                experimental_media_enabled=experimental_media_enabled,
                tagger_model=tagger_model,
                wd_general_threshold=wd_general_threshold,
            )
            base = _classify_from_scores(
                path,
                scores,
                matched_tags,
                confidence_threshold,
                hybrid_real_life=False,
            )
            results[idx] = _apply_hybrid_real_life_filter(base, style=style)
        except Exception:
            logger.exception("hybrid_media_failed path=%s", path)
            results[idx] = _ImageInferenceResult(
                image_path=path,
                scores={},
                primary_tag=None,
                primary_score=None,
                secondary=[],
                needs_review=True,
                reason="Inference failed for this image; requires manual review.",
                inference_failed=True,
            )

    return [r if r is not None else _hybrid_reject(p, {}, "hybrid_internal_miss") for r, p in zip(results, image_paths)]


def _apply_experimental_style_gate(
    result: _ImageInferenceResult,
    matched_tags: set[str],
) -> _ImageInferenceResult:
    """Override taxonomy routing when dedicated real-vs-anime detector fires."""
    from .style_detectors import (
        BUCKET_ANIME,
        BUCKET_UNCERTAIN,
        REAL_LIFE_FOLDER,
        detect_production_style,
    )

    if result.inference_failed:
        return result
    try:
        style = detect_production_style(result.image_path)
    except Exception:
        logger.exception("experimental_style_detector_failed path=%s", result.image_path)
        return _ImageInferenceResult(
            image_path=result.image_path,
            scores=result.scores,
            primary_tag=result.primary_tag,
            primary_score=result.primary_score,
            secondary=result.secondary,
            needs_review=True,
            reason=(
                (result.reason + " · " if result.reason else "")
                + "Experimental style detector failed; manual review required."
            ),
            inference_failed=False,
        )

    style_note = f"style={style.label} ({style.confidence:.3f} via {style.method})"
    if style.bucket == BUCKET_ANIME:
        if result.needs_review and result.reason:
            return _ImageInferenceResult(
                image_path=result.image_path,
                scores=result.scores,
                primary_tag=result.primary_tag,
                primary_score=result.primary_score,
                secondary=result.secondary,
                needs_review=True,
                reason=f"{result.reason} · {style_note}",
                inference_failed=False,
            )
        return result

    if style.bucket == BUCKET_UNCERTAIN:
        secondary = list(result.secondary or [])
        if result.primary_tag is not None and result.primary_score is not None:
            secondary = [
                {"tag": result.primary_tag, "score": float(result.primary_score)},
                *secondary,
            ][:4]
        return _ImageInferenceResult(
            image_path=result.image_path,
            scores=result.scores,
            primary_tag=None,
            primary_score=None,
            secondary=secondary,
            needs_review=True,
            reason=(
                "Experimental style detector uncertain "
                f"(real={style.scores.get('real', 0):.3f}, "
                f"anime={style.scores.get('anime', 0):.3f}); manual review."
            ),
            inference_failed=False,
        )

    has_real_life = REAL_LIFE_FOLDER in matched_tags or any(
        str(t).lower().replace(" ", "_") in {REAL_LIFE_FOLDER, "photo"}
        for t in matched_tags
    )
    if not has_real_life:
        return _ImageInferenceResult(
            image_path=result.image_path,
            scores=result.scores,
            primary_tag=None,
            primary_score=None,
            secondary=result.secondary,
            needs_review=True,
            reason=(
                f"Experimental style detector: real photo ({style.confidence:.3f}) "
                "but real_life is not among selected destinations."
            ),
            inference_failed=False,
        )

    secondary = list(result.secondary or [])
    if result.primary_tag and result.primary_tag != REAL_LIFE_FOLDER:
        secondary = [
            {"tag": result.primary_tag, "score": float(result.primary_score or 0.0)},
            *secondary,
        ][:4]
    return _ImageInferenceResult(
        image_path=result.image_path,
        scores=result.scores,
        primary_tag=REAL_LIFE_FOLDER,
        primary_score=float(style.confidence),
        secondary=secondary,
        needs_review=False,
        reason=None,
        inference_failed=False,
    )


def _infer_one_image(
    image_path: Path,
    matched_tags: set[str],
    confidence_threshold: float,
    experimental_media_enabled: bool = False,
    tagger_model: str = "wd_swinv2_v3",
    wd_general_threshold: float = 0.35,
    experimental_style_detector_enabled: bool = False,
    hybrid_real_life: bool = False,
    hybrid_ml_on_review: bool = False,
) -> _ImageInferenceResult:
    if hybrid_real_life:
        return _infer_hybrid_one_image(
            image_path,
            matched_tags,
            confidence_threshold,
            experimental_media_enabled,
            tagger_model,
            wd_general_threshold,
        )
    try:
        if experimental_media_enabled and is_experimental_media(image_path):
            # Normal classify/reclassify: duration-scaled frames (up to ~24).
            # real_life filter mode caps via _extract_scores_for_hybrid instead.
            scores = extract_scores_with_experimental_media(
                image_path,
                experimental_media_enabled,
                tagger_model=tagger_model,
                wd_general_threshold=wd_general_threshold,
            )
        else:
            scores = extract_scores(
                image_path,
                tagger_model=tagger_model,
                wd_general_threshold=wd_general_threshold,
            )
        result = _classify_from_scores(
            image_path,
            scores,
            matched_tags,
            confidence_threshold,
            experimental_style_detector_enabled=experimental_style_detector_enabled,
            hybrid_real_life=False,
        )
        return _maybe_hybrid_ml_rescue(
            result,
            matched_tags,
            confidence_threshold,
            experimental_media_enabled,
            tagger_model,
            wd_general_threshold,
            experimental_style_detector_enabled,
            hybrid_ml_on_review,
        )
    except Exception as err:
        if _is_provider_related_error(err):
            logger.exception("inference_provider_failure image=%s", image_path)
        else:
            logger.exception("inference_failed image=%s", image_path)
        return _ImageInferenceResult(
            image_path=image_path,
            scores={},
            primary_tag=None,
            primary_score=None,
            secondary=[],
            needs_review=True,
            reason="Inference failed for this image; requires manual review.",
            inference_failed=True,
        )


def _infer_batch_with_fallback(
    image_paths: list[Path],
    matched_tags: set[str],
    confidence_threshold: float,
    requested_mode: str,
    experimental_media_enabled: bool = False,
    tagger_model: str = "wd_swinv2_v3",
    wd_general_threshold: float = 0.35,
    experimental_style_detector_enabled: bool = False,
    hybrid_real_life: bool = False,
    hybrid_ml_on_review: bool = False,
) -> tuple[list[_ImageInferenceResult], float, str]:
    if not image_paths:
        return [], 0.0, "none"

    if hybrid_real_life:
        start = time.perf_counter()
        if requested_mode != "batch" or len(image_paths) == 1:
            rows = [
                _infer_hybrid_one_image(
                    image_paths[0],
                    matched_tags,
                    confidence_threshold,
                    experimental_media_enabled,
                    tagger_model,
                    wd_general_threshold,
                )
            ]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            mode = "single" if requested_mode == "single" else "hybrid_single"
            return rows, elapsed_ms, mode
        rows = _infer_hybrid_batch(
            image_paths,
            matched_tags,
            confidence_threshold,
            experimental_media_enabled,
            tagger_model,
            wd_general_threshold,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return rows, elapsed_ms, "hybrid_batch"

    if requested_mode != "batch" or len(image_paths) == 1:
        start = time.perf_counter()
        rows = [
            _infer_one_image(
                image_paths[0],
                matched_tags,
                confidence_threshold,
                experimental_media_enabled,
                tagger_model,
                wd_general_threshold,
                experimental_style_detector_enabled,
                False,
                hybrid_ml_on_review,
            )
        ]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        mode = "single" if requested_mode == "single" else "single_fallback"
        return rows, elapsed_ms, mode

    # Batch inference currently supports image files only.
    if any(is_experimental_media(p) for p in image_paths):
        start = time.perf_counter()
        rows = [
            _infer_one_image(
                p,
                matched_tags,
                confidence_threshold,
                experimental_media_enabled,
                tagger_model,
                wd_general_threshold,
                experimental_style_detector_enabled,
                False,
                hybrid_ml_on_review,
            )
            for p in image_paths
        ]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return rows, elapsed_ms, "single_fallback"

    start = time.perf_counter()
    try:
        scores_by_image = extract_scores_batch(
            image_paths,
            tagger_model=tagger_model,
            wd_general_threshold=wd_general_threshold,
        )
        if len(scores_by_image) != len(image_paths):
            raise RuntimeError("Batch inference result count mismatch")
        rows = [
            _maybe_hybrid_ml_rescue(
                _classify_from_scores(
                    image_path,
                    scores,
                    matched_tags,
                    confidence_threshold,
                    experimental_style_detector_enabled=experimental_style_detector_enabled,
                    hybrid_real_life=False,
                ),
                matched_tags,
                confidence_threshold,
                experimental_media_enabled,
                tagger_model,
                wd_general_threshold,
                experimental_style_detector_enabled,
                hybrid_ml_on_review,
            )
            for image_path, scores in zip(image_paths, scores_by_image)
        ]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return rows, elapsed_ms, "batch"
    except Exception as err:
        if _is_provider_related_error(err):
            logger.exception("batch_inference_provider_failure batch_size=%d", len(image_paths))
        else:
            logger.exception("batch_inference_failed batch_size=%d", len(image_paths))
        if len(image_paths) > 1:
            mid = len(image_paths) // 2
            left_rows, left_ms, _ = _infer_batch_with_fallback(
                image_paths[:mid],
                matched_tags,
                confidence_threshold,
                "batch",
                experimental_media_enabled,
                tagger_model,
                wd_general_threshold,
                experimental_style_detector_enabled,
                False,
                hybrid_ml_on_review,
            )
            right_rows, right_ms, _ = _infer_batch_with_fallback(
                image_paths[mid:],
                matched_tags,
                confidence_threshold,
                "batch",
                experimental_media_enabled,
                tagger_model,
                wd_general_threshold,
                experimental_style_detector_enabled,
                False,
                hybrid_ml_on_review,
            )
            return left_rows + right_rows, left_ms + right_ms, "batch_fallback"
        row = _infer_one_image(
            image_paths[0],
            matched_tags,
            confidence_threshold,
            experimental_media_enabled,
            tagger_model,
            wd_general_threshold,
            experimental_style_detector_enabled,
            False,
            hybrid_ml_on_review,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return [row], elapsed_ms, "single_fallback"


def _settings_from_db() -> AppSettings:
    row = fetch_one("SELECT * FROM settings WHERE id = 1")
    if not row:
        return AppSettings()
    selected_raw = row.get("selected_tags_json") or "[]"
    selected_tags = from_json(selected_raw, default=[])
    if not isinstance(selected_tags, list):
        selected_tags = []
    selected_tags = [str(t).strip() for t in selected_tags if str(t).strip()]
    tagger_model = str(row.get("tagger_model") or "wd_swinv2_v3").strip()
    if tagger_model not in {"ml_danbooru", "wd_swinv2_v3", "wd_eva02_large"}:
        tagger_model = "wd_swinv2_v3"
    return AppSettings(
        root_repo=row["root_repo"],
        categories_root=row["categories_root"],
        confidence_threshold=float(row["confidence_threshold"]),
        default_migrate_mode=row["default_migrate_mode"],
        scan_recursive=bool(row.get("scan_recursive", 1)),
        experimental_media_enabled=bool(row.get("experimental_media_enabled", 0)),
        experimental_style_detector_enabled=bool(
            row.get("experimental_style_detector_enabled", 0)
        ),
        hybrid_ml_on_review=bool(row.get("hybrid_ml_on_review", 1)),
        selected_tags=selected_tags,
        max_inference_workers=int(row.get("max_inference_workers") or 2),
        inference_batch_size=int(row.get("inference_batch_size") or 4),
        force_cpu_inference=bool(row.get("force_cpu_inference", 0)),
        tagger_model=tagger_model,  # type: ignore[arg-type]
        wd_general_threshold=float(row.get("wd_general_threshold") or 0.35),
    )


def _apply_runtime_inference_env(settings: AppSettings) -> None:
    """Mirror persisted settings into env knobs used by the run executor."""
    os.environ["MAX_INFERENCE_WORKERS"] = str(settings.max_inference_workers)
    os.environ["INFERENCE_BATCH_SIZE"] = str(settings.inference_batch_size)
    # Settings batch size is authoritative. A stuck INFERENCE_MODE=single from an
    # earlier batch=1 save must not keep future runs in single-image mode.
    os.environ["INFERENCE_MODE"] = (
        "batch" if settings.inference_batch_size > 1 else "single"
    )
    prev_force = os.environ.get("FORCE_CPU_INFERENCE")
    if settings.force_cpu_inference:
        os.environ["FORCE_CPU_INFERENCE"] = "true"
        os.environ["ONNX_MODE"] = "cpu"
    else:
        os.environ.pop("FORCE_CPU_INFERENCE", None)
        os.environ.pop("ONNX_MODE", None)
    new_force = os.environ.get("FORCE_CPU_INFERENCE")
    if prev_force != new_force:
        from .inference_engine import reset_engine

        reset_engine()


def _item_from_row(row: dict, include_full_scores: bool = False) -> ClassifiedItem:
    scores = from_json(row.get("full_scores_json") or "{}", default={})
    if not isinstance(scores, dict):
        scores = {}
    return ClassifiedItem(
        id=row["id"],
        run_id=row["run_id"],
        file_path=row["file_path"],
        relative_path=row["relative_path"],
        primary_tag=row["primary_tag"],
        primary_score=row["primary_score"],
        secondary_suggestions=from_json(row.get("secondary_json") or "[]", default=[]),
        global_top_tags=global_top_tags({str(k): float(v) for k, v in scores.items()}),
        full_scores=scores if include_full_scores else None,
        suggested_destination=row["suggested_destination"],
        final_tag=row["final_tag"],
        final_destination=row["final_destination"],
        status=row["status"],
        needs_review=bool(row["needs_review"]),
        review_reason=row["review_reason"],
        migrated_to=row["migrated_to"],
    )


def estimate_run_eta(
    *,
    status: str,
    total_images: int,
    processed_images: int,
    started_at: str | None,
    avg_infer_ms_per_image: float | None,
    now: datetime | None = None,
) -> tuple[float | None, str | None]:
    """Estimate remaining seconds and UTC finish time for an active run.

    Prefers wall-clock throughput once a few images have finished (accounts for
    parallel workers). Falls back to avg_infer_ms_per_image when needed.
    """
    status_key = (status or "").strip().lower()
    if status_key not in {"pending", "running"}:
        return None, None
    total = max(0, int(total_images))
    processed = max(0, int(processed_images))
    remaining = total - processed
    if total <= 0 or remaining <= 0:
        return None, None

    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    eta_seconds: float | None = None

    if started_at and processed >= 2:
        try:
            started = datetime.fromisoformat(str(started_at))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            elapsed = max(0.0, (now_utc - started.astimezone(timezone.utc)).total_seconds())
            if elapsed >= 1.0:
                rate = processed / elapsed  # images / second (wall clock)
                if rate > 0:
                    eta_seconds = remaining / rate
        except ValueError:
            eta_seconds = None

    if eta_seconds is None and avg_infer_ms_per_image is not None:
        avg_ms = float(avg_infer_ms_per_image)
        if avg_ms > 0:
            eta_seconds = remaining * (avg_ms / 1000.0)

    if eta_seconds is None:
        return None, None

    # Clamp absurd spikes early in a run while still allowing long jobs.
    eta_seconds = max(0.0, min(float(eta_seconds), 7 * 24 * 3600))
    finish_at = (now_utc + timedelta(seconds=eta_seconds)).isoformat()
    return eta_seconds, finish_at


def _run_status_from_row(row: dict) -> RunStatusResponse:
    total = int(row.get("total_images") or 0)
    processed = int(row.get("processed_images") or 0)
    pct = 0.0 if total <= 0 else min(100.0, (processed / total) * 100.0)
    item_count_row = fetch_one("SELECT COUNT(*) AS cnt FROM items WHERE run_id = ?", (row["id"],))
    has_items = bool(item_count_row and int(item_count_row["cnt"]) > 0)
    telemetry = _get_run_telemetry(row["id"])
    avg_ms = telemetry.get("avg_infer_ms_per_image")
    eta_seconds, eta_finish_at = estimate_run_eta(
        status=row.get("status") or "pending",
        total_images=total,
        processed_images=processed,
        started_at=row.get("started_at"),
        avg_infer_ms_per_image=float(avg_ms) if avg_ms is not None else None,
    )
    return RunStatusResponse(
        run_id=row["id"],
        status=row.get("status") or "pending",
        total_images=total,
        processed_images=processed,
        failed_images=int(row.get("failed_images") or 0),
        progress_pct=pct,
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        last_error=row.get("last_error"),
        cancel_requested=bool(row.get("cancel_requested") or 0),
        has_items=has_items,
        inference_mode=telemetry.get("inference_mode"),
        batch_size=telemetry.get("batch_size"),
        avg_infer_ms_per_image=avg_ms,
        eta_seconds_remaining=eta_seconds,
        eta_finish_at=eta_finish_at,
        queue_seed=telemetry.get("queue_seed"),
        tagger_model=row.get("tagger_model"),
    )


def _is_cancel_requested(run_id: int) -> bool:
    row = fetch_one("SELECT cancel_requested FROM runs WHERE id = ?", (run_id,))
    return bool(row and row.get("cancel_requested"))


def _update_run_progress(run_id: int, processed: int, failed: int) -> None:
    execute(
        "UPDATE runs SET processed_images = ?, failed_images = ? WHERE id = ?",
        (processed, failed, run_id),
    )


def _execute_run(
    run_id: int,
    root_repo: Path,
    categories_root: Path,
    confidence_threshold: float,
    matched_tags: set[str],
    scan_recursive: bool = True,
    experimental_media_enabled: bool = False,
    max_inference_workers: int = 2,
    inference_batch_size: int = 1,
    tagger_model: str = "wd_swinv2_v3",
    wd_general_threshold: float = 0.35,
    experimental_style_detector_enabled: bool = False,
    real_life_filter: bool = False,
    hybrid_ml_on_review: bool = False,
) -> None:
    try:
        # Mark running immediately so clients can cancel during provider probe / scan.
        execute(
            "UPDATE runs SET status = 'running', started_at = ?, last_error = NULL WHERE id = ?",
            (_now_iso(), run_id),
        )
        if _is_cancel_requested(run_id):
            execute("DELETE FROM items WHERE run_id = ?", (run_id,))
            execute(
                """
                UPDATE runs
                SET status = 'cancelled',
                    finished_at = ?,
                    total_images = 0,
                    processed_images = 0,
                    failed_images = 0
                WHERE id = ?
                """,
                (_now_iso(), run_id),
            )
            return
        provider_state = probe_execution_providers()
        logger.info("run_provider_state run_id=%d state=%s", run_id, provider_state)
        if _is_cancel_requested(run_id):
            execute(
                """
                UPDATE runs
                SET status = 'cancelled',
                    finished_at = ?,
                    total_images = 0,
                    processed_images = 0,
                    failed_images = 0
                WHERE id = ?
                """,
                (_now_iso(), run_id),
            )
            return
        scan_output = scan_images(
            root_repo,
            exclude_dirs=categories_exclude_dirs(root_repo, categories_root),
            recursive=scan_recursive,
            experimental_media_enabled=experimental_media_enabled,
        )
        execute("UPDATE runs SET total_images = ? WHERE id = ?", (scan_output.stats.eligible_images, run_id))
        logger.info(
            "run_scan_complete run_id=%d total_files=%d eligible=%d "
            "ignored_gif=%d ignored_unsupported=%d failed_to_read=%d",
            run_id,
            scan_output.stats.total_files,
            scan_output.stats.eligible_images,
            scan_output.stats.ignored_gif,
            scan_output.stats.ignored_unsupported,
            scan_output.stats.failed_to_read,
        )

        queue_shuffle_enabled = _get_queue_shuffle_enabled()
        queue_seed = _get_queue_shuffle_seed(run_id)
        ordered_paths = list(scan_output.image_paths)
        if queue_shuffle_enabled:
            rng = random.Random(queue_seed)
            rng.shuffle(ordered_paths)
        configured_batch_size = _get_inference_batch_size(inference_batch_size)
        # Settings/arg batch size is authoritative for WD (env INFERENCE_MODE is mirrored from it).
        if tagger_model == "ml_danbooru" or configured_batch_size <= 1:
            inference_mode = "single"
            batch_size = 1
        else:
            inference_mode = "batch"
            batch_size = configured_batch_size
        _set_run_telemetry(
            run_id,
            queue_seed=queue_seed,
            inference_mode=inference_mode,
            batch_size=batch_size,
            avg_infer_ms_per_image=0.0,
        )
        logger.info(
            "run_queue_config run_id=%d shuffle=%s seed=%d mode=%s batch_size=%d",
            run_id,
            queue_shuffle_enabled,
            queue_seed,
            inference_mode,
            batch_size,
        )

        processed = 0
        failed = 0
        cancelled = False
        infer_elapsed_ms_total = 0.0
        infer_sample_count = 0
        max_workers = _get_max_inference_workers(max_inference_workers)
        if real_life_filter:
            # Style-first cascade benefits from a bit more CPU overlap.
            max_workers = max(max_workers, 3)
            max_workers = min(max_workers, 4)
        logger.info(
            "run_inference_workers run_id=%d workers=%d total_images=%d device=%s hybrid=%s",
            run_id,
            max_workers,
            scan_output.stats.eligible_images,
            provider_state.get("likely_device"),
            real_life_filter,
        )

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="infer") as executor:
            pending: dict[Future[tuple[list[_ImageInferenceResult], float, str]], list[Path]] = {}
            # Keep stills in true WD batches; process GIF/video one-at-a-time so a
            # single media file cannot collapse an entire batch to single mode.
            if experimental_media_enabled or real_life_filter:
                stills = [p for p in ordered_paths if not is_experimental_media(p)]
                media = [p for p in ordered_paths if is_experimental_media(p)]
                batches = [
                    stills[idx : idx + batch_size]
                    for idx in range(0, len(stills), batch_size)
                ]
                batches.extend([[p] for p in media])
                logger.info(
                    "run_queue_split run_id=%d stills=%d media=%d still_batches=%d",
                    run_id,
                    len(stills),
                    len(media),
                    len(batches) - len(media),
                )
            else:
                batches = [
                    ordered_paths[idx : idx + batch_size]
                    for idx in range(0, len(ordered_paths), batch_size)
                ]
            iterator = iter(batches)

            def _submit_until_capacity() -> None:
                while len(pending) < max_workers:
                    try:
                        next_batch = next(iterator)
                    except StopIteration:
                        return
                    future = executor.submit(
                        _infer_batch_with_fallback,
                        next_batch,
                        matched_tags,
                        confidence_threshold,
                        inference_mode,
                        experimental_media_enabled,
                        tagger_model,
                        wd_general_threshold,
                        experimental_style_detector_enabled,
                        real_life_filter,
                        False if real_life_filter else hybrid_ml_on_review,
                    )
                    pending[future] = next_batch

            _submit_until_capacity()
            while pending:
                if _is_cancel_requested(run_id):
                    cancelled = True
                    for future in pending:
                        future.cancel()
                    break

                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    if future.cancelled():
                        continue
                    batch_results, elapsed_ms, used_mode = future.result()
                    infer_elapsed_ms_total += elapsed_ms
                    infer_sample_count += len(batch_results)
                    avg_ms = (
                        infer_elapsed_ms_total / infer_sample_count if infer_sample_count else 0.0
                    )
                    _set_run_telemetry(
                        run_id,
                        inference_mode=used_mode if used_mode != "single_fallback" else "single",
                        batch_size=batch_size,
                        avg_infer_ms_per_image=avg_ms,
                    )

                    for result in batch_results:
                        relative_path = str(result.image_path.relative_to(root_repo))
                        is_real_life_hit = (
                            result.primary_tag is not None
                            and normalize_tag_name(result.primary_tag)
                            in {"real_life", "photo"}
                            and not result.needs_review
                            and not result.inference_failed
                        )
                        # Filter mode: only persist real_life hits (approved).
                        if real_life_filter and not is_real_life_hit:
                            processed += 1
                            if result.inference_failed:
                                failed += 1
                            _update_run_progress(run_id, processed, failed)
                            continue

                        suggested_destination = (
                            str(destination_path(categories_root, result.primary_tag))
                            if result.primary_tag is not None
                            else None
                        )
                        status = "approved" if not result.needs_review else "proposed"
                        if real_life_filter and is_real_life_hit:
                            status = "approved"
                        execute(
                            """
                            INSERT INTO items (
                                run_id, file_path, relative_path, primary_tag, primary_score, secondary_json,
                                full_scores_json, suggested_destination, final_tag, final_destination, status, needs_review, review_reason
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                run_id,
                                str(result.image_path),
                                relative_path,
                                result.primary_tag,
                                result.primary_score,
                                to_json(result.secondary),
                                to_json(result.scores),
                                suggested_destination,
                                result.primary_tag,
                                suggested_destination,
                                status,
                                1 if result.needs_review and not real_life_filter else 0,
                                result.reason,
                            ),
                        )
                        processed += 1
                        if result.inference_failed:
                            failed += 1
                        _update_run_progress(run_id, processed, failed)
                        if processed % 10 == 0:
                            logger.info(
                                "run_progress run_id=%d processed=%d total=%d avg_infer_ms=%.2f",
                                run_id,
                                processed,
                                scan_output.stats.eligible_images,
                                avg_ms,
                            )
                _submit_until_capacity()

        if cancelled:
            # Reset partial run artifacts so cancelled runs do not look like
            # "missing file" runs with incomplete queues.
            execute("DELETE FROM items WHERE run_id = ?", (run_id,))
            execute(
                """
                UPDATE runs
                SET status = 'cancelled',
                    finished_at = ?,
                    total_images = 0,
                    processed_images = 0,
                    failed_images = 0
                WHERE id = ?
                """,
                (_now_iso(), run_id),
            )
            logger.info("run_cancelled run_id=%d processed=%d", run_id, processed)
            return

        execute(
            "UPDATE runs SET status = 'completed', finished_at = ? WHERE id = ?",
            (_now_iso(), run_id),
        )
        logger.info("run_completed run_id=%d processed=%d failed=%d", run_id, processed, failed)
    except Exception as err:
        logger.exception("run_failed run_id=%d", run_id)
        execute(
            "UPDATE runs SET status = 'failed', finished_at = ?, last_error = ? WHERE id = ?",
            (_now_iso(), str(err), run_id),
        )


def _count_inference_failed_items(run_id: int) -> int:
    rows = fetch_all(
        "SELECT review_reason FROM items WHERE run_id = ? AND needs_review = 1",
        (run_id,),
    )
    failed = 0
    for row in rows:
        reason = str(row.get("review_reason") or "").lower()
        if "inference failed" in reason or "no tag scores" in reason:
            failed += 1
    return failed


def _eligible_reclassify_rows(
    run_id: int, item_ids: list[int] | None = None
) -> list[dict]:
    query = """
        SELECT * FROM items
        WHERE run_id = ? AND needs_review = 1 AND status = 'proposed'
    """
    params: list = [run_id]
    if item_ids:
        placeholders = ",".join("?" for _ in item_ids)
        query += f" AND id IN ({placeholders})"
        params.extend(item_ids)
    query += " ORDER BY id ASC"
    return fetch_all(query, tuple(params))


def _reclassify_review_reason(tagger_model: str, result: _ImageInferenceResult) -> str | None:
    if not result.needs_review:
        return None
    prefix = f"Reclassified with {tagger_model}."
    if result.reason:
        return f"{prefix} {result.reason}"
    return prefix


def _finish_reclassify_run(run_id: int, *, cancelled: bool, eligible: int, failed: int) -> None:
    """Reclassify is a pass over an already-classified run.

    Cancelling the pass must not poison the run into a terminal `cancelled` state
    that blocks migrate / another reclassify — restore `completed` instead.
    """
    execute(
        """
        UPDATE runs
        SET status = 'completed',
            finished_at = ?,
            cancel_requested = 0,
            last_error = CASE
                WHEN ? THEN 'Reclassify cancelled; partial updates kept.'
                ELSE NULL
            END
        WHERE id = ?
        """,
        (_now_iso(), 1 if cancelled else 0, run_id),
    )
    if cancelled:
        logger.info(
            "reclassify_cancelled run_id=%d eligible=%d failed=%d (restored completed)",
            run_id,
            eligible,
            failed,
        )
    else:
        logger.info(
            "reclassify_completed run_id=%d eligible=%d failed=%d",
            run_id,
            eligible,
            failed,
        )


def _execute_reclassify(
    run_id: int,
    root_repo: Path,
    categories_root: Path,
    confidence_threshold: float,
    matched_tags: set[str],
    item_ids: list[int] | None,
    experimental_media_enabled: bool = False,
    max_inference_workers: int = 2,
    inference_batch_size: int = 1,
    tagger_model: str = "wd_eva02_large",
    wd_general_threshold: float = 0.35,
    experimental_style_detector_enabled: bool = False,
    hybrid_ml_on_review: bool = False,
) -> None:
    try:
        # Claim already set status=running; only refresh bookkeeping here.
        # Do NOT clear cancel_requested — a cancel clicked between claim and
        # worker start must still win.
        execute(
            """
            UPDATE runs
            SET status = 'running',
                last_error = NULL,
                finished_at = NULL,
                tagger_model = ?
            WHERE id = ?
            """,
            (tagger_model, run_id),
        )
        if _is_cancel_requested(run_id):
            _finish_reclassify_run(run_id, cancelled=True, eligible=0, failed=0)
            return

        rows = _eligible_reclassify_rows(run_id, item_ids)
        if not rows:
            _finish_reclassify_run(run_id, cancelled=False, eligible=0, failed=0)
            return

        # Drop rows whose source file is already gone (e.g. migrated/moved).
        existing_rows: list[dict] = []
        for row in rows:
            path = Path(row["file_path"])
            if path.exists():
                existing_rows.append(row)
            else:
                logger.warning(
                    "reclassify_skip_missing run_id=%d item_id=%s path=%s",
                    run_id,
                    row.get("id"),
                    path,
                )
        rows = existing_rows
        if not rows:
            _finish_reclassify_run(run_id, cancelled=False, eligible=0, failed=0)
            return

        probe_execution_providers()
        ordered_paths = [Path(row["file_path"]) for row in rows]
        path_to_row = {Path(row["file_path"]): row for row in rows}

        configured_batch_size = _get_inference_batch_size(inference_batch_size)
        if tagger_model == "ml_danbooru" or configured_batch_size <= 1:
            inference_mode = "single"
            batch_size = 1
        else:
            inference_mode = "batch"
            batch_size = configured_batch_size

        _set_run_telemetry(
            run_id,
            inference_mode=inference_mode,
            batch_size=batch_size,
            avg_infer_ms_per_image=0.0,
        )
        # Progress reflects this reclassify pass (not the original classify totals).
        execute(
            """
            UPDATE runs
            SET total_images = ?, processed_images = 0, failed_images = 0, started_at = ?
            WHERE id = ?
            """,
            (len(rows), _now_iso(), run_id),
        )

        max_workers = _get_max_inference_workers(max_inference_workers)
        # EVA02 is heavy; keep workers modest. SwinV2/ML can overlap a bit more.
        if tagger_model == "wd_eva02_large":
            max_workers = min(max_workers, 2)
        else:
            max_workers = max(max_workers, 3)
            max_workers = min(max_workers, 4)
        cancelled = False
        infer_elapsed_ms_total = 0.0
        infer_sample_count = 0
        processed = 0

        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="reclass")
        pending: dict[Future[tuple[list[_ImageInferenceResult], float, str]], list[Path]] = {}
        try:
            # Keep stills in true WD batches; media one-at-a-time so one GIF/video
            # cannot collapse an entire batch to single-file fallback.
            if experimental_media_enabled:
                stills = [p for p in ordered_paths if not is_experimental_media(p)]
                media = [p for p in ordered_paths if is_experimental_media(p)]
                batches = [
                    stills[idx : idx + batch_size]
                    for idx in range(0, len(stills), batch_size)
                ]
                batches.extend([[p] for p in media])
                logger.info(
                    "reclassify_queue_split run_id=%d stills=%d media=%d workers=%d model=%s",
                    run_id,
                    len(stills),
                    len(media),
                    max_workers,
                    tagger_model,
                )
            else:
                batches = [
                    ordered_paths[idx : idx + batch_size]
                    for idx in range(0, len(ordered_paths), batch_size)
                ]
            iterator = iter(batches)

            def _submit_until_capacity() -> None:
                while len(pending) < max_workers:
                    if _is_cancel_requested(run_id):
                        return
                    try:
                        next_batch = next(iterator)
                    except StopIteration:
                        return
                    future = executor.submit(
                        _infer_batch_with_fallback,
                        next_batch,
                        matched_tags,
                        confidence_threshold,
                        inference_mode,
                        experimental_media_enabled,
                        tagger_model,
                        wd_general_threshold,
                        experimental_style_detector_enabled,
                        False,
                        hybrid_ml_on_review,
                    )
                    pending[future] = next_batch

            _submit_until_capacity()
            while pending:
                if _is_cancel_requested(run_id):
                    cancelled = True
                    for future in pending:
                        future.cancel()
                    break

                # Timeout so cancel_requested is polled even while a batch is in flight.
                done, _ = wait(
                    set(pending.keys()),
                    timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue
                for future in done:
                    pending.pop(future, None)
                    if future.cancelled():
                        continue
                    try:
                        batch_results, elapsed_ms, used_mode = future.result()
                    except Exception:
                        logger.exception("reclassify_future_failed run_id=%d", run_id)
                        continue
                    if cancelled or _is_cancel_requested(run_id):
                        cancelled = True
                        continue
                    infer_elapsed_ms_total += elapsed_ms
                    infer_sample_count += len(batch_results)
                    avg_ms = (
                        infer_elapsed_ms_total / infer_sample_count if infer_sample_count else 0.0
                    )
                    _set_run_telemetry(
                        run_id,
                        inference_mode=used_mode if used_mode != "single_fallback" else "single",
                        batch_size=batch_size,
                        avg_infer_ms_per_image=avg_ms,
                    )
                    for result in batch_results:
                        row = path_to_row.get(result.image_path)
                        if row is None:
                            continue
                        suggested_destination = (
                            str(destination_path(categories_root, result.primary_tag))
                            if result.primary_tag is not None
                            else None
                        )
                        new_status = "approved" if not result.needs_review else "proposed"
                        review_reason = _reclassify_review_reason(tagger_model, result)
                        # Preserve a manually set final_tag; otherwise mirror classification.
                        existing_final = row.get("final_tag")
                        if existing_final:
                            new_final_tag = existing_final
                            new_final_destination = row.get("final_destination")
                        else:
                            new_final_tag = result.primary_tag
                            new_final_destination = suggested_destination
                        execute(
                            """
                            UPDATE items
                            SET primary_tag = ?,
                                primary_score = ?,
                                secondary_json = ?,
                                full_scores_json = ?,
                                suggested_destination = ?,
                                final_tag = ?,
                                final_destination = ?,
                                status = ?,
                                needs_review = ?,
                                review_reason = ?
                            WHERE id = ? AND run_id = ? AND status = 'proposed' AND needs_review = 1
                            """,
                            (
                                result.primary_tag,
                                result.primary_score,
                                to_json(result.secondary),
                                to_json(result.scores),
                                suggested_destination,
                                new_final_tag,
                                new_final_destination,
                                new_status,
                                1 if result.needs_review else 0,
                                review_reason,
                                row["id"],
                                run_id,
                            ),
                        )
                        processed += 1
                        if result.inference_failed:
                            # Count refreshed below for accuracy across workers.
                            pass
                        failed = _count_inference_failed_items(run_id)
                        _update_run_progress(run_id, processed, failed)
                        if processed % 10 == 0:
                            logger.info(
                                "reclassify_progress run_id=%d processed=%d total=%d "
                                "avg_infer_ms=%.2f model=%s",
                                run_id,
                                processed,
                                len(rows),
                                avg_ms,
                                tagger_model,
                            )
                if not cancelled:
                    _submit_until_capacity()
        finally:
            # On cancel, do not block the API worker on in-flight GPU batches.
            executor.shutdown(wait=not cancelled, cancel_futures=True)

        failed = _count_inference_failed_items(run_id)
        _update_run_progress(run_id, processed, failed)
        _finish_reclassify_run(
            run_id,
            cancelled=cancelled,
            eligible=len(rows),
            failed=failed,
        )
    except Exception as err:
        logger.exception("reclassify_failed run_id=%d", run_id)
        execute(
            "UPDATE runs SET status = 'failed', finished_at = ?, last_error = ? WHERE id = ?",
            (_now_iso(), str(err), run_id),
        )


@router.get("/settings", response_model=AppSettings)
def get_settings() -> AppSettings:
    return _settings_from_db()


@router.put("/settings", response_model=AppSettings)
def save_settings(payload: SaveSettingsRequest) -> AppSettings:
    known = load_known_tags(TAGS_CSV)
    known_by_norm = {normalize_tag_name(t): t for t in known}
    cleaned_tags: list[str] = []
    for tag in payload.selected_tags:
        value = tag.strip()
        if not value:
            continue
        tax = resolve_taxonomy_folder(value)
        if tax is not None:
            if tax.folder not in cleaned_tags:
                cleaned_tags.append(tax.folder)
            continue
        matched = value if value in known else known_by_norm.get(normalize_tag_name(value))
        if matched and matched not in cleaned_tags:
            cleaned_tags.append(matched)
    payload.selected_tags = cleaned_tags
    try:
        execute(
            """
            UPDATE settings
            SET root_repo = ?, categories_root = ?, confidence_threshold = ?,
                default_migrate_mode = ?, scan_recursive = ?, experimental_media_enabled = ?,
                experimental_style_detector_enabled = ?, hybrid_ml_on_review = ?,
                selected_tags_json = ?, max_inference_workers = ?, inference_batch_size = ?,
                force_cpu_inference = ?, tagger_model = ?, wd_general_threshold = ?
            WHERE id = 1
            """,
            (
                payload.root_repo,
                payload.categories_root,
                payload.confidence_threshold,
                payload.default_migrate_mode,
                1 if payload.scan_recursive else 0,
                1 if payload.experimental_media_enabled else 0,
                1 if payload.experimental_style_detector_enabled else 0,
                1 if payload.hybrid_ml_on_review else 0,
                to_json(cleaned_tags),
                int(payload.max_inference_workers),
                int(payload.inference_batch_size),
                1 if payload.force_cpu_inference else 0,
                payload.tagger_model,
                float(payload.wd_general_threshold),
            ),
        )
    except Exception:
        logger.exception("failed to save settings")
        raise HTTPException(status_code=500, detail="Failed to persist settings")
    _apply_runtime_inference_env(payload)
    return payload


@router.get("/tags")
def search_tags(query: str = Query("", min_length=0), limit: int = 50) -> dict:
    """Suggest destination folders (taxonomy) first, then tags.csv matches."""
    q = (query or "").strip().lower()
    limit = max(1, min(int(limit), 200))

    def _matches(name: str) -> bool:
        return (not q) or (q in name.lower())

    items: list[str] = []
    seen: set[str] = set()
    # Taxonomy destinations (e.g. Voyeur, Voyeur/panties) before raw danbooru tags
    # like voyeurism, so users pick the folder they want to migrate into.
    for name in taxonomy_folder_names():
        if _matches(name) and name not in seen:
            items.append(name)
            seen.add(name)
    for name in sorted(load_known_tags(TAGS_CSV)):
        if _matches(name) and name not in seen:
            items.append(name)
            seen.add(name)
    return {"items": items[:limit], "count": len(items)}


@router.get("/scan/preview")
def scan_preview() -> dict:
    """Run image discovery on the configured root_repo and return stats without inference."""
    settings = _settings_from_db()
    if not settings.root_repo:
        raise HTTPException(status_code=400, detail="root_repo is not configured in settings")
    root_repo = Path(settings.root_repo).expanduser()
    cats = (
        Path(settings.categories_root).expanduser()
        if settings.categories_root
        else None
    )
    exclude_dirs = categories_exclude_dirs(root_repo, cats)
    try:
        output = scan_images(
            root_repo,
            exclude_dirs=exclude_dirs or None,
            recursive=settings.scan_recursive,
            experimental_media_enabled=settings.experimental_media_enabled,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    return {
        "root_repo": str(root_repo),
        "recursive": settings.scan_recursive,
        "experimental_media_enabled": settings.experimental_media_enabled,
        "excluded_dirs": [str(d) for d in exclude_dirs],
        "stats": output.stats.model_dump(),
        "sample_paths": [str(p) for p in output.image_paths[:20]],
    }


@router.get("/providers")
def get_providers() -> dict[str, object]:
    settings = _settings_from_db()
    _apply_runtime_inference_env(settings)
    info = probe_execution_providers()
    info["tagger_model"] = settings.tagger_model
    info["note"] = (
        "CUDA usability reflects ORT GPU runtime readiness; "
        "the active tagger model is selected separately in settings."
    )
    return info


@router.post("/runs/start", response_model=StartRunResponse)
def start_run(payload: StartRunRequest) -> StartRunResponse:
    logger.info("run_start_requested mode=%s", payload.run_mode)
    current = _settings_from_db()
    _apply_runtime_inference_env(current)
    resolved = resolve_settings(
        current, payload.root_repo, payload.categories_root, payload.confidence_threshold
    )
    root_repo = Path(resolved.root_repo).expanduser()
    categories_root = Path(resolved.categories_root).expanduser()
    real_life_filter = payload.run_mode == "real_life_filter"

    if not resolved.root_repo or not resolved.categories_root:
        raise HTTPException(status_code=400, detail="root_repo and categories_root are required")
    if not root_repo.exists() or not root_repo.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"root_repo does not exist or is not a directory: {root_repo}",
        )
    if categories_root.exists() and not categories_root.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"categories_root exists but is not a directory: {categories_root}",
        )
    if not categories_root.exists():
        try:
            categories_root.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to create categories_root ({categories_root}): {err}",
            )

    # Real-life filter: GIF/video on + hybrid WD realism × style detector.
    experimental_media_enabled = (
        True if real_life_filter else bool(resolved.experimental_media_enabled)
    )
    style_detector_enabled = (
        True if real_life_filter else bool(current.experimental_style_detector_enabled)
    )
    if real_life_filter:
        selected_folders = ["real_life"]
    else:
        selected_folders = payload.selected_folders
        if not selected_folders:
            selected_folders = list(current.selected_tags)

    try:
        known_tags = load_known_tags(TAGS_CSV)
        mappings = discover_tag_folders(categories_root, known_tags, selected_folders)
        matched_tags = {m.matched_tag for m in mappings if m.matched and m.matched_tag}
        if style_detector_enabled or real_life_filter:
            matched_tags.add("real_life")
        if not matched_tags:
            raise HTTPException(
                status_code=400,
                detail="No selected tags map to known tags.csv entries. Save tags in settings first.",
            )
    except HTTPException:
        raise
    except ValueError as err:
        logger.warning("failed to prepare run: %s", err)
        raise HTTPException(status_code=400, detail=str(err))
    except Exception:
        logger.exception("failed to prepare run")
        raise HTTPException(status_code=500, detail="Failed to prepare classification run")

    try:
        run_id = execute(
            """
            INSERT INTO runs (
                root_repo, categories_root, confidence_threshold, status,
                total_images, processed_images, failed_images, cancel_requested, tagger_model
            ) VALUES (?, ?, ?, 'pending', 0, 0, 0, 0, ?)
            """,
            (
                str(root_repo),
                str(categories_root),
                resolved.confidence_threshold,
                current.tagger_model,
            ),
        )
    except Exception:
        logger.exception("failed to create run row")
        raise HTTPException(status_code=500, detail="Failed to queue run")

    worker = threading.Thread(
        target=_execute_run,
        kwargs={
            "run_id": run_id,
            "root_repo": root_repo,
            "categories_root": categories_root,
            "confidence_threshold": resolved.confidence_threshold,
            "matched_tags": matched_tags,
            "scan_recursive": resolved.scan_recursive,
            "experimental_media_enabled": experimental_media_enabled,
            "max_inference_workers": current.max_inference_workers,
            "inference_batch_size": current.inference_batch_size,
            "tagger_model": current.tagger_model,
            "wd_general_threshold": current.wd_general_threshold,
            "experimental_style_detector_enabled": style_detector_enabled,
            "real_life_filter": real_life_filter,
            "hybrid_ml_on_review": bool(current.hybrid_ml_on_review),
        },
        daemon=True,
    )
    worker.start()
    mode_note = (
        " Real-life filter: experimental media on; hybrid WD realism + style detector; "
        "only real_life hits are kept (auto-approved)."
        if real_life_filter
        else ""
    )
    return StartRunResponse(
        run_id=run_id,
        status="pending",
        mappings=mappings,
        unmatched_folders=[m.folder_name for m in mappings if not m.matched],
        message=f"Run queued; poll /api/runs/{run_id}/status for progress.{mode_note}",
    )


@router.get("/runs/{run_id}")
def get_run(run_id: int) -> dict:
    run = fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    counts = fetch_all(
        "SELECT status, COUNT(*) AS count FROM items WHERE run_id = ? GROUP BY status", (run_id,)
    )
    return {"run": run, "counts": counts}


@router.get("/runs/{run_id}/status", response_model=RunStatusResponse)
def get_run_status(run_id: int) -> RunStatusResponse:
    run = fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_status_from_row(run)


@router.post("/runs/{run_id}/cancel", response_model=RunStatusResponse)
def cancel_run(run_id: int) -> RunStatusResponse:
    run = fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") in {"completed", "failed", "cancelled"}:
        return _run_status_from_row(run)
    execute("UPDATE runs SET cancel_requested = 1 WHERE id = ?", (run_id,))
    updated = fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    return _run_status_from_row(updated)


@router.post("/runs/{run_id}/reclassify", response_model=ReclassifyResponse)
def reclassify_run(run_id: int, payload: ReclassifyRequest) -> ReclassifyResponse:
    run = fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    status = str(run.get("status") or "")
    if status in {"pending", "running"}:
        raise HTTPException(
            status_code=400,
            detail="Cannot reclassify while a run is still pending or running.",
        )
    # cancelled is allowed: a prior reclassify cancel restores completed, but older
    # runs may still be cancelled with proposed items left to retry.
    if status not in {"completed", "failed", "cancelled"}:
        raise HTTPException(status_code=400, detail=f"Run status '{status}' cannot be reclassified.")

    eligible = _eligible_reclassify_rows(run_id, payload.item_ids)
    if not eligible:
        raise HTTPException(
            status_code=400,
            detail="No eligible needs-review items to reclassify "
            "(only proposed + needs_review items are updated).",
        )

    current = _settings_from_db()
    _apply_runtime_inference_env(current)
    root_repo = Path(str(run["root_repo"])).expanduser()
    categories_root = Path(str(run["categories_root"])).expanduser()
    confidence_threshold = float(run.get("confidence_threshold") or current.confidence_threshold)

    selected_folders = list(current.selected_tags)
    if not selected_folders:
        raise HTTPException(
            status_code=400,
            detail="No selected tags in settings; save destination tags before reclassifying.",
        )
    try:
        known_tags = load_known_tags(TAGS_CSV)
        mappings = discover_tag_folders(categories_root, known_tags, selected_folders)
        matched_tags = {m.matched_tag for m in mappings if m.matched and m.matched_tag}
        if current.experimental_style_detector_enabled:
            matched_tags.add("real_life")
        if not matched_tags:
            raise HTTPException(
                status_code=400,
                detail="No selected tags map to known tags.csv entries. Save tags in settings first.",
            )
    except HTTPException:
        raise
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception:
        logger.exception("failed to prepare reclassify run_id=%d", run_id)
        raise HTTPException(status_code=500, detail="Failed to prepare reclassify")

    # Claim the run before spawning the worker so a second reclassify cannot race in.
    execute(
        """
        UPDATE runs
        SET status = 'running',
            last_error = NULL,
            cancel_requested = 0,
            finished_at = NULL,
            tagger_model = ?
        WHERE id = ? AND status IN ('completed', 'failed', 'cancelled')
        """,
        (payload.tagger_model, run_id),
    )
    claimed = fetch_one("SELECT status FROM runs WHERE id = ?", (run_id,))
    if not claimed or claimed.get("status") != "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot reclassify while a run is still pending or running.",
        )

    worker = threading.Thread(
        target=_execute_reclassify,
        args=(
            run_id,
            root_repo,
            categories_root,
            confidence_threshold,
            matched_tags,
            payload.item_ids,
            current.experimental_media_enabled,
            current.max_inference_workers,
            current.inference_batch_size,
            payload.tagger_model,
            current.wd_general_threshold,
            current.experimental_style_detector_enabled,
            current.hybrid_ml_on_review,
        ),
        daemon=True,
    )
    worker.start()
    return ReclassifyResponse(
        run_id=run_id,
        status="running",
        eligible_count=len(eligible),
        tagger_model=payload.tagger_model,
        message=(
            f"Reclassify queued for {len(eligible)} item(s) with {payload.tagger_model}; "
            f"poll /api/runs/{run_id}/status for progress."
        ),
    )


@router.get("/runs/{run_id}/items", response_model=list[ClassifiedItem])
def get_run_items(
    run_id: int,
    status: str | None = None,
    needs_review: bool | None = None,
    include_scores: bool = Query(False),
) -> list[ClassifiedItem]:
    query = "SELECT * FROM items WHERE run_id = ?"
    params: list = [run_id]
    if status:
        query += " AND status = ?"
        params.append(status)
    if needs_review is not None:
        query += " AND needs_review = ?"
        params.append(1 if needs_review else 0)
    query += " ORDER BY id ASC"
    rows = fetch_all(query, tuple(params))
    return [_item_from_row(r, include_full_scores=include_scores) for r in rows]


def _secondary_tags_from_row(row: dict) -> list[str]:
    secondary = from_json(row.get("secondary_json") or "[]", default=[])
    tags: list[str] = []
    if isinstance(secondary, list):
        for entry in secondary:
            if isinstance(entry, dict) and entry.get("tag"):
                tags.append(str(entry["tag"]).strip())
            elif isinstance(entry, str) and entry.strip():
                tags.append(entry.strip())
    return [t for t in tags if t]


def _resolve_item_assignment(
    row: dict,
    *,
    final_tag_override: str | None = None,
    categories_root: Path | None = None,
) -> tuple[str | None, str | None]:
    """Pick (final_tag, final_destination) for approve/migrate.

    Needs-review items often have primary_tag cleared but keep a secondary
    suggestion — approve must still resolve a folder or migrate cannot run.
    """
    tag = (final_tag_override or "").strip() or None
    if not tag:
        for candidate in (
            row.get("final_tag"),
            row.get("primary_tag"),
            *(_secondary_tags_from_row(row)[:1]),
        ):
            if candidate and str(candidate).strip():
                tag = str(candidate).strip()
                break

    destination = row.get("final_destination") or row.get("suggested_destination")
    if destination and tag and categories_root is not None:
        expected = destination_path(categories_root, tag)
        dest_path = Path(str(destination))
        try:
            if dest_path.resolve() == expected.resolve():
                return tag, str(dest_path)
        except OSError:
            if dest_path == expected:
                return tag, str(dest_path)
    elif destination and tag:
        # Fallback when categories_root unknown: leaf name match (flat folders).
        dest_path = Path(str(destination))
        leaf = tag.replace("\\", "/").rstrip("/").split("/")[-1]
        if sanitize_folder_name(dest_path.name) == sanitize_folder_name(leaf):
            return tag, str(dest_path)
    if tag and categories_root is not None:
        return tag, str(destination_path(categories_root, tag))
    if destination:
        return tag, str(destination)
    return tag, None


@router.patch("/items/{item_id}", response_model=ClassifiedItem)
def update_item(item_id: int, payload: UpdateItemRequest) -> ClassifiedItem:
    row = fetch_one("SELECT * FROM items WHERE id = ?", (item_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    run = fetch_one("SELECT categories_root FROM runs WHERE id = ?", (row["run_id"],))
    categories_root = Path(run["categories_root"]) if run and run.get("categories_root") else None
    new_status = payload.status if payload.status is not None else row["status"]
    tag_override = payload.final_tag if payload.final_tag is not None else None

    if new_status == "rejected":
        # Reject = leave file in place; never invent a migrate destination.
        new_final_tag = tag_override.strip() if isinstance(tag_override, str) and tag_override.strip() else None
        new_final_destination = None
        new_needs_review = 0
        new_review_reason = "Rejected by user"
    else:
        new_final_tag, new_final_destination = _resolve_item_assignment(
            row,
            final_tag_override=tag_override,
            categories_root=categories_root,
        )
        # Approving without any resolvable destination cannot migrate later.
        if new_status == "approved" and not new_final_destination:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot approve without a destination tag. "
                    "Set Final tag (or ensure a secondary suggestion exists) first."
                ),
            )
        new_needs_review = (
            0
            if new_status in {"reviewed", "approved", "migrated"}
            else row["needs_review"]
        )
        new_review_reason = (
            None
            if new_status in {"reviewed", "approved", "migrated"}
            else row["review_reason"]
        )

    execute(
        """
        UPDATE items
        SET status = ?, final_tag = ?, final_destination = ?, needs_review = ?, review_reason = ?
        WHERE id = ?
        """,
        (
            new_status,
            new_final_tag,
            new_final_destination,
            new_needs_review,
            new_review_reason,
            item_id,
        ),
    )
    updated = fetch_one("SELECT * FROM items WHERE id = ?", (item_id,))
    return _item_from_row(updated)


@router.get("/items/{item_id}/scores")
def get_item_scores(item_id: int) -> dict:
    row = fetch_one("SELECT id, full_scores_json FROM items WHERE id = ?", (item_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    return {
        "item_id": row["id"],
        "full_scores": from_json(row.get("full_scores_json") or "{}", default={}),
    }


@router.get("/items/{item_id}/preview", response_model=None)
def get_item_preview(
    item_id: int,
    raw: bool = Query(
        False,
        description="Serve original bytes (video/gif). Default returns a JPEG still for media.",
    ),
):
    row = fetch_one("SELECT file_path, migrated_to FROM items WHERE id = ?", (item_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    candidates = []
    for key in ("file_path", "migrated_to"):
        raw_path = row.get(key)
        if raw_path:
            candidates.append(Path(str(raw_path)).expanduser())
    image_path = next((p for p in candidates if p.exists() and p.is_file()), None)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Preview media not found")
    suffix = image_path.suffix.lower()
    media_type = SUPPORTED_PREVIEW_SUFFIXES.get(suffix)
    if not media_type:
        raise HTTPException(status_code=415, detail="Unsupported media type for preview")

    # Table thumbs must stay tiny: full MP4/GIF as media elements collapses under
    # parallel loads. Default to a cached JPEG still; ?raw=1 keeps original bytes.
    needs_still = (not raw) and (suffix == ".gif" or suffix in VIDEO_EXTENSIONS)
    if needs_still:
        try:
            jpeg = media_preview_still_jpeg(image_path)
        except Exception as err:
            logger.warning(
                "preview_still_failed item_id=%s path=%s err=%s",
                item_id,
                image_path,
                err,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Unable to build media thumbnail: {err}"
            ) from err
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'inline; filename="{image_path.stem}_thumb.jpg"',
                "Cache-Control": "private, max-age=86400",
            },
        )

    # Must be inline so <img>/<video> can render; "attachment" + filename makes
    # browsers refuse to display the bytes in media elements.
    return FileResponse(
        path=image_path,
        media_type=media_type,
        filename=image_path.name,
        content_disposition_type="inline",
    )


@router.post("/runs/{run_id}/batch")
def batch_update(run_id: int, payload: BatchUpdateRequest) -> dict:
    if not payload.item_ids:
        return {"updated": 0, "skipped": 0}
    run = fetch_one("SELECT categories_root FROM runs WHERE id = ?", (run_id,))
    categories_root = Path(run["categories_root"]) if run and run.get("categories_root") else None
    updated = 0
    skipped = 0
    for item_id in payload.item_ids:
        row = fetch_one("SELECT * FROM items WHERE id = ? AND run_id = ?", (item_id, run_id))
        if not row:
            skipped += 1
            continue
        new_status = payload.status if payload.status is not None else row["status"]
        tag_override = payload.final_tag if payload.final_tag is not None else None
        if new_status == "rejected":
            new_final_tag = (
                tag_override.strip()
                if isinstance(tag_override, str) and tag_override.strip()
                else None
            )
            new_final_destination = None
            new_needs_review = 0
            new_review_reason = "Rejected by user"
        else:
            new_final_tag, new_final_destination = _resolve_item_assignment(
                row,
                final_tag_override=tag_override,
                categories_root=categories_root,
            )
            if new_status == "approved" and not new_final_destination:
                skipped += 1
                continue
            new_needs_review = (
                0
                if new_status in {"reviewed", "approved", "migrated"}
                else row["needs_review"]
            )
            new_review_reason = (
                None
                if new_status in {"reviewed", "approved", "migrated"}
                else row["review_reason"]
            )
        execute(
            """
            UPDATE items
            SET status = ?, final_tag = ?, final_destination = ?, needs_review = ?, review_reason = ?
            WHERE id = ?
            """,
            (
                new_status,
                new_final_tag,
                new_final_destination,
                new_needs_review,
                new_review_reason,
                item_id,
            ),
        )
        updated += 1
    return {"updated": updated, "skipped": skipped}


@router.post("/runs/{run_id}/migrate", response_model=MigrateResponse)
def migrate_run(run_id: int, payload: MigrateRequest) -> MigrateResponse:
    run = fetch_one("SELECT status, categories_root FROM runs WHERE id = ?", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in {"completed", "cancelled"}:
        raise HTTPException(status_code=409, detail="Run must be completed/cancelled before migration")
    categories_root = Path(run["categories_root"]) if run.get("categories_root") else None
    rows = fetch_all("SELECT * FROM items WHERE run_id = ? AND status = 'approved'", (run_id,))
    results = []
    migrated_count = 0
    failed_count = 0
    for row in rows:
        source = Path(row["file_path"])
        resolved_tag, resolved_dest = _resolve_item_assignment(
            row, categories_root=categories_root
        )
        final_destination = resolved_dest or row["final_destination"]
        # Persist repaired assignment so retries and UI stay consistent.
        if resolved_tag and resolved_dest and (
            row.get("final_tag") != resolved_tag or row.get("final_destination") != resolved_dest
        ):
            execute(
                "UPDATE items SET final_tag = ?, final_destination = ? WHERE id = ?",
                (resolved_tag, resolved_dest, row["id"]),
            )
        if not final_destination:
            failed_count += 1
            results.append(
                {
                    "item_id": row["id"],
                    "source": str(source),
                    "destination": None,
                    "success": False,
                    "error": "No destination configured",
                }
            )
            continue

        destination_folder = Path(final_destination)
        destination = destination_folder / source.name

        # Source already gone but file sits at destination (prior move / manual).
        if not source.exists():
            if destination.exists() and destination.is_file():
                try:
                    execute(
                        "UPDATE items SET status = 'migrated', migrated_to = ? WHERE id = ?",
                        (str(destination), row["id"]),
                    )
                    migrated_count += 1
                except Exception:
                    logger.exception(
                        "failed to persist already-migrated status item_id=%s", row["id"]
                    )
                    failed_count += 1
                    results.append(
                        {
                            "item_id": row["id"],
                            "source": str(source),
                            "destination": str(destination),
                            "success": False,
                            "error": "Already at destination but DB update failed",
                        }
                    )
            else:
                failed_count += 1
                results.append(
                    {
                        "item_id": row["id"],
                        "source": str(source),
                        "destination": str(destination),
                        "success": False,
                        "error": "Source file does not exist",
                    }
                )
            continue

        try:
            if payload.create_missing_folders:
                destination_folder.mkdir(parents=True, exist_ok=True)
            elif not destination_folder.exists():
                failed_count += 1
                results.append(
                    {
                        "item_id": row["id"],
                        "source": str(source),
                        "destination": str(destination_folder),
                        "success": False,
                        "error": "Destination folder does not exist",
                    }
                )
                continue
            elif not destination_folder.is_dir():
                failed_count += 1
                results.append(
                    {
                        "item_id": row["id"],
                        "source": str(source),
                        "destination": str(destination_folder),
                        "success": False,
                        "error": "Destination path exists but is not a folder",
                    }
                )
                continue
        except Exception:
            failed_count += 1
            results.append(
                {
                    "item_id": row["id"],
                    "source": str(source),
                    "destination": str(destination_folder),
                    "success": False,
                    "error": "Unable to create destination folder",
                }
            )
            continue

        migration_result = migrate_file(source, destination, payload.mode)
        migration_result.item_id = row["id"]
        if migration_result.success:
            if not Path(migration_result.destination).exists():
                migration_result.success = False
                migration_result.error = "Destination file missing after migration"
            else:
                try:
                    execute(
                        "UPDATE items SET status = 'migrated', migrated_to = ? WHERE id = ?",
                        (migration_result.destination, row["id"]),
                    )
                    migrated_count += 1
                except Exception:
                    logger.exception("failed to persist migration status item_id=%s", row["id"])
                    migration_result.success = False
                    migration_result.error = (
                        "File moved but DB update failed; please refresh and reconcile."
                    )
        if not migration_result.success:
            failed_count += 1
            # Only return failures — full success lists are multi‑MB on large runs
            # and cause the UI request to time out after the work already finished.
            results.append(migration_result.model_dump())

    return MigrateResponse(
        mode=payload.mode,
        total_candidates=len(rows),
        migrated_count=migrated_count,
        failed_count=failed_count,
        results=results,
    )


@router.get("/debug/sfw-sources")
def debug_sfw_sources() -> dict[str, object]:
    from .sfw_sources import list_sources

    return {
        "sources": [
            {
                "id": s.id,
                "label": s.label,
                "sfw_policy": s.sfw_policy,
                "max_content_tags": s.max_content_tags,
            }
            for s in list_sources()
        ]
    }


@router.post("/debug/sfw-eval", response_model=SfwDebugEvalResponse)
def debug_sfw_eval(payload: SfwDebugEvalRequest) -> SfwDebugEvalResponse:
    from .debug_eval import run_sfw_eval
    from .sfw_sources import get_source

    settings = _settings_from_db()
    _apply_runtime_inference_env(settings)
    try:
        get_source(payload.source)
    except KeyError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    try:
        result = run_sfw_eval(
            source_id=payload.source,
            tags=payload.tags,
            count=payload.count,
            settings=settings,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        logger.exception("debug_sfw_eval_failed")
        raise HTTPException(status_code=500, detail=f"SFW eval failed: {err}") from err
    return SfwDebugEvalResponse.model_validate(result)


@router.post("/debug/realism-eval", response_model=None)
def debug_realism_eval(payload: RealismDebugEvalRequest):
    """People-photo vs anime separation eval (remote samples + taxonomy routing)."""
    from .realism_eval import run_realism_eval, run_realism_eval_multi_model

    settings = _settings_from_db()
    _apply_runtime_inference_env(settings)
    try:
        if payload.compare_models:
            multi = run_realism_eval_multi_model(
                count_per_class=payload.count_per_class,
                settings=settings,
            )
            # Surface best single-model report at top-level for the UI table.
            best_id = multi.get("best_model")
            best = next(
                (r for r in multi.get("reports") or [] if r.get("tagger_model") == best_id),
                (multi.get("reports") or [None])[0],
            )
            if not isinstance(best, dict):
                raise RuntimeError("multi-model realism eval produced no reports")
            best = {**best, "multi_model": multi}
            return best
        result = run_realism_eval(
            count_per_class=payload.count_per_class,
            settings=settings,
            tagger_model=payload.tagger_model,
        )
        return RealismDebugEvalResponse.model_validate(result)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        logger.exception("debug_realism_eval_failed")
        raise HTTPException(
            status_code=500, detail=f"Realism eval failed: {err}"
        ) from err


@router.get("/debug/style-detectors")
def debug_style_detectors():
    """List debug-only real-vs-anime style detectors available for compare."""
    from .style_detectors import list_style_detectors

    return {"detectors": list_style_detectors()}


@router.post("/debug/style-eval", response_model=None)
def debug_style_eval(payload: StyleDebugEvalRequest):
    """Compare dedicated style detectors vs WD taxonomy on the realism corpus."""
    from .style_eval import run_style_detector_eval

    settings = _settings_from_db()
    _apply_runtime_inference_env(settings)
    try:
        return run_style_detector_eval(
            count_per_class=payload.count_per_class,
            settings=settings,
            detector_ids=payload.detectors,
            tagger_model=payload.tagger_model,
            uncertain_threshold=float(payload.uncertain_threshold),
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        logger.exception("debug_style_eval_failed")
        raise HTTPException(
            status_code=500, detail=f"Style detector eval failed: {err}"
        ) from err


@router.get("/debug/realism-eval/preview/{source_id}/{file_name}")
def debug_realism_eval_preview(source_id: str, file_name: str) -> FileResponse:
    from .realism_sources import resolve_realism_cached_file

    try:
        path = resolve_realism_cached_file(source_id, file_name)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="Preview not found") from err
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    suffix = path.suffix.lower()
    media = SUPPORTED_PREVIEW_SUFFIXES.get(suffix, "application/octet-stream")
    return FileResponse(
        path,
        media_type=media,
        filename=path.name,
        content_disposition_type="inline",
    )


@router.get("/debug/sfw-eval/preview/{source_id}/{file_name}")
def debug_sfw_eval_preview(source_id: str, file_name: str) -> FileResponse:
    from .sfw_sources import resolve_cached_file

    try:
        path = resolve_cached_file(source_id, file_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Preview not found")
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    suffix = path.suffix.lower()
    media = SUPPORTED_PREVIEW_SUFFIXES.get(suffix, "application/octet-stream")
    return FileResponse(
        path,
        media_type=media,
        filename=path.name,
        content_disposition_type="inline",
    )


@router.post("/debug/tag-recall-eval", response_model=None)
def debug_tag_recall_eval(payload: TagRecallEvalRequest):
    """Curated-suite tag recall@threshold and recall@top-K across taggers."""
    from .tag_recall_eval import run_tag_recall_eval

    settings = _settings_from_db()
    _apply_runtime_inference_env(settings)
    try:
        return run_tag_recall_eval(
            models=[str(m) for m in payload.models],
            threshold=float(payload.threshold),
            top_k=int(payload.top_k),
            refresh_cache=bool(payload.refresh_cache),
            wd_general_threshold=float(settings.wd_general_threshold),
            include_items=bool(payload.include_items),
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        logger.exception("debug_tag_recall_eval_failed")
        raise HTTPException(
            status_code=500, detail=f"Tag recall eval failed: {err}"
        ) from err


@router.get("/debug/tag-recall-eval/preview/{source_id}/{file_name}")
def debug_tag_recall_eval_preview(source_id: str, file_name: str) -> FileResponse:
    from .tag_recall_eval import resolve_cached_file

    try:
        path = resolve_cached_file(source_id, file_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Preview not found")
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    suffix = path.suffix.lower()
    media = SUPPORTED_PREVIEW_SUFFIXES.get(suffix, "application/octet-stream")
    return FileResponse(
        path,
        media_type=media,
        filename=path.name,
        content_disposition_type="inline",
    )


@router.post("/debug/tag-fp-eval", response_model=None)
def debug_tag_fp_eval(payload: TagFpEvalRequest):
    """Preferred-tag false-positive rates on the curated suite (+ folder route FPs)."""
    from .tag_fp_eval import run_tag_fp_eval

    settings = _settings_from_db()
    _apply_runtime_inference_env(settings)
    tag_thr = (
        float(payload.tag_threshold)
        if payload.tag_threshold is not None
        else float(settings.confidence_threshold)
    )
    route_thr = (
        float(payload.route_threshold)
        if payload.route_threshold is not None
        else tag_thr
    )
    try:
        return run_tag_fp_eval(
            selected_tags=list(settings.selected_tags or []),
            models=[str(m) for m in payload.models],
            tag_threshold=tag_thr,
            route_threshold=route_thr,
            min_weight=float(payload.min_weight),
            also_wd_threshold=bool(payload.also_wd_threshold),
            wd_general_threshold=float(settings.wd_general_threshold),
            refresh_cache=bool(payload.refresh_cache),
            include_items=False,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        logger.exception("debug_tag_fp_eval_failed")
        raise HTTPException(
            status_code=500, detail=f"Tag FP eval failed: {err}"
        ) from err

