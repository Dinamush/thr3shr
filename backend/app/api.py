from __future__ import annotations

import os
import logging
import random
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .schemas import (
    AppSettings,
    BatchUpdateRequest,
    ClassifiedItem,
    MigrateRequest,
    MigrateResponse,
    RunStatusResponse,
    SaveSettingsRequest,
    StartRunRequest,
    StartRunResponse,
    UpdateItemRequest,
)
from .services import (
    choose_best_tags,
    discover_tag_folders,
    extract_scores,
    extract_scores_batch,
    extract_scores_with_experimental_media,
    global_top_tags,
    is_experimental_media,
    load_known_tags,
    migrate_file,
    normalize_tag_name,
    resolve_settings,
    sanitize_folder_name,
    scan_images,
)
from .providers import clear_provider_probe_cache, probe_execution_providers
from .storage import execute, fetch_all, fetch_one, from_json, to_json

router = APIRouter(prefix="/api")
REPO_ROOT = Path(__file__).resolve().parents[2]
TAGS_CSV = REPO_ROOT / "tags.csv"
logger = logging.getLogger(__name__)
SUPPORTED_PREVIEW_SUFFIXES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
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
    raw = os.getenv("INFERENCE_MODE", "single").strip().lower()
    return "single" if raw == "single" else "batch"


def _get_inference_batch_size(settings_batch: int | None = None) -> int:
    # Default 1: imgutils cannot true-batch; small batches keep workers saturated.
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
) -> _ImageInferenceResult:
    primary_tag, primary_score, secondary = choose_best_tags(scores, matched_tags)
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
            # Weak selected-tag winners are suggestions only — do not present as the label.
            secondary = [{"tag": primary_tag, "score": float(primary_score)}, *secondary][:4]
            primary_tag = None
            primary_score = None
    return _ImageInferenceResult(
        image_path=image_path,
        scores=scores,
        primary_tag=primary_tag,
        primary_score=primary_score,
        secondary=secondary,
        needs_review=needs_review,
        reason=reason,
        inference_failed=False,
    )


def _infer_one_image(
    image_path: Path,
    matched_tags: set[str],
    confidence_threshold: float,
    experimental_media_enabled: bool = False,
    tagger_model: str = "wd_swinv2_v3",
    wd_general_threshold: float = 0.35,
) -> _ImageInferenceResult:
    try:
        if experimental_media_enabled and is_experimental_media(image_path):
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
        return _classify_from_scores(image_path, scores, matched_tags, confidence_threshold)
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
) -> tuple[list[_ImageInferenceResult], float, str]:
    if not image_paths:
        return [], 0.0, "none"

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
            _classify_from_scores(image_path, scores, matched_tags, confidence_threshold)
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
            )
            right_rows, right_ms, _ = _infer_batch_with_fallback(
                image_paths[mid:],
                matched_tags,
                confidence_threshold,
                "batch",
                experimental_media_enabled,
                tagger_model,
                wd_general_threshold,
            )
            return left_rows + right_rows, left_ms + right_ms, "batch_fallback"
        row = _infer_one_image(
            image_paths[0],
            matched_tags,
            confidence_threshold,
            experimental_media_enabled,
            tagger_model,
            wd_general_threshold,
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
        selected_tags=selected_tags,
        max_inference_workers=int(row.get("max_inference_workers") or 2),
        inference_batch_size=int(row.get("inference_batch_size") or 1),
        force_cpu_inference=bool(row.get("force_cpu_inference", 0)),
        tagger_model=tagger_model,  # type: ignore[arg-type]
        wd_general_threshold=float(row.get("wd_general_threshold") or 0.35),
    )


def _apply_runtime_inference_env(settings: AppSettings) -> None:
    """Mirror persisted settings into env knobs used by the run executor."""
    os.environ["MAX_INFERENCE_WORKERS"] = str(settings.max_inference_workers)
    os.environ["INFERENCE_BATCH_SIZE"] = str(settings.inference_batch_size)
    if settings.force_cpu_inference:
        os.environ["FORCE_CPU_INFERENCE"] = "true"
    else:
        os.environ.pop("FORCE_CPU_INFERENCE", None)
    clear_provider_probe_cache()


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


def _run_status_from_row(row: dict) -> RunStatusResponse:
    total = int(row.get("total_images") or 0)
    processed = int(row.get("processed_images") or 0)
    pct = 0.0 if total <= 0 else min(100.0, (processed / total) * 100.0)
    item_count_row = fetch_one("SELECT COUNT(*) AS cnt FROM items WHERE run_id = ?", (row["id"],))
    has_items = bool(item_count_row and int(item_count_row["cnt"]) > 0)
    telemetry = _get_run_telemetry(row["id"])
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
        avg_infer_ms_per_image=telemetry.get("avg_infer_ms_per_image"),
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
) -> None:
    try:
        provider_state = probe_execution_providers()
        logger.info("run_provider_state run_id=%d state=%s", run_id, provider_state)
        execute(
            "UPDATE runs SET status = 'running', started_at = ?, last_error = NULL WHERE id = ?",
            (_now_iso(), run_id),
        )
        scan_output = scan_images(
            root_repo,
            exclude_dirs={categories_root},
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
        inference_mode = _get_inference_mode()
        configured_batch_size = _get_inference_batch_size(inference_batch_size)
        # Prefer single-image tasks so worker pool stays saturated (no true ORT batch).
        batch_size = 1 if inference_mode == "single" else configured_batch_size
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
        logger.info(
            "run_inference_workers run_id=%d workers=%d total_images=%d device=%s",
            run_id,
            max_workers,
            scan_output.stats.eligible_images,
            provider_state.get("likely_device"),
        )

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="infer") as executor:
            pending: dict[Future[tuple[list[_ImageInferenceResult], float, str]], list[Path]] = {}
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
                        suggested_destination = (
                            str(categories_root / sanitize_folder_name(result.primary_tag))
                            if result.primary_tag is not None
                            else None
                        )
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
                                "approved" if not result.needs_review else "proposed",
                                1 if result.needs_review else 0,
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
    known = sorted(load_known_tags(TAGS_CSV))
    if query:
        known = [t for t in known if query.lower() in t.lower()]
    return {"items": known[:limit], "count": len(known)}


@router.get("/scan/preview")
def scan_preview() -> dict:
    """Run image discovery on the configured root_repo and return stats without inference."""
    settings = _settings_from_db()
    if not settings.root_repo:
        raise HTTPException(status_code=400, detail="root_repo is not configured in settings")
    root_repo = Path(settings.root_repo).expanduser()
    exclude_dirs: set[Path] = set()
    if settings.categories_root:
        exclude_dirs.add(Path(settings.categories_root).expanduser())
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
    logger.info("run_start_requested")
    current = _settings_from_db()
    _apply_runtime_inference_env(current)
    resolved = resolve_settings(
        current, payload.root_repo, payload.categories_root, payload.confidence_threshold
    )
    root_repo = Path(resolved.root_repo).expanduser()
    categories_root = Path(resolved.categories_root).expanduser()

    if not resolved.root_repo or not resolved.categories_root:
        raise HTTPException(status_code=400, detail="root_repo and categories_root are required")

    selected_folders = payload.selected_folders
    if not selected_folders:
        selected_folders = list(current.selected_tags)

    try:
        known_tags = load_known_tags(TAGS_CSV)
        mappings = discover_tag_folders(categories_root, known_tags, selected_folders)
        matched_tags = {m.matched_tag for m in mappings if m.matched and m.matched_tag}
        if not matched_tags:
            raise HTTPException(
                status_code=400,
                detail="No selected tags map to known tags.csv entries. Save tags in settings first.",
            )
    except HTTPException:
        raise
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
        args=(
            run_id,
            root_repo,
            categories_root,
            resolved.confidence_threshold,
            matched_tags,
            resolved.scan_recursive,
            resolved.experimental_media_enabled,
            current.max_inference_workers,
            current.inference_batch_size,
            current.tagger_model,
            current.wd_general_threshold,
        ),
        daemon=True,
    )
    worker.start()
    return StartRunResponse(
        run_id=run_id,
        status="pending",
        mappings=mappings,
        unmatched_folders=[m.folder_name for m in mappings if not m.matched],
        message="Run queued; poll /api/runs/{run_id}/status for progress.",
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


@router.patch("/items/{item_id}", response_model=ClassifiedItem)
def update_item(item_id: int, payload: UpdateItemRequest) -> ClassifiedItem:
    row = fetch_one("SELECT * FROM items WHERE id = ?", (item_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    new_status = payload.status if payload.status is not None else row["status"]
    new_final_tag = payload.final_tag if payload.final_tag is not None else row["final_tag"]
    new_final_destination = (
        str(Path(row["suggested_destination"]).parent / sanitize_folder_name(new_final_tag))
        if row["suggested_destination"] and new_final_tag
        else row["final_destination"]
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
            0 if new_status in {"reviewed", "approved", "rejected", "migrated"} else row["needs_review"],
            None if new_status in {"reviewed", "approved", "rejected", "migrated"} else row["review_reason"],
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


@router.get("/items/{item_id}/preview")
def get_item_preview(item_id: int) -> FileResponse:
    row = fetch_one("SELECT file_path FROM items WHERE id = ?", (item_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    image_path = Path(row["file_path"]).expanduser()
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Preview image not found")
    media_type = SUPPORTED_PREVIEW_SUFFIXES.get(image_path.suffix.lower())
    if not media_type:
        raise HTTPException(status_code=415, detail="Unsupported image type for preview")
    return FileResponse(path=image_path, media_type=media_type, filename=image_path.name)


@router.post("/runs/{run_id}/batch")
def batch_update(run_id: int, payload: BatchUpdateRequest) -> dict:
    if not payload.item_ids:
        return {"updated": 0, "skipped": 0}
    updated = 0
    skipped = 0
    for item_id in payload.item_ids:
        row = fetch_one("SELECT * FROM items WHERE id = ? AND run_id = ?", (item_id, run_id))
        if not row:
            skipped += 1
            continue
        new_status = payload.status if payload.status is not None else row["status"]
        new_final_tag = payload.final_tag if payload.final_tag is not None else row["final_tag"]
        new_final_destination = (
            str(Path(row["suggested_destination"]).parent / sanitize_folder_name(new_final_tag))
            if row["suggested_destination"] and new_final_tag
            else row["final_destination"]
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
                0 if new_status in {"reviewed", "approved", "rejected", "migrated"} else row["needs_review"],
                None if new_status in {"reviewed", "approved", "rejected", "migrated"} else row["review_reason"],
                item_id,
            ),
        )
        updated += 1
    return {"updated": updated, "skipped": skipped}


@router.post("/runs/{run_id}/migrate", response_model=MigrateResponse)
def migrate_run(run_id: int, payload: MigrateRequest) -> MigrateResponse:
    run = fetch_one("SELECT status FROM runs WHERE id = ?", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in {"completed", "cancelled"}:
        raise HTTPException(status_code=409, detail="Run must be completed/cancelled before migration")
    rows = fetch_all("SELECT * FROM items WHERE run_id = ? AND status = 'approved'", (run_id,))
    results = []
    migrated_count = 0
    failed_count = 0
    for row in rows:
        source = Path(row["file_path"])
        final_destination = row["final_destination"]
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
        if not source.exists():
            failed_count += 1
            results.append(
                {
                    "item_id": row["id"],
                    "source": str(source),
                    "destination": final_destination,
                    "success": False,
                    "error": "Source file does not exist",
                }
            )
            continue

        destination_folder = Path(final_destination)
        try:
            if payload.create_missing_folders:
                destination_folder.mkdir(parents=True, exist_ok=True)
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

        destination = destination_folder / source.name
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
        results.append(migration_result.model_dump())

    return MigrateResponse(
        mode=payload.mode,
        total_candidates=len(rows),
        migrated_count=migrated_count,
        failed_count=failed_count,
        results=results,
    )
