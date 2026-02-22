from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

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
    load_known_tags,
    migrate_file,
    resolve_settings,
    scan_images,
)
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _settings_from_db() -> AppSettings:
    row = fetch_one("SELECT * FROM settings WHERE id = 1")
    if not row:
        return AppSettings()
    return AppSettings(
        root_repo=row["root_repo"],
        categories_root=row["categories_root"],
        confidence_threshold=float(row["confidence_threshold"]),
        default_migrate_mode=row["default_migrate_mode"],
    )


def _item_from_row(row: dict) -> ClassifiedItem:
    return ClassifiedItem(
        id=row["id"],
        run_id=row["run_id"],
        file_path=row["file_path"],
        relative_path=row["relative_path"],
        primary_tag=row["primary_tag"],
        primary_score=row["primary_score"],
        secondary_suggestions=from_json(row.get("secondary_json") or "[]", default=[]),
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
) -> None:
    try:
        execute(
            "UPDATE runs SET status = 'running', started_at = ?, last_error = NULL WHERE id = ?",
            (_now_iso(), run_id),
        )
        scan_output = scan_images(root_repo)
        execute("UPDATE runs SET total_images = ? WHERE id = ?", (scan_output.stats.eligible_images, run_id))

        processed = 0
        failed = 0
        for image_path in scan_output.image_paths:
            if _is_cancel_requested(run_id):
                execute(
                    "UPDATE runs SET status = 'cancelled', finished_at = ? WHERE id = ?",
                    (_now_iso(), run_id),
                )
                logger.info("run_cancelled run_id=%d processed=%d", run_id, processed)
                return

            relative_path = str(image_path.relative_to(root_repo))
            try:
                scores = extract_scores(image_path)
                primary_tag, primary_score, secondary = choose_best_tags(scores, matched_tags)
                needs_review = False
                reason = None
            except Exception:
                logger.exception("inference_failed run_id=%d image=%s", run_id, image_path)
                primary_tag, primary_score, secondary = None, None, []
                needs_review = True
                reason = "Inference failed for this image; requires manual review."
                failed += 1

            if primary_tag is None and reason is None:
                needs_review = True
                reason = "No matching tags found in selected folders."
            elif (
                primary_score is not None
                and primary_score < confidence_threshold
                and reason is None
            ):
                needs_review = True
                reason = f"Below threshold ({primary_score:.3f} < {confidence_threshold:.3f})."

            suggested_destination = (
                str(categories_root / primary_tag) if primary_tag is not None else None
            )
            execute(
                """
                INSERT INTO items (
                    run_id, file_path, relative_path, primary_tag, primary_score, secondary_json,
                    suggested_destination, final_tag, final_destination, status, needs_review, review_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(image_path),
                    relative_path,
                    primary_tag,
                    primary_score,
                    to_json(secondary),
                    suggested_destination,
                    primary_tag,
                    suggested_destination,
                    "approved" if not needs_review else "proposed",
                    1 if needs_review else 0,
                    reason,
                ),
            )
            processed += 1
            _update_run_progress(run_id, processed, failed)
            if processed % 10 == 0:
                logger.info("run_progress run_id=%d processed=%d total=%d", run_id, processed, scan_output.stats.eligible_images)

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
    try:
        execute(
            """
            UPDATE settings
            SET root_repo = ?, categories_root = ?, confidence_threshold = ?, default_migrate_mode = ?
            WHERE id = 1
            """,
            (
                payload.root_repo,
                payload.categories_root,
                payload.confidence_threshold,
                payload.default_migrate_mode,
            ),
        )
    except Exception:
        logger.exception("failed to save settings")
        raise HTTPException(status_code=500, detail="Failed to persist settings")
    return payload


@router.get("/tags")
def search_tags(query: str = Query("", min_length=0), limit: int = 50) -> dict:
    known = sorted(load_known_tags(TAGS_CSV))
    if query:
        known = [t for t in known if query.lower() in t.lower()]
    return {"items": known[:limit], "count": len(known)}


@router.post("/runs/start", response_model=StartRunResponse)
def start_run(payload: StartRunRequest) -> StartRunResponse:
    logger.info("run_start_requested")
    current = _settings_from_db()
    resolved = resolve_settings(
        current, payload.root_repo, payload.categories_root, payload.confidence_threshold
    )
    root_repo = Path(resolved.root_repo).expanduser()
    categories_root = Path(resolved.categories_root).expanduser()

    if not resolved.root_repo or not resolved.categories_root:
        raise HTTPException(status_code=400, detail="root_repo and categories_root are required")

    try:
        known_tags = load_known_tags(TAGS_CSV)
        mappings = discover_tag_folders(categories_root, known_tags, payload.selected_folders)
        matched_tags = {m.matched_tag for m in mappings if m.matched and m.matched_tag}
        if not matched_tags:
            raise HTTPException(status_code=400, detail="No folder names map to known tags")
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
                total_images, processed_images, failed_images, cancel_requested
            ) VALUES (?, ?, ?, 'pending', 0, 0, 0, 0)
            """,
            (str(root_repo), str(categories_root), resolved.confidence_threshold),
        )
    except Exception:
        logger.exception("failed to create run row")
        raise HTTPException(status_code=500, detail="Failed to queue run")

    worker = threading.Thread(
        target=_execute_run,
        args=(run_id, root_repo, categories_root, resolved.confidence_threshold, matched_tags),
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
    return [_item_from_row(r) for r in rows]


@router.patch("/items/{item_id}", response_model=ClassifiedItem)
def update_item(item_id: int, payload: UpdateItemRequest) -> ClassifiedItem:
    row = fetch_one("SELECT * FROM items WHERE id = ?", (item_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    new_status = payload.status if payload.status is not None else row["status"]
    new_final_tag = payload.final_tag if payload.final_tag is not None else row["final_tag"]
    new_final_destination = (
        str(Path(row["suggested_destination"]).parent / new_final_tag)
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
            str(Path(row["suggested_destination"]).parent / new_final_tag)
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
