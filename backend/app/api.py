from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from .schemas import (
    AppSettings,
    BatchUpdateRequest,
    ClassifiedItem,
    MigrateRequest,
    MigrateResponse,
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
from .storage import execute, execute_many, fetch_all, fetch_one, from_json, to_json

router = APIRouter(prefix="/api")
REPO_ROOT = Path(__file__).resolve().parents[2]
TAGS_CSV = REPO_ROOT / "tags.csv"


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
        secondary_suggestions=from_json(row["secondary_json"]),
        suggested_destination=row["suggested_destination"],
        final_tag=row["final_tag"],
        final_destination=row["final_destination"],
        status=row["status"],
        needs_review=bool(row["needs_review"]),
        review_reason=row["review_reason"],
        migrated_to=row["migrated_to"],
    )


@router.get("/settings", response_model=AppSettings)
def get_settings() -> AppSettings:
    return _settings_from_db()


@router.put("/settings", response_model=AppSettings)
def save_settings(payload: SaveSettingsRequest) -> AppSettings:
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
    return payload


@router.get("/tags")
def search_tags(query: str = Query("", min_length=0), limit: int = 50) -> dict:
    known = sorted(load_known_tags(TAGS_CSV))
    if query:
        known = [t for t in known if query.lower() in t.lower()]
    return {"items": known[:limit], "count": len(known)}


@router.post("/runs/start", response_model=StartRunResponse)
def start_run(payload: StartRunRequest) -> StartRunResponse:
    current = _settings_from_db()
    resolved = resolve_settings(
        current, payload.root_repo, payload.categories_root, payload.confidence_threshold
    )
    root_repo = Path(resolved.root_repo).expanduser()
    categories_root = Path(resolved.categories_root).expanduser()

    if not resolved.root_repo or not resolved.categories_root:
        raise HTTPException(status_code=400, detail="root_repo and categories_root are required")

    known_tags = load_known_tags(TAGS_CSV)
    mappings = discover_tag_folders(categories_root, known_tags, payload.selected_folders)
    matched_tags = {m.matched_tag for m in mappings if m.matched and m.matched_tag}
    if not matched_tags:
        raise HTTPException(status_code=400, detail="No folder names map to known tags")

    scan_output = scan_images(root_repo)

    run_id = execute(
        "INSERT INTO runs (root_repo, categories_root, confidence_threshold) VALUES (?, ?, ?)",
        (str(root_repo), str(categories_root), resolved.confidence_threshold),
    )

    rows_to_insert: list[tuple] = []
    for image_path in scan_output.image_paths:
        relative_path = str(image_path.relative_to(root_repo))
        scores = extract_scores(image_path)
        primary_tag, primary_score, secondary = choose_best_tags(scores, matched_tags)
        needs_review = False
        reason = None
        status = "proposed"

        if primary_tag is None:
            needs_review = True
            reason = "No matching tags found in selected folders."
        elif primary_score is not None and primary_score < resolved.confidence_threshold:
            needs_review = True
            reason = (
                f"Below threshold ({primary_score:.3f} < {resolved.confidence_threshold:.3f})."
            )

        suggested_destination = (
            str(categories_root / primary_tag) if primary_tag is not None else None
        )
        rows_to_insert.append(
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
                status,
                1 if needs_review else 0,
                reason,
            )
        )

    if rows_to_insert:
        execute_many(
            """
            INSERT INTO items (
                run_id, file_path, relative_path, primary_tag, primary_score, secondary_json,
                suggested_destination, final_tag, final_destination, status, needs_review, review_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )

    return StartRunResponse(
        run_id=run_id,
        stats=scan_output.stats,
        mappings=mappings,
        unmatched_folders=[m.folder_name for m in mappings if not m.matched],
        created_items=len(rows_to_insert),
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


@router.post("/runs/{run_id}/batch")
def batch_update(run_id: int, payload: BatchUpdateRequest) -> dict:
    if not payload.item_ids:
        return {"updated": 0}
    updated = 0
    for item_id in payload.item_ids:
        row = fetch_one("SELECT * FROM items WHERE id = ? AND run_id = ?", (item_id, run_id))
        if not row:
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
    return {"updated": updated}


@router.post("/runs/{run_id}/migrate", response_model=MigrateResponse)
def migrate_run(run_id: int, payload: MigrateRequest) -> MigrateResponse:
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

        destination_folder = Path(final_destination)
        if payload.create_missing_folders:
            destination_folder.mkdir(parents=True, exist_ok=True)
        destination = destination_folder / source.name
        migration_result = migrate_file(source, destination, payload.mode)
        migration_result.item_id = row["id"]
        if migration_result.success:
            migrated_count += 1
            execute(
                "UPDATE items SET status = 'migrated', migrated_to = ? WHERE id = ?",
                (migration_result.destination, row["id"]),
            )
        else:
            failed_count += 1
        results.append(migration_result.model_dump())

    return MigrateResponse(
        mode=payload.mode,
        total_candidates=len(rows),
        migrated_count=migrated_count,
        failed_count=failed_count,
        results=results,
    )
