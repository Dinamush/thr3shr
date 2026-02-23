from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ItemStatus = Literal["proposed", "reviewed", "approved", "rejected", "migrated"]
MigrateMode = Literal["move", "copy"]
RunLifecycleStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class AppSettings(BaseModel):
    root_repo: str = ""
    categories_root: str = ""
    confidence_threshold: float = 0.6
    default_migrate_mode: MigrateMode = "copy"
    scan_recursive: bool = True
    experimental_media_enabled: bool = False


class SaveSettingsRequest(AppSettings):
    pass


class FolderMapping(BaseModel):
    folder_name: str
    normalized_name: str
    matched_tag: str | None = None
    matched: bool


class ScanStats(BaseModel):
    total_files: int
    eligible_images: int
    ignored_unsupported: int
    ignored_gif: int
    failed_to_read: int


class StartRunRequest(BaseModel):
    root_repo: str | None = None
    categories_root: str | None = None
    confidence_threshold: float | None = None
    selected_folders: list[str] | None = None


class SecondarySuggestion(BaseModel):
    tag: str
    score: float


class ClassifiedItem(BaseModel):
    id: int
    run_id: int
    file_path: str
    relative_path: str
    primary_tag: str | None
    primary_score: float | None
    secondary_suggestions: list[SecondarySuggestion] = Field(default_factory=list)
    full_scores: dict[str, float] | None = None
    suggested_destination: str | None
    final_tag: str | None
    final_destination: str | None
    status: ItemStatus
    needs_review: bool
    review_reason: str | None
    migrated_to: str | None


class StartRunResponse(BaseModel):
    run_id: int
    status: RunLifecycleStatus
    stats: ScanStats | None = None
    mappings: list[FolderMapping] = Field(default_factory=list)
    unmatched_folders: list[str] = Field(default_factory=list)
    created_items: int = 0
    message: str = "Run queued"


class RunStatusResponse(BaseModel):
    run_id: int
    status: RunLifecycleStatus
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    progress_pct: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    last_error: str | None = None
    cancel_requested: bool = False
    has_items: bool = False
    inference_mode: str | None = None
    batch_size: int | None = None
    avg_infer_ms_per_image: float | None = None
    queue_seed: int | None = None


class UpdateItemRequest(BaseModel):
    status: ItemStatus | None = None
    final_tag: str | None = None


class BatchUpdateRequest(BaseModel):
    item_ids: list[int]
    status: ItemStatus | None = None
    final_tag: str | None = None


class MigrateRequest(BaseModel):
    mode: MigrateMode
    create_missing_folders: bool = True


class MigrationResult(BaseModel):
    item_id: int
    source: str
    destination: str | None
    success: bool
    error: str | None = None


class MigrateResponse(BaseModel):
    mode: MigrateMode
    total_candidates: int
    migrated_count: int
    failed_count: int
    results: list[MigrationResult]
