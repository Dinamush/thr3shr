from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ItemStatus = Literal["proposed", "reviewed", "approved", "rejected", "migrated"]
MigrateMode = Literal["move", "copy"]


class AppSettings(BaseModel):
    root_repo: str = ""
    categories_root: str = ""
    confidence_threshold: float = 0.6
    default_migrate_mode: MigrateMode = "copy"


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
    suggested_destination: str | None
    final_tag: str | None
    final_destination: str | None
    status: ItemStatus
    needs_review: bool
    review_reason: str | None
    migrated_to: str | None


class StartRunResponse(BaseModel):
    run_id: int
    stats: ScanStats
    mappings: list[FolderMapping]
    unmatched_folders: list[str]
    created_items: int


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
