from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ItemStatus = Literal["proposed", "reviewed", "approved", "rejected", "migrated"]
MigrateMode = Literal["move", "copy"]
RunLifecycleStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
TaggerModel = Literal["ml_danbooru", "wd_swinv2_v3", "wd_eva02_large"]


class AppSettings(BaseModel):
    root_repo: str = ""
    categories_root: str = ""
    confidence_threshold: float = 0.6
    default_migrate_mode: MigrateMode = "copy"
    scan_recursive: bool = True
    experimental_media_enabled: bool = False
    # Dedicated real-vs-anime ONNX gate (imgutils / deepghs anime_real_cls).
    experimental_style_detector_enabled: bool = False
    # Shuck3r-style persisted preferences (survive reload / restart).
    selected_tags: list[str] = Field(default_factory=list)
    # Shared ORT run lock; preprocess overlaps across workers. Prefer 2 on GPU.
    max_inference_workers: int = Field(default=2, ge=1, le=16)
    inference_batch_size: int = Field(default=4, ge=1, le=64)
    force_cpu_inference: bool = False
    tagger_model: TaggerModel = "wd_swinv2_v3"
    wd_general_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


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
    # classify = normal multi-folder run; real_life_filter = only keep real_life hits.
    run_mode: Literal["classify", "real_life_filter"] = "classify"


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
    global_top_tags: list[SecondarySuggestion] = Field(default_factory=list)
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


class ReclassifyRequest(BaseModel):
    tagger_model: TaggerModel
    item_ids: list[int] | None = None


class ReclassifyResponse(BaseModel):
    run_id: int
    status: RunLifecycleStatus
    eligible_count: int
    tagger_model: TaggerModel
    message: str = "Reclassify queued"


class SfwDebugEvalRequest(BaseModel):
    source: str = "safebooru"
    tags: list[str] = Field(default_factory=list)
    count: int = Field(default=10, ge=5, le=30)


class SfwDebugRecallRow(BaseModel):
    tag: str
    present_in_posts: int
    hits_at_threshold: int
    hit_rate: float | None = None


class SfwDebugEvalItem(BaseModel):
    source: str
    post_id: str
    rating: str
    file_name: str
    known_tags: list[str] = Field(default_factory=list)
    pull_tag_scores: dict[str, float | None] = Field(default_factory=dict)
    global_top_tags: list[SecondarySuggestion] = Field(default_factory=list)
    primary_tag: str | None = None
    primary_score: float | None = None
    needs_review: bool = True
    review_reason: str | None = None
    suggested_folder: str | None = None
    secondary_suggestions: list[SecondarySuggestion] = Field(default_factory=list)


class SfwDebugEvalResponse(BaseModel):
    source: str
    source_label: str
    sfw_policy: str
    query: str
    tags: list[str]
    count_requested: int
    count_evaluated: int
    tagger_model: str
    confidence_threshold: float
    destination_tags: list[str] = Field(default_factory=list)
    recall: list[SfwDebugRecallRow] = Field(default_factory=list)
    items: list[SfwDebugEvalItem] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RealismDebugEvalRequest(BaseModel):
    count_per_class: int = Field(default=12, ge=5, le=40)
    tagger_model: str | None = None
    compare_models: bool = False


class StyleDebugEvalRequest(BaseModel):
    count_per_class: int = Field(default=12, ge=5, le=40)
    tagger_model: str | None = None
    detectors: list[str] | None = None
    uncertain_threshold: float = Field(default=0.85, ge=0.5, le=0.99)


class RealismDebugEvalItem(BaseModel):
    sample_id: str
    label: str
    bucket: str
    source: str
    query: str = ""
    title: str = ""
    file_name: str
    predicted_bucket: str
    correct: bool
    primary_folder: str | None = None
    primary_score: float | None = None
    evidence_scores: dict[str, float] = Field(default_factory=dict)
    global_top_tags: list[SecondarySuggestion] = Field(default_factory=list)


class RealismDebugEvalResponse(BaseModel):
    count_per_class_requested: int
    count_photos_fetched: int = 0
    count_anime_fetched: int = 0
    count_evaluated: int
    tagger_model: str
    wd_general_threshold: float
    selected_destinations: list[str] = Field(default_factory=list)
    metrics: dict[str, object] = Field(default_factory=dict)
    by_label: dict[str, object] = Field(default_factory=dict)
    conclusion: dict[str, object] = Field(default_factory=dict)
    items: list[RealismDebugEvalItem] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    photo_source: str = ""
    anime_source: str = ""
    # Present when compare_models=true (loose dict payload).
    multi_model: dict[str, object] | None = None


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
    eta_seconds_remaining: float | None = None
    eta_finish_at: str | None = None
    queue_seed: int | None = None
    tagger_model: str | None = None


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
