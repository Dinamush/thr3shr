from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .schemas import AppSettings, FolderMapping, MigrationResult, ScanStats

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
}


def normalize_tag_name(value: str) -> str:
    text = value.strip().lower()
    normalized = []
    for ch in text:
        if ch.isalnum():
            normalized.append(ch)
        elif ch in {" ", "-", ".", "/"}:
            normalized.append("_")
    return "".join(normalized).strip("_")


def load_known_tags(tags_csv_path: Path) -> set[str]:
    tags: set[str] = set()
    with tags_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = (row.get("tag") or "").strip()
            if tag:
                tags.add(tag)
    return tags


def discover_tag_folders(
    categories_root: Path, known_tags: set[str], selected_folders: list[str] | None = None
) -> list[FolderMapping]:
    if not categories_root.exists() or not categories_root.is_dir():
        raise ValueError(f"categories_root does not exist or is not a directory: {categories_root}")

    if selected_folders:
        folder_names = selected_folders
    else:
        folder_names = sorted([p.name for p in categories_root.iterdir() if p.is_dir()])

    mappings: list[FolderMapping] = []
    for folder in folder_names:
        normalized = normalize_tag_name(folder)
        matched_tag = normalized if normalized in known_tags else None
        mappings.append(
            FolderMapping(
                folder_name=folder,
                normalized_name=normalized,
                matched_tag=matched_tag,
                matched=matched_tag is not None,
            )
        )
    return mappings


@dataclass
class ScanOutput:
    image_paths: list[Path]
    stats: ScanStats


def scan_images(root_repo: Path) -> ScanOutput:
    if not root_repo.exists() or not root_repo.is_dir():
        raise ValueError(f"root_repo does not exist or is not a directory: {root_repo}")

    total_files = 0
    eligible: list[Path] = []
    ignored_unsupported = 0
    ignored_gif = 0
    failed_to_read = 0

    for path in root_repo.rglob("*"):
        if not path.is_file():
            continue
        total_files += 1
        ext = path.suffix.lower()
        if ext == ".gif":
            ignored_gif += 1
            continue
        if ext in VIDEO_EXTENSIONS:
            ignored_unsupported += 1
            continue
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            ignored_unsupported += 1
            continue

        try:
            with Image.open(path) as img:
                img.verify()
            eligible.append(path)
        except Exception:
            failed_to_read += 1

    return ScanOutput(
        image_paths=eligible,
        stats=ScanStats(
            total_files=total_files,
            eligible_images=len(eligible),
            ignored_unsupported=ignored_unsupported,
            ignored_gif=ignored_gif,
            failed_to_read=failed_to_read,
        ),
    )


def extract_scores(image_path: Path) -> dict[str, float]:
    from imgutils.tagging import get_mldanbooru_tags

    raw = get_mldanbooru_tags(
        str(image_path),
        threshold=0.0,
        size=448,
        keep_ratio=True,
        drop_overlap=True,
        use_real_name=False,
    )

    scores: dict[str, float] = {}
    if isinstance(raw, dict):
        for tag, score in raw.items():
            scores[str(tag)] = float(score)
        return scores

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                scores[str(item[0])] = float(item[1])
    return scores


def choose_best_tags(
    scores: dict[str, float], allowed_tags: set[str], max_secondary: int = 3
) -> tuple[str | None, float | None, list[dict[str, float]]]:
    candidates = [(tag, score) for tag, score in scores.items() if tag in allowed_tags]
    candidates.sort(key=lambda x: x[1], reverse=True)
    if not candidates:
        return None, None, []
    primary_tag, primary_score = candidates[0]
    secondary = [{"tag": t, "score": s} for t, s in candidates[1 : 1 + max_secondary]]
    return primary_tag, float(primary_score), secondary


def ensure_collision_free_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def migrate_file(source: Path, destination: Path, mode: str) -> MigrationResult:
    try:
        target = ensure_collision_free_destination(destination)
        if mode == "copy":
            shutil.copy2(source, target)
        elif mode == "move":
            shutil.move(source, target)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return MigrationResult(
            item_id=-1,
            source=str(source),
            destination=str(target),
            success=True,
        )
    except Exception as err:
        return MigrationResult(
            item_id=-1,
            source=str(source),
            destination=str(destination),
            success=False,
            error=str(err),
        )


def resolve_settings(
    current: AppSettings,
    root_repo: str | None,
    categories_root: str | None,
    confidence_threshold: float | None,
) -> AppSettings:
    return AppSettings(
        root_repo=root_repo if root_repo is not None else current.root_repo,
        categories_root=categories_root if categories_root is not None else current.categories_root,
        confidence_threshold=(
            confidence_threshold if confidence_threshold is not None else current.confidence_threshold
        ),
        default_migrate_mode=current.default_migrate_mode,
    )
