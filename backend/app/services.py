from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .schemas import AppSettings, FolderMapping, MigrationResult, ScanStats

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".tiff"}
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
logger = logging.getLogger(__name__)
_BATCH_INFERENCE_SUPPORTED: bool | None = None
_WINDOWS_FORBIDDEN_CHARS = set('<>:"/\\|?*')


def normalize_tag_name(value: str) -> str:
    text = value.strip().lower()
    normalized = []
    for ch in text:
        if ch.isalnum():
            normalized.append(ch)
        elif ch in {" ", "-", ".", "/", "_"}:
            normalized.append("_")
    return "".join(normalized).strip("_")


def sanitize_folder_name(value: str) -> str:
    """
    Convert an arbitrary tag into a filesystem-safe folder name.
    Keeps names readable while replacing common invalid path characters.
    """
    cleaned = []
    for ch in value.strip():
        if ch in _WINDOWS_FORBIDDEN_CHARS or ord(ch) < 32:
            cleaned.append("_")
        else:
            cleaned.append(ch)
    result = "".join(cleaned).strip(" .")
    return result or "_"


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

    known_by_normalized: dict[str, str] = {}
    for tag in sorted(known_tags):
        normalized_tag = normalize_tag_name(tag)
        # Keep first stable mapping for normalized fallback.
        if normalized_tag not in known_by_normalized:
            known_by_normalized[normalized_tag] = tag

    mappings: list[FolderMapping] = []
    for folder in folder_names:
        normalized = normalize_tag_name(folder)
        # Prefer exact known tag match first for selected tags that contain
        # special syntax (e.g. `remodel_(kantai_collection)`).
        matched_tag = folder if folder in known_tags else known_by_normalized.get(normalized)
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


def scan_images(
    root_repo: Path,
    exclude_dirs: set[Path] | None = None,
    recursive: bool = True,
) -> ScanOutput:
    if not root_repo.exists() or not root_repo.is_dir():
        raise ValueError(f"root_repo does not exist or is not a directory: {root_repo}")

    # Resolve exclude_dirs to absolute paths so prefix matching is reliable.
    resolved_excludes: set[Path] = set()
    if exclude_dirs:
        for d in exclude_dirs:
            try:
                resolved_excludes.add(d.resolve())
            except OSError:
                pass

    total_files = 0
    eligible: list[Path] = []
    ignored_unsupported = 0
    ignored_gif = 0
    failed_to_read = 0

    iterator = root_repo.rglob("*") if recursive else root_repo.iterdir()
    for path in iterator:
        try:
            if not path.is_file():
                continue
        except OSError:
            failed_to_read += 1
            continue

        # Skip any path that lives inside an excluded directory.
        if resolved_excludes:
            try:
                resolved_path = path.resolve()
                if any(
                    resolved_path == exc or resolved_path.is_relative_to(exc)
                    for exc in resolved_excludes
                ):
                    continue
            except OSError:
                pass

        total_files += 1
        ext = path.suffix.lower()
        if ext == ".gif":
            ignored_gif += 1
            continue
        if ext in VIDEO_EXTENSIONS:
            ignored_unsupported += 1
            continue

        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            # Known image extension: accept without Pillow verification.
            # Corrupt or unreadable files are handled gracefully during inference.
            eligible.append(path)
            continue

        # Unknown or missing extension: try content-based detection.
        try:
            with Image.open(path) as img:
                if not img.format:
                    raise ValueError("Not a recognizable image format")
            eligible.append(path)
        except Exception:
            if ext == "":
                failed_to_read += 1
            else:
                ignored_unsupported += 1

    logger.info(
        "scan_complete root=%s recursive=%s total_files=%d eligible=%d "
        "ignored_gif=%d ignored_unsupported=%d failed_to_read=%d excluded_dirs=%d",
        root_repo,
        recursive,
        total_files,
        len(eligible),
        ignored_gif,
        ignored_unsupported,
        failed_to_read,
        len(resolved_excludes),
    )
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
        drop_overlap=False,
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


def extract_scores_batch(image_paths: list[Path]) -> list[dict[str, float]]:
    if not image_paths:
        return []
    global _BATCH_INFERENCE_SUPPORTED

    if _BATCH_INFERENCE_SUPPORTED is False:
        return [extract_scores(p) for p in image_paths]

    from imgutils.tagging import get_mldanbooru_tags

    try:
        raw = get_mldanbooru_tags(
            [str(p) for p in image_paths],
            threshold=0.0,
            size=448,
            keep_ratio=True,
            drop_overlap=False,
            use_real_name=False,
        )
    except TypeError as err:
        # Current imgutils build treats list input as invalid image type.
        if "Unknown image type" in str(err):
            if _BATCH_INFERENCE_SUPPORTED is not False:
                logger.warning("batch inference not supported by imgutils; using per-image fallback")
            _BATCH_INFERENCE_SUPPORTED = False
            return [extract_scores(p) for p in image_paths]
        raise

    parsed: list[dict[str, float]] = []
    if isinstance(raw, list) and len(raw) == len(image_paths):
        _BATCH_INFERENCE_SUPPORTED = True
        for entry in raw:
            if isinstance(entry, dict):
                parsed.append({str(k): float(v) for k, v in entry.items()})
                continue
            if isinstance(entry, list):
                scores: dict[str, float] = {}
                for item in entry:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        scores[str(item[0])] = float(item[1])
                parsed.append(scores)
                continue
            raise TypeError("Unexpected batch inference entry format")
        return parsed

    # If the backend or library returns an unexpected shape, fall back to per-image inference.
    logger.warning("batch inference unsupported format; falling back to per-image path")
    _BATCH_INFERENCE_SUPPORTED = False
    return [extract_scores(p) for p in image_paths]


def choose_best_tags(
    scores: dict[str, float], allowed_tags: set[str], max_secondary: int = 3
) -> tuple[str | None, float | None, list[dict[str, float]]]:
    allowed_by_normalized = {normalize_tag_name(tag): tag for tag in allowed_tags}
    best_by_allowed: dict[str, float] = {}
    for tag, score in scores.items():
        resolved = allowed_by_normalized.get(tag) or allowed_by_normalized.get(normalize_tag_name(tag))
        if resolved is None:
            continue
        current = best_by_allowed.get(resolved)
        if current is None or score > current:
            best_by_allowed[resolved] = float(score)
    candidates = list(best_by_allowed.items())
    candidates.sort(key=lambda x: x[1], reverse=True)
    if not candidates:
        return None, None, []
    primary_tag, primary_score = candidates[0]
    secondary = [{"tag": t, "score": s} for t, s in candidates[1 : 1 + max_secondary]]
    return primary_tag, float(primary_score), secondary


def ensure_collision_free_destination(destination: Path, max_attempts: int = 10_000) -> Path:
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    idx = 1
    while idx <= max_attempts:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1
    raise RuntimeError("Unable to resolve collision-free destination name")


def migrate_file(source: Path, destination: Path, mode: str) -> MigrationResult:
    try:
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        target = ensure_collision_free_destination(destination)
        if mode == "copy":
            shutil.copy2(source, target)
        elif mode == "move":
            shutil.move(source, target)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        if not target.exists():
            raise RuntimeError("Migration reported success but target file is missing")
        return MigrationResult(
            item_id=-1,
            source=str(source),
            destination=str(target),
            success=True,
        )
    except Exception as err:
        logger.warning("file migration failed source=%s destination=%s error=%s", source, destination, err)
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
        scan_recursive=current.scan_recursive,
    )
