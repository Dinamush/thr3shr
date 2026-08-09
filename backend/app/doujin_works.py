"""Doujin / comic-work classification: folder or archive as one unit."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .media_pooling import pool_presence
from .services import SUPPORTED_IMAGE_EXTENSIONS, extract_scores
from .taxonomy import choose_best_destination, resolve_taxonomy_folder

logger = logging.getLogger(__name__)

ARCHIVE_EXTENSIONS = {".cbz", ".zip"}

DOUJIN_FAVOURITE_FOLDERS: tuple[str, ...] = (
    "loli",
    "shota",
    "milf",
    "group_sex",
    "fertilization",
    "monster_girl",
    "incest",
    "bestiality",
    "Pokemon",
    "NTR",
    "tentacles",
    "furry",
    "android",
)

DEFAULT_SAMPLE_COUNT = 12
MIN_SAMPLE_COUNT = 4
MAX_SAMPLE_COUNT = 16


@dataclass
class DoujinWork:
    path: Path
    name: str
    kind: str  # "folder" | "archive"


@dataclass
class DoujinClassifyResult:
    work: DoujinWork
    scores: dict[str, float] = field(default_factory=dict)
    primary_tag: str | None = None
    primary_score: float | None = None
    category_tags: list[dict[str, float]] = field(default_factory=list)
    needs_review: bool = True
    reason: str | None = None
    inference_failed: bool = False


def scan_doujin_works(root: Path) -> list[DoujinWork]:
    """Immediate child folders with images, plus top-level archives."""
    root = root.expanduser()
    if not root.is_dir():
        raise ValueError(f"Doujin root is not a directory: {root}")
    works: list[DoujinWork] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            if _folder_image_paths(child):
                works.append(DoujinWork(path=child, name=child.name, kind="folder"))
        elif child.is_file() and child.suffix.lower() in ARCHIVE_EXTENSIONS:
            works.append(DoujinWork(path=child, name=child.name, kind="archive"))
    return works


def sample_count_for_pages(page_count: int, requested: int = DEFAULT_SAMPLE_COUNT) -> int:
    if page_count <= 0:
        return 0
    n = max(MIN_SAMPLE_COUNT, min(MAX_SAMPLE_COUNT, int(requested)))
    return min(page_count, n)


def even_page_indices(page_count: int, sample_count: int) -> list[int]:
    if page_count <= 0 or sample_count <= 0:
        return []
    if sample_count >= page_count:
        return list(range(page_count))
    if sample_count == 1:
        return [0]
    idxs = {0}
    for i in range(1, sample_count):
        idxs.add(round(i * (page_count - 1) / (sample_count - 1)))
    return sorted(idxs)


def load_sample_images(
    work: DoujinWork, *, sample_count: int = DEFAULT_SAMPLE_COUNT
) -> list[Image.Image]:
    if work.kind == "folder":
        paths = _folder_image_paths(work.path)
        idxs = even_page_indices(
            len(paths), sample_count_for_pages(len(paths), sample_count)
        )
        images: list[Image.Image] = []
        for i in idxs:
            try:
                with Image.open(paths[i]) as im:
                    images.append(im.convert("RGB"))
            except Exception:
                logger.warning("doujin_sample_failed path=%s", paths[i])
        return images

    members = _archive_image_members(work.path)
    idxs = even_page_indices(
        len(members), sample_count_for_pages(len(members), sample_count)
    )
    images = []
    with zipfile.ZipFile(work.path, "r") as zf:
        for i in idxs:
            try:
                data = zf.read(members[i])
                with Image.open(io.BytesIO(data)) as im:
                    images.append(im.convert("RGB"))
            except Exception:
                logger.warning(
                    "doujin_archive_sample_failed path=%s member=%s",
                    work.path,
                    members[i],
                )
    return images


def classify_doujin_work(
    work: DoujinWork,
    *,
    matched_tags: set[str],
    confidence_threshold: float,
    tagger_model: str = "wd_swinv2_v3",
    wd_general_threshold: float = 0.35,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> DoujinClassifyResult:
    try:
        images = load_sample_images(work, sample_count=sample_count)
        if not images:
            return DoujinClassifyResult(
                work=work,
                needs_review=True,
                reason="No readable sample pages in this work.",
                inference_failed=True,
            )
        frame_scores: list[tuple[float | None, dict[str, float]]] = []
        for idx, image in enumerate(images):
            scores = extract_scores(
                image,
                tagger_model=tagger_model,
                wd_general_threshold=wd_general_threshold,
            )
            frame_scores.append((float(idx), scores))
        min_hits = 2 if len(frame_scores) >= 2 else 1
        pooled = pool_presence(
            frame_scores,
            support_thr=min(0.35, float(wd_general_threshold)),
            min_hits=min_hits,
            top_k=3,
            scene_gap_s=None,
        )
        primary, primary_score, secondary = choose_best_destination(
            pooled, matched_tags, max_secondary=max(1, len(matched_tags))
        )
        category_tags: list[dict[str, float]] = []
        if primary is not None and primary_score is not None:
            category_tags.append({"tag": primary, "score": float(primary_score)})
        for row in secondary:
            category_tags.append({"tag": row["tag"], "score": float(row["score"])})

        needs_review = False
        reason = None
        if primary is None:
            needs_review = True
            reason = "No matching favourite tags for this work."
        elif primary_score is not None and primary_score < confidence_threshold:
            needs_review = True
            reason = (
                f"Below threshold ({primary_score:.3f} < {confidence_threshold:.3f})."
            )

        return DoujinClassifyResult(
            work=work,
            scores=pooled,
            primary_tag=primary,
            primary_score=primary_score,
            category_tags=category_tags,
            needs_review=needs_review,
            reason=reason,
            inference_failed=False,
        )
    except Exception:
        logger.exception("doujin_classify_failed work=%s", work.path)
        return DoujinClassifyResult(
            work=work,
            needs_review=True,
            reason="Inference failed for this work; requires manual review.",
            inference_failed=True,
        )


def doujin_tag_folder_name(tag: str) -> str:
    folder = resolve_taxonomy_folder(tag)
    return folder.folder if folder is not None else tag


def doujin_destination_folder(categories_root: Path, primary_tag: str) -> Path:
    """Parent folder for a work: ``categories_root/Doujins/<tag>/``."""
    return categories_root.expanduser() / "Doujins" / doujin_tag_folder_name(primary_tag)


def doujin_destination(
    categories_root: Path, primary_tag: str, work_name: str
) -> Path:
    return doujin_destination_folder(categories_root, primary_tag) / work_name


def create_tag_link(link_path: Path, target: Path) -> None:
    """Directory junction (folders) or hardlink (files) on Windows."""
    link_path = link_path.expanduser()
    target = target.expanduser().resolve()
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            link_path.symlink_to(target, target_is_directory=True)
        return
    try:
        os.link(target, link_path)
    except OSError:
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/H", str(link_path), str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            link_path.symlink_to(target)


def write_tags_sidecar(work_path: Path, tags: list[str]) -> None:
    if work_path.is_dir():
        sidecar = work_path / "tags.json"
    else:
        sidecar = work_path.parent / f"{work_path.name}.tags.json"
    try:
        sidecar.write_text(json.dumps({"tags": tags}, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("doujin_sidecar_write_failed path=%s", sidecar)


def resolve_work_cover_file(work_path: Path) -> Path | None:
    work_path = work_path.expanduser()
    if work_path.is_dir():
        images = _folder_image_paths(work_path)
        return images[0] if images else None
    if work_path.is_file() and work_path.suffix.lower() in ARCHIVE_EXTENSIONS:
        members = _archive_image_members(work_path)
        if not members:
            return None
        cache_dir = work_path.parent / ".doujin_preview_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = cache_dir / f"{work_path.stem}_cover.jpg"
        if out.exists():
            return out
        with zipfile.ZipFile(work_path, "r") as zf:
            data = zf.read(members[0])
        with Image.open(io.BytesIO(data)) as im:
            im.convert("RGB").save(out, format="JPEG", quality=85)
        return out
    if work_path.is_file():
        return work_path
    return None


def _folder_image_paths(folder: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(folder.rglob("*"), key=lambda x: str(x).lower()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            files.append(p)
    return files


def _archive_image_members(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive, "r") as zf:
        names = [
            n
            for n in zf.namelist()
            if not n.endswith("/")
            and Path(n).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            and not Path(n).name.startswith(".")
        ]
    return sorted(names, key=lambda n: n.lower())
