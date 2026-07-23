from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile, ImageSequence

from .schemas import AppSettings, FolderMapping, MigrationResult, ScanStats

# Animated/corrupt downloads often truncate; allow decode of usable prefix frames.
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
# Length-scaled sampling for GIF/video. Fixed MEDIA_SAMPLE_FRAMES overrides scaling.
DEFAULT_MEDIA_SAMPLE_MIN = 4
DEFAULT_MEDIA_SAMPLE_MAX = 24
# Videos: aim for about one sample every N seconds.
DEFAULT_MEDIA_SAMPLE_SECONDS = 0.75
# GIFs: aim for about one sample every N source frames (before min/max clamp).
DEFAULT_MEDIA_GIF_FRAME_STRIDE = 3
logger = logging.getLogger(__name__)
_BATCH_INFERENCE_SUPPORTED: bool | None = None
_WINDOWS_FORBIDDEN_CHARS = set('<>:"/\\|?*')
# Legacy lock retained for any callers that still expect it. The owned
# InferenceEngine serializes only session.run internally now.
_INFERENCE_LOCK = threading.Lock()


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(lo, min(value, hi))


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(lo, min(value, hi))


def _media_sample_bounds() -> tuple[int, int]:
    minimum = _env_int("MEDIA_SAMPLE_FRAMES_MIN", DEFAULT_MEDIA_SAMPLE_MIN, lo=1, hi=32)
    maximum = _env_int("MEDIA_SAMPLE_FRAMES_MAX", DEFAULT_MEDIA_SAMPLE_MAX, lo=1, hi=48)
    if maximum < minimum:
        maximum = minimum
    return minimum, maximum


def scaled_media_sample_count(
    *,
    total_frames: int | None = None,
    duration_seconds: float | None = None,
) -> int:
    """Choose how many frames to sample from media length.

    Longer clips get more samples so late-starting action is less likely to be
    missed. Caps keep inference cost bounded.
    """
    minimum, maximum = _media_sample_bounds()
    # Optional hard override (old knob): exact count, ignoring duration scaling.
    override = os.getenv("MEDIA_SAMPLE_FRAMES", "").strip()
    if override:
        try:
            return max(1, min(int(override), maximum))
        except ValueError:
            pass

    target = minimum
    if duration_seconds is not None and duration_seconds > 0:
        seconds_per = _env_float(
            "MEDIA_SAMPLE_SECONDS",
            DEFAULT_MEDIA_SAMPLE_SECONDS,
            lo=0.25,
            hi=5.0,
        )
        target = int(round(float(duration_seconds) / seconds_per))
    elif total_frames is not None and total_frames > 0:
        stride = _env_int(
            "MEDIA_GIF_FRAME_STRIDE",
            DEFAULT_MEDIA_GIF_FRAME_STRIDE,
            lo=1,
            hi=30,
        )
        target = int((int(total_frames) + stride - 1) // stride)

    target = max(minimum, min(target, maximum))
    if total_frames is not None and total_frames > 0:
        target = min(target, int(total_frames))
    return max(1, target)


def _even_frame_indices(total_frames: int, sample_count: int) -> list[int]:
    """Pick up to sample_count indices evenly across [0, total_frames)."""
    if total_frames <= 0:
        return []
    n = min(max(1, sample_count), total_frames)
    if n == 1:
        return [0]
    if n == total_frames:
        return list(range(total_frames))
    # Midpoints of n equal bins — covers start/middle/end without double-counting.
    return sorted({min(total_frames - 1, int((i + 0.5) * total_frames / n)) for i in range(n)})


def pool_frame_scores(score_maps: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-frame tag scores into one map.

    Uses mean across frames (missing tag => 0.0) and keeps a presence-aware
    boost so tags strong on a minority of frames are not washed out:
    final = max(mean, max_score * (hits / n_frames)).
    """
    if not score_maps:
        return {}
    n = len(score_maps)
    keys: set[str] = set()
    for scores in score_maps:
        keys.update(scores)
    pooled: dict[str, float] = {}
    for key in keys:
        values = [float(scores.get(key, 0.0)) for scores in score_maps]
        mean = sum(values) / n
        peak = max(values)
        hits = sum(1 for value in values if value > 0.0)
        presence = peak * (hits / n)
        pooled[key] = max(mean, presence)
    return pooled


def sample_gif_frames(image_path: Path, sample_count: int | None = None) -> list[Image.Image]:
    """Load evenly spaced RGB frames from a GIF (handles truncated files)."""
    frames: list[Image.Image] = []
    with Image.open(image_path) as gif:
        total = int(getattr(gif, "n_frames", 0) or 0)
        if total < 1:
            # Some GIFs omit n_frames; fall back to sequence length.
            total = sum(1 for _ in ImageSequence.Iterator(gif))
            gif.seek(0)
        # Approximate duration from per-frame delays when available.
        duration_s = None
        try:
            # Pillow duration is milliseconds per frame (often constant).
            delay_ms = gif.info.get("duration")
            if delay_ms and total > 0:
                duration_s = (float(delay_ms) * float(total)) / 1000.0
        except Exception:
            duration_s = None
        count = (
            max(1, sample_count)
            if sample_count is not None
            else scaled_media_sample_count(total_frames=total, duration_seconds=duration_s)
        )
        indices = _even_frame_indices(total, count)
        if not indices:
            gif.seek(0)
            frames.append(gif.convert("RGB"))
            return frames
        for index in indices:
            try:
                gif.seek(index)
                frames.append(gif.convert("RGB"))
            except Exception:
                logger.warning(
                    "gif_frame_skip path=%s index=%s", image_path, index, exc_info=True
                )
    if not frames:
        raise RuntimeError(f"Unable to decode any frames from GIF: {image_path}")
    return frames


def sample_video_frames(image_path: Path, sample_count: int | None = None) -> list[Image.Image]:
    """Load evenly spaced RGB frames from a video via OpenCV (bundled FFmpeg)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(image_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for frame sampling: {image_path}")

    try:
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration_s = None
        if reported > 0 and fps > 1e-3:
            duration_s = float(reported) / fps
        count = (
            max(1, sample_count)
            if sample_count is not None
            else scaled_media_sample_count(
                total_frames=reported if reported > 0 else None,
                duration_seconds=duration_s,
            )
        )
        indices = _even_frame_indices(reported, count) if reported > 0 else []
        frames: list[Image.Image] = []

        if indices:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(np.asarray(rgb)))
        else:
            # Unknown length: grab up to `count` frames spaced by skipping.
            grabbed: list[Image.Image] = []
            step = 1
            pos = 0
            while len(grabbed) < count:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                if pos % step == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    grabbed.append(Image.fromarray(np.asarray(rgb)))
                pos += 1
                if pos > 0 and pos % (count * 4) == 0:
                    step = min(step + 1, 30)
            frames = grabbed

        if not frames:
            raise RuntimeError(f"Unable to decode any frames from video: {image_path}")
        logger.info(
            "video_frames_sampled path=%s frames=%d reported=%d duration_s=%s",
            image_path,
            len(frames),
            reported,
            f"{duration_s:.2f}" if duration_s is not None else "unknown",
        )
        return frames
    finally:
        cap.release()


def _score_pil_frames(
    frames: list[Image.Image],
    *,
    tagger_model: str,
    wd_general_threshold: float,
) -> dict[str, float]:
    from .inference_engine import get_engine
    from .providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

    ensure_nvidia_dll_search_path()
    preload_onnx_runtime_dlls()
    if not frames:
        return {}
    maps = get_engine().score_many(
        list(frames),
        tagger_model=tagger_model,
        wd_general_threshold=wd_general_threshold,
        batch_size=min(8, len(frames)),
    )
    return pool_frame_scores(maps)


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


def is_experimental_media(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext == ".gif" or ext in VIDEO_EXTENSIONS


def _preview_cache_dir() -> Path:
    root = Path(__file__).resolve().parent.parent / ".preview_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def media_preview_still_jpeg(path: Path, *, max_edge: int = 320) -> bytes:
    """Extract a small JPEG still for GIF/video table thumbnails.

    Serving full MP4s as ``<video>`` thumbs fails under load (hundreds of
    parallel range requests). A cached JPEG works with a normal ``<img>``.
    """
    resolved = path.resolve()
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    cache_key = hashlib.sha1(
        f"{resolved}|{mtime_ns}|{max_edge}".encode("utf-8", errors="replace")
    ).hexdigest()
    cache_path = _preview_cache_dir() / f"{cache_key}.jpg"
    if cache_path.is_file():
        return cache_path.read_bytes()

    suffix = resolved.suffix.lower()
    if suffix == ".gif":
        frames = sample_gif_frames(resolved, sample_count=1)
        image = frames[0]
    elif suffix in VIDEO_EXTENSIONS:
        frames = sample_video_frames(resolved, sample_count=1)
        image = frames[0]
    else:
        with Image.open(resolved) as opened:
            image = opened.convert("RGB")

    image = image.convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=72, optimize=True)
    data = buffer.getvalue()
    try:
        cache_path.write_bytes(data)
    except OSError:
        logger.warning("preview_cache_write_failed path=%s", cache_path, exc_info=True)
    return data


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
    if selected_folders:
        # Selected-tag runs may target a destination root that does not exist yet.
        if categories_root.exists() and not categories_root.is_dir():
            raise ValueError(f"categories_root exists but is not a directory: {categories_root}")
        folder_names = selected_folders
    else:
        if not categories_root.exists() or not categories_root.is_dir():
            raise ValueError(
                f"categories_root does not exist or is not a directory: {categories_root}"
            )
        folder_names = sorted([p.name for p in categories_root.iterdir() if p.is_dir()])

    known_by_normalized: dict[str, str] = {}
    for tag in sorted(known_tags):
        normalized_tag = normalize_tag_name(tag)
        # Keep first stable mapping for normalized fallback.
        if normalized_tag not in known_by_normalized:
            known_by_normalized[normalized_tag] = tag

    from .taxonomy import resolve_taxonomy_folder

    mappings: list[FolderMapping] = []
    for folder in folder_names:
        normalized = normalize_tag_name(folder)
        # Prefer exact known tag match first for selected tags that contain
        # special syntax (e.g. `remodel_(kantai_collection)`).
        matched_tag = folder if folder in known_tags else known_by_normalized.get(normalized)
        if matched_tag is None:
            # Taxonomy destinations (e.g. Pokemon → pokemon_(creature) evidence).
            tax = resolve_taxonomy_folder(folder)
            if tax is not None:
                matched_tag = tax.folder
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
    experimental_media_enabled: bool = False,
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
            if not experimental_media_enabled:
                ignored_gif += 1
                continue
            eligible.append(path)
            continue
        if ext in VIDEO_EXTENSIONS:
            if not experimental_media_enabled:
                ignored_unsupported += 1
                continue
            eligible.append(path)
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


TAGGER_MODEL_ML = "ml_danbooru"
TAGGER_MODEL_WD_SWINV2 = "wd_swinv2_v3"
TAGGER_MODEL_WD_EVA02 = "wd_eva02_large"
WD_MODEL_NAMES = {
    TAGGER_MODEL_WD_SWINV2: "SwinV2_v3",
    TAGGER_MODEL_WD_EVA02: "EVA02_Large",
}


def _parse_mldanbooru_raw(raw: object) -> dict[str, float]:
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


def _normalize_score_tags(scores: dict[str, float]) -> dict[str, float]:
    """Normalize tag keys to underscore form so they match tags.csv / selected tags."""
    normalized: dict[str, float] = {}
    for tag, score in scores.items():
        key = normalize_tag_name(str(tag))
        if not key:
            continue
        current = normalized.get(key)
        if current is None or float(score) > current:
            normalized[key] = float(score)
    return normalized


def _parse_wd14_raw(raw: object) -> dict[str, float]:
    scores: dict[str, float] = {}
    if isinstance(raw, dict):
        for tag, score in raw.items():
            scores[str(tag)] = float(score)
        return _normalize_score_tags(scores)
    if isinstance(raw, (list, tuple)):
        # fmt tuple returns ordered parts; flatten dict parts only.
        for part in raw:
            if isinstance(part, dict):
                for tag, score in part.items():
                    scores[str(tag)] = float(score)
        return _normalize_score_tags(scores)
    return {}


def _run_mldanbooru(image: Path | Image.Image | str) -> dict[str, float]:
    from .inference_engine import get_engine

    return get_engine().score_one(
        image,
        tagger_model=TAGGER_MODEL_ML,
        wd_general_threshold=0.35,
    )


def _run_wd14(
    image: Path | Image.Image | str,
    *,
    model_name: str,
    general_threshold: float,
) -> dict[str, float]:
    from .inference_engine import get_engine

    tagger_model = next(
        (key for key, value in WD_MODEL_NAMES.items() if value == model_name),
        None,
    )
    if tagger_model is None:
        raise ValueError(f"Unsupported WD model_name: {model_name}")
    return get_engine().score_one(
        image,
        tagger_model=tagger_model,
        wd_general_threshold=general_threshold,
    )


def extract_scores(
    image_path: Path,
    *,
    tagger_model: str = TAGGER_MODEL_WD_SWINV2,
    wd_general_threshold: float = 0.35,
) -> dict[str, float]:
    from .providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

    ensure_nvidia_dll_search_path()
    preload_onnx_runtime_dlls()

    scores = _extract_scores_unlocked(
        image_path,
        tagger_model=tagger_model,
        wd_general_threshold=wd_general_threshold,
    )
    # One retry: concurrent/GPU glitches occasionally return an empty map.
    if not scores:
        logger.warning(
            "empty_scores_retry path=%s tagger_model=%s", image_path, tagger_model
        )
        scores = _extract_scores_unlocked(
            image_path,
            tagger_model=tagger_model,
            wd_general_threshold=wd_general_threshold,
        )
    return scores


def _extract_scores_unlocked(
    image: Path | Image.Image | str,
    *,
    tagger_model: str,
    wd_general_threshold: float,
) -> dict[str, float]:
    if tagger_model == TAGGER_MODEL_ML:
        return _run_mldanbooru(image)
    wd_name = WD_MODEL_NAMES.get(tagger_model)
    if wd_name is None:
        raise ValueError(f"Unsupported tagger_model: {tagger_model}")
    return _run_wd14(
        image,
        model_name=wd_name,
        general_threshold=wd_general_threshold,
    )


def _extract_scores_from_pil_image(
    image: Image.Image,
    *,
    tagger_model: str = TAGGER_MODEL_WD_SWINV2,
    wd_general_threshold: float = 0.35,
) -> dict[str, float]:
    from .providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

    ensure_nvidia_dll_search_path()
    preload_onnx_runtime_dlls()
    return _extract_scores_unlocked(
        image,
        tagger_model=tagger_model,
        wd_general_threshold=wd_general_threshold,
    )


def extract_scores_with_experimental_media(
    image_path: Path,
    experimental_media_enabled: bool = False,
    *,
    tagger_model: str = TAGGER_MODEL_WD_SWINV2,
    wd_general_threshold: float = 0.35,
) -> dict[str, float]:
    """
    Experimental path: GIF/video via multi-frame sampling + pooled tag scores.

    Samples evenly spaced frames (default 8), runs the tagger on each, then
    pools with mean/presence so brief but strong cues are not washed out.
    Videos use OpenCV (bundled FFmpeg); system ffmpeg CLI is optional fallback.
    """
    if not experimental_media_enabled or not is_experimental_media(image_path):
        return extract_scores(
            image_path,
            tagger_model=tagger_model,
            wd_general_threshold=wd_general_threshold,
        )

    ext = image_path.suffix.lower()
    try:
        if ext == ".gif":
            frames = sample_gif_frames(image_path)
        else:
            try:
                frames = sample_video_frames(image_path)
            except Exception as cv_err:
                # Optional CLI ffmpeg fallback when OpenCV cannot decode.
                frames = _sample_video_frames_ffmpeg(image_path)
                if not frames:
                    raise RuntimeError(
                        f"Video frame extraction failed (OpenCV: {cv_err})"
                    ) from cv_err
        logger.info(
            "experimental_media_sampled path=%s frames=%d model=%s",
            image_path,
            len(frames),
            tagger_model,
        )
        return _score_pil_frames(
            frames,
            tagger_model=tagger_model,
            wd_general_threshold=wd_general_threshold,
        )
    except Exception:
        logger.exception("experimental_media_failed path=%s", image_path)
        raise


def _sample_video_frames_ffmpeg(
    image_path: Path, sample_count: int | None = None
) -> list[Image.Image]:
    """Fallback: extract evenly spaced frames with system ffmpeg if available."""
    if shutil.which("ffmpeg") is None:
        return []
    # Without a reliable probe here, assume a mid-length clip for scaling.
    count = (
        max(1, sample_count)
        if sample_count is not None
        else scaled_media_sample_count(duration_seconds=30.0)
    )
    frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory(prefix="media_frames_") as temp_dir:
        pattern = str(Path(temp_dir) / "frame_%03d.png")
        # fps filter approximates even coverage across ~duration.
        seconds_per = _env_float(
            "MEDIA_SAMPLE_SECONDS",
            DEFAULT_MEDIA_SAMPLE_SECONDS,
            lo=0.25,
            hi=5.0,
        )
        fps = max(1.0 / seconds_per, 0.1)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(image_path),
            "-vf",
            f"fps={fps:.4f}",
            "-frames:v",
            str(count),
            pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "ffmpeg_frame_extract_failed path=%s err=%s",
                image_path,
                (result.stderr or result.stdout or "").strip(),
            )
            return []
        for path in sorted(Path(temp_dir).glob("frame_*.png")):
            with Image.open(path) as img:
                frames.append(img.convert("RGB"))
    return frames


def extract_scores_batch(
    image_paths: list[Path],
    *,
    tagger_model: str = TAGGER_MODEL_WD_SWINV2,
    wd_general_threshold: float = 0.35,
) -> list[dict[str, float]]:
    """
    WD models support true ORT batching (fixed NHWC). ML-Danbooru stays
    sequential Run because keep_ratio yields variable HxW tensors.
    """
    if not image_paths:
        return []

    from .inference_engine import get_engine
    from .providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

    ensure_nvidia_dll_search_path()
    preload_onnx_runtime_dlls()

    global _BATCH_INFERENCE_SUPPORTED
    if tagger_model == TAGGER_MODEL_ML:
        if _BATCH_INFERENCE_SUPPORTED is not False:
            logger.info("ml_danbooru batch uses sequential ORT runs (variable HxW)")
            _BATCH_INFERENCE_SUPPORTED = False
        return [
            extract_scores(
                path,
                tagger_model=tagger_model,
                wd_general_threshold=wd_general_threshold,
            )
            for path in image_paths
        ]

    _BATCH_INFERENCE_SUPPORTED = True
    return get_engine().score_many(
        list(image_paths),
        tagger_model=tagger_model,
        wd_general_threshold=wd_general_threshold,
        batch_size=max(1, len(image_paths)),
    )


def global_top_tags(scores: dict[str, float], limit: int = 5) -> list[dict[str, float]]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [{"tag": tag, "score": float(score)} for tag, score in ranked]


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
    return current.model_copy(
        update={
            "root_repo": root_repo if root_repo is not None else current.root_repo,
            "categories_root": (
                categories_root if categories_root is not None else current.categories_root
            ),
            "confidence_threshold": (
                confidence_threshold
                if confidence_threshold is not None
                else current.confidence_threshold
            ),
        }
    )
