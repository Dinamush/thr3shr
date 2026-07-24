"""Probe, budget, plan, and decode GIF/video frames for media classification."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageFile, ImageSequence

from .media_types import MediaProbe, SampledFrame, SamplingBudget

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

DEFAULT_MEDIA_SAMPLE_MIN = 4
DEFAULT_MEDIA_SAMPLE_MAX = 48
DEFAULT_MEDIA_SAMPLE_SECONDS = 0.75
DEFAULT_MEDIA_GIF_FRAME_STRIDE = 3
DEFAULT_CANDIDATE_MULTIPLIER = 4
DEFAULT_CANDIDATE_CAP = 192
FILTER_MEDIA_SAMPLE_MAX = 8
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


def budget_for_duration(
    probe: MediaProbe | None = None,
    *,
    duration_seconds: float | None = None,
    total_frames: int | None = None,
    override: int | None = None,
) -> SamplingBudget:
    """Choose tagged/candidate budgets from duration bands or override."""
    minimum, maximum = _media_sample_bounds()
    hard = os.getenv("MEDIA_SAMPLE_FRAMES", "").strip()
    if override is not None:
        tagged = max(1, min(int(override), maximum))
    elif hard:
        try:
            tagged = max(1, min(int(hard), maximum))
        except ValueError:
            tagged = _tagged_from_length(
                duration_seconds=duration_seconds
                if duration_seconds is not None
                else (probe.duration_s if probe else None),
                total_frames=total_frames
                if total_frames is not None
                else (probe.total_frames if probe else None),
                minimum=minimum,
                maximum=maximum,
            )
    else:
        tagged = _tagged_from_length(
            duration_seconds=duration_seconds
            if duration_seconds is not None
            else (probe.duration_s if probe else None),
            total_frames=total_frames
            if total_frames is not None
            else (probe.total_frames if probe else None),
            minimum=minimum,
            maximum=maximum,
        )

    if probe and probe.total_frames and probe.total_frames > 0:
        tagged = min(tagged, int(probe.total_frames))
    elif total_frames is not None and total_frames > 0:
        tagged = min(tagged, int(total_frames))

    mult = _env_int(
        "MEDIA_CANDIDATE_MULTIPLIER",
        DEFAULT_CANDIDATE_MULTIPLIER,
        lo=1,
        hi=16,
    )
    cand_cap = _env_int(
        "MEDIA_CANDIDATE_CAP",
        DEFAULT_CANDIDATE_CAP,
        lo=8,
        hi=256,
    )
    candidate = min(cand_cap, max(tagged, tagged * mult))
    if probe and probe.total_frames and probe.total_frames > 0:
        candidate = min(candidate, int(probe.total_frames))
    return SamplingBudget(tagged_max=max(1, tagged), candidate_max=max(1, candidate))


def _tagged_from_length(
    *,
    duration_seconds: float | None,
    total_frames: int | None,
    minimum: int,
    maximum: int,
) -> int:
    if duration_seconds is not None and duration_seconds > 0:
        d = float(duration_seconds)
        if d < 10.0:
            target = 8
        elif d < 60.0:
            target = 12
        elif d < 300.0:
            target = 24
        elif d < 1800.0:
            target = 36
        else:
            target = 48
        return max(minimum, min(target, maximum))

    if total_frames is not None and total_frames > 0:
        stride = _env_int(
            "MEDIA_GIF_FRAME_STRIDE",
            DEFAULT_MEDIA_GIF_FRAME_STRIDE,
            lo=1,
            hi=30,
        )
        target = int((int(total_frames) + stride - 1) // stride)
        return max(minimum, min(target, maximum))

    return minimum


def scaled_media_sample_count(
    *,
    total_frames: int | None = None,
    duration_seconds: float | None = None,
) -> int:
    """Back-compat: tagged frame count only."""
    return budget_for_duration(
        duration_seconds=duration_seconds,
        total_frames=total_frames,
    ).tagged_max


def even_frame_indices(total_frames: int, sample_count: int) -> list[int]:
    """Pick up to sample_count indices evenly across [0, total_frames)."""
    if total_frames <= 0:
        return []
    n = min(max(1, sample_count), total_frames)
    if n == 1:
        return [0]
    if n == total_frames:
        return list(range(total_frames))
    return sorted(
        {min(total_frames - 1, int((i + 0.5) * total_frames / n)) for i in range(n)}
    )


def plan_candidate_timestamps(probe: MediaProbe, budget: SamplingBudget) -> list[float]:
    """Uniform timestamps in [0, duration) for candidate decode."""
    n = max(1, int(budget.candidate_max))
    duration = probe.duration_s
    if duration is not None and duration > 0:
        if n == 1:
            return [0.0]
        return [((i + 0.5) * duration / n) for i in range(n)]

    total = probe.total_frames or 0
    if total <= 0:
        # Unknown: plan relative slots; decode will sequential-fill.
        return [float(i) for i in range(n)]

    indices = even_frame_indices(total, n)
    if probe.fps and probe.fps > 1e-3:
        return [idx / float(probe.fps) for idx in indices]
    if probe.per_frame_delays_ms:
        cum = _cumulative_seconds(probe.per_frame_delays_ms)
        return [cum[min(idx, len(cum) - 1)] for idx in indices]
    return [float(idx) for idx in indices]


def probe_media(path: Path) -> MediaProbe:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        return _probe_gif(path)
    if suffix in VIDEO_EXTENSIONS:
        return _probe_video(path)
    return MediaProbe(
        duration_s=None,
        total_frames=None,
        time_base="unknown",
        kind="unknown",
    )


def decode_frames(
    path: Path,
    timestamps: list[float],
    probe: MediaProbe,
    *,
    neighbor_retry: bool = True,
) -> list[SampledFrame]:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        return _decode_gif_frames(path, timestamps, probe, neighbor_retry=neighbor_retry)
    if suffix in VIDEO_EXTENSIONS:
        try:
            frames = _decode_video_opencv(
                path, timestamps, probe, neighbor_retry=neighbor_retry
            )
            if frames:
                return frames
        except Exception as cv_err:
            logger.warning(
                "opencv_decode_failed path=%s err=%s; trying ffmpeg",
                path,
                cv_err,
            )
        frames = _decode_video_ffmpeg(path, timestamps, probe)
        if not frames:
            raise RuntimeError(f"Unable to decode any frames from video: {path}")
        return frames
    raise RuntimeError(f"Unsupported media type for decode: {path}")


def sample_gif_frames(image_path: Path, sample_count: int | None = None) -> list[Image.Image]:
    """Back-compat wrapper returning bare PIL frames."""
    probe = probe_media(image_path)
    budget = budget_for_duration(probe, override=sample_count)
    # For forced small counts (style/preview), plan tagged_max timestamps only.
    plan_budget = SamplingBudget(
        tagged_max=budget.tagged_max,
        candidate_max=budget.tagged_max if sample_count is not None else budget.candidate_max,
    )
    stamps = plan_candidate_timestamps(probe, plan_budget)
    sampled = decode_frames(image_path, stamps, probe, neighbor_retry=True)
    return [f.image for f in sampled if f.decode_ok]


def sample_video_frames(image_path: Path, sample_count: int | None = None) -> list[Image.Image]:
    """Back-compat wrapper returning bare PIL frames."""
    probe = probe_media(image_path)
    budget = budget_for_duration(probe, override=sample_count)
    plan_budget = SamplingBudget(
        tagged_max=budget.tagged_max,
        candidate_max=budget.tagged_max if sample_count is not None else budget.candidate_max,
    )
    stamps = plan_candidate_timestamps(probe, plan_budget)
    sampled = decode_frames(image_path, stamps, probe, neighbor_retry=True)
    return [f.image for f in sampled if f.decode_ok]


def _probe_gif(path: Path) -> MediaProbe:
    with Image.open(path) as gif:
        total = int(getattr(gif, "n_frames", 0) or 0)
        delays: list[float] = []
        if total < 1:
            total = 0
            for frame in ImageSequence.Iterator(gif):
                total += 1
                delays.append(float(frame.info.get("duration") or gif.info.get("duration") or 40))
            gif.seek(0)
        else:
            for i in range(total):
                try:
                    gif.seek(i)
                    delays.append(
                        float(gif.info.get("duration") or 40)
                    )
                except Exception:
                    delays.append(40.0)
            gif.seek(0)
        if not delays and total > 0:
            delay = float(gif.info.get("duration") or 40)
            delays = [delay] * total
        duration_s = sum(delays) / 1000.0 if delays else None
        return MediaProbe(
            duration_s=duration_s if duration_s and duration_s > 0 else None,
            total_frames=total if total > 0 else None,
            time_base="gif_delays" if delays else "unknown",
            per_frame_delays_ms=delays or None,
            kind="gif",
        )


def _probe_video(path: Path) -> MediaProbe:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        # Try ffprobe without OpenCV.
        duration = _ffprobe_duration(path)
        return MediaProbe(
            duration_s=duration,
            total_frames=None,
            time_base="fps" if duration else "unknown",
            fps=None,
            kind="video",
        )
    try:
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration_s = None
        if reported > 0 and fps > 1e-3:
            duration_s = float(reported) / fps
        if duration_s is None or duration_s <= 0:
            probed = _ffprobe_duration(path)
            if probed:
                duration_s = probed
        time_base: str = "fps" if (fps > 1e-3 or duration_s) else "unknown"
        return MediaProbe(
            duration_s=duration_s,
            total_frames=reported if reported > 0 else None,
            time_base=time_base,  # type: ignore[arg-type]
            fps=fps if fps > 1e-3 else None,
            kind="video",
        )
    finally:
        cap.release()


def _ffprobe_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    raw = (result.stdout or "").strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _cumulative_seconds(delays_ms: list[float]) -> list[float]:
    out: list[float] = []
    acc = 0.0
    for delay in delays_ms:
        out.append(acc / 1000.0)
        acc += max(1.0, float(delay))
    return out


def _timestamp_to_gif_index(probe: MediaProbe, timestamp_s: float) -> int:
    total = probe.total_frames or 0
    if total <= 0:
        return 0
    delays = probe.per_frame_delays_ms
    if delays:
        cum = _cumulative_seconds(delays)
        # Pick last frame whose start time <= timestamp.
        idx = 0
        for i, start in enumerate(cum):
            if start <= timestamp_s:
                idx = i
            else:
                break
        return min(idx, total - 1)
    if probe.duration_s and probe.duration_s > 0:
        frac = min(1.0, max(0.0, timestamp_s / probe.duration_s))
        return min(total - 1, int(frac * total))
    return min(total - 1, int(timestamp_s))


def _timestamp_to_video_index(probe: MediaProbe, timestamp_s: float) -> int:
    total = probe.total_frames or 0
    if probe.fps and probe.fps > 1e-3:
        idx = int(round(timestamp_s * probe.fps))
        if total > 0:
            return min(max(0, idx), total - 1)
        return max(0, idx)
    if total > 0 and probe.duration_s and probe.duration_s > 0:
        frac = min(1.0, max(0.0, timestamp_s / probe.duration_s))
        return min(total - 1, int(frac * total))
    return max(0, int(timestamp_s))


def _decode_gif_frames(
    path: Path,
    timestamps: list[float],
    probe: MediaProbe,
    *,
    neighbor_retry: bool,
) -> list[SampledFrame]:
    frames: list[SampledFrame] = []
    with Image.open(path) as gif:
        total = probe.total_frames or int(getattr(gif, "n_frames", 0) or 0)
        for ts in timestamps:
            primary = _timestamp_to_gif_index(probe, float(ts))
            candidates = [primary]
            if neighbor_retry and total > 0:
                for delta in (1, -1, 2, -2, 3, -3):
                    alt = primary + delta
                    if 0 <= alt < total:
                        candidates.append(alt)
            decoded = None
            used_idx = primary
            for idx in candidates:
                try:
                    gif.seek(idx)
                    decoded = gif.convert("RGB")
                    used_idx = idx
                    break
                except Exception:
                    continue
            if decoded is None:
                logger.warning("gif_frame_skip path=%s ts=%s", path, ts)
                continue
            frames.append(
                SampledFrame(
                    source_index=used_idx,
                    image=decoded,
                    timestamp_s=float(ts),
                    requested_timestamp_s=float(ts),
                    decode_ok=True,
                )
            )
    if not frames:
        raise RuntimeError(f"Unable to decode any frames from GIF: {path}")
    return frames


def _decode_video_opencv(
    path: Path,
    timestamps: list[float],
    probe: MediaProbe,
    *,
    neighbor_retry: bool,
) -> list[SampledFrame]:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for frame sampling: {path}")

    frames: list[SampledFrame] = []
    try:
        total = probe.total_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0 and (probe.duration_s is None or probe.duration_s <= 0):
            # Unknown length: discover stream span via grab(), reopen, subsample.
            return _decode_video_unknown_length(path, len(timestamps))

        if total <= 0 and (probe.fps is None or probe.fps <= 1e-3):
            # Duration known (ffprobe) but no frame count / fps: frame-index
            # seeking would treat seconds as indices and sample only the
            # opening. Seek by media time instead.
            return _decode_video_by_msec(cap, timestamps)

        for ts in timestamps:
            primary = _timestamp_to_video_index(probe, float(ts))
            candidates = [primary]
            if neighbor_retry and total > 0:
                for delta in (1, -1, 2, -2, 3, -3):
                    alt = primary + delta
                    if 0 <= alt < total:
                        candidates.append(alt)
            decoded = None
            used_idx = primary
            for idx in candidates:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded = Image.fromarray(np.asarray(rgb))
                used_idx = idx
                break
            if decoded is None:
                continue
            frames.append(
                SampledFrame(
                    source_index=used_idx,
                    image=decoded,
                    timestamp_s=float(ts),
                    requested_timestamp_s=float(ts),
                    decode_ok=True,
                )
            )
        return frames
    finally:
        cap.release()


def _decode_video_by_msec(cap, timestamps: list[float]) -> list[SampledFrame]:
    """Seek by media time for streams with unknown frame count / fps."""
    import cv2
    import numpy as np

    frames: list[SampledFrame] = []
    for i, ts in enumerate(timestamps):
        decoded = None
        for delta_s in (0.0, 0.5, -0.5, 1.0):
            target = max(0.0, float(ts) + delta_s)
            cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded = Image.fromarray(np.asarray(rgb))
                break
        if decoded is None:
            continue
        frames.append(
            SampledFrame(
                source_index=i,
                image=decoded,
                timestamp_s=float(ts),
                requested_timestamp_s=float(ts),
                decode_ok=True,
            )
        )
    return frames


def _decode_video_unknown_length(path: Path, count: int) -> list[SampledFrame]:
    """Sample evenly across a stream with unknown frame count / duration.

    Uses a cheap grab()-only pass to discover length (bounded), then reopens
    and decodes only the selected indices so late content is not starved.
    """
    import cv2
    import numpy as np

    max_scan = max(int(count) * 250, 3000)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    try:
        n = 0
        while n < max_scan:
            if not cap.grab():
                break
            n += 1
    finally:
        cap.release()

    if n <= 0:
        return []

    indices = even_frame_indices(n, max(1, int(count)))
    wanted = set(indices)
    frames: list[SampledFrame] = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    try:
        pos = 0
        while pos < n and len(frames) < len(indices):
            if not cap.grab():
                break
            if pos in wanted:
                ok, bgr = cap.retrieve()
                if ok and bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames.append(
                        SampledFrame(
                            source_index=pos,
                            image=Image.fromarray(np.asarray(rgb)),
                            timestamp_s=float(pos),
                            requested_timestamp_s=float(pos),
                            decode_ok=True,
                        )
                    )
            pos += 1
    finally:
        cap.release()
    return frames


def _decode_video_ffmpeg(
    path: Path,
    timestamps: list[float],
    probe: MediaProbe,
) -> list[SampledFrame]:
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg_absent path=%s", path)
        return []

    duration = probe.duration_s or _ffprobe_duration(path)
    count = max(1, len(timestamps)) if timestamps else 8
    if not timestamps and duration and duration > 0:
        timestamps = [((i + 0.5) * duration / count) for i in range(count)]
    elif not timestamps:
        timestamps = [float(i) * 0.75 for i in range(count)]

    frames: list[SampledFrame] = []
    with tempfile.TemporaryDirectory(prefix="media_frames_") as temp_dir:
        for i, ts in enumerate(timestamps):
            out_path = Path(temp_dir) / f"frame_{i:03d}.png"
            # Seek then grab one frame — distributes across full duration.
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, float(ts)):.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not out_path.exists():
                continue
            with Image.open(out_path) as img:
                frames.append(
                    SampledFrame(
                        source_index=i,
                        image=img.convert("RGB"),
                        timestamp_s=float(ts),
                        requested_timestamp_s=float(ts),
                        decode_ok=True,
                    )
                )
    if not frames:
        logger.warning("ffmpeg_frame_extract_failed path=%s", path)
    return frames
