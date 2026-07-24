"""Orchestrate robust GIF/video presence classification."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .media_pooling import pool_presence
from .media_quality import compute_frame_quality
from .media_sampling import (
    budget_for_duration,
    decode_frames,
    plan_candidate_timestamps,
    probe_media,
)
from .media_selection import select_tagged_frames
from .media_types import SampledFrame, SamplingBudget

logger = logging.getLogger(__name__)


def extract_media_scores(
    path: Path,
    *,
    tagger_model: str,
    wd_general_threshold: float = 0.35,
    budget_override: int | None = None,
) -> dict[str, float]:
    """Probe → candidates → quality → select → raw score → presence pool."""
    probe = probe_media(path)
    budget = budget_for_duration(probe, override=budget_override)
    timestamps = plan_candidate_timestamps(probe, budget)
    candidates = decode_frames(path, timestamps, probe, neighbor_retry=True)
    if not candidates:
        raise RuntimeError(f"Unable to decode any frames from media: {path}")

    scored_quality = [(frame, compute_frame_quality(frame.image)) for frame in candidates]
    selected = select_tagged_frames(scored_quality, budget)
    if not selected:
        selected = [candidates[0]]

    rejected = sum(1 for _, q in scored_quality if q.verdict != "ok")
    logger.info(
        "media_pipeline path=%s duration_s=%s candidates=%d rejected=%d tagged=%d budget=%d",
        path,
        f"{probe.duration_s:.2f}" if probe.duration_s is not None else "unknown",
        len(candidates),
        rejected,
        len(selected),
        budget.tagged_max,
    )
    return score_and_pool_frames(
        selected,
        tagger_model=tagger_model,
        wd_general_threshold=wd_general_threshold,
    )


def score_and_pool_frames(
    frames: list[SampledFrame],
    *,
    tagger_model: str,
    wd_general_threshold: float,
) -> dict[str, float]:
    from .inference_engine import (
        WD_REALISM_ALWAYS_TAGS,
        WD_REALISM_FLOOR,
        get_engine,
    )
    from .providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

    ensure_nvidia_dll_search_path()
    preload_onnx_runtime_dlls()
    if not frames:
        return {}

    images = [frame.image for frame in frames]
    maps = get_engine().score_many(
        images,
        tagger_model=tagger_model,
        wd_general_threshold=wd_general_threshold,
        batch_size=min(8, len(images)),
        raw_general=True,
    )
    support_thr = _support_threshold(wd_general_threshold)
    paired = [
        (frame.timestamp_s, scores) for frame, scores in zip(frames, maps)
    ]
    pooled = pool_presence(
        paired,
        support_thr=support_thr,
        min_hits=2,
        top_k=3,
        # Selected frames are already time-spaced by binning; do not collapse
        # short GIFs into a single scene (that would defeat corroboration).
        scene_gap_s=None,
    )
    # Still path always surfaces weak realistic/photorealistic (>= floor) for
    # hybrid real_life blending. Mirror that here: pool realism tags with the
    # floor as support so ~0.2 photo signal survives presence suppression.
    realism_floor = min(support_thr, float(WD_REALISM_FLOOR))
    realism_paired = [
        (ts, {k: v for k, v in scores.items() if k in WD_REALISM_ALWAYS_TAGS})
        for ts, scores in paired
    ]
    realism_pooled = pool_presence(
        realism_paired,
        support_thr=realism_floor,
        min_hits=2,
        top_k=3,
        scene_gap_s=None,
    )
    for key, value in realism_pooled.items():
        if value > pooled.get(key, 0.0):
            pooled[key] = value
    return pooled


def _support_threshold(wd_general_threshold: float) -> float:
    raw = os.getenv("MEDIA_SUPPORT_THRESHOLD", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(wd_general_threshold)
