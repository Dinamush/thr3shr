"""Shared types for GIF/video media classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PIL import Image

TimeBase = Literal["gif_delays", "fps", "unknown"]
QualityVerdict = Literal["ok", "black", "blank", "low_info"]


@dataclass(frozen=True)
class MediaProbe:
    duration_s: float | None
    total_frames: int | None
    time_base: TimeBase
    per_frame_delays_ms: list[float] | None = None
    fps: float | None = None
    kind: Literal["gif", "video", "unknown"] = "unknown"


@dataclass(frozen=True)
class SamplingBudget:
    tagged_max: int
    candidate_max: int


@dataclass
class SampledFrame:
    source_index: int
    image: Image.Image
    timestamp_s: float | None = None
    requested_timestamp_s: float | None = None
    decode_ok: bool = True


@dataclass(frozen=True)
class FrameQuality:
    mean_y: float
    std_y: float
    p05_y: float
    p95_y: float
    entropy_bits: float
    edge_density: float
    verdict: QualityVerdict
