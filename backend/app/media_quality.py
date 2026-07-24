"""Frame quality metrics for rejecting black / blank / low-info media frames."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .media_types import FrameQuality, SampledFrame


def compute_frame_quality(image: Image.Image, *, thumb_edge: int = 192) -> FrameQuality:
    gray = _thumbnail_gray(image, thumb_edge=thumb_edge)
    flat = gray.astype(np.float64).reshape(-1) / 255.0
    mean_y = float(flat.mean())
    std_y = float(flat.std())
    p05_y = float(np.percentile(flat, 5))
    p95_y = float(np.percentile(flat, 95))
    entropy_bits = _histogram_entropy(flat)
    edge_density = _edge_density(gray)
    verdict = _verdict(mean_y, std_y, p05_y, p95_y, entropy_bits, edge_density)
    return FrameQuality(
        mean_y=mean_y,
        std_y=std_y,
        p05_y=p05_y,
        p95_y=p95_y,
        entropy_bits=entropy_bits,
        edge_density=edge_density,
        verdict=verdict,
    )


def is_rejected(q: FrameQuality) -> bool:
    return q.verdict != "ok"


def quality_rank(q: FrameQuality) -> float:
    return float(q.edge_density) * float(q.entropy_bits)


def pick_best_quality_frame(frames: list[SampledFrame]) -> SampledFrame:
    if not frames:
        raise ValueError("pick_best_quality_frame requires at least one frame")
    best = frames[0]
    best_rank = -1.0
    for frame in frames:
        if not frame.decode_ok:
            continue
        q = compute_frame_quality(frame.image)
        rank = quality_rank(q)
        if rank > best_rank:
            best_rank = rank
            best = frame
    return best


def _thumbnail_gray(image: Image.Image, *, thumb_edge: int) -> np.ndarray:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if max(w, h) > thumb_edge:
        scale = thumb_edge / float(max(w, h))
        rgb = rgb.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
    arr = np.asarray(rgb, dtype=np.uint8)
    # Rec. 601 luminance.
    return (
        0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    ).astype(np.uint8)


def _histogram_entropy(flat01: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(flat01, bins=bins, range=(0.0, 1.0))
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    probs = hist.astype(np.float64) / total
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= float(p) * math.log2(float(p))
    return float(entropy)


def _edge_density(gray_u8: np.ndarray) -> float:
    g = gray_u8.astype(np.float64)
    if g.shape[0] < 2 or g.shape[1] < 2:
        return 0.0
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    # Fraction of strong horizontal/vertical gradients.
    hx = float((dx > 18.0).mean()) if dx.size else 0.0
    hy = float((dy > 18.0).mean()) if dy.size else 0.0
    return max(hx, hy)


def _verdict(
    mean_y: float,
    std_y: float,
    p05_y: float,
    p95_y: float,
    entropy_bits: float,
    edge_density: float,
) -> str:
    # Dark anime scenes with structure must not be rejected as black.
    low_light = p95_y < 0.08 or (mean_y < 0.05 and std_y < 0.025)
    if low_light and edge_density < 0.003 and entropy_bits < 2.0:
        return "black"
    if p05_y > 0.92 and std_y < 0.025 and edge_density < 0.003:
        return "blank"
    if entropy_bits < 2.0 and edge_density < 0.003:
        return "low_info"
    return "ok"
