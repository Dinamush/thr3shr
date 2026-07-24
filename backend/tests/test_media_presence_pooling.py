"""Unit tests for presence pooling, quality, selection, and media budgets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.media_pooling import pool_presence
from app.media_quality import compute_frame_quality, is_rejected, quality_rank
from app.media_sampling import (
    budget_for_duration,
    even_frame_indices,
    plan_candidate_timestamps,
    probe_media,
    sample_gif_frames,
)
from app.media_selection import select_tagged_frames
from app.media_types import FrameQuality, MediaProbe, SampledFrame, SamplingBudget
# NOTE: import the model constant from services, not inference_engine —
# inference_engine needs huggingface_hub at import time and would break
# test collection in ML-dependency-free environments.
from app.services import TAGGER_MODEL_WD_SWINV2, extract_scores_with_experimental_media


def test_pool_presence_single_hit_suppressed() -> None:
    pooled = pool_presence(
        [
            (0.0, {"loli": 0.9, "1girl": 0.8}),
            (1.0, {"1girl": 0.7}),
            (2.0, {"1girl": 0.6}),
            (3.0, {}),
        ],
        support_thr=0.35,
        min_hits=2,
        scene_gap_s=None,
    )
    assert "loli" not in pooled
    assert abs(pooled["1girl"] - ((0.8 + 0.7 + 0.6) / 3)) < 1e-9


def test_pool_presence_two_hits_top_k_mean() -> None:
    pooled = pool_presence(
        [
            (0.0, {"loli": 0.9}),
            (1.0, {"loli": 0.1}),
            (2.0, {"loli": 0.8}),
            (3.0, {"loli": 0.2}),
        ],
        support_thr=0.35,
        min_hits=2,
        top_k=2,
        scene_gap_s=None,
    )
    assert abs(pooled["loli"] - ((0.9 + 0.8) / 2)) < 1e-9


def test_pool_presence_black_injection_stable() -> None:
    # Two strong hits remain after black (empty) frames.
    content = [
        (0.0, {}),
        (1.0, {"fellatio": 0.91}),
        (2.0, {}),
        (3.0, {"fellatio": 0.88}),
        (4.0, {}),
        (5.0, {}),
    ]
    pooled = pool_presence(content, support_thr=0.35, min_hits=2, top_k=2, scene_gap_s=None)
    assert abs(pooled["fellatio"] - ((0.91 + 0.88) / 2)) < 1e-9


def test_pool_presence_short_media_still_semantics() -> None:
    pooled = pool_presence(
        [(0.0, {"loli": 0.92})],
        support_thr=0.35,
        min_hits=2,
    )
    assert abs(pooled["loli"] - 0.92) < 1e-9


def test_pool_presence_short_media_fallback_thresholds_raw_junk() -> None:
    # Media path scores with raw (dense) probs; a 1-frame clip must not leak
    # sub-threshold junk tags through the still-semantics fallback.
    pooled = pool_presence(
        [(0.0, {"loli": 0.92, "junk_a": 0.12, "junk_b": 0.30})],
        support_thr=0.35,
        min_hits=2,
    )
    assert abs(pooled["loli"] - 0.92) < 1e-9
    assert "junk_a" not in pooled
    assert "junk_b" not in pooled


def test_pool_presence_omits_suppressed_keys() -> None:
    pooled = pool_presence(
        [
            (0.0, {"loli": 0.9}),
            (1.0, {"1girl": 0.8}),
            (2.0, {"1girl": 0.7}),
        ],
        support_thr=0.35,
        min_hits=2,
        scene_gap_s=None,
    )
    assert "loli" not in pooled
    assert "1girl" in pooled


def test_unknown_length_video_subsamples_across_discovered_span(tmp_path: Path) -> None:
    """When OpenCV reports no frame count, sampling must not stay in the opening."""
    import cv2
    import numpy as np

    path = tmp_path / "unknown_len.mp4"
    w, h, n, fps = 32, 32, 200, 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Unique marker in late frames only.
        if i >= 160:
            frame[:, :] = (0, 255, 0)
            frame[5:25, 5:7] = (255, 255, 255)
        else:
            frame[:, :] = (10, 10, 10)
        writer.write(frame)
    writer.release()

    from app.media_sampling import _decode_video_unknown_length

    frames = _decode_video_unknown_length(path, count=8)
    assert len(frames) >= 4
    assert max(f.source_index for f in frames) > 100


def test_quality_rejects_black_and_keeps_pattern() -> None:
    black = Image.new("RGB", (64, 64), color=(0, 0, 0))
    white = Image.new("RGB", (64, 64), color=(255, 255, 255))
    # Checkerboard-ish pattern with edges.
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[::2, ::2] = 220
    arr[1::2, 1::2] = 40
    patterned = Image.fromarray(arr)

    qb = compute_frame_quality(black)
    qw = compute_frame_quality(white)
    qp = compute_frame_quality(patterned)
    assert is_rejected(qb)
    assert qb.verdict == "black"
    assert is_rejected(qw)
    assert qw.verdict == "blank"
    assert not is_rejected(qp)
    assert quality_rank(qp) > quality_rank(qb)


def test_quality_dark_with_edges_not_rejected() -> None:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :] = 8
    arr[10:54, 10:12] = 180
    arr[10:12, 10:54] = 180
    dark_edged = Image.fromarray(arr)
    q = compute_frame_quality(dark_edged)
    assert q.verdict == "ok"


def test_budget_bands(monkeypatch) -> None:
    monkeypatch.delenv("MEDIA_SAMPLE_FRAMES", raising=False)
    monkeypatch.setenv("MEDIA_SAMPLE_FRAMES_MIN", "4")
    monkeypatch.setenv("MEDIA_SAMPLE_FRAMES_MAX", "48")

    assert budget_for_duration(duration_seconds=5.0).tagged_max == 8
    assert budget_for_duration(duration_seconds=30.0).tagged_max == 12
    assert budget_for_duration(duration_seconds=120.0).tagged_max == 24
    assert budget_for_duration(duration_seconds=600.0).tagged_max == 36
    assert budget_for_duration(duration_seconds=4000.0).tagged_max == 48
    assert budget_for_duration(duration_seconds=4000.0, override=8).tagged_max == 8


def test_plan_timestamps_cover_duration() -> None:
    probe = MediaProbe(duration_s=100.0, total_frames=1000, time_base="fps", fps=10.0, kind="video")
    budget = SamplingBudget(tagged_max=10, candidate_max=10)
    stamps = plan_candidate_timestamps(probe, budget)
    assert len(stamps) == 10
    assert stamps[0] < stamps[-1]
    assert stamps[0] >= 0
    assert stamps[-1] < 100.0


def test_select_prefers_non_rejected_and_covers_end() -> None:
    frames = []
    qualities = []
    for i in range(20):
        # Late frames have content; early are black.
        color = (0, 0, 0) if i < 16 else (200, 40, 40)
        img = Image.new("RGB", (32, 32), color=color)
        if i >= 16:
            # Add edges so quality passes.
            pixels = img.load()
            for x in range(32):
                pixels[x, 8] = (255, 255, 255)
                pixels[8, x] = (255, 255, 255)
        frame = SampledFrame(source_index=i, image=img, timestamp_s=float(i))
        q = compute_frame_quality(img)
        frames.append(frame)
        qualities.append(q)
    selected = select_tagged_frames(list(zip(frames, qualities)), SamplingBudget(8, 20))
    assert selected
    assert max(f.source_index for f in selected) >= 16


def test_variable_delay_gif_time_sampling(tmp_path: Path) -> None:
    path = tmp_path / "var.gif"
    # Dense short delays then long delays — index-uniform would bias early half.
    frames = []
    durations = []
    for i in range(20):
        color = (255, 0, 0) if i < 10 else (0, 255, 0)
        frames.append(Image.new("RGB", (8, 8), color=color))
        durations.append(10 if i < 10 else 200)
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    probe = probe_media(path)
    assert probe.duration_s is not None
    assert probe.per_frame_delays_ms is not None
    # Mid-time should land in the long-delay (green) half.
    mid_t = probe.duration_s * 0.75
    budget = SamplingBudget(tagged_max=1, candidate_max=1)
    stamps = [mid_t]
    from app.media_sampling import decode_frames

    sampled = decode_frames(path, stamps, probe)
    assert sampled
    # Green dominant pixel.
    px = sampled[0].image.getpixel((4, 4))
    assert px[1] > px[0]


def test_sample_gif_frames_evenly(tmp_path: Path) -> None:
    path = tmp_path / "anim.gif"
    frames = [
        Image.new("RGB", (8, 8), color=c)
        for c in ("red", "green", "blue", "yellow", "purple", "cyan")
    ]
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
    )
    sampled = sample_gif_frames(path, sample_count=3)
    assert len(sampled) == 3
    assert all(frame.mode == "RGB" for frame in sampled)


def test_experimental_media_gif_uses_presence_pool(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "multi.gif"
    frames = [Image.new("RGB", (16, 16), color=(40 + i * 30, 20, 20)) for i in range(6)]
    for img in frames:
        px = img.load()
        for x in range(16):
            px[x, 4] = (255, 255, 255)
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
    )

    def fake_score_many(images, **kwargs):
        assert kwargs.get("raw_general") is True
        out = []
        for idx, _img in enumerate(images):
            # Two frames above support for loli.
            out.append({"loli": 0.9 if idx in (0, 2) else 0.05, "1girl": 0.5})
        return out

    class _Engine:
        def score_many(self, images, **kwargs):
            return fake_score_many(images, **kwargs)

    monkeypatch.setattr("app.inference_engine.get_engine", lambda: _Engine())
    monkeypatch.setattr("app.media_classify.get_engine", lambda: _Engine(), raising=False)

    scores = extract_scores_with_experimental_media(
        path,
        experimental_media_enabled=True,
        tagger_model=TAGGER_MODEL_WD_SWINV2,
        sample_count=4,
    )
    assert scores["loli"] > 0.5
    assert scores["1girl"] >= 0.5


def test_media_pooling_preserves_weak_realism_floor(monkeypatch, tmp_path: Path) -> None:
    """Hybrid real_life blending needs realistic/photorealistic >= floor (~0.18)
    to survive presence pooling even though they sit below the 0.35 support."""
    path = tmp_path / "real.gif"
    frames = [Image.new("RGB", (16, 16), color=(60 + i * 20, 90, 120)) for i in range(6)]
    for img in frames:
        px = img.load()
        for x in range(16):
            px[x, 7] = (255, 255, 255)
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
    )

    def fake_score_many(images, **kwargs):
        assert kwargs.get("raw_general") is True
        # Weak-but-consistent photo signal, below general support threshold.
        return [{"realistic": 0.22, "1girl": 0.6, "junk": 0.05} for _ in images]

    class _Engine:
        def score_many(self, images, **kwargs):
            return fake_score_many(images, **kwargs)

    monkeypatch.setattr("app.inference_engine.get_engine", lambda: _Engine())

    scores = extract_scores_with_experimental_media(
        path,
        experimental_media_enabled=True,
        tagger_model=TAGGER_MODEL_WD_SWINV2,
        sample_count=4,
    )
    assert scores.get("realistic", 0.0) >= 0.18
    assert scores["1girl"] >= 0.5
    # Sub-floor junk stays suppressed.
    assert scores.get("junk", 0.0) == 0.0


def test_even_frame_indices_covers_span() -> None:
    assert even_frame_indices(1, 8) == [0]
    assert even_frame_indices(8, 8) == list(range(8))
    idxs = even_frame_indices(100, 4)
    assert len(idxs) == 4
    assert idxs[0] < idxs[-1]
