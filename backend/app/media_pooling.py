"""Presence-oriented pooling for multi-frame media tag scores."""

from __future__ import annotations

from collections import defaultdict


def pool_presence(
    frame_scores: list[tuple[float | None, dict[str, float]]],
    *,
    support_thr: float = 0.35,
    min_hits: int = 2,
    top_k: int = 3,
    scene_gap_s: float | None = 1.0,
) -> dict[str, float]:
    """Aggregate per-frame tag maps into clip-level presence scores.

    - Collapse near-adjacent timestamps into scenes (max within gap).
    - Require ``min_hits`` scenes at/above ``support_thr`` (else 0).
    - Score = mean of top-k scene scores among hits.
    - If fewer than ``min_hits`` frames were scored at all, fall back to
      single-frame still semantics (max across frames, thresholded at
      ``support_thr``) so 1-frame GIFs work. Thresholding matters because the
      media path scores with raw (dense) probabilities; without it a short
      clip would leak thousands of sub-threshold junk tags downstream.
    """
    if not frame_scores:
        return {}

    usable = [(ts, dict(scores)) for ts, scores in frame_scores if scores is not None]
    if not usable:
        return {}

    # Short-media exception: not enough frames to corroborate.
    if len(usable) < min_hits:
        thr_fallback = float(support_thr)
        keys: set[str] = set()
        for _, scores in usable:
            keys.update(scores)
        out: dict[str, float] = {}
        for key in keys:
            best = max(float(scores.get(key, 0.0)) for _, scores in usable)
            if best >= thr_fallback:
                out[key] = best
        return out

    scene_maps = _collapse_scenes(usable, scene_gap_s=scene_gap_s)
    keys = set()
    for scores in scene_maps:
        keys.update(scores)

    pooled: dict[str, float] = {}
    k = max(1, int(top_k))
    thr = float(support_thr)
    for key in keys:
        scene_vals = [float(scores.get(key, 0.0)) for scores in scene_maps]
        hits = [v for v in scene_vals if v >= thr]
        if len(hits) < min_hits:
            # Omit suppressed tags — do not emit dense 0.0 noise into taxonomy.
            continue
        ranked = sorted(hits, reverse=True)
        take = ranked[: min(k, len(ranked))]
        pooled[key] = sum(take) / float(len(take))
    return pooled


def _collapse_scenes(
    usable: list[tuple[float | None, dict[str, float]]],
    *,
    scene_gap_s: float | None,
) -> list[dict[str, float]]:
    if not usable:
        return []
    # No timestamps or no gap → each frame is its own scene.
    if scene_gap_s is None or scene_gap_s <= 0 or all(ts is None for ts, _ in usable):
        return [scores for _, scores in usable]

    # Sort by timestamp; missing ts stay in relative order at end.
    indexed = list(enumerate(usable))
    indexed.sort(
        key=lambda item: (
            item[1][0] is None,
            item[1][0] if item[1][0] is not None else 0.0,
            item[0],
        )
    )

    scenes: list[dict[str, float]] = []
    cur_ts: float | None = None
    cur: dict[str, float] = {}
    gap = float(scene_gap_s)

    def flush() -> None:
        nonlocal cur, cur_ts
        if cur:
            scenes.append(cur)
        cur = {}
        cur_ts = None

    for _, (ts, scores) in indexed:
        if ts is None:
            flush()
            scenes.append(dict(scores))
            continue
        if cur_ts is None:
            cur_ts = float(ts)
            cur = dict(scores)
            continue
        if abs(float(ts) - cur_ts) <= gap:
            for key, value in scores.items():
                prev = cur.get(key, 0.0)
                if float(value) > prev:
                    cur[key] = float(value)
            # Keep earliest timestamp as scene anchor.
        else:
            flush()
            cur_ts = float(ts)
            cur = dict(scores)
    flush()
    return scenes
