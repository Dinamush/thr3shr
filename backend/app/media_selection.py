"""Select tagged frames from quality-scored candidates."""

from __future__ import annotations

from .media_quality import is_rejected, quality_rank
from .media_types import FrameQuality, SampledFrame, SamplingBudget


def select_tagged_frames(
    candidates: list[tuple[SampledFrame, FrameQuality]],
    budget: SamplingBudget,
) -> list[SampledFrame]:
    """Pick up to ``budget.tagged_max`` frames via time-bin best-quality.

    Rejected frames are skipped when alternatives exist. If everything is
    rejected, keep the single highest-ranked candidate.
    """
    if not candidates:
        return []

    tagged_max = max(1, int(budget.tagged_max))
    usable = [(f, q) for f, q in candidates if f.decode_ok and not is_rejected(q)]
    if not usable:
        ranked = sorted(
            ((f, q) for f, q in candidates if f.decode_ok),
            key=lambda item: quality_rank(item[1]),
            reverse=True,
        )
        return [ranked[0][0]] if ranked else [candidates[0][0]]

    # Sort by timestamp (fallback: source_index).
    usable.sort(
        key=lambda item: (
            item[0].timestamp_s if item[0].timestamp_s is not None else float(item[0].source_index),
            item[0].source_index,
        )
    )

    if len(usable) <= tagged_max:
        return [f for f, _ in usable]

    bins: list[list[tuple[SampledFrame, FrameQuality]]] = [[] for _ in range(tagged_max)]
    n = len(usable)
    for i, item in enumerate(usable):
        bin_idx = min(tagged_max - 1, int(i * tagged_max / n))
        bins[bin_idx].append(item)

    selected: list[SampledFrame] = []
    seen_idx: set[int] = set()
    for bucket in bins:
        if not bucket:
            continue
        best_f, _best_q = max(bucket, key=lambda item: quality_rank(item[1]))
        if best_f.source_index in seen_idx:
            continue
        selected.append(best_f)
        seen_idx.add(best_f.source_index)

    # Coverage fill if some bins empty.
    if len(selected) < tagged_max:
        leftovers = [
            f for f, _ in usable if f.source_index not in seen_idx
        ]
        # Prefer frames farthest from already selected timestamps.
        while leftovers and len(selected) < tagged_max:
            pick = _farthest_frame(leftovers, selected)
            selected.append(pick)
            seen_idx.add(pick.source_index)
            leftovers = [f for f in leftovers if f.source_index not in seen_idx]

    selected.sort(
        key=lambda f: (
            f.timestamp_s if f.timestamp_s is not None else float(f.source_index),
            f.source_index,
        )
    )
    return selected[:tagged_max]


def _farthest_frame(
    candidates: list[SampledFrame],
    selected: list[SampledFrame],
) -> SampledFrame:
    def ts(frame: SampledFrame) -> float:
        if frame.timestamp_s is not None:
            return float(frame.timestamp_s)
        return float(frame.source_index)

    if not selected:
        return candidates[0]
    best = candidates[0]
    best_dist = -1.0
    for cand in candidates:
        dist = min(abs(ts(cand) - ts(s)) for s in selected)
        if dist > best_dist:
            best_dist = dist
            best = cand
    return best
