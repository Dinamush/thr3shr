# Robust GIF/Video Presence Classification — Design

**Date:** 2026-07-24  
**Status:** Draft — awaiting user review  
**Repo:** `thr3shr`

## Problem

Experimental GIF/video classification samples evenly spaced midpoint frames and aggregates with a misleading “mean/presence” pool that mathematically reduces to:

```text
pooled[tag] = peak × (frames_with_tag > 0) / N
```

Weaknesses:

1. **Black / fade / blank frames** count in `N` and suppress valid tags (especially WD, where sub-threshold scores are dropped before pooling).
2. **Uniform midpoints** miss brief events between samples; endpoints are usually skipped; long clips hard-cap at 24 tagged frames (~one sample / 150s on a 1h video).
3. **Presence intent is wrong for mean** — user wants folder routing when a selected tag **appears somewhere**, not when it dominates the clip.
4. **Unknown-length / FFmpeg fallback** undersamples openings only.
5. **Style gate** uses frame 0 only (titles/black opens can reject the whole clip).
6. **Tests** cover sampling math and GIF decode but not black pollution, corroboration, long sparse events, or variable-delay GIFs.

## Goals (product)

| Choice | Decision |
|--------|----------|
| Routing semantics | **Presence** — route if tag appears in the clip |
| Corroboration | Require **≥2** independent frames/scenes |
| Long-media budget | Cap ~**48** tagged frames + cheap candidate oversampling (no default deep-scan) |

Success criteria:

- Injecting black/fade frames must not materially bury a real mid-clip tag.
- A tag present in only one sampled frame must **not** auto-route (needs review / suppressed).
- A tag present in ≥2 corroborating samples must be able to route under the existing confidence threshold path.
- Inference frames tagged per file stay ≤ 48 (env-overridable, hard max 48 for v1).
- Existing still-image classification behavior unchanged.

## Non-goals (v1)

- Optional deep-scan mode (~1 tagged frame / 5–10s).
- New UI settings beyond the existing experimental-media toggle (env knobs OK).
- PySceneDetect / TransNetV2 dependencies.
- Changing taxonomy evidence weights or destination priority order.
- Guaranteeing recall of 2-second events in multi-hour videos under the 48-frame cap.

## Current → target architecture

```text
TODAY
  decode → N midpoint frames → (WD threshold per frame) → peak×hit_rate → taxonomy

TARGET
  probe duration
    → candidate pool (~4N, ≤192) time-uniform
    → quality filter (reject black/blank/low-info)
    → scene/bin pick ≤ tagged budget (≤48)
    → ONNX with RAW probabilities
    → scene-collapse + top-K presence + ≥2 support
    → taxonomy / confidence gates (unchanged)
```

## Components

### 1. Duration / time-base probing

**GIF**

- Prefer cumulative per-frame `duration` (ms) via `ImageSequence` when available.
- Fall back to constant `info["duration"] × n_frames`.
- Sample in **playback time**, not raw frame index, when delays vary.

**Video (OpenCV)**

- `duration = frame_count / fps` when both valid.
- Keep frame-index seeking for known counts.
- Unknown length: do not grab consecutive opening frames; either:
  - probe with FFmpeg/`ffprobe` duration when available, or
  - sequential read with increasing skip until budget filled across the opened stream,
  - and log `duration=unknown` for metrics.

**FFmpeg fallback**

- Prefer `ffprobe` duration; distribute samples across full duration.
- Stop assuming fixed 30s / first-~18s only.

### 2. Tagged-frame budget (duration bands)

Defaults (env-overridable; clamp tagged max to 48):

| Duration | Tagged frames |
|----------|---------------|
| &lt; 10s | 8 |
| 10s–60s | 12 |
| 1–5 min | 24 |
| 5–30 min | 36 |
| &gt; 30 min | 48 |

Candidate pool size ≈ `min(192, 4 × tagged_budget)`.

Retain `MEDIA_SAMPLE_FRAMES` as a hard override of tagged count (still clamped).

Raise `DEFAULT_MEDIA_SAMPLE_MAX` from 24 → 48. Update `MEDIA_SAMPLE_FRAMES_MAX` clamp accordingly (already hi=48).

Real-life hybrid path that currently forces 8 frames: keep an explicit cap for that path unless product later asks otherwise; document it. Style gate sampling is separate (quality-picked single frame).

### 3. Frame quality metrics

Operate on a downscaled grayscale thumbnail (~160–256 px long side).

Signals:

- mean / std luminance `Y ∈ [0,1]`
- p05 / p95 luminance
- 32-bin histogram entropy (bits)
- Canny edge density (edges / pixels)
- optional Laplacian variance (blur) — nice-to-have, not required for reject

**Reject rules** (multi-signal; start conservative):

```text
black:
  p95(Y) < 0.08
  OR (mean(Y) < 0.05 AND std(Y) < 0.025)

near-white blank:
  p05(Y) > 0.92 AND std(Y) < 0.025

low-information:
  entropy < 2.0 AND edge_density < 0.003
```

If every candidate is rejected, keep the **highest-quality** candidate (edge density × entropy ranking) so all-black clips still decode something and fail gracefully downstream.

Do **not** reject dark anime scenes solely on low mean luminance when edges/entropy remain.

### 4. Candidate selection (scene/bin)

1. Place candidate timestamps uniformly across `[0, duration)`.
2. Decode candidates (GIF seek / video seek).
3. Drop quality rejects; replace from neighbors in the same bin when possible (one retry per slot).
4. Partition remaining into `tagged_budget` time bins.
5. Per bin, pick the best-quality frame (prefer interior of bin over exact edges when ties).
6. Ensure coverage: if fewer than `min(tagged_budget, usable)` frames, fill with next-best remaining candidates by time spacing.

Failed seeks must attempt a neighbor index; do not silently shrink the denominator without replacement when alternatives exist.

### 5. Scoring: raw probabilities for media

Today WD drops tags below `wd_general_threshold` **before** pooling inside `score_many`. For media aggregation that is harmful.

**Change:**

- Media path requests raw general-tag probabilities (new flag or `wd_general_threshold=0` only on media score path, then apply support threshold in the pooler).
- Still images keep current thresholding behavior.
- ML-Danbooru already emits dense sigmoids — unchanged.

### 6. Presence pooling with corroboration

Replace `pool_frame_scores` for media with presence-oriented pooling (keep old function name as wrapper or rename and update call sites/tests).

Per tag:

1. Collect per-frame scores (zeros for missing).
2. Optionally collapse adjacent frames within a small time gap into one “scene” by taking max (when timestamps available; else treat each sampled frame as its own scene if temporally spaced by construction).
3. Let `support_thr` default to `wd_general_threshold` (0.35) or `MEDIA_SUPPORT_THRESHOLD`.
4. `hits = scenes with score >= support_thr`.
5. If `hits < 2`: **suppress pooled score to 0** for taxonomy input so one-frame spikes cannot win destinations. (Peak diagnostics can be added later; not required for v1.)
6. If `hits >= 2`:  
   `presence = mean(top_k scene scores)` where `k = min(3, hits)` (top-2/3).
7. **Short-media exception:** if fewer than 2 usable tagged frames remain after quality filtering (e.g. 1-frame GIF), fall back to single-frame still semantics (allow 1-hit / max score) so tiny clips are not forced to zero.
8. Feed `presence` into existing taxonomy.

**Not used as primary:** plain mean, median, peak×hit_rate, noisy-OR.

Document that still-image path does not use this pooler.

### 7. Style gate

When experimental style detector runs on GIF/video:

- Sample **one quality-picked** frame from a small candidate set (e.g. 8 time-uniform → quality filter → best), not forced frame 0.
- If all candidates fail quality, use best-of-pool fallback.

### 8. API / settings surface

- No new required UI toggles for v1.
- Env knobs: budget bands / max / candidate multiplier / support threshold / quality thresholds.
- README: fix “sample a frame” → describe multi-frame presence pooling + quality filter.
- Logging: candidates kept/rejected, tagged count, duration, corroboration hit counts for top destinations (debug level OK).

## File touch map (expected)

| Area | Files |
|------|--------|
| Sampling / quality / pool | `backend/app/services.py` (possibly split `media_sampling.py` if file grows too large) |
| Raw WD for media | `backend/app/inference_engine.py`, media call in `services.py` |
| Style first frame | `backend/app/style_detectors.py` |
| Caps / hybrid 8-frame | `backend/app/api.py` (only if needed for max consistency) |
| Tests | `backend/tests/test_services.py`, new `test_media_presence_pooling.py`, synthetic fixture helpers |
| Docs | this spec; README media section |

## Testing plan (autonomous, comprehensive)

### Unit

- Quality metrics: pure black, white blank, normal anime-like synthetic pattern, dark-but-edged frame.
- Budget bands vs duration.
- Time-based GIF indices with variable delays.
- Pooler: 1-hit → 0; 2-hit → ~top-k mean; black frames injected into score maps → stable presence when ≥2 real hits remain.
- Neighbor seek replacement when primary index fails (mocked).

### Integration / synthetic media (Pillow + OpenCV, no GPU required for pool/quality; optional ONNX smoke)

Build temporary GIF/MP4 fixtures:

1. Content mid-clip + 50% black frames → presence of content tags survives vs old pooler baseline (compare functions on known score maps; decode path checked separately).
2. Tag signal only at begin / mid / end (paint distinct regions or inject score maps with timestamps).
3. Single-frame spike → no route (pool → 0).
4. Two separated spikes → routes.
5. All-black clip → does not crash; scores empty/weak → needs review path.
6. Variable-delay GIF: denser frames in one half must not bias time sampling toward that half.
7. Long sparse: mock duration 40 min with event in last bin → selected indices include late coverage under 48 cap.
8. FFmpeg fallback duration distribution (mock subprocess / skip if no ffmpeg).

### Regression

- Full `pytest tests/` must pass.
- Existing still-image taxonomy / classify accuracy tests unchanged.
- `test_media_sample_cap_scope` updated for new max / hybrid 8-frame policy.

### Optional live bench (manual / script)

If sample GIFs/videos exist on disk, script comparing old vs new destinations — not blocking merge if fixtures cover logic.

## Rollout

1. Implement sampling + quality + new pooler behind same `experimental_media_enabled` flag (no separate flag).
2. Update tests first or alongside (TDD preferred for pooler/quality).
3. Ship; deep-scan deferred.

## Risks

| Risk | Mitigation |
|------|------------|
| Suppressing 1-hit tags lowers recall on very short GIFs with only 1 usable frame | If usable tagged frames &lt; 2 after quality filter, fall back to single-frame still semantics (allow 1-hit) |
| Dark scenes over-rejected | Multi-signal reject; edges/entropy save |
| Raw WD maps larger / slower | Media only; still path unchanged |
| Taxonomy thresholds calibrated on stills | Presence scores are still in [0,1]; use same confidence gates initially; tune if review rate spikes |
| 48-frame cost on GPU | Batch as today (8); duration bands keep short clips cheaper |

## Open points resolved in design review

- Presence routing: yes  
- Corroboration: ≥2 frames/scenes  
- Long media: B (48 + oversample), not default deep-scan  
- Approach: quality + scene/bin + top-K presence (not plain mean)

## Approval

User approved conversational design 2026-07-24 (“go ahead”). This written spec is the review gate before implementation planning.
