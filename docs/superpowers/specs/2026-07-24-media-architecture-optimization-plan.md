# Media Classification — Architectural Optimization Plan

**Date:** 2026-07-24  
**Auditor:** Claude Fable 5 (controlled subagent)  
**Companion design:** `2026-07-24-robust-media-presence-classification-design.md`

## Verdict

Implement the design behind a **module split first**, not by growing `services.py`. Ship in phases so P0 (raw WD + presence pooling) lands before the heavier sampling rebuild.

## Critical findings

1. **`services.py` is a god-module** (~924 lines): sampling, pooling, scoring, preview, scan, migration, settings. The new pipeline would push it toward ~1400 lines if done in place.
2. **WD thresholding inside `score_many` destroys pooler input** — media cannot do correct presence pooling until raw probabilities are available.
3. **Frames lose timestamps** — `list[Image.Image]` cannot support scene collapse, time-based GIF sampling, or bin replacement.
4. **FFmpeg / unknown-duration paths sample openings only** — not full-span.
5. **`sample_count=1` ⇒ frame 0** for style gate, preview, and single-frame paths — black opens poison style + thumbnails.
6. **Old pooler tests lock in broken behavior** — must be replaced, not preserved.

## Preserve

- Owned `InferenceEngine` ORT locking/batching
- Media vs still queue split in run/reclassify
- Taxonomy consuming plain `dict[str, float]`
- Still-image extract paths, taxonomy.json, hybrid blend thresholds (except which frame feeds style)

## Proposed layout

| Module | Role |
|--------|------|
| `media_types.py` | `MediaProbe`, `SampledFrame`, `FrameQuality`, `SamplingBudget` |
| `media_sampling.py` | probe, budget bands, candidate timestamps, decode + neighbor retry, ffmpeg/ffprobe |
| `media_quality.py` | thumbnail metrics, reject/rank, best-frame pick |
| `media_selection.py` | bin pick + coverage fill |
| `media_pooling.py` | `pool_presence` (≥2 hits, top-k, short-clip fallback) |
| Thin orchestrator | `extract_scores_with_experimental_media` stays in `services.py` as ~5-line delegator |

Surgical only: `inference_engine.score_many(..., raw_general=True)`; style gate uses quality-picked frame; move `FILTER_MEDIA_SAMPLE_MAX` into budget policy.

## Pipeline boundaries

```text
probe → plan timestamps → decode(SampledFrame) → quality → select ≤48
  → score(raw) → pool_presence → taxonomy (unchanged)
```

Decode is the only partial-failure stage; later stages are pure/deterministic.

## Priorities

### P0 — required for correctness
1. Raw WD probabilities for media
2. `pool_presence` with ≥2 corroboration + short-clip fallback
3. Timestamped `SampledFrame` records

### P1 — required for design success criteria
4. Quality filter + candidate oversampling + bin selection (cap 48)
5. Duration probing (variable-delay GIF, ffprobe, full-span ffmpeg)
6. Neighbor-retry on failed seeks

### P2
7. Style gate quality-picked frame  
8. Preview best-frame reuse (optional)  
9. Constant/normalizer dedup  
10. Debug logging counters  

## Phased delivery

1. **Seams + raw scores** — new modules dark; old pooler still live  
2. **Cutover pooling** — presence pool live on old sampler; rewrite old pooler tests  
3. **Sampling rebuild** — quality/bins/budget/ffmpeg fixes + synthetic fixtures  
4. **Style gate + README**  
5. **Optional hygiene**

Phases 1–2 are independently shippable and remove single-frame spike routing + threshold-censored pooling with a small diff.

## Do not refactor in this work

Still-image extract/retry, taxonomy evidence/priority, InferenceEngine internals (except raw flag), run orchestration/batch fallback ladder, preview cache/migration/scan (except optional preview quality later).
