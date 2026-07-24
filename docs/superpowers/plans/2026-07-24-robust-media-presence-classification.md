# Robust Media Presence Classification — Implementation Plan

> **For agentic workers:** Execute task-by-task. Checkboxes track progress.

**Goal:** Presence-based GIF/video tagging with quality-filtered sampling, raw WD probs, and ≥2-frame corroboration.

**Architecture:** Split media into `media_types` / `media_sampling` / `media_quality` / `media_selection` / `media_pooling` / thin `media_classify` orchestrator; keep `extract_scores_with_experimental_media` as stable delegator in `services.py`.

**Tech Stack:** Pillow, OpenCV, numpy, ONNX Runtime (existing), pytest.

## Global Constraints

- Still-image path unchanged
- Taxonomy unchanged
- Tagged frame hard max 48; hybrid real_life override stays 8
- Presence routing + min_hits=2; short-clip (&lt;2 usable) still semantics
- No new UI toggles; env knobs OK
- Full `pytest tests/` green at each phase end

## Tasks

- [x] Phase 1: `media_types.py`, `media_pooling.py` + unit tests, `raw_general` on `score_many`
- [x] Phase 2: Wire media path to raw + `pool_presence`; delete old pooler tests; rewrite GIF pool test
- [x] Phase 3: `media_sampling` / `media_quality` / `media_selection` / `media_classify`; budget 48; ffmpeg/ffprobe; synthetic fixtures
- [x] Phase 4: Style gate quality-picked frame; README media section
- [x] Regression: full pytest suite (194 passed; 2 pre-existing failures unrelated to media — Voyeur taxonomy matrix + migrate edge case)
- [x] Fable follow-up audit: short-clip raw junk threshold; realism-floor merge for hybrid; POS_MSEC seek when fps unknown; lazy test import (44 media tests green)
- [x] Outstanding backlog: unknown-length span sample; omit 0.0 pool keys; quality preview stills; Voyeur soft `pussy` weight; migrate test matches destination repair

## Out of scope

Deep-scan mode, preview best-frame reuse, normalizer dedup.
