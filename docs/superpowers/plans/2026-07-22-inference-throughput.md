# Inference Throughput Core — Implementation Plan

> **For agentic workers:** Steps use TDD. Existing suite must stay green.

**Goal:** Own ORT sessions, preprocess outside lock, WD batching, startup warm; validate on SFW Danbooru `rating:g` samples.

**Files:**
- Create: `backend/app/inference_engine.py`
- Create: `backend/tests/test_inference_engine.py`
- Create: `backend/scripts/bench_sfw_sample.py`
- Modify: `backend/app/services.py`, `backend/app/main.py`, `backend/app/api.py` (batch path comments / mode)

## Task 1: Engine + equivalence tests
## Task 2: Wire services + warm startup
## Task 3: Bench script + run on SFW samples
## Task 4: Full pytest
