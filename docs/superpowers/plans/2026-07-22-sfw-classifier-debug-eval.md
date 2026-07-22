# SFW Classifier Debug Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Settings debug panel that pulls SFW images from Safebooru/Danbooru for chosen tags and reports recall + mini classify preview.

**Architecture:** Pluggable SFW board clients download into `sample_data/debug_evals/`; a sync `POST /api/debug/sfw-eval` scores with the current tagger and reuses `_classify_from_scores`. Frontend adds a collapsible Settings section.

**Tech Stack:** FastAPI, existing inference/`extract_scores`, React settings UI, urllib/json board APIs.

## Global Constraints

- Sources v1: `safebooru` (default) + `danbooru` (`rating:g` only); interface must allow more plugins
- Count: 5–30 inclusive, default 10
- No run/migrate DB side effects
- Cache under `sample_data/debug_evals/` (gitignored)
- Semicolons avoided in new JS; follow existing backend style

---

## File map

| File | Responsibility |
|------|----------------|
| `backend/app/sfw_sources.py` | Source registry + Safebooru/Danbooru fetch/download |
| `backend/app/debug_eval.py` | Orchestrate download + score + recall + classify preview |
| `backend/app/schemas.py` | Request/response models |
| `backend/app/api.py` | `/api/debug/sfw-sources`, `/api/debug/sfw-eval`, preview route |
| `backend/tests/test_sfw_debug_eval.py` | Unit/API tests with mocks |
| `frontend/src/api.js` | Client + mock |
| `frontend/src/App.jsx` | Settings debug panel UI |
| `frontend/src/styles.css` | Minimal panel styles |

---

### Task 1: SFW source plugins

- [ ] Create `sfw_sources.py` with `SfwPost`, `SfwSource` protocol/ABC, `list_sources()`, `get_source(id)`
- [ ] Implement Safebooru (`rating:safe`) and Danbooru (`rating:g`) using patterns from `scripts/bench_sfw_sample.py`
- [ ] Reject posts whose rating is not the forced SFW value
- [ ] Test: query strings contain forced rating; unsafe rating skipped

### Task 2: Eval orchestrator + API

- [ ] Create `debug_eval.py` `run_sfw_eval(...)` using settings + `extract_scores` + `_classify_from_scores`
- [ ] Add schemas + routes; validate tags non-empty, count 5–30
- [ ] Preview GET confined to debug_evals dir
- [ ] Test: mocked downloads/scores; promote recall; 400 on bad input

### Task 3: Frontend debug panel

- [ ] API helpers + offline mock
- [ ] Settings section: source, tags, count, Fetch & evaluate, results table
- [ ] Manual smoke: open settings, run mock or live eval

### Task 4: Verify

- [ ] `pytest tests/test_sfw_debug_eval.py` green
- [ ] Related API tests still green (ignore known shuffle flake)
