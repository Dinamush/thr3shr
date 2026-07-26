# Doujin Works Implementation Plan

> **For agentic workers:** Implement task-by-task; mark checkboxes done as you go.

**Goal:** Folder/cbz work classification with sample-pool tagging, review UX, move+junction migrate.

**Architecture:** New `backend/app/doujin_works.py` for scan/sample/infer; `run_mode=doujin_works` in API; taxonomy milf+inseki; migrate preserves work unit + junctions; UI start + display.

## Task 1: Taxonomy milf + inseki

- Add `milf` bucket; alias `inseki` on `incest`
- Tests for routing

## Task 2: doujin_works module

- `scan_works`, `sample_work_images`, `classify_work` (pool + choose destinations)
- Unit tests with tmp folders / tiny zip

## Task 3: API run + migrate

- `StartRunRequest.run_mode` includes `doujin_works`
- Execute path inserts one item per work
- Migrate moves folder/cbz; junctions for secondaries under `Doujins/<tag>/`

## Task 4: Frontend

- Start Doujin works run button
- Show work name + primary + category tags

## Task 5: Verify

- pytest targeted suite; restart server
