# Backend scripts

Utility / probe scripts for tagging accuracy, throughput, and model audits.

## Tag recall benchmark (ML vs WD)

UI: **Debug → Tag recall benchmark**.

Pinned suite: `backend/app/data/tag_recall_suite.json`  
Cache (gitignored): `sample_data/tag_recall_suite/`

```text
.venv\Scripts\python.exe backend\scripts\fetch_tag_recall_suite.py --rebuild
.venv\Scripts\python.exe backend\scripts\run_tag_recall_eval.py
```

Latest report: `backend/scripts/out/tag_recall_eval_latest.json`.

False-positive rates on the same suite (selected Settings folders → taxonomy
evidence tags):

```text
.venv\Scripts\python.exe backend\scripts\run_tag_fp_eval.py --also-wd-threshold
```

UI: **Debug → Tag false-positive benchmark**.  
Latest report: `backend/scripts/out/tag_fp_eval_latest.json`.

## Thresh3r debug: real-life vs anime eval

UI: **Debug → Real-life vs anime**.

API: `POST /api/debug/realism-eval` with
`{"count_per_class": 20, "compare_models": true}`.

CLI (thorough, multi-model + local video probe):

```text
.venv\Scripts\python.exe backend\scripts\run_realism_debug_eval.py 20
```

Latest report: `backend/scripts/out/realism_debug_eval_latest.json`.

Photos: RandomUser people portraits (remote). Anime: Safebooru typical +
`realistic` / `photorealistic` / `3d` edges. Routing uses production
`real_life` taxonomy.

## Real-life vs anime (`real_life` bucket)

Production `backend/app/data/taxonomy.json` includes a priority-`0` `real_life`
folder (aliases: `photo`, `Real Life`, …). Select that folder in the UI to
route accidental camera photos away from content buckets.

### Probe

```text
.venv\Scripts\python.exe backend\scripts\debug_realism_probe.py
```

Report: `backend/scripts/out/realism_probe_report.json`.

Findings (12 photos + 12 anime):

- **GO** with WD, best model **`wd_eva02_large`**
- WD v3 has **`realistic` / `photorealistic`** only (no `photo_(medium)` / `3d`)
- Anime max `realistic` ≈ 0.004 on EVA02; photos often 0.11–0.99
- Inference always includes `realistic`/`photorealistic` when score ≥ **0.10**
  even if below `wd_general_threshold` (so moon/sky shots are not dropped)

### Evidence policy

| Tag | Weight | Notes |
|-----|--------|--------|
| `photorealistic` | 1.0 | Hard (WD + ML) |
| `realistic` | 1.0 | Hard on WD (probe-safe vs anime) |
| `photo_(medium)` | 1.0 | Hard, **ML only** |
| `3d` | 0.45 | Soft, **ML only** |

Prop/framing tags (`photo_(object)`, `selfie`, `photo_background`, …) are
ignored so they never win the folder alone.
