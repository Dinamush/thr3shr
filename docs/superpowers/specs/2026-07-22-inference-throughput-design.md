# Inference Throughput Core (Scope A) — Design

Approved: Approach 1 (own ORT sessions). SFW sample evaluation only.

## Goals

1. Own ONNX Runtime sessions per tagger model with correct CUDA/CPU provider selection (`FORCE_CPU_INFERENCE`).
2. Preprocess (decode/resize/tensorize) outside the inference lock; serialize only `session.run`.
3. True batch `Run` for WD14 models (fixed NHWC shape). ML-Danbooru stays sequential run with overlapped preprocess (variable HxW when `keep_ratio=True`).
4. Warm the active tagger session at API startup.
5. Preserve score semantics vs current imgutils-backed path (normalized tags, WD `fmt=general` + threshold, ML `threshold=0`).

## Non-goals (this pass)

- SSE polling, slim DB payloads, EVA02 review UX, TensorRT EP, FP16 export.

## Architecture

```
workers (CPU)                lock                  workers
-----------                  ----                  -------
load + preprocess   ->   session.run(batch)  ->  postprocess + classify + DB
```

New module: `backend/app/inference_engine.py`

- `get_engine()` singleton
- `warm(tagger_model)` — download weights/labels, create session, dummy forward
- `score_paths(paths, tagger_model, wd_general_threshold)` — preprocess all, locked batched/serial run, postprocess
- Providers: CUDA+CPU unless `FORCE_CPU_INFERENCE`, then CPU only
- DLL search via existing `providers.ensure_nvidia_dll_search_path` / `preload_onnx_runtime_dlls`

## Integration

- `services.extract_scores` / `extract_scores_batch` / PIL path → engine
- Keep `_run_mldanbooru` / `_run_wd14` as thin wrappers for existing unit mocks OR route tests through engine hooks
- `main.startup` warms settings’ `tagger_model` (best-effort; log failures)
- Default `INFERENCE_MODE=batch` with batch size from settings when engine supports it (WD); ML forces effective batch 1 for Run

## Validation

- Unit: preprocess/postprocess equivalence vs imgutils on fixture images
- Existing pytest suite green
- Script: download SFW sample set with expected top tags; report latency before/after semantics match + speedup

## Risks

- Numerical drift vs imgutils → assert close scores (atol ~1e-4) on shared images
- Dynamic batch dim unsupported → fall back to serial `run` inside lock
- Startup warm downloads large models → catch/log, do not block health
