---
title: ML-Danbooru ONNX Webapp
emoji: "🖼️"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
tags:
- image-classification
- onnx
- anime
- tagging
- danbooru
- deep-learning
- computer-vision
---

# ML-Danbooru ONNX Web UI

This repo includes:

- ML-Danbooru ONNX model assets and tag data (`tags.csv`)
- A FastAPI backend for scanning, tagging, review, and migration
- A React frontend Web UI for interactive classification workflows

## What The Web UI Does

- Configure source image root and category destination root
- Select output classes from real tags in `tags.csv` (search/select + comma-separated add)
- Scan only supported images and ignore unsupported files (videos, GIFs, unreadable files)
- Run non-blocking classification with live status/progress and cancel support
- Rank only the user-selected tags for assignment:
  - highest selected tag = primary
  - next selected tags = secondary
  - non-selected global top tags do not override selected output classes
- Show image previews and per-item full score JSON for debugging
- Approve/reject/review items and migrate approved files with `copy` or `move`

## Quick Start

### 1) Create Python environment and install backend dependencies

```bash
cd /path/to/thr3shr
python3 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2) Start backend API (port 8000)

```bash
cd /path/to/thr3shr/backend
../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 3) Start frontend Web UI (port 5173)

```bash
cd /path/to/thr3shr/frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Frontend Offline Mode

The frontend automatically falls back to local mock mode when backend requests fail. This lets UI flows work even if the backend is not running, but file preview/migration are mock-only in that mode.

## Classification Behavior Notes

- Backend inference uses `get_mldanbooru_tags(..., threshold=0.0, drop_overlap=False)` so umbrella tags like `monster_girl` are preserved in scores.
- Assignment considers only selected tags/folders that match known tags.
- Threshold is applied to the winning selected tag:
  - above threshold -> auto `approved`
  - below threshold or no selected match -> `needs_review`

## Review Workflow

1. Save settings (root repo, categories root, threshold, default migrate mode)
2. Add selected tags (type/search/select or comma-separated input)
3. Start run
4. Watch live run status (`pending`/`running`/`completed`/`failed`/`cancelled`)
5. Review table:
   - image preview
   - primary and secondary selected-tag ranking
   - debug scores JSON
6. Approve/reject/batch update and run migration (`copy` or `move`)

## Supported And Ignored Files

- Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`
- Ignored: `.gif`, common video formats, and unreadable/corrupt images

## GPU Acceleration

Inference attempts to use ONNX Runtime CUDA provider when available. If CUDA/cuDNN dependencies are missing, runtime falls back to CPU and logs provider errors.

### Verify GPU visibility

```bash
curl http://127.0.0.1:8000/health/providers
curl http://127.0.0.1:8000/api/providers
```

`likely_device: "gpu"` means CUDA provider is visible and CPU is not being force-disabled.

### Runtime controls

- `MAX_INFERENCE_WORKERS` (default `2`, clamped to `1..16`): controls conservative thread-pool parallelism for per-image inference.
- `FORCE_CPU_INFERENCE=true`: force reported/expected CPU path even if CUDA provider is available.
- `INFERENCE_MODE=batch|single` (default `batch`): prefer GPU-first batched inference or legacy single-image inference.
- `INFERENCE_BATCH_SIZE` (default `8`, clamped to `1..64`): request batch size for batched inference. If batch inference fails, runtime auto-falls back by splitting batches down to single-image.
- `QUEUE_SHUFFLE_ENABLED=true|false` (default `true`): stochastic queue ordering toggle.
- `QUEUE_SHUFFLE_SEED=<int>` (default `run_id`): deterministic seed for reproducible queue shuffling.

Examples:

```bash
# CPU-safe baseline
FORCE_CPU_INFERENCE=true MAX_INFERENCE_WORKERS=1 ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Throughput mode (tune workers to your CPU/GPU memory limits)
MAX_INFERENCE_WORKERS=4 ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# GPU-first batching + seeded stochastic queue
INFERENCE_MODE=batch INFERENCE_BATCH_SIZE=8 QUEUE_SHUFFLE_ENABLED=true QUEUE_SHUFFLE_SEED=1337 MAX_INFERENCE_WORKERS=4 ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## API Surface (High Level)

- `GET /health`
- `GET /health/providers`
- `GET /api/providers`
- `GET/PUT /api/settings`
- `GET /api/tags`
- `POST /api/runs/start`
- `GET /api/runs/{run_id}/status`
- `POST /api/runs/{run_id}/cancel`
- `GET /api/runs/{run_id}/items`
- `PATCH /api/items/{item_id}`
- `GET /api/items/{item_id}/preview`
- `GET /api/items/{item_id}/scores`
- `POST /api/runs/{run_id}/batch`
- `POST /api/runs/{run_id}/migrate`

## Model Assets

This repo contains multiple ONNX variants, including:

- `ml_caformer_m36_dec-5-97527.onnx`
- `ml_caformer_m36_dec-3-80000.onnx`
- `TResnet-D-FLq_ema_2-40000.onnx`
- `TResnet-D-FLq_ema_4-10000.onnx`
- `TResnet-D-FLq_ema_6-10000.onnx`
- `TResnet-D-FLq_ema_6-30000.onnx`
- `caformer_m36-3-80000.onnx`

`tags.csv` contains the canonical tag list used for matching and validation.
