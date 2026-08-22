---
title: THR3SHR
emoji: "🌾"
colorFrom: yellow
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

# THR3SHR

Local FastAPI + React workflow that threshes a media dump into destination bins — anime/GIF/video tagging, review, and folder migration (ML-Danbooru / WD taggers).

**© 2026 Dinamush** — software under [MIT](./LICENSE); brand, UI creative materials, and curated docs under [CC BY 4.0](./CREATIVE_COMMONS.md). Third-party model owners are credited in [ATTRIBUTION.md](./ATTRIBUTION.md).

This repo includes:

- Tag vocabulary (`tags.csv`)
- FastAPI backend for scan → tag → review → migrate
- React SPA with sectioned settings, run dashboard, and review table
- Multi-model inference via `dghs-imgutils` (ML-Danbooru ONNX and WD14 taggers)

ONNX weights are **not** stored in this git tree (too large for GitHub). They are downloaded into the Hugging Face Hub cache on first use. See [Model weights](#model-weights) for direct links.

## What The Web UI Does

1. **Settings** — paths, tagger model, thresholds, workers; sticky unsaved save bar
2. **Tag selection** — search chips from `tags.csv` (auto-persisted)
3. **Preview scan** — eligible image counts before starting a run
4. **Run dashboard** — status, totals, failed/review counts, progress bar, cancel
5. **Review table** — preview, primary (or “needs review”), secondary suggestions, **global top-5 tags**, final tag, approve/migrate

## Tagger Models

Choose in Settings (persisted in SQLite as `tagger_model`):

| Setting value | Backend call | Notes |
|---|---|---|
| `wd_swinv2_v3` (default) | `get_wd14_tags(..., model_name="SwinV2_v3")` | Recommended accuracy baseline |
| `wd_eva02_large` | `get_wd14_tags(..., model_name="EVA02_Large")` | Larger / slower WD tagger |
| `ml_danbooru` | `get_mldanbooru_tags(..., threshold=0.0, size=448)` | Original ML-Danbooru path |

Related settings:

- `confidence_threshold` (default `0.6`) — assignment gate for selected tags
- `wd_general_threshold` (default `0.35`) — WD14 general-tag cutoff at inference (WD models only)
- `max_inference_workers` (default `2`) — keep low; ORT session is serialized under a lock
- `force_cpu_inference` — force CPU path

## Classification Rules

- Scores are produced for the whole vocabulary (model-dependent).
- **Folder assignment ranks only user-selected tags** that map to `tags.csv`.
- Empty scores → failed / needs review.
- Best selected score **&lt; noise floor** `max(0.15, confidence_threshold * 0.5)` → no primary (weak noise winner discarded; kept as secondary suggestion).
- Best selected score **&lt; confidence_threshold`** → no primary; suggestion in secondary.
- Review payload always includes **global top-5** tags so mis-assignments are diagnosable.
- Each run stores `tagger_model` for provenance.

## Quick Start

### 1) Create Python environment and install backend dependencies

```bash
cd /path/to/thr3shr
python3 -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
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

The frontend falls back to local mock mode when the backend is unreachable. Preview/migration are mock-only in that mode.

## Review Workflow

1. Set **root repository** (unsorted images) and **categories root** (destination folders — not `tags.csv`)
2. Pick a tagger model and thresholds → **Save settings**
3. Optionally **Preview scan**
4. Select destination tags → **Start run**
5. Watch the run dashboard; cancel if needed
6. Review: primary / secondary / global tops → approve → migrate (`copy` or `move`)

## Supported And Ignored Files

- Supported image extensions: `.jpg`, `.jpeg`, `.jfif`, `.png`, `.bmp`, `.webp`, `.tiff`
- Extensionless / odd extensions are included when PIL can decode them
- By default ignored: `.gif` and common video formats
- Optional **experimental media**: quality-filtered multi-frame sampling with presence pooling (require ≥2 corroborating frames; tagged budget up to 48; black/blank frames rejected) when enabled in settings

## GPU Acceleration

Inference uses ONNX Runtime CUDA when the pip CUDA/cuDNN wheels are installed and discoverable.

```bash
pip install -r backend/requirements.txt
# includes: onnxruntime-gpu[cuda,cudnn]==1.26.0
```

On Windows the API prepends `site-packages/nvidia/*/bin` to the DLL search path before creating sessions. Without that, ORT may *list* CUDA then fall back to CPU on the first Conv.

CUDA usability is independent of which tagger model is selected; both ML-Danbooru and WD14 share the ORT runtime path.

### Verify GPU is actually usable

```bash
curl http://127.0.0.1:8000/health/providers
curl http://127.0.0.1:8000/api/providers
```

Look for `"cuda_usable": true`, `"likely_device": "gpu"`, and `active_providers` containing `CUDAExecutionProvider`. Listing CUDA alone is not enough. The providers payload also echoes the active `tagger_model`.

### Runtime controls

Settings UI values are mirrored into env knobs used by the run executor:

- `MAX_INFERENCE_WORKERS` (default `2`, clamped `1..16`)
- `FORCE_CPU_INFERENCE=true`
- `INFERENCE_MODE=batch|single` (default `single` in practice for imgutils; batch falls back to per-image)
- `INFERENCE_BATCH_SIZE` (default `1`, clamped `1..64`)
- `QUEUE_SHUFFLE_ENABLED=true|false` (default `true`)
- `QUEUE_SHUFFLE_SEED=<int>` (default `run_id`)

Examples:

```bash
# CPU-safe baseline
FORCE_CPU_INFERENCE=true MAX_INFERENCE_WORKERS=1 ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Conservative GPU (recommended)
MAX_INFERENCE_WORKERS=2 ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## API Surface (High Level)

- `GET /health`
- `GET /health/providers`
- `GET /api/providers`
- `GET/PUT /api/settings` — includes `tagger_model`, `wd_general_threshold`, `selected_tags`, workers, etc.
- `GET /api/tags`
- `GET /api/scan/preview` — discovery stats without inference
- `POST /api/runs/start`
- `GET /api/runs/{run_id}/status` — includes `tagger_model`, progress telemetry
- `POST /api/runs/{run_id}/cancel`
- `GET /api/runs/{run_id}/items` — includes `global_top_tags`, secondary suggestions
- `PATCH /api/items/{item_id}`
- `GET /api/items/{item_id}/preview`
- `GET /api/items/{item_id}/scores`
- `POST /api/runs/{run_id}/batch`
- `POST /api/runs/{run_id}/migrate`

## Tests

```bash
cd backend
../.venv/bin/python -m pytest tests -q
```

## Model weights

Tagger ONNX files are pulled automatically via `huggingface_hub` on first inference. Hosted copies (not in this repo):

| Setting value | Weights / labels | Size note |
|---|---|---|
| `wd_swinv2_v3` (default) | [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) · mirror [deepghs/wd14_tagger_with_embeddings](https://huggingface.co/deepghs/wd14_tagger_with_embeddings) (`…/model.onnx`) | ~hundreds of MB |
| `wd_eva02_large` | [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) · same deepghs mirror path | larger / slower |
| `ml_danbooru` | [deepghs/ml-danbooru-onnx](https://huggingface.co/deepghs/ml-danbooru-onnx) (`ml_caformer_m36_dec-5-97527.onnx`) · tags [deepghs/imgutils-models](https://huggingface.co/deepghs/imgutils-models) (`mldanbooru/mldanbooru_tags.csv`) | ONNX + CSV |

Cache location (typical): `~/.cache/huggingface/hub/`.

## License and attribution

| What | Owner | License |
|---|---|---|
| THR3SHR source code | Dinamush | [MIT](./LICENSE) |
| THR3SHR brand, creative docs, curated taxonomies | Dinamush | [CC BY 4.0](./CREATIVE_COMMONS.md) |
| WD / ML-Danbooru / style / optional VLM weights | Upstream authors (SmilingWolf, deepghs, bartowski, porntech, …) | See [ATTRIBUTION.md](./ATTRIBUTION.md) |

Model weights are **not** shipped in this repository. Preserve `LICENSE`, `CREATIVE_COMMONS.md`, and `ATTRIBUTION.md` when redistributing.

## Remotes

| Remote | URL |
|---|---|
| `origin` (Hugging Face Hub) | https://huggingface.co/Dinamus/thr3shr |
| `github` | https://github.com/Dinamush/thr3shr |

```bash
git push origin main    # Hugging Face
git push github main    # GitHub
```

The Gradio `app.py` entry in the YAML front matter is a legacy Space config; the primary workflow documented here is the FastAPI + React app.
