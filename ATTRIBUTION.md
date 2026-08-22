# Attribution

**THR3SHR** is owned and maintained by **Dinamush**
([GitHub](https://github.com/Dinamush) · [Hugging Face](https://huggingface.co/Dinamus)).

This document lists project ownership, licensing, and required credit for
third-party models and libraries used at runtime.

---

## 1. Project ownership

| Asset | Owner | License |
|---|---|---|
| Application source code (FastAPI backend, React frontend, tooling) | Dinamush | [MIT](./LICENSE) |
| THR3SHR brand, UI creative direction, original docs, curated taxonomies | Dinamush | [CC BY 4.0](./CREATIVE_COMMONS.md) |
| This Hub / GitHub repository layout and packaging | Dinamush | MIT + CC BY 4.0 as above |

Suggested software notice:

> Copyright (c) 2026 Dinamush. THR3SHR is licensed under the MIT License.
> Creative works © 2026 Dinamush, CC BY 4.0.

Repository mirrors:

- https://github.com/Dinamush/thr3shr
- https://huggingface.co/Dinamus/thr3shr

---

## 2. Third-party models (not redistributed here)

ONNX / GGUF / PyTorch weights are **downloaded on first use** into the local
Hugging Face Hub cache (`~/.cache/huggingface/hub/`). They are **not** committed
to this git tree. Use of each model is subject to its upstream license.

### Primary taggers

| THR3SHR setting | Upstream | Role | License (Hub metadata) |
|---|---|---|---|
| `wd_swinv2_v3` (default) | [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) | WD14 / Danbooru-style tagger (SwinV2 v3) | Apache-2.0 |
| `wd_eva02_large` | [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) | WD14 / Danbooru-style tagger (EVA02 Large) | Apache-2.0 |
| WD ONNX runtime path | [deepghs/wd14_tagger_with_embeddings](https://huggingface.co/deepghs/wd14_tagger_with_embeddings) | ONNX mirror of WD taggers used for inference | Apache-2.0 |
| `ml_danbooru` | [deepghs/ml-danbooru-onnx](https://huggingface.co/deepghs/ml-danbooru-onnx) | ML-Danbooru ONNX weights | MIT |
| ML-Danbooru label CSV | [deepghs/imgutils-models](https://huggingface.co/deepghs/imgutils-models) (`mldanbooru/mldanbooru_tags.csv`) | Tag vocabulary for ML-Danbooru | MIT |

**Credits:**

- **SmilingWolf** — WD tagger v3 family (training, release, `selected_tags.csv`)
- **deepghs** — ONNX packaging / mirrors and imgutils model assets
- ML-Danbooru lineage — community ML-Danbooru tagger work converted/packaged for
  ONNX by deepghs (see the Hub model card for upstream pointers)

### Style / realism helpers

| Component | Upstream | Role | License (Hub metadata) |
|---|---|---|---|
| Anime vs real classifier | [deepghs/anime_real_cls](https://huggingface.co/deepghs/anime_real_cls) | Used via `dghs-imgutils` (`anime_real_score`) | OpenRAIL |

**Credit:** deepghs — anime/real classification models and imgutils integration.

### Optional real-life adult tagging domain

Enabled only when optional deps are installed; defaults:

| Component | Upstream | Role | License (Hub metadata) |
|---|---|---|---|
| NSFW caption VLM (GGUF) | [bartowski/thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-GGUF](https://huggingface.co/bartowski/thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-GGUF) | Optional llama.cpp multimodal tagger | Apache-2.0 |
| Sex-position classifier | [porntech/sex-position](https://huggingface.co/porntech/sex-position) | Optional timm image classifier | MIT |

**Credits:**

- **bartowski** — GGUF quantization / packaging of the NSFW caption VLM
- Upstream fine-tune authors referenced on that model card (thesby / Qwen2.5-VL lineage)
- **porntech** — sex-position classifier
- Qwen / Alibaba — base Qwen2.5-VL architecture (see Apache-2.0 notices on Hub cards)

---

## 3. Third-party libraries (selected)

Runtime packaging does not vendor these licenses in-tree; installers pull them
from PyPI / npm. Notable dependencies:

| Package | Use in THR3SHR | Typical license |
|---|---|---|
| [dghs-imgutils](https://github.com/deepghs/imgutils) | Tagging helpers, anime/real validation | See package |
| [onnxruntime](https://onnxruntime.ai/) / `onnxruntime-gpu` | ONNX inference | MIT |
| [huggingface_hub](https://huggingface.co/docs/huggingface_hub) | Model download / cache | Apache-2.0 |
| [FastAPI](https://fastapi.tiangolo.com/) | Backend API | MIT |
| [React](https://react.dev/) + [Vite](https://vitejs.dev/) | Frontend SPA | MIT |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (optional) | VLM runtime | MIT |
| [timm](https://github.com/huggingface/pytorch-image-models) (optional) | Position classifier | Apache-2.0 |

Full dependency trees: `backend/requirements.txt`, `frontend/package-lock.json`.

---

## 4. Data sources (debug / eval only)

Debug and evaluation paths may fetch sample posts from public APIs such as
Danbooru and Safebooru. Those sites’ content remains under their respective
terms of service and copyright holders. THR3SHR does not claim ownership of
fetched media.

---

## 5. What you must preserve when redistributing THR3SHR

1. Keep this file (or an equivalent notice) with redistributions of the source.
2. Keep the MIT copyright notice for Dinamush software (`LICENSE`).
3. For reuse of THR3SHR **creative** materials, provide CC BY 4.0 attribution
   to Dinamush (`CREATIVE_COMMONS.md`).
4. Do **not** imply endorsement by SmilingWolf, deepghs, bartowski, porntech,
   Qwen/Alibaba, or other model authors.
5. When redistributing **model weights** yourself, follow each upstream Hub
   license (Apache-2.0, MIT, OpenRAIL, etc.) — this project intentionally avoids
   shipping those binaries.

---

## 6. Contact / identity

- GitHub: [@Dinamush](https://github.com/Dinamush)
- Hugging Face: [@Dinamus](https://huggingface.co/Dinamus)
- Project: THR3SHR
