from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .providers import preload_onnx_runtime_dlls, probe_execution_providers
from .storage import init_db

app = FastAPI(title="THR3SHR API", version="0.1.0")
logger = logging.getLogger("thr3shr_api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _get_provider_snapshot() -> dict[str, object]:
    return probe_execution_providers()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "request_failed method=%s path=%s elapsed_ms=%d",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "request_complete method=%s path=%s status=%d elapsed_ms=%d",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.on_event("startup")
def startup() -> None:
    _configure_logging()
    preload_onnx_runtime_dlls()
    provider_state = _get_provider_snapshot()
    logger.info("onnx_provider_state state=%s", provider_state)
    logger.info("initializing database at startup")
    init_db()
    try:
        from .api import _settings_from_db
        from .inference_engine import get_engine

        settings = _settings_from_db()
        if settings.force_cpu_inference:
            import os

            os.environ["FORCE_CPU_INFERENCE"] = "true"
        else:
            import os

            os.environ.pop("FORCE_CPU_INFERENCE", None)
        get_engine().warm(settings.tagger_model)
    except Exception:
        logger.exception("inference_engine_warm_failed")
    logger.info("startup complete")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/providers")
def health_providers() -> dict[str, object]:
    return {"status": "ok", **_get_provider_snapshot()}


@app.get("/attribution")
def attribution() -> dict[str, object]:
    """Machine-readable ownership and third-party model credits."""
    return {
        "project": "THR3SHR",
        "owner": "Dinamush",
        "owner_urls": {
            "github": "https://github.com/Dinamush",
            "huggingface": "https://huggingface.co/Dinamus",
        },
        "licenses": {
            "software": "MIT",
            "creative_works": "CC-BY-4.0",
            "software_url": "https://opensource.org/licenses/MIT",
            "creative_url": "https://creativecommons.org/licenses/by/4.0/",
            "notice": "NOTICE",
            "attribution_doc": "ATTRIBUTION.md",
            "creative_doc": "CREATIVE_COMMONS.md",
        },
        "copyright": "Copyright (c) 2026 Dinamush",
        "models_not_redistributed": True,
        "models": [
            {
                "setting": "wd_swinv2_v3",
                "credit": "SmilingWolf",
                "repo": "SmilingWolf/wd-swinv2-tagger-v3",
                "license": "Apache-2.0",
                "url": "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3",
            },
            {
                "setting": "wd_eva02_large",
                "credit": "SmilingWolf",
                "repo": "SmilingWolf/wd-eva02-large-tagger-v3",
                "license": "Apache-2.0",
                "url": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3",
            },
            {
                "setting": "wd_onnx_mirror",
                "credit": "deepghs",
                "repo": "deepghs/wd14_tagger_with_embeddings",
                "license": "Apache-2.0",
                "url": "https://huggingface.co/deepghs/wd14_tagger_with_embeddings",
            },
            {
                "setting": "ml_danbooru",
                "credit": "deepghs",
                "repo": "deepghs/ml-danbooru-onnx",
                "license": "MIT",
                "url": "https://huggingface.co/deepghs/ml-danbooru-onnx",
            },
            {
                "setting": "ml_danbooru_labels",
                "credit": "deepghs",
                "repo": "deepghs/imgutils-models",
                "license": "MIT",
                "url": "https://huggingface.co/deepghs/imgutils-models",
            },
            {
                "setting": "anime_real_cls",
                "credit": "deepghs",
                "repo": "deepghs/anime_real_cls",
                "license": "OpenRAIL",
                "url": "https://huggingface.co/deepghs/anime_real_cls",
            },
            {
                "setting": "optional_real_life_vlm",
                "credit": "bartowski (GGUF); upstream NSFW caption VLM / Qwen2.5-VL lineage",
                "repo": "bartowski/thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-GGUF",
                "license": "Apache-2.0",
                "url": "https://huggingface.co/bartowski/thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-GGUF",
                "optional": True,
            },
            {
                "setting": "optional_sex_position",
                "credit": "porntech",
                "repo": "porntech/sex-position",
                "license": "MIT",
                "url": "https://huggingface.co/porntech/sex-position",
                "optional": True,
            },
        ],
        "repositories": {
            "github": "https://github.com/Dinamush/thr3shr",
            "huggingface": "https://huggingface.co/Dinamus/thr3shr",
        },
    }


app.include_router(router)
