from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .api import router
from .providers import preload_onnx_runtime_dlls, probe_execution_providers
from .storage import fail_interrupted_runs, init_db

logger = logging.getLogger("thr3shr_api")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    startup()
    yield


# The API reads and moves local files without authentication, so only the
# local UI may call it. Any web page could otherwise read settings and paths
# (CORS), send a simple cross-site POST that starts a move-migration (Origin and
# Sec-Fetch-Site), or reach it through DNS rebinding (Host). Override with
# comma-separated lists.
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost")


def _env_list(name: str, default: tuple[str, ...]) -> list[str]:
    values = [part.strip() for part in os.getenv(name, "").split(",") if part.strip()]
    return values or list(default)


UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

app = FastAPI(title="THR3SHR API", version=__version__, lifespan=lifespan)
cors_origins = _env_list("THR3SHR_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)


@app.middleware("http")
async def reject_cross_site_writes(request: Request, call_next):
    if request.method in UNSAFE_METHODS and "*" not in cors_origins:
        origin = request.headers.get("origin")
        if origin is not None:
            cross_site = origin not in cors_origins
        else:
            cross_site = request.headers.get("sec-fetch-site") == "cross-site"
        if cross_site:
            return JSONResponse(status_code=403, content={"detail": "Cross-site request refused"})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_env_list("THR3SHR_ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS),
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


def startup() -> None:
    _configure_logging()
    logger.info("thr3shr_version version=%s", __version__)
    logger.info("initializing database at startup")
    init_db()
    interrupted = fail_interrupted_runs()
    if interrupted:
        logger.warning("interrupted_runs_failed count=%d", interrupted)
    from .api import _settings_from_db

    settings = _settings_from_db()
    if settings.force_cpu_inference:
        os.environ["FORCE_CPU_INFERENCE"] = "true"
    else:
        os.environ.pop("FORCE_CPU_INFERENCE", None)
    _start_engine_warmup(settings.tagger_model)
    logger.info("startup complete")


def _warm_engine(tagger_model: str) -> None:
    from .inference_engine import get_engine, trim_process_heap

    try:
        preload_onnx_runtime_dlls()
        logger.info("onnx_provider_state state=%s", _get_provider_snapshot())
        get_engine().warm(tagger_model)
    except Exception:
        logger.exception("inference_engine_warm_failed")
    finally:
        # Model loading leaves hundreds of MB of freed parse buffers resident.
        trim_process_heap()


def _start_engine_warmup(tagger_model: str) -> None:
    """Probe providers and load the tagger in the background.

    Both take seconds; the API answers meanwhile, and a run or probe that needs
    them first waits on the engine and probe locks instead of loading twice.
    """
    threading.Thread(
        target=_warm_engine, args=(tagger_model,), name="engine-warmup", daemon=True
    ).start()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/health/providers")
def health_providers() -> dict[str, object]:
    return {"status": "ok", **_get_provider_snapshot()}


@app.get("/attribution")
def attribution() -> dict[str, object]:
    """Machine-readable ownership and third-party model credits."""
    return {
        "project": "THR3SHR",
        "version": __version__,
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
