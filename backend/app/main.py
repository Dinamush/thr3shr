from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .storage import init_db

app = FastAPI(title="Image Classifier Workflow API", version="0.1.0")
logger = logging.getLogger("image_classifier_api")

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
    force_cpu = os.getenv("FORCE_CPU_INFERENCE", "").strip().lower() in {"1", "true", "yes", "on"}
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        cuda_available = "CUDAExecutionProvider" in providers
        return {
            "available_providers": providers,
            "cuda_available": cuda_available,
            "cpu_available": "CPUExecutionProvider" in providers,
            "forced_cpu": force_cpu,
            "likely_device": "cpu" if force_cpu else ("gpu" if cuda_available else "cpu"),
        }
    except Exception as err:
        return {
            "available_providers": [],
            "cuda_available": False,
            "cpu_available": True,
            "forced_cpu": force_cpu,
            "likely_device": "cpu",
            "error": str(err),
        }


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
    provider_state = _get_provider_snapshot()
    logger.info("onnx_provider_state state=%s", provider_state)
    logger.info("initializing database at startup")
    init_db()
    logger.info("startup complete")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/providers")
def health_providers() -> dict[str, object]:
    return {"status": "ok", **_get_provider_snapshot()}


app.include_router(router)
