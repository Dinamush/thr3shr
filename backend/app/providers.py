from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)
_DLL_DIRS_CONFIGURED = False
_DLL_DIRS_CACHE: list[str] = []
_ORT_DLLS_PRELOADED = False


def _nvidia_bin_dirs() -> list[Path]:
    try:
        import onnxruntime as ort

        site_packages = Path(ort.__file__).resolve().parents[1]
    except Exception:
        site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.is_dir():
        return []
    return sorted(p for p in nvidia_root.glob("*/bin") if p.is_dir())


def ensure_nvidia_dll_search_path() -> list[str]:
    """
    Make pip-bundled CUDA/cuDNN DLLs discoverable on Windows.

    ort.preload_dlls() alone is not always enough for cudnn_engines_* libs;
    without PATH / add_dll_directory, Conv ops fall back to CPU at runtime.
    """
    global _DLL_DIRS_CONFIGURED, _DLL_DIRS_CACHE
    if _DLL_DIRS_CONFIGURED:
        return list(_DLL_DIRS_CACHE)

    bin_dirs = [str(p.resolve()) for p in _nvidia_bin_dirs()]
    if not bin_dirs:
        _DLL_DIRS_CONFIGURED = True
        _DLL_DIRS_CACHE = []
        return []

    path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    # Prepend missing bins so the loader finds them first.
    for d in reversed(bin_dirs):
        if d not in path_parts:
            path_parts.insert(0, d)
    os.environ["PATH"] = os.pathsep.join(path_parts)

    if hasattr(os, "add_dll_directory"):
        for d in bin_dirs:
            try:
                os.add_dll_directory(d)
            except (FileNotFoundError, OSError) as err:
                logger.warning("add_dll_directory failed for %s: %s", d, err)

    logger.info("nvidia_dll_dirs configured count=%d dirs=%s", len(bin_dirs), bin_dirs)
    _DLL_DIRS_CACHE = bin_dirs
    _DLL_DIRS_CONFIGURED = True
    return list(_DLL_DIRS_CACHE)


def preload_onnx_runtime_dlls() -> None:
    """Load pip-bundled CUDA/cuDNN DLLs before creating ORT sessions."""
    global _ORT_DLLS_PRELOADED
    ensure_nvidia_dll_search_path()
    if _ORT_DLLS_PRELOADED:
        return
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
            logger.info("onnxruntime preload_dlls completed")
        _ORT_DLLS_PRELOADED = True
    except Exception:
        logger.exception("onnxruntime preload_dlls failed")


def _force_cpu() -> bool:
    return os.getenv("FORCE_CPU_INFERENCE", "").strip().lower() in {"1", "true", "yes", "on"}


def _find_probe_model() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    matches = sorted(hub.glob("models--deepghs--ml-danbooru-onnx/**/*.onnx"))
    return matches[0] if matches else None


@lru_cache(maxsize=1)
def probe_execution_providers() -> dict[str, object]:
    """
    Probe *active* providers by creating a real ORT session and running it.

    Listing CUDAExecutionProvider is not enough — Windows often lists it even when
    CUDA/cuDNN DLLs fail during Conv execution and ORT silently falls back to CPU.

    Note: CUDA usability is independent of which tagger model is selected in settings
    (ML-Danbooru vs WD14 variants); both share the same ORT runtime path.
    """
    force_cpu = _force_cpu()
    result: dict[str, object] = {
        "available_providers": [],
        "active_providers": [],
        "cuda_available": False,
        "cuda_usable": False,
        "cpu_available": True,
        "forced_cpu": force_cpu,
        "likely_device": "cpu",
        "provider_error": None,
        "ort_version": None,
        "nvidia_dll_dirs": [],
    }
    try:
        import numpy as np
        import onnxruntime as ort

        result["nvidia_dll_dirs"] = ensure_nvidia_dll_search_path()
        preload_onnx_runtime_dlls()
        result["ort_version"] = ort.__version__
        available = list(ort.get_available_providers())
        result["available_providers"] = available
        result["cuda_available"] = "CUDAExecutionProvider" in available
        result["cpu_available"] = "CPUExecutionProvider" in available

        if force_cpu:
            result["likely_device"] = "cpu"
            result["active_providers"] = ["CPUExecutionProvider"]
            return result

        model_path = _find_probe_model()
        if model_path is None:
            result["likely_device"] = "gpu" if result["cuda_available"] else "cpu"
            result["provider_error"] = (
                "No cached ML-Danbooru ONNX model found to verify CUDA; "
                "run one tagging pass to download it."
            )
            return result

        requested = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if result["cuda_available"]
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(model_path), providers=requested)
        active = list(session.get_providers())
        result["active_providers"] = active

        # Execute a real forward pass — session creation can claim CUDA while Conv fails.
        if "CUDAExecutionProvider" in active:
            inputs = session.get_inputs()[0]
            shape = []
            for dim in inputs.shape:
                if isinstance(dim, int) and dim > 0:
                    shape.append(dim)
                else:
                    shape.append(1 if len(shape) == 0 else 448)
            # Model expects NCHW float; use 1x3x448x448 when dynamic.
            if len(shape) != 4:
                shape = [1, 3, 448, 448]
            feed = {inputs.name: np.zeros(shape, dtype=np.float32)}
            try:
                session.run(None, feed)
                result["cuda_usable"] = True
            except Exception as run_err:
                result["cuda_usable"] = False
                result["provider_error"] = (
                    "CUDAExecutionProvider loaded but inference failed "
                    f"(cuDNN/runtime). Falling back to CPU. Detail: {run_err}"
                )
                logger.warning("provider_probe_cuda_run_failed error=%s", run_err)
        else:
            result["cuda_usable"] = False
            if result["cuda_available"]:
                result["provider_error"] = (
                    "CUDAExecutionProvider is listed but failed to initialize; "
                    "sessions are using CPU. Ensure onnxruntime-gpu[cuda,cudnn] "
                    "DLLs are installed and restart the API."
                )

        result["likely_device"] = "gpu" if result["cuda_usable"] else "cpu"
        if result["cuda_usable"]:
            logger.info("provider_probe_ok active=%s model=%s", active, model_path)
        return result
    except Exception as err:
        logger.exception("provider_probe_failed")
        result["provider_error"] = str(err)
        result["likely_device"] = "cpu"
        return result


def clear_provider_probe_cache() -> None:
    probe_execution_providers.cache_clear()
