"""Measure the cost of probe_execution_providers() and verify its cache keys.

A cold probe builds a real ORT session, so it costs seconds. This checks that the
cache survives an unrelated settings save but still re-probes when force-CPU flips.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers import clear_provider_probe_cache, probe_execution_providers


def timed() -> tuple[float, dict]:
    start = time.perf_counter()
    state = probe_execution_providers()
    return (time.perf_counter() - start) * 1000, state


def main() -> None:
    from app.api import _apply_runtime_inference_env, _settings_from_db
    from app.storage import init_db

    init_db()

    clear_provider_probe_cache()
    cold, state = timed()
    print(f"cold probe          {cold:8.1f} ms  device={state.get('likely_device')}")

    warm, _ = timed()
    print(f"warm probe          {warm:8.3f} ms")

    settings = _settings_from_db()
    _apply_runtime_inference_env(settings)
    after_save, _ = timed()
    print(f"after settings save {after_save:8.3f} ms  <- must stay warm")

    os.environ["FORCE_CPU_INFERENCE"] = "true"
    forced, forced_state = timed()
    print(f"after force-cpu on  {forced:8.1f} ms  device={forced_state.get('likely_device')}")
    os.environ.pop("FORCE_CPU_INFERENCE", None)

    restored, restored_state = timed()
    print(f"after force-cpu off {restored:8.3f} ms  device={restored_state.get('likely_device')}")

    ok = after_save < 50 and restored < 50 and forced_state.get("likely_device") == "cpu"
    print("\nRESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
