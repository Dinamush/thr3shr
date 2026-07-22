"""
End-to-end verification of all tagger models against a small image folder.

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/verify_all_taggers.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8000/api"
ROOT = Path(r"/examples/probe")
CATS = ROOT / "cats"
# Keep selected tags broad enough that anime stills usually hit something.
SELECTED = ["1girl", "solo", "long_hair", "blush", "smile"]
MODELS = ["ml_danbooru", "wd_swinv2_v3", "wd_eva02_large"]


def req(method: str, path: str, body: dict | None = None, timeout: float = 120.0) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {err.code}: {detail}") from err


def wait_run(run_id: int, timeout_s: float = 1800.0) -> dict:
    start = time.time()
    last = None
    while time.time() - start < timeout_s:
        last = req("GET", f"/runs/{run_id}/status", timeout=60.0)
        status = last.get("status")
        processed = last.get("processed_images")
        total = last.get("total_images")
        print(
            f"  … run#{run_id} {status} {processed}/{total} "
            f"failed={last.get('failed_images')} "
            f"avg_ms={last.get('avg_infer_ms_per_image')}",
            flush=True,
        )
        if status in {"completed", "failed", "cancelled"}:
            return last
        time.sleep(2.0)
    raise TimeoutError(f"run {run_id} did not finish: {last}")


def main() -> int:
    if not ROOT.is_dir():
        print(f"ERROR: probe root missing: {ROOT}", file=sys.stderr)
        return 2
    CATS.mkdir(parents=True, exist_ok=True)

    providers = req("GET", "/providers", timeout=120.0)
    print(
        "providers:",
        {
            k: providers.get(k)
            for k in ("likely_device", "cuda_usable", "tagger_model", "ort_version")
        },
    )
    if not providers.get("cuda_usable"):
        print("WARNING: CUDA not usable; continuing on CPU (slow).", flush=True)

    settings = req("GET", "/settings")
    # Restore durable user paths after pytest clobber, but verify on probe set.
    settings.update(
        {
            "root_repo": str(ROOT),
            "categories_root": str(CATS),
            "confidence_threshold": 0.6,
            "wd_general_threshold": 0.35,
            "max_inference_workers": 2,
            "inference_batch_size": 1,
            "force_cpu_inference": False,
            "scan_recursive": True,
            "experimental_media_enabled": False,
            "selected_tags": SELECTED,
            "default_migrate_mode": "copy",
        }
    )

    results: dict[str, dict] = {}
    for model in MODELS:
        print(f"\n=== MODEL {model} ===", flush=True)
        settings["tagger_model"] = model
        saved = req("PUT", "/settings", settings)
        assert saved["tagger_model"] == model, saved
        reloaded = req("GET", "/settings")
        assert reloaded["tagger_model"] == model

        preview = req("GET", "/scan/preview", timeout=120.0)
        eligible = int(preview["stats"]["eligible_images"])
        print(f"  preview eligible={eligible}")
        assert eligible > 0, preview

        start = req(
            "POST",
            "/runs/start",
            {
                "root_repo": str(ROOT),
                "categories_root": str(CATS),
                "confidence_threshold": 0.6,
                "selected_folders": SELECTED,
            },
            timeout=60.0,
        )
        run_id = int(start["run_id"])
        final = wait_run(run_id)
        items = req("GET", f"/runs/{run_id}/items", timeout=60.0)
        if not isinstance(items, list):
            raise RuntimeError(f"unexpected items payload: {items}")

        empty_scores = 0
        with_primary = 0
        with_global = 0
        failed_items = 0
        sample = []
        for item in items:
            scores_dbg = None
            if item.get("needs_review") and "no tag scores" in (item.get("review_reason") or "").lower():
                empty_scores += 1
            if item.get("inference_failed") or (
                item.get("review_reason") or ""
            ).lower().startswith("inference failed"):
                failed_items += 1
            if item.get("primary_tag"):
                with_primary += 1
            tops = item.get("global_top_tags") or []
            if tops:
                with_global += 1
            if len(sample) < 3:
                sample.append(
                    {
                        "file": item.get("relative_path"),
                        "primary": item.get("primary_tag"),
                        "score": item.get("primary_score"),
                        "tops": [f"{t['tag']}={t['score']:.3f}" for t in tops[:3]],
                        "reason": item.get("review_reason"),
                    }
                )

        ok = (
            final.get("status") == "completed"
            and final.get("tagger_model") == model
            and int(final.get("processed_images") or 0) == eligible
            and empty_scores == 0
            and with_global == len(items)
            and int(final.get("failed_images") or 0) == 0
        )
        results[model] = {
            "ok": ok,
            "run_id": run_id,
            "status": final.get("status"),
            "tagger_model": final.get("tagger_model"),
            "processed": final.get("processed_images"),
            "failed": final.get("failed_images"),
            "avg_ms": final.get("avg_infer_ms_per_image"),
            "items": len(items),
            "with_primary": with_primary,
            "with_global_tops": with_global,
            "empty_scores": empty_scores,
            "failed_items": failed_items,
            "sample": sample,
        }
        print(json.dumps(results[model], indent=2), flush=True)
        if not ok:
            print(f"FAIL model={model}", file=sys.stderr)

    # Restore the large Organize library paths for the user UI (tags preserved).
    user_settings = req("GET", "/settings")
    user_settings.update(
        {
            "root_repo": r"/examples/library",
            "categories_root": r"/examples/categories",
            "selected_tags": [],
            "tagger_model": "wd_swinv2_v3",
            "confidence_threshold": 0.6,
            "wd_general_threshold": 0.35,
            "max_inference_workers": 2,
        }
    )
    req("PUT", "/settings", user_settings)
    print("\nRestored UI settings to Organize library + wd_swinv2_v3", flush=True)

    print("\n=== SUMMARY ===")
    all_ok = True
    for model, row in results.items():
        mark = "PASS" if row["ok"] else "FAIL"
        all_ok = all_ok and row["ok"]
        print(
            f"{mark} {model}: processed={row['processed']} "
            f"primary={row['with_primary']}/{row['items']} "
            f"empty={row['empty_scores']} avg_ms={row['avg_ms']}"
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
