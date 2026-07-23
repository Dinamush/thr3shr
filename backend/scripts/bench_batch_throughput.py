#!/usr/bin/env python3
"""Compare WD single vs batched ORT throughput on local images.

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/bench_batch_throughput.py
  ../.venv/Scripts/python.exe scripts/bench_batch_throughput.py --root "E:\\path" --n 24
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_images(root: Path, limit: int) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
            if len(paths) >= limit:
                break
    return paths


def _bench_batches(images: list[Path], batch_size: int, tagger_model: str, thr: float) -> dict:
    from app.services import extract_scores, extract_scores_batch

    # Warmup
    extract_scores(images[0], tagger_model=tagger_model, wd_general_threshold=thr)

    start = time.perf_counter()
    if batch_size <= 1:
        for image in images:
            extract_scores(image, tagger_model=tagger_model, wd_general_threshold=thr)
    else:
        for idx in range(0, len(images), batch_size):
            chunk = images[idx : idx + batch_size]
            extract_scores_batch(
                chunk,
                tagger_model=tagger_model,
                wd_general_threshold=thr,
            )
    elapsed = time.perf_counter() - start
    n = len(images)
    return {
        "batch_size": batch_size,
        "images": n,
        "elapsed_s": elapsed,
        "ms_per_image": (elapsed / n) * 1000.0,
        "images_per_min": (n / elapsed) * 60.0 if elapsed else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=r"/examples/inbox",
    )
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--model", default="wd_swinv2_v3")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--batches", default="1,4,8")
    args = parser.parse_args()

    root = Path(args.root)
    images = _collect_images(root, args.n)
    if len(images) < 4:
        raise SystemExit(f"Need at least 4 images under {root}, found {len(images)}")

    print(f"root={root}")
    print(f"model={args.model} n={len(images)} thr={args.threshold}")
    print(f"sample={[p.name for p in images[:3]]}")

    results = []
    for raw in args.batches.split(","):
        batch_size = int(raw.strip())
        print(f"\n=== batch_size={batch_size} ===", flush=True)
        row = _bench_batches(images, batch_size, args.model, args.threshold)
        results.append(row)
        print(
            f"  elapsed={row['elapsed_s']:.2f}s  "
            f"ms/img={row['ms_per_image']:.0f}  "
            f"img/min={row['images_per_min']:.1f}",
            flush=True,
        )

    baseline = next((r for r in results if r["batch_size"] == 1), None)
    if baseline:
        print("\n=== vs batch=1 ===")
        for row in results:
            if row["batch_size"] == 1:
                continue
            speedup = baseline["ms_per_image"] / row["ms_per_image"]
            print(f"  batch={row['batch_size']}: {speedup:.2f}x faster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
