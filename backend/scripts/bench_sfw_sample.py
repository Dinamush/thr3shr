#!/usr/bin/env python3
"""
Download SFW Safebooru samples with known tags and benchmark the owned
InferenceEngine vs imgutils baseline.

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/bench_sfw_sample.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "sample_data" / "sfw_safebooru"
META_PATH = SAMPLE_DIR / "manifest.json"
API = "https://safebooru.org/index.php"
QUERY = "1girl solo rating:safe"
LIMIT = 6


def _http_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "thr3shr-bench/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "thr3shr-bench/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as resp:
        dest.write_bytes(resp.read())


def fetch_samples() -> list[dict]:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))

    params = urllib.parse.urlencode(
        {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "limit": str(LIMIT),
            "tags": QUERY,
        }
    )
    posts = _http_json(f"{API}?{params}")
    if not isinstance(posts, list) or not posts:
        raise RuntimeError("No Safebooru posts returned")

    manifest: list[dict] = []
    for post in posts:
        file_url = post.get("sample_url") or post.get("file_url")
        if not file_url:
            continue
        post_id = post["id"]
        ext = Path(str(file_url)).suffix or ".jpg"
        dest = SAMPLE_DIR / f"{post_id}{ext}"
        if not dest.exists():
            print(f"downloading {post_id} …", flush=True)
            _download(str(file_url), dest)
        tags = str(post.get("tags") or "").split()
        manifest.append(
            {
                "id": post_id,
                "path": str(dest),
                "rating": post.get("rating"),
                "known_general": tags,
                "must_have": [t for t in ("1girl", "solo") if t in tags],
            }
        )
    META_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def top_n(scores: dict[str, float], n: int = 10) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:n]


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from imgutils.tagging import get_wd14_tags

    from app.inference_engine import InferenceEngine, reset_engine
    from app.services import _parse_wd14_raw

    manifest = fetch_samples()
    print(f"samples={len(manifest)} dir={SAMPLE_DIR}")
    for row in manifest:
        rating = str(row.get("rating") or "").lower()
        assert rating in {"safe", "s", "g", "general", ""}, rating
        print(f"  #{row['id']} must_have={row['must_have']} file={Path(row['path']).name}")

    paths = [Path(row["path"]) for row in manifest]
    reset_engine()
    engine = InferenceEngine()
    engine.warm("wd_swinv2_v3")

    t0 = time.perf_counter()
    baseline: list[dict[str, float]] = []
    for path in paths:
        raw = get_wd14_tags(
            str(path),
            model_name="SwinV2_v3",
            general_threshold=0.35,
            no_underline=False,
            drop_overlap=False,
            fmt="general",
        )
        baseline.append(_parse_wd14_raw(raw))
    baseline_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    serial = [
        engine.score_one(path, tagger_model="wd_swinv2_v3", wd_general_threshold=0.35)
        for path in paths
    ]
    serial_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    batched = engine.score_many(
        paths,
        tagger_model="wd_swinv2_v3",
        wd_general_threshold=0.35,
        batch_size=min(4, len(paths)),
    )
    batch_s = time.perf_counter() - t0

    print("\n=== Latency ===")
    print(f"imgutils serial : {baseline_s:.3f}s  ({baseline_s / len(paths) * 1000:.1f} ms/img)")
    print(f"engine serial   : {serial_s:.3f}s  ({serial_s / len(paths) * 1000:.1f} ms/img)")
    print(f"engine batch    : {batch_s:.3f}s  ({batch_s / len(paths) * 1000:.1f} ms/img)")
    if batch_s > 0:
        print(f"speedup vs imgutils: {baseline_s / batch_s:.2f}x")

    print("\n=== Known-tag recall + equivalence ===")
    equiv_failures = 0
    recall_hits = 0
    recall_total = 0
    for row, base, eng_s, eng_b in zip(manifest, baseline, serial, batched):
        must = set(row["must_have"])
        recall_total += 1
        hit_s = must <= set(eng_s)
        hit_b = must <= set(eng_b)
        if hit_s and hit_b:
            recall_hits += 1
        shared = set(base) & set(eng_s)
        max_delta = max((abs(base[t] - eng_s[t]) for t in shared), default=0.0)
        # Also require identical tag sets vs imgutils at this threshold.
        set_match = set(base) == set(eng_s) == set(eng_b)
        ok = set_match and max_delta < 1e-3
        status = "OK" if ok else "FAIL"
        if not ok:
            equiv_failures += 1
        print(
            f"  #{row['id']} equiv={status} known_hit={hit_s and hit_b} "
            f"max_delta={max_delta:.6f} top={top_n(eng_b, 5)}"
        )

    print(
        f"\nknown-tag recall@{0.35}: {recall_hits}/{recall_total} "
        f"(model may miss site tags; equivalence is the gate)"
    )
    print(f"equivalence failures: {equiv_failures}")
    return 1 if equiv_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
