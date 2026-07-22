"""
Offline + optional live audit of taxonomy routing and tagger outputs.

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/audit_classify_accuracy.py
  ../.venv/Scripts/python.exe scripts/audit_classify_accuracy.py --images DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.taxonomy import choose_best_destination, reload_taxonomy  # noqa: E402

SELECTED = {
    "fertilization",
    "NTR",
    "incest",
    "nakadashi",
    "fellatio",
    "loli",
    "shota",
    "monster_girl",
    "furry",
    "Pokemon",
}

FIXTURES: list[tuple[str, dict[str, float], str | None]] = [
    ("loli_hard", {"loli": 0.92, "flat_chest": 0.99}, "loli"),
    ("fashion_fp", {"lolita_fashion": 0.99, "gothic_lolita": 0.95}, None),
    ("shota_hard", {"shota": 0.88, "1boy": 0.99}, "shota"),
    ("ntr_hard", {"netorare": 0.8}, "NTR"),
    ("incest_hard", {"incest": 0.8, "siblings": 0.99}, "incest"),
    ("siblings_fp", {"siblings": 0.99}, None),
    ("nakadashi", {"internal_cumshot": 0.9}, "nakadashi"),
    ("fert_over_creampie", {"fertilization": 0.8, "cum_in_pussy": 0.99}, "fertilization"),
    ("fellatio_impl", {"deepthroat": 0.9}, "fellatio"),
    ("monster_girl", {"monster_girl": 0.9, "horns": 0.99}, "monster_girl"),
    ("parts_fp", {"horns": 0.99, "wings": 0.98}, None),
    ("furry", {"furry_female": 0.9, "animal_ears": 0.99}, "furry"),
    ("pokemon", {"pokemon_(creature)": 0.9}, "Pokemon"),
]


def audit_fixtures() -> int:
    reload_taxonomy()
    failed = 0
    print("=== TAXONOMY FIXTURE AUDIT ===")
    for name, scores, expected in FIXTURES:
        folder, score, _ = choose_best_destination(scores, SELECTED)
        ok = folder == expected
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{mark} {name}: got={folder}({score}) expected={expected}")
    return failed


def audit_images(image_dir: Path, models: list[str], limit: int) -> int:
    from app.services import extract_scores

    paths = sorted(
        p
        for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )[:limit]
    if not paths:
        print(f"ERROR: no images under {image_dir}", file=sys.stderr)
        return 1

    print(f"\n=== MODEL IMAGE AUDIT ({len(paths)} images) ===")
    failed = 0
    for model in models:
        empty = 0
        routed = 0
        errors = 0
        latencies: list[float] = []
        samples: list[dict] = []
        for path in paths:
            t0 = time.perf_counter()
            try:
                scores = extract_scores(path, tagger_model=model, wd_general_threshold=0.35)
            except Exception as err:
                errors += 1
                print(f"  ERR {model} {path.name}: {err}")
                continue
            latencies.append((time.perf_counter() - t0) * 1000)
            if not scores:
                empty += 1
            folder, score, _ = choose_best_destination(scores, SELECTED)
            if folder is not None:
                routed += 1
            if len(samples) < 3:
                top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
                samples.append(
                    {
                        "file": path.name,
                        "folder": folder,
                        "folder_score": score,
                        "top": [f"{t}={s:.3f}" for t, s in top],
                    }
                )
        avg = sum(latencies) / len(latencies) if latencies else None
        ok = errors == 0 and empty == 0
        if not ok:
            failed += 1
        print(
            json.dumps(
                {
                    "model": model,
                    "ok": ok,
                    "images": len(paths),
                    "empty_scores": empty,
                    "errors": errors,
                    "taxonomy_routed": routed,
                    "avg_ms": round(avg, 1) if avg is not None else None,
                    "samples": samples,
                },
                indent=2,
            )
        )
    return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ml_danbooru", "wd_swinv2_v3", "wd_eva02_large"],
    )
    args = parser.parse_args()
    failed = audit_fixtures()
    if args.images:
        failed += audit_images(args.images, args.models, args.limit)
    else:
        print("\n(skip image audit: pass --images DIR for live tagger accuracy smoke)")
    print(f"\n=== DONE failed={failed} ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
