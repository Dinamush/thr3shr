#!/usr/bin/env python3
"""Local calibration eval for real-life adult tagging.

Usage:
  python scripts/run_real_life_adult_eval.py --root E:\\photos\\sample --labels labels.json

labels.json format:
  {
    "photo1.jpg": ["creampie", "Asian"],
    "clip.mp4": ["blowjob"]
  }

Prints per-tag precision/recall against model proposals (primary + secondary).
Does not migrate files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.real_life_engine import get_real_life_engine  # noqa: E402
from app.real_life_taxonomy import real_life_folder_names  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Folder of sample media")
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="JSON map of relative filename -> expected tag list",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Confidence threshold (same as run setting)",
    )
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    engine = get_real_life_engine()
    print(json.dumps(engine.status(), indent=2))

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    rows = []

    for rel, expected in labels.items():
        path = args.root / rel
        if not path.exists():
            print(f"MISSING {rel}", file=sys.stderr)
            continue
        result = engine.classify_path(
            path,
            categories_root=args.root.parent / "_eval_cats",
            selected_folders=set(real_life_folder_names()),
            confidence_threshold=args.threshold,
        )
        predicted = set()
        if result.primary_tag:
            predicted.add(result.primary_tag)
        for item in result.secondary:
            predicted.add(str(item["tag"]))
        expected_set = {str(x) for x in expected}
        for tag in predicted & expected_set:
            tp[tag] += 1
        for tag in predicted - expected_set:
            fp[tag] += 1
        for tag in expected_set - predicted:
            fn[tag] += 1
        rows.append(
            {
                "file": rel,
                "expected": sorted(expected_set),
                "predicted": sorted(predicted),
                "primary": result.primary_tag,
                "reason": result.reason,
                "needs_review": result.needs_review,
            }
        )

    print("\n=== Items ===")
    for row in rows:
        print(json.dumps(row))

    all_tags = sorted(set(tp) | set(fp) | set(fn))
    print("\n=== Per-tag metrics ===")
    for tag in all_tags:
        precision = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) else None
        recall = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) else None
        print(
            f"{tag}: P={precision!s} R={recall!s} "
            f"(tp={tp[tag]} fp={fp[tag]} fn={fn[tag]})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
