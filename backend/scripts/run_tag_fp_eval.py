"""False-positive rates on the curated suite vs preferred (selected) taxonomy tags."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api import _settings_from_db  # noqa: E402
from app.tag_fp_eval import run_tag_fp_eval  # noqa: E402
from app.tag_recall_eval import DEFAULT_SUITE_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_tag_fp_eval")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ml_danbooru", "wd_swinv2_v3"],
    )
    parser.add_argument(
        "--tag-threshold",
        type=float,
        default=None,
        help="Score threshold for tag FPs (default: settings confidence_threshold)",
    )
    parser.add_argument(
        "--route-threshold",
        type=float,
        default=None,
        help="Threshold for folder routing (default: same as tag-threshold)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.85,
        help="Only probe evidence tags with weight >= this",
    )
    parser.add_argument(
        "--also-wd-threshold",
        action="store_true",
        help="Also evaluate at wd_general_threshold (0.35)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "scripts" / "out" / "tag_fp_eval_latest.json",
    )
    args = parser.parse_args()

    settings = _settings_from_db()
    tag_thr = float(
        args.tag_threshold
        if args.tag_threshold is not None
        else settings.confidence_threshold
    )
    route_thr = float(
        args.route_threshold if args.route_threshold is not None else tag_thr
    )

    result = run_tag_fp_eval(
        selected_tags=list(settings.selected_tags or []),
        models=args.models,
        tag_threshold=tag_thr,
        route_threshold=route_thr,
        min_weight=args.min_weight,
        also_wd_threshold=bool(args.also_wd_threshold),
        wd_general_threshold=float(settings.wd_general_threshold),
        suite_path=args.suite,
        include_items=False,
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.out)

    for rep in result.get("reports") or []:
        tf = rep["tag_fp"]
        rf = rep["folder_route_fp"]
        print(
            f"\n[{rep['tagger_model']}] tag_thr={rep['tag_threshold']} "
            f"route_thr={rep['route_threshold']}"
        )
        print(
            f"  tag micro FPR: {tf['micro_fpr']:.3f} "
            f"({tf['false_positives']}/{tf['negatives']}) "
            f"macro={tf['macro_fpr']:.3f}  probe_tags={tf['probe_tag_count']}"
        )
        print(
            f"  folder route FPR: {rf['fpr']:.3f} "
            f"({rf['false_positives']}/{rf['evaluated']}) "
            f"by={rf['by_folder']}"
        )
        print("  top FP tags:")
        for row in (rep.get("top_false_positive_tags") or [])[:15]:
            print(
                f"    {row['tag']:28} fp={row['false_positives']:3}/{row['negatives']:<3} "
                f"fpr={row['fpr']:.2f}  -> {(row.get('folders') or [])[:2]}"
            )

    alt = result.get("at_wd_general_threshold")
    if alt:
        print("\n--- also at wd_general_threshold ---")
        for rep in alt.get("reports") or []:
            tf = rep["tag_fp"]
            rf = rep["folder_route_fp"]
            print(
                f"[{rep['tagger_model']}] micro_fpr={tf['micro_fpr']:.3f} "
                f"route_fpr={rf['fpr']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
