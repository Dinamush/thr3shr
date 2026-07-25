"""Run ML vs WD tag-recall benchmark on the curated suite."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tag_recall_eval import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_SUITE_PATH,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    run_tag_recall_eval,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_tag_recall_eval")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Tagger models to compare",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download suite images before scoring",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "scripts" / "out" / "tag_recall_eval_latest.json",
    )
    args = parser.parse_args()

    result = run_tag_recall_eval(
        models=args.models,
        threshold=args.threshold,
        top_k=args.top_k,
        refresh_cache=args.refresh_cache,
        suite_path=args.suite,
        include_items=True,
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.out)

    comparison = result.get("comparison") or {}
    print(f"best_model={comparison.get('best_model')}")
    for row in comparison.get("ranking") or []:
        print(
            f"  {row['tagger_model']}: "
            f"micro@thr={row.get('micro_recall_at_threshold')} "
            f"micro@topK={row.get('micro_recall_at_top_k')}"
        )
    for pair in comparison.get("pairwise") or []:
        print(
            f"  pairwise {pair['model_a']} vs {pair['model_b']}: "
            f"a_wins={pair['a_wins']} b_wins={pair['b_wins']} ties={pair['ties']}"
        )
    for report in result.get("reports") or []:
        summary = report.get("summary") or {}
        print(
            f"[{report['tagger_model']}] "
            f"n={report.get('count_evaluated')} "
            f"opp={summary.get('opportunities')} "
            f"thr_hits={summary.get('hits_at_threshold')} "
            f"top_hits={summary.get('hits_at_top_k')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
