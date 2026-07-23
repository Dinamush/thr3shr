"""CLI: compare style detectors on the realism corpus (debug)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.style_eval import run_style_detector_eval  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_style_debug_eval")


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    from app.api import _settings_from_db

    settings = _settings_from_db()
    logger.info("starting style detector compare count_per_class=%s", count)
    result = run_style_detector_eval(count_per_class=count, settings=settings)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latest = out_dir / "style_debug_eval_latest.json"
    stamped = out_dir / f"style_debug_eval_{stamp}.json"
    text = json.dumps(result, indent=2)
    latest.write_text(text, encoding="utf-8")
    stamped.write_text(text, encoding="utf-8")
    oc = result.get("overall_conclusion") or {}
    print(
        f"decision={oc.get('decision')} best={oc.get('best_detector')} "
        f"wrote={latest}"
    )
    for report in result.get("reports") or []:
        t = (report.get("conclusion") or {}).get("typical_metrics") or {}
        print(
            f"  {report.get('detector_id')}: {report.get('conclusion', {}).get('decision')} "
            f"P={t.get('precision')} R={t.get('recall')} F1={t.get('f1')} "
            f"animeFP={t.get('anime_false_positive_rate')} "
            f"uncertain={report.get('conclusion', {}).get('uncertain_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
