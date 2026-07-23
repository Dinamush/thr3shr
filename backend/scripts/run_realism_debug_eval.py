"""Thorough remote real-life vs anime eval (multi-model) + optional local video check.

Writes JSON under scripts/out/ and prints a definitive GO/NO_GO conclusion.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from app.realism_eval import predict_real_life, run_realism_eval_multi_model  # noqa: E402
from app.schemas import AppSettings  # noqa: E402
from app.services import VIDEO_EXTENSIONS, extract_scores_with_experimental_media  # noqa: E402
from app.storage import init_db  # noqa: E402
from app.taxonomy import reload_taxonomy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_realism_debug_eval")


def _settings_from_db() -> AppSettings:
    from app.api import _settings_from_db as load

    return load()


def _probe_local_videos(settings: AppSettings, model: str, limit: int = 8) -> dict:
    """Score local Organize videos with multi-frame pooling if present."""
    roots = [
        Path(settings.root_repo).expanduser() if settings.root_repo else None,
        Path(r"/examples/library"),
    ]
    videos: list[Path] = []
    for root in roots:
        if root is None or not root.is_dir():
            continue
        for path in root.iterdir():
            if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
                videos.append(path)
        if videos:
            break
    videos = sorted(videos)[:limit]
    items = []
    for path in videos:
        try:
            scores = extract_scores_with_experimental_media(
                path,
                experimental_media_enabled=True,
                tagger_model=model,
                wd_general_threshold=settings.wd_general_threshold,
            )
            is_rl, folder, score, evidence = predict_real_life(
                scores,
                selected={"real_life", "photo"},
            )
            items.append(
                {
                    "path": str(path),
                    "predicted_real_life": is_rl,
                    "folder": folder,
                    "score": score,
                    "evidence": evidence,
                }
            )
        except Exception as err:
            items.append({"path": str(path), "error": str(err)})
    return {
        "model": model,
        "count": len(items),
        "predicted_real_life": sum(1 for i in items if i.get("predicted_real_life")),
        "items": items,
    }


def main() -> int:
    init_db()
    reload_taxonomy()
    settings = _settings_from_db()
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    count = max(8, min(count, 40))

    logger.info("starting multi-model realism eval count_per_class=%s", count)
    multi = run_realism_eval_multi_model(count_per_class=count, settings=settings)
    best = multi.get("best_model") or "wd_eva02_large"
    logger.info("probing local videos with best_model=%s", best)
    video_probe = _probe_local_videos(settings, best, limit=10)

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "multi_model": multi,
        "local_video_probe": video_probe,
    }
    out_path = out_dir / f"realism_debug_eval_{stamp}.json"
    latest = out_dir / "realism_debug_eval_latest.json"
    text = json.dumps(payload, indent=2)
    out_path.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")

    overall = multi.get("overall_conclusion") or {}
    print("=== REALISM DEBUG EVAL ===")
    print("decision:", overall.get("decision"))
    print("best_model:", overall.get("best_model"))
    print("summary:", overall.get("summary"))
    print("video:", overall.get("video_and_gif"))
    for report in multi.get("reports") or []:
        m = report.get("metrics") or {}
        print(
            f"- {report.get('tagger_model')}: {report.get('conclusion', {}).get('decision')} "
            f"P={m.get('precision')} R={m.get('recall')} F1={m.get('f1')} "
            f"animeFP={m.get('anime_false_positive_rate')} n={report.get('count_evaluated')}"
        )
    print("local_video_predicted_real_life:", video_probe.get("predicted_real_life"), "/", video_probe.get("count"))
    print("wrote", out_path)
    return 0 if overall.get("decision") == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
