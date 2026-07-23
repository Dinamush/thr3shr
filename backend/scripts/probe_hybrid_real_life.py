"""Autonomous probe: hybrid WD + style detector on a small photo/anime set."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe_hybrid")


def main() -> int:
    from app.api import _apply_hybrid_real_life_filter, _ImageInferenceResult
    from app.realism_sources import collect_realism_samples, download_realism_sample
    from app.services import extract_scores
    from app.style_detectors import blend_real_life_scores, detect_production_style, wd_realism_score

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    samples = collect_realism_samples(count_per_class=n)
    rows = []
    y_true = []
    y_pred = []
    for sample in samples:
        path = download_realism_sample(sample)
        if path is None:
            continue
        scores = extract_scores(path, tagger_model="wd_swinv2_v3", wd_general_threshold=0.35)
        style = detect_production_style(path, uncertain_threshold=None)
        accepted, hybrid, detail = blend_real_life_scores(scores, style)
        base = _ImageInferenceResult(
            image_path=path,
            scores=scores,
            primary_tag=None,
            primary_score=None,
            secondary=[],
            needs_review=False,
            reason=None,
            inference_failed=False,
        )
        out = _apply_hybrid_real_life_filter(base)
        truth = "photo" if sample.label == "photo" else "anime"
        pred = "photo" if out.primary_tag == "real_life" else "anime"
        y_true.append(truth)
        y_pred.append(pred)
        rows.append(
            {
                "label": sample.label,
                "pred": pred,
                "accepted": accepted,
                "hybrid": hybrid,
                "wd": wd_realism_score(scores),
                "style_real": detail["style_real"],
                "style_anime": detail["style_anime"],
                "path": str(path),
            }
        )
        logger.info(
            "%s -> %s hybrid=%.3f wd=%.3f style_r=%.3f style_a=%.3f",
            sample.label,
            pred,
            hybrid,
            detail["wd_realism"],
            detail["style_real"],
            detail["style_anime"],
        )

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "photo" and p == "photo")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "photo" and p == "photo")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != "photo" and p != "photo")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "photo" and p != "photo")
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    summary = {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "anime_fp_rate": fp / max(1, fp + tn),
        "rows": rows,
    }
    out_path = ROOT / "scripts" / "out" / "hybrid_probe_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "SUMMARY n=%d P=%s R=%s anime_fp=%.3f -> %s",
        summary["n"],
        f"{precision:.3f}" if precision is not None else "n/a",
        f"{recall:.3f}" if recall is not None else "n/a",
        summary["anime_fp_rate"],
        out_path,
    )
    # GO if recall decent and anime FP low
    go = (
        summary["n"] >= 6
        and (recall or 0) >= 0.75
        and summary["anime_fp_rate"] <= 0.15
        and (precision or 0) >= 0.8
    )
    logger.info("VERDICT %s", "GO" if go else "NO_GO")
    return 0 if go else 2


if __name__ == "__main__":
    raise SystemExit(main())
