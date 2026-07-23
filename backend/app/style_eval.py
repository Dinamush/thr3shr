"""Debug eval: compare style detectors on the realism photo vs anime corpus."""

from __future__ import annotations

import logging

from .realism_eval import _metrics
from .realism_sources import collect_realism_samples, download_realism_sample
from .schemas import AppSettings
from .style_detectors import (
    BUCKET_PHOTO,
    BUCKET_UNCERTAIN,
    DEFAULT_STYLE_DETECTOR_IDS,
    build_detector,
    list_style_detectors,
)
from .taxonomy import reload_taxonomy

logger = logging.getLogger(__name__)


def _selected_destinations(settings: AppSettings) -> set[str]:
    selected = {"real_life", "photo"}
    for name in settings.selected_tags:
        text = str(name).strip()
        if text:
            selected.add(text)
    if len(selected) < 3:
        selected.update({"SFW", "scenery", "1girl"})
    return selected


def _conclusion_for_items(items: list[dict], detector_id: str) -> dict:
    typical_items = [i for i in items if i["label"] in {"photo", "anime"}]
    # Binary metrics: uncertain counts as not-photo (photo miss / anime OK).
    typical_metrics = _metrics(
        [i["bucket"] for i in typical_items],
        [
            BUCKET_PHOTO if i["predicted_bucket"] == BUCKET_PHOTO else "anime"
            for i in typical_items
        ],
    )
    overall_metrics = _metrics(
        [i["bucket"] for i in items],
        [
            BUCKET_PHOTO if i["predicted_bucket"] == BUCKET_PHOTO else "anime"
            for i in items
        ],
    )
    uncertain_n = sum(1 for i in items if i["predicted_bucket"] == BUCKET_UNCERTAIN)
    edge_items = [i for i in items if str(i["label"]).startswith("edge_")]
    edge_fp = sum(
        1
        for i in edge_items
        if i["bucket"] == "anime" and i["predicted_bucket"] == BUCKET_PHOTO
    )
    go_typical = (
        len(typical_items) >= 10
        and typical_metrics["precision"] is not None
        and typical_metrics["recall"] is not None
        and typical_metrics["precision"] >= 0.95
        and typical_metrics["recall"] >= 0.90
        and typical_metrics["anime_false_positive_rate"] <= 0.05
    )
    return {
        "decision": "GO" if go_typical else "NO_GO",
        "decision_scope": "typical_photo_vs_anime",
        "detector_id": detector_id,
        "summary": (
            "GO: detector separates typical people photos from typical anime on this corpus."
            if go_typical
            else "NO_GO: inspect misses / uncertain band before trusting this detector."
        ),
        "gates": {
            "typical_precision_ge_0.95": bool(
                typical_metrics["precision"] and typical_metrics["precision"] >= 0.95
            ),
            "typical_recall_ge_0.90": bool(
                typical_metrics["recall"] and typical_metrics["recall"] >= 0.90
            ),
            "typical_anime_fp_le_0.05": typical_metrics["anime_false_positive_rate"]
            <= 0.05,
            "min_typical_samples_10": len(typical_items) >= 10,
        },
        "typical_metrics": typical_metrics,
        "overall_metrics_including_edges": overall_metrics,
        "uncertain_count": uncertain_n,
        "edge_quarantine_count": edge_fp,
        "edge_sample_count": len(edge_items),
    }


def run_style_detector_eval(
    *,
    count_per_class: int,
    settings: AppSettings,
    detector_ids: list[str] | None = None,
    tagger_model: str | None = None,
    uncertain_threshold: float = 0.85,
) -> dict:
    reload_taxonomy()
    n = max(5, min(int(count_per_class), 40))
    detectors = list(detector_ids or DEFAULT_STYLE_DETECTOR_IDS)
    unknown = [d for d in detectors if d not in {x["id"] for x in list_style_detectors()}]
    if unknown:
        raise ValueError(f"Unknown style detectors: {', '.join(unknown)}")

    model = tagger_model or settings.tagger_model
    selected = _selected_destinations(settings)
    samples = collect_realism_samples(count_per_class=n)

    paths: dict[str, object] = {}
    errors: list[str] = []
    for sample in samples:
        try:
            paths[sample.sample_id] = (sample, download_realism_sample(sample))
        except Exception as err:
            logger.warning(
                "style_eval_download_failed id=%s err=%s", sample.sample_id, err
            )
            errors.append(f"{sample.sample_id}: download failed ({err})")

    reports: list[dict] = []
    for detector_id in detectors:
        logger.info("style_eval_detector_start detector=%s n=%s", detector_id, n)
        predict = build_detector(
            detector_id,
            tagger_model=model,
            wd_general_threshold=float(settings.wd_general_threshold),
            selected=selected,
            uncertain_threshold=float(uncertain_threshold),
        )
        items: list[dict] = []
        det_errors: list[str] = []
        for sample_id, (sample, path) in paths.items():
            try:
                pred = predict(path)
            except Exception as err:
                logger.exception(
                    "style_eval_infer_failed detector=%s id=%s", detector_id, sample_id
                )
                det_errors.append(f"{sample_id}: {detector_id} failed ({err})")
                continue
            # For correctness on binary buckets, uncertain is never "correct".
            correct = (
                pred.bucket == sample.bucket
                if pred.bucket != BUCKET_UNCERTAIN
                else False
            )
            items.append(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "bucket": sample.bucket,
                    "source": sample.source,
                    "query": sample.query,
                    "title": sample.title,
                    "file_name": path.name,
                    "predicted_label": pred.label,
                    "predicted_bucket": pred.bucket,
                    "correct": correct,
                    "confidence": pred.confidence,
                    "scores": pred.scores,
                    "method": pred.method,
                    "detail": pred.detail,
                }
            )

        conclusion = _conclusion_for_items(items, detector_id)
        by_label: dict[str, dict] = {}
        for label in sorted({i["label"] for i in items}):
            subset = [i for i in items if i["label"] == label]
            by_label[label] = {
                "count": len(subset),
                "accuracy": sum(1 for i in subset if i["correct"]) / max(1, len(subset)),
                "uncertain": sum(
                    1 for i in subset if i["predicted_bucket"] == BUCKET_UNCERTAIN
                ),
            }

        report = {
            "detector_id": detector_id,
            "method": items[0]["method"] if items else detector_id,
            "tagger_model": model if detector_id == "wd_taxonomy" else None,
            "count_evaluated": len(items),
            "metrics": conclusion["overall_metrics_including_edges"],
            "by_label": by_label,
            "conclusion": conclusion,
            "items": items,
            "errors": det_errors,
        }
        reports.append(report)
        logger.info(
            "style_eval_detector_done detector=%s decision=%s f1=%s",
            detector_id,
            conclusion["decision"],
            (conclusion.get("typical_metrics") or {}).get("f1"),
        )

    def _rank_key(r: dict) -> tuple:
        c = r.get("conclusion") or {}
        typical = c.get("typical_metrics") or {}
        go = 1 if c.get("decision") == "GO" else 0
        # Prefer dedicated imgutils over WD on ties; prefer hard labels over band.
        prefer = {
            "imgutils_caformer": 3,
            "imgutils_mobilenet": 2,
            "imgutils_caformer_band": 1,
            "wd_taxonomy": 0,
        }.get(r.get("detector_id"), 0)
        return (
            go,
            float(typical.get("f1") or 0.0),
            float(typical.get("recall") or 0.0),
            prefer,
        )

    ranked = sorted(reports, key=_rank_key, reverse=True)
    best = ranked[0] if ranked else None
    overall_go = bool(best and best["conclusion"]["decision"] == "GO")

    return {
        "count_per_class_requested": n,
        "count_photos_fetched": sum(1 for s in samples if s.bucket == "photo"),
        "count_anime_fetched": sum(1 for s in samples if s.bucket == "anime"),
        "count_paths": len(paths),
        "tagger_model": model,
        "uncertain_threshold": float(uncertain_threshold),
        "detectors": detectors,
        "available_detectors": list_style_detectors(),
        "reports": reports,
        "best_detector": best["detector_id"] if best else None,
        "overall_conclusion": {
            "decision": "GO" if overall_go else "NO_GO",
            "best_detector": best["detector_id"] if best else None,
            "best_typical_metrics": (best.get("conclusion") or {}).get("typical_metrics")
            if best
            else None,
            "summary": (
                f"GO: {best['detector_id']} clears typical photo-vs-anime gates."
                if overall_go and best
                else "NO_GO: no style detector met typical gates on this corpus."
            ),
            "note": (
                "Debug compare + optional production gate. Enable "
                "'Experimental: real vs anime style gate' in Classifier settings "
                "to route real photos to real_life (uncertain → needs review)."
            ),
        },
        "errors": errors,
        "photo_source": "randomuser portraits (+ commons/local fallback)",
        "anime_source": "safebooru (typical + realistic/photorealistic/3d edges)",
    }
