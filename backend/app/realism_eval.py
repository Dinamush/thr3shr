"""Real-life vs anime debug evaluation using production taxonomy routing."""

from __future__ import annotations

import logging
from pathlib import Path

from .realism_sources import (
    RealismSample,
    collect_realism_samples,
    download_realism_sample,
)
from .schemas import AppSettings
from .services import extract_scores
from .taxonomy import choose_best_destination, get_taxonomy, reload_taxonomy

logger = logging.getLogger(__name__)

REALISM_EVIDENCE_TAGS = (
    "realistic",
    "photorealistic",
    "photo_(medium)",
    "3d",
    "cellphone_photo",
    "photo_background",
)


def _tag_score(scores: dict[str, float], tag: str) -> float:
    if tag in scores:
        return float(scores[tag])
    wanted = tag.lower().replace(" ", "_")
    best = 0.0
    for key, value in scores.items():
        if str(key).lower().replace(" ", "_") == wanted:
            best = max(best, float(value))
    return best


def predict_real_life(
    scores: dict[str, float],
    *,
    selected: set[str] | None = None,
) -> tuple[bool, str | None, float | None, dict[str, float]]:
    """Return (is_real_life, folder, score, evidence_scores) via taxonomy."""
    cfg = get_taxonomy()
    dest = selected or {"real_life", "photo"}
    folder, score, _secondary = choose_best_destination(scores, dest, taxonomy=cfg)
    evidence = {tag: _tag_score(scores, tag) for tag in REALISM_EVIDENCE_TAGS}
    is_rl = folder is not None and str(folder).lower().replace(" ", "_") in {
        "real_life",
        "photo",
    }
    return is_rl, folder, score, evidence


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Binary metrics treating photo as positive class."""
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        pos_t = truth == "photo"
        pos_p = pred == "photo"
        if pos_t and pos_p:
            tp += 1
        elif not pos_t and pos_p:
            fp += 1
        elif not pos_t and not pos_p:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "anime_false_positive_rate": fp / max(1, fp + tn),
        "photo_miss_rate": fn / max(1, fn + tp),
    }


def run_realism_eval(
    *,
    count_per_class: int,
    settings: AppSettings,
    tagger_model: str | None = None,
    include_edge_breakdown: bool = True,
) -> dict:
    reload_taxonomy()
    model = tagger_model or settings.tagger_model
    n = max(5, min(int(count_per_class), 40))
    samples = collect_realism_samples(count_per_class=n)
    errors: list[str] = []
    items: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    selected = {"real_life", "photo"}
    # Compete with content buckets so accidental routing is visible.
    for name in settings.selected_tags:
        text = str(name).strip()
        if text:
            selected.add(text)
    if len(selected) < 3:
        # Generic competitors only — never seed personal destination prefs.
        selected.update({"SFW", "scenery", "1girl"})

    for sample in samples:
        try:
            path = download_realism_sample(sample)
        except Exception as err:
            logger.warning(
                "realism_download_failed id=%s err=%s", sample.sample_id, err
            )
            errors.append(f"{sample.sample_id}: download failed ({err})")
            continue
        try:
            scores = extract_scores(
                path,
                tagger_model=model,
                wd_general_threshold=float(settings.wd_general_threshold),
            )
        except Exception as err:
            logger.exception("realism_infer_failed path=%s", path)
            errors.append(f"{sample.sample_id}: inference failed ({err})")
            continue

        is_rl, folder, folder_score, evidence = predict_real_life(
            scores, selected=selected
        )
        pred_bucket = "photo" if is_rl else "anime"
        y_true.append(sample.bucket)
        y_pred.append(pred_bucket)
        top_global = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        items.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "bucket": sample.bucket,
                "source": sample.source,
                "query": sample.query,
                "title": sample.title,
                "file_name": path.name,
                "predicted_bucket": pred_bucket,
                "correct": pred_bucket == sample.bucket,
                "primary_folder": folder,
                "primary_score": folder_score,
                "evidence_scores": evidence,
                "global_top_tags": [
                    {"tag": t, "score": float(s)} for t, s in top_global
                ],
            }
        )

    metrics = _metrics(y_true, y_pred)
    by_label: dict[str, dict] = {}
    if include_edge_breakdown:
        for label in sorted({i["label"] for i in items}):
            subset = [i for i in items if i["label"] == label]
            yt = [i["bucket"] for i in subset]
            yp = [i["predicted_bucket"] for i in subset]
            by_label[label] = {
                "count": len(subset),
                "accuracy": sum(1 for i in subset if i["correct"]) / max(1, len(subset)),
                **_metrics(yt, yp),
            }

    # Primary product question: people photos vs *typical* anime (not tagged
    # realistic/photorealistic). Edge photoreal anime is expected quarantine.
    typical_items = [i for i in items if i["label"] in {"photo", "anime"}]
    typical_metrics = _metrics(
        [i["bucket"] for i in typical_items],
        [i["predicted_bucket"] for i in typical_items],
    )
    edge_items = [i for i in items if str(i["label"]).startswith("edge_")]
    edge_fp = sum(
        1 for i in edge_items if i["bucket"] == "anime" and i["predicted_bucket"] == "photo"
    )
    go_typical = (
        len(typical_items) >= 10
        and typical_metrics["precision"] is not None
        and typical_metrics["recall"] is not None
        and typical_metrics["precision"] >= 0.95
        and typical_metrics["recall"] >= 0.90
        and typical_metrics["anime_false_positive_rate"] <= 0.05
    )
    conclusion = {
        "decision": "GO" if go_typical else "NO_GO",
        "decision_scope": "typical_photo_vs_anime",
        "summary": (
            "GO for accidental people-photo vs typical anime: production real_life "
            "routing is reliable on this corpus. Photoreal/3d *anime* edge cases may "
            "also quarantine into real_life (by design for ambiguous media)."
            if go_typical
            else "NO_GO for typical photo vs anime on this sample; inspect misses "
            "before trusting automatic routing."
        ),
        "recommended_model": model,
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
        "edge_quarantine_count": edge_fp,
        "edge_sample_count": len(edge_items),
        "overall_metrics_including_edges": metrics,
        "video_note": (
            "GIF/video already share this path when experimental media sampling is on: "
            "pooled frame scores feed real_life evidence. Prefer wd_eva02_large. "
            "Anime videos usually keep content-folder winners; semi-real clips can "
            "still quarantine to real_life — keep review on."
        ),
    }

    return {
        "count_per_class_requested": n,
        "count_photos_fetched": sum(1 for s in samples if s.bucket == "photo"),
        "count_anime_fetched": sum(1 for s in samples if s.bucket == "anime"),
        "count_evaluated": len(items),
        "tagger_model": model,
        "wd_general_threshold": float(settings.wd_general_threshold),
        "selected_destinations": sorted(selected),
        "metrics": metrics,
        "by_label": by_label,
        "conclusion": conclusion,
        "items": items,
        "errors": errors,
        "photo_source": "randomuser portraits (+ commons/local fallback)",
        "anime_source": "safebooru (typical + realistic/photorealistic/3d edges)",
    }


def run_realism_eval_multi_model(
    *,
    count_per_class: int,
    settings: AppSettings,
    models: list[str] | None = None,
) -> dict:
    models = models or ["wd_eva02_large", "wd_swinv2_v3", "ml_danbooru"]
    reports = []
    for model in models:
        logger.info("realism_eval_model_start model=%s n=%s", model, count_per_class)
        report = run_realism_eval(
            count_per_class=count_per_class,
            settings=settings,
            tagger_model=model,
        )
        reports.append(report)
        logger.info(
            "realism_eval_model_done model=%s decision=%s f1=%s",
            model,
            report["conclusion"]["decision"],
            report["metrics"].get("f1"),
        )

    # Prefer models that pass typical GO gates, then typical F1, then overall F1.
    def _key(r: dict) -> tuple:
        conclusion = r.get("conclusion") or {}
        typical = conclusion.get("typical_metrics") or r.get("metrics") or {}
        overall = r.get("metrics") or {}
        go = 1 if conclusion.get("decision") == "GO" else 0
        return (
            go,
            float(typical.get("f1") or 0.0),
            float(typical.get("precision") or 0.0),
            float(overall.get("f1") or 0.0),
            # Prefer EVA02 on ties (stronger people-photo signal in prior probes).
            1 if r.get("tagger_model") == "wd_eva02_large" else 0,
        )

    ranked = sorted(reports, key=_key, reverse=True)
    best = ranked[0] if ranked else None
    overall_go = bool(best and best["conclusion"]["decision"] == "GO")
    return {
        "count_per_class_requested": count_per_class,
        "models": models,
        "reports": reports,
        "best_model": best["tagger_model"] if best else None,
        "overall_conclusion": {
            "decision": "GO" if overall_go else "NO_GO",
            "best_model": best["tagger_model"] if best else None,
            "best_metrics": (best.get("conclusion") or {}).get("typical_metrics")
            if best
            else None,
            "best_metrics_including_edges": best["metrics"] if best else None,
            "summary": (
                f"GO: {best['tagger_model']} separates remote people photos from "
                f"typical anime. Edge photoreal anime may quarantine into real_life."
                if overall_go and best
                else "NO_GO: no model met typical photo-vs-anime gates on this corpus."
            ),
            "video_and_gif": (
                best.get("conclusion", {}).get("video_note")
                if best
                else "Enable experimental media sampling so GIF/video use pooled frames."
            ),
            "wiring": (
                "Select destination folder real_life (alias photo) alongside content "
                "folders. Prefer tagger wd_eva02_large. real_life is already in "
                "taxonomy.json at priority 0; GIF/video work via multi-frame pooling."
            ),
        },
    }
