"""False-positive rates on the curated suite vs preferred taxonomy tags."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from .services import normalize_tag_name
from .tag_recall_eval import (
    DEFAULT_SUITE_PATH,
    SUPPORTED_MODELS,
    _score_image,
    _tag_score,
    ensure_suite_cache,
    load_suite,
)
from .taxonomy import choose_best_destination, get_taxonomy, reload_taxonomy

logger = logging.getLogger(__name__)


def _selected_norms(selected: list[str] | set[str]) -> set[str]:
    return {normalize_tag_name(t) for t in selected if str(t).strip()}


def _bucket_matches_selected(bucket, selected: set[str]) -> bool:
    names = {
        normalize_tag_name(bucket.folder),
        normalize_tag_name(bucket.id),
        normalize_tag_name(bucket.folder.split("/")[0]),
    }
    names |= {normalize_tag_name(a) for a in bucket.aliases}
    return bool(names & selected)


def preferred_probe_tags(
    selected: set[str],
    *,
    min_weight: float,
) -> dict[str, list[str]]:
    """tag -> list of folder names that use it as evidence."""
    cfg = get_taxonomy()
    out: dict[str, list[str]] = defaultdict(list)
    for bucket in cfg.buckets:
        if not _bucket_matches_selected(bucket, selected):
            continue
        for ev in bucket.evidence:
            if float(ev.weight) < float(min_weight):
                continue
            out[ev.tag].append(bucket.folder)
    return dict(out)


def preferred_folder_names(selected: set[str]) -> set[str]:
    cfg = get_taxonomy()
    folders: set[str] = set()
    for bucket in cfg.buckets:
        if _bucket_matches_selected(bucket, selected):
            folders.add(bucket.folder)
    folders |= selected
    return folders


def _folder_root(folder: str | None) -> str | None:
    if not folder:
        return None
    return normalize_tag_name(str(folder).split("/")[0])


def _truth_folders_for_post(post_tags: list[str], selected: set[str]) -> set[str]:
    """Folders whose evidence intersects the post's known tags."""
    cfg = get_taxonomy()
    post = {normalize_tag_name(t) for t in post_tags}
    roots: set[str] = set()
    for bucket in cfg.buckets:
        if not _bucket_matches_selected(bucket, selected):
            continue
        for ev in bucket.evidence:
            if normalize_tag_name(ev.tag) in post:
                roots.add(_folder_root(bucket.folder) or "")
                break
    return {r for r in roots if r}


def evaluate_fps(
    samples: list[dict[str, Any]],
    *,
    models: list[str],
    tag_threshold: float,
    route_threshold: float,
    probe_by_tag: dict[str, list[str]],
    selected_folders: set[str],
    wd_general_threshold: float,
) -> dict[str, Any]:
    probe_tags = sorted(probe_by_tag)
    reports: list[dict[str, Any]] = []

    for model in models:
        tag_neg: dict[str, int] = defaultdict(int)
        tag_fp: dict[str, int] = defaultdict(int)
        route_n = 0
        route_fp = 0
        route_fp_by: dict[str, int] = defaultdict(int)
        items: list[dict[str, Any]] = []
        errors: list[str] = []

        for sample in samples:
            key = f"{sample['source']}:{sample['post_id']}"
            try:
                scores = _score_image(
                    sample["path"],
                    tagger_model=model,
                    wd_general_threshold=wd_general_threshold,
                )
            except Exception as err:
                errors.append(f"{key}: {err}")
                continue

            post_norms = {normalize_tag_name(t) for t in (sample.get("post_tags") or [])}
            fp_tags: list[dict[str, Any]] = []
            for tag in probe_tags:
                if normalize_tag_name(tag) in post_norms:
                    continue
                tag_neg[tag] += 1
                score = _tag_score(scores, tag)
                if score is not None and float(score) >= float(tag_threshold):
                    tag_fp[tag] += 1
                    fp_tags.append({"tag": tag, "score": float(score)})

            truth_roots = _truth_folders_for_post(
                sample.get("post_tags") or [], selected_folders
            )
            folder, score, _ = choose_best_destination(scores, selected_folders)
            if folder is not None and (
                score is None or float(score) < float(route_threshold)
            ):
                folder, score = None, None
            pred_root = _folder_root(folder)
            route_n += 1
            is_route_fp = bool(
                pred_root and truth_roots and pred_root not in truth_roots
            )
            if pred_root and not truth_roots:
                is_route_fp = True
            if is_route_fp and pred_root:
                route_fp += 1
                route_fp_by[pred_root] += 1

            items.append(
                {
                    "bucket_id": sample["bucket_id"],
                    "source": sample["source"],
                    "post_id": sample["post_id"],
                    "file_name": sample["file_name"],
                    "truth_folder_roots": sorted(truth_roots),
                    "predicted_folder": folder,
                    "predicted_score": score,
                    "route_fp": is_route_fp,
                    "false_positive_tags": fp_tags,
                }
            )

        per_tag: dict[str, Any] = {}
        rates: list[float] = []
        for tag in probe_tags:
            neg = tag_neg[tag]
            fp = tag_fp[tag]
            rate = (fp / neg) if neg else None
            if rate is not None:
                rates.append(rate)
            per_tag[tag] = {
                "negatives": neg,
                "false_positives": fp,
                "fpr": rate,
                "folders": probe_by_tag[tag],
            }

        micro_neg = sum(tag_neg.values())
        micro_fp = sum(tag_fp.values())
        reports.append(
            {
                "tagger_model": model,
                "tag_threshold": tag_threshold,
                "route_threshold": route_threshold,
                "count_evaluated": len(items),
                "tag_fp": {
                    "probe_tag_count": len(probe_tags),
                    "negatives": micro_neg,
                    "false_positives": micro_fp,
                    "micro_fpr": (micro_fp / micro_neg) if micro_neg else None,
                    "macro_fpr": (sum(rates) / len(rates)) if rates else None,
                    "per_tag": per_tag,
                },
                "folder_route_fp": {
                    "evaluated": route_n,
                    "false_positives": route_fp,
                    "fpr": (route_fp / route_n) if route_n else None,
                    "by_folder": dict(sorted(route_fp_by.items())),
                },
                "items": items,
                "errors": errors,
            }
        )

    return {"reports": reports, "probe_tags": probe_tags}


def _slim_report(rep: dict[str, Any], *, top_n: int = 40) -> dict[str, Any]:
    top_fp = sorted(
        (
            (t, row)
            for t, row in (rep["tag_fp"]["per_tag"] or {}).items()
            if (row.get("false_positives") or 0) > 0
        ),
        key=lambda kv: (-kv[1]["false_positives"], -((kv[1]["fpr"] or 0))),
    )[:top_n]
    return {
        "tagger_model": rep["tagger_model"],
        "tag_threshold": rep["tag_threshold"],
        "route_threshold": rep["route_threshold"],
        "count_evaluated": rep["count_evaluated"],
        "tag_fp": {
            k: v
            for k, v in rep["tag_fp"].items()
            if k != "per_tag"
        },
        "folder_route_fp": rep["folder_route_fp"],
        "top_false_positive_tags": [{"tag": t, **row} for t, row in top_fp],
        "sample_route_fps": [
            {
                "source": i["source"],
                "post_id": i["post_id"],
                "file_name": i["file_name"],
                "bucket_id": i["bucket_id"],
                "truth": i["truth_folder_roots"],
                "predicted": i["predicted_folder"],
                "score": i["predicted_score"],
            }
            for i in rep.get("items") or []
            if i.get("route_fp")
        ][:30],
        "errors": rep.get("errors") or [],
    }


def run_tag_fp_eval(
    *,
    selected_tags: list[str],
    models: list[str] | None = None,
    tag_threshold: float = 0.6,
    route_threshold: float | None = None,
    min_weight: float = 0.85,
    also_wd_threshold: bool = True,
    wd_general_threshold: float = 0.35,
    refresh_cache: bool = False,
    suite_path: Path | None = None,
    include_items: bool = False,
) -> dict[str, Any]:
    """Run preferred-tag FP benchmark; returns UI-friendly payload by default."""
    reload_taxonomy()
    selected = _selected_norms(selected_tags)
    if not selected:
        raise ValueError("selected_tags is empty — pick folders in Settings first")

    model_list = [
        str(m).strip() for m in (models or ["ml_danbooru", "wd_swinv2_v3"]) if str(m).strip()
    ]
    if not model_list:
        raise ValueError("At least one model is required")
    for model in model_list:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported tagger_model: {model}")

    if tag_threshold <= 0 or tag_threshold > 1:
        raise ValueError("tag_threshold must be in (0, 1]")
    route_thr = float(route_threshold if route_threshold is not None else tag_threshold)
    if route_thr <= 0 or route_thr > 1:
        raise ValueError("route_threshold must be in (0, 1]")
    if min_weight < 0 or min_weight > 1:
        raise ValueError("min_weight must be in [0, 1]")

    probe = preferred_probe_tags(selected, min_weight=min_weight)
    if not probe:
        raise ValueError(
            f"No taxonomy evidence tags at min_weight={min_weight} for selected folders"
        )
    folders = preferred_folder_names(selected)

    suite = load_suite(suite_path or DEFAULT_SUITE_PATH)
    samples, cache_errors = ensure_suite_cache(suite, refresh=refresh_cache)
    if not samples:
        raise ValueError(
            "No suite samples available. Run scripts/fetch_tag_recall_suite.py --rebuild first."
        )

    raw = evaluate_fps(
        samples,
        models=model_list,
        tag_threshold=float(tag_threshold),
        route_threshold=route_thr,
        probe_by_tag=probe,
        selected_folders=folders,
        wd_general_threshold=float(wd_general_threshold),
    )

    reports = (
        raw["reports"]
        if include_items
        else [_slim_report(r) for r in raw["reports"]]
    )

    out: dict[str, Any] = {
        "selected_folders": sorted(selected),
        "min_weight": float(min_weight),
        "tag_threshold": float(tag_threshold),
        "route_threshold": route_thr,
        "models": model_list,
        "probe_tag_count": len(raw["probe_tags"]),
        "count_samples": len(samples),
        "cache_errors": cache_errors,
        "reports": reports,
        "errors": list(cache_errors)
        + [e for r in raw["reports"] for e in (r.get("errors") or [])],
    }

    if also_wd_threshold:
        alt_thr = float(wd_general_threshold)
        if abs(alt_thr - float(tag_threshold)) > 1e-9:
            alt = evaluate_fps(
                samples,
                models=model_list,
                tag_threshold=alt_thr,
                route_threshold=alt_thr,
                probe_by_tag=probe,
                selected_folders=folders,
                wd_general_threshold=float(wd_general_threshold),
            )
            out["at_wd_general_threshold"] = {
                "tag_threshold": alt_thr,
                "reports": [
                    {
                        "tagger_model": r["tagger_model"],
                        "tag_fp": {
                            k: v for k, v in r["tag_fp"].items() if k != "per_tag"
                        },
                        "folder_route_fp": r["folder_route_fp"],
                        "top_false_positive_tags": _slim_report(r)[
                            "top_false_positive_tags"
                        ][:15],
                    }
                    for r in alt["reports"]
                ],
            }

    return out
