from __future__ import annotations

import logging
from pathlib import Path

from .schemas import AppSettings
from .services import (
    discover_tag_folders,
    extract_scores,
    load_known_tags,
    normalize_tag_name,
    sanitize_folder_name,
)
from .sfw_sources import download_post, get_source

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TAGS_CSV = REPO_ROOT / "tags.csv"


def _matched_tags_for_settings(settings: AppSettings) -> set[str]:
    selected = [t for t in settings.selected_tags if str(t).strip()]
    if not selected:
        return set()
    categories_root = Path(settings.categories_root or ".").expanduser()
    known = load_known_tags(TAGS_CSV)
    mappings = discover_tag_folders(categories_root, known, selected)
    return {m.matched_tag for m in mappings if m.matched and m.matched_tag}


def _tag_score(scores: dict[str, float], tag: str) -> float | None:
    if tag in scores:
        return float(scores[tag])
    norm = normalize_tag_name(tag)
    for key, value in scores.items():
        if normalize_tag_name(key) == norm:
            return float(value)
    return None


def run_sfw_eval(
    *,
    source_id: str,
    tags: list[str],
    count: int,
    settings: AppSettings,
) -> dict:
    # Lazy import avoids circular dependency with api.py routes.
    from .api import _classify_from_scores

    source = get_source(source_id)
    pull_tags = [t.strip().replace(" ", "_") for t in tags if str(t).strip()]
    if not pull_tags:
        raise ValueError("At least one tag is required")
    max_tags = source.info.max_content_tags
    if max_tags is not None and len(pull_tags) > max_tags:
        raise ValueError(
            f"{source.info.label} allows at most {max_tags} content tag(s) "
            "without an API key (plus forced SFW rating)."
        )
    if count < 5 or count > 30:
        raise ValueError("count must be between 5 and 30")

    matched_dest = _matched_tags_for_settings(settings)
    threshold = float(settings.confidence_threshold)

    posts = source.fetch_posts(pull_tags, count)
    errors: list[str] = []
    items: list[dict] = []

    hits: dict[str, int] = {t: 0 for t in pull_tags}
    present: dict[str, int] = {t: 0 for t in pull_tags}

    for post in posts:
        try:
            path = download_post(post)
        except Exception as err:
            logger.warning("debug_eval_download_failed post=%s err=%s", post.post_id, err)
            errors.append(f"{post.post_id}: download failed ({err})")
            continue

        try:
            scores = extract_scores(
                path,
                tagger_model=settings.tagger_model,
                wd_general_threshold=settings.wd_general_threshold,
            )
        except Exception as err:
            logger.exception("debug_eval_infer_failed path=%s", path)
            errors.append(f"{post.post_id}: inference failed ({err})")
            continue

        known_set = {normalize_tag_name(t) for t in post.tags}
        pull_scores: dict[str, float | None] = {}
        for tag in pull_tags:
            score = _tag_score(scores, tag)
            pull_scores[tag] = score
            if normalize_tag_name(tag) in known_set:
                present[tag] += 1
                if score is not None and score >= threshold:
                    hits[tag] += 1

        classified = _classify_from_scores(path, scores, matched_dest, threshold)
        suggested = (
            sanitize_folder_name(classified.primary_tag) if classified.primary_tag else None
        )
        top_global = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:8]

        items.append(
            {
                "source": post.source_id,
                "post_id": post.post_id,
                "rating": post.rating,
                "file_name": path.name,
                "known_tags": [t for t in pull_tags if normalize_tag_name(t) in known_set],
                "pull_tag_scores": pull_scores,
                "global_top_tags": [{"tag": t, "score": float(s)} for t, s in top_global],
                "primary_tag": classified.primary_tag,
                "primary_score": classified.primary_score,
                "needs_review": classified.needs_review,
                "review_reason": classified.reason,
                "suggested_folder": suggested,
                "secondary_suggestions": classified.secondary,
            }
        )

    recall = []
    for tag in pull_tags:
        denom = present[tag]
        recall.append(
            {
                "tag": tag,
                "present_in_posts": denom,
                "hits_at_threshold": hits[tag],
                "hit_rate": (hits[tag] / denom) if denom else None,
            }
        )

    return {
        "source": source.info.id,
        "source_label": source.info.label,
        "sfw_policy": source.info.sfw_policy,
        "query": source.build_query(pull_tags),
        "tags": pull_tags,
        "count_requested": count,
        "count_evaluated": len(items),
        "tagger_model": settings.tagger_model,
        "confidence_threshold": threshold,
        "destination_tags": list(settings.selected_tags),
        "recall": recall,
        "items": items,
        "errors": errors,
    }
