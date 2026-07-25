"""Fixed-suite tag recall benchmark: ML-Danbooru vs WD (and peers)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .services import (
    TAGGER_MODEL_ML,
    TAGGER_MODEL_WD_EVA02,
    TAGGER_MODEL_WD_SWINV2,
    normalize_tag_name,
)
from .sfw_sources import (
    USER_AGENT,
    DanbooruSource,
    SafebooruSource,
    SfwPost,
    _download_bytes,
    _http_json,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_PATH = Path(__file__).resolve().parent / "data" / "tag_recall_suite.json"
CACHE_ROOT = REPO_ROOT / "sample_data" / "tag_recall_suite"

DEFAULT_MODELS = (TAGGER_MODEL_ML, TAGGER_MODEL_WD_SWINV2)
DEFAULT_THRESHOLD = 0.35
DEFAULT_TOP_K = 20

SUPPORTED_MODELS = frozenset(
    {TAGGER_MODEL_ML, TAGGER_MODEL_WD_SWINV2, TAGGER_MODEL_WD_EVA02}
)
_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif", ".jfif"}
)


def _is_image_url(file_url: str) -> bool:
    path = urllib.parse.urlparse(str(file_url)).path.lower()
    suffix = Path(path).suffix
    return suffix in _IMAGE_SUFFIXES


def load_suite(path: Path | None = None) -> dict[str, Any]:
    suite_path = path or DEFAULT_SUITE_PATH
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("buckets"), list):
        raise ValueError("suite must be an object with a buckets array")
    return raw


def cache_dir_for(source_id: str) -> Path:
    path = CACHE_ROOT / str(source_id).strip().lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_cached_file(source_id: str, filename: str) -> Path:
    if not source_id or not filename:
        raise ValueError("source and filename are required")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("invalid filename")
    base = cache_dir_for(source_id).resolve()
    candidate = (base / filename).resolve()
    if not str(candidate).startswith(str(base)):
        raise ValueError("path escapes tag-recall cache")
    if not candidate.is_file():
        raise FileNotFoundError(filename)
    return candidate


def _tag_score(scores: dict[str, float], tag: str) -> float | None:
    if tag in scores:
        return float(scores[tag])
    wanted = normalize_tag_name(tag)
    best: float | None = None
    for key, value in scores.items():
        if normalize_tag_name(key) == wanted:
            score = float(value)
            best = score if best is None else max(best, score)
    return best


def _top_k_tags(scores: dict[str, float], k: int) -> list[str]:
    ranked = sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return [normalize_tag_name(t) for t, _ in ranked[: max(0, int(k))]]


def _fetch_danbooru_post(post_id: str) -> SfwPost:
    url = f"https://danbooru.donmai.us/posts/{urllib.parse.quote(post_id)}.json"
    try:
        payload = _http_json(url)
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"Danbooru HTTP {err.code} for post {post_id}") from err
    if not isinstance(payload, dict):
        raise RuntimeError(f"Danbooru unexpected payload for post {post_id}")
    if payload.get("is_deleted"):
        raise RuntimeError(f"Danbooru post {post_id} is deleted")
    file_url = (
        payload.get("file_url")
        or payload.get("large_file_url")
        or payload.get("preview_file_url")
    )
    if not file_url:
        raise RuntimeError(f"Danbooru post {post_id} has no file_url")
    tags = str(payload.get("tag_string") or "").split()
    return SfwPost(
        source_id="danbooru",
        post_id=str(payload.get("id") or post_id),
        file_url=str(file_url),
        rating=str(payload.get("rating") or ""),
        tags=tags,
    )


def _fetch_safebooru_post(post_id: str) -> SfwPost:
    params = urllib.parse.urlencode(
        {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "id": str(post_id),
        }
    )
    url = f"https://safebooru.org/index.php?{params}"
    try:
        payload = _http_json(url)
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"Safebooru HTTP {err.code} for post {post_id}") from err
    rows = payload if isinstance(payload, list) else []
    if not rows or not isinstance(rows[0], dict):
        raise RuntimeError(f"Safebooru post {post_id} not found")
    row = rows[0]
    file_url = row.get("file_url")
    if not file_url:
        raise RuntimeError(f"Safebooru post {post_id} has no file_url")
    if not str(file_url).startswith("http"):
        file_url = f"https:{file_url}"
    tags = str(row.get("tags") or "").split()
    return SfwPost(
        source_id="safebooru",
        post_id=str(row.get("id") or post_id),
        file_url=str(file_url),
        rating=str(row.get("rating") or "safe"),
        tags=tags,
    )


def fetch_post(source_id: str, post_id: str) -> SfwPost:
    key = str(source_id or "").strip().lower()
    pid = str(post_id or "").strip()
    if not pid:
        raise ValueError("post_id is required")
    if key == "danbooru":
        return _fetch_danbooru_post(pid)
    if key == "safebooru":
        return _fetch_safebooru_post(pid)
    raise ValueError(f"Unsupported source: {source_id}")


def download_suite_post(post: SfwPost, *, force: bool = False) -> Path:
    dest_dir = cache_dir_for(post.source_id)
    ext = Path(urllib.parse.urlparse(post.file_url).path).suffix or ".jpg"
    if len(ext) > 8:
        ext = ".jpg"
    dest = dest_dir / f"{post.post_id}{ext}"
    if force and dest.exists():
        dest.unlink(missing_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    _download_bytes(post.file_url, dest)
    return dest


def iter_suite_samples(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten buckets into sample dicts with resolved desired_tags."""
    out: list[dict[str, Any]] = []
    for bucket in suite.get("buckets") or []:
        if not isinstance(bucket, dict):
            continue
        bucket_id = str(bucket.get("id") or "").strip() or "unknown"
        bucket_desired = [
            str(t).strip().replace(" ", "_")
            for t in (bucket.get("desired_tags") or [])
            if str(t).strip()
        ]
        for sample in bucket.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            source = str(sample.get("source") or "").strip().lower()
            post_id = str(sample.get("post_id") or "").strip()
            if not source or not post_id:
                continue
            sample_desired = [
                str(t).strip().replace(" ", "_")
                for t in (sample.get("desired_tags") or bucket_desired)
                if str(t).strip()
            ]
            if not sample_desired:
                continue
            out.append(
                {
                    "bucket_id": bucket_id,
                    "source": source,
                    "post_id": post_id,
                    "desired_tags": sample_desired,
                }
            )
    return out


def ensure_suite_cache(
    suite: dict[str, Any],
    *,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Download pinned posts. Returns (resolved samples with path/tags, errors)."""
    resolved: list[dict[str, Any]] = []
    errors: list[str] = []
    for sample in iter_suite_samples(suite):
        key = f"{sample['source']}:{sample['post_id']}"
        try:
            post = fetch_post(sample["source"], sample["post_id"])
            if not _is_image_url(post.file_url):
                errors.append(f"{key}: skipped non-image ({post.file_url})")
                continue
            path = download_suite_post(post, force=refresh)
        except Exception as err:
            logger.warning("tag_recall_cache_failed sample=%s err=%s", key, err)
            errors.append(f"{key}: {err}")
            continue
        # Prefer manifest desired tags that appear on the post when available.
        post_norms = {normalize_tag_name(t) for t in post.tags}
        desired = list(sample["desired_tags"])
        verified = [t for t in desired if normalize_tag_name(t) in post_norms]
        if not verified:
            # Keep requested tags anyway — still a valid probe target.
            verified = desired
        resolved.append(
            {
                **sample,
                "desired_tags": verified,
                "post_tags": post.tags,
                "rating": post.rating,
                "path": path,
                "file_name": path.name,
            }
        )
    return resolved, errors


def _score_image(
    path: Path,
    *,
    tagger_model: str,
    wd_general_threshold: float,
) -> dict[str, float]:
    from .inference_engine import get_engine
    from .providers import ensure_nvidia_dll_search_path, preload_onnx_runtime_dlls

    ensure_nvidia_dll_search_path()
    preload_onnx_runtime_dlls()
    # Dense scores so threshold/top-K are comparable across models.
    return get_engine().score_one(
        path,
        tagger_model=tagger_model,
        wd_general_threshold=wd_general_threshold,
        raw_general=True,
    )


def _empty_counters() -> dict[str, Any]:
    return {
        "opportunities": 0,
        "hits_at_threshold": 0,
        "hits_at_top_k": 0,
        "per_tag": {},
    }


def _bump_tag(counters: dict[str, Any], tag: str, *, thr: bool, top: bool) -> None:
    row = counters["per_tag"].setdefault(
        tag,
        {
            "opportunities": 0,
            "hits_at_threshold": 0,
            "hits_at_top_k": 0,
        },
    )
    counters["opportunities"] += 1
    row["opportunities"] += 1
    if thr:
        counters["hits_at_threshold"] += 1
        row["hits_at_threshold"] += 1
    if top:
        counters["hits_at_top_k"] += 1
        row["hits_at_top_k"] += 1


def _finalize_counters(counters: dict[str, Any]) -> dict[str, Any]:
    opp = int(counters["opportunities"])
    thr = int(counters["hits_at_threshold"])
    top = int(counters["hits_at_top_k"])
    per_tag_out: dict[str, Any] = {}
    macro_thr: list[float] = []
    macro_top: list[float] = []
    for tag, row in sorted(counters["per_tag"].items()):
        t_opp = int(row["opportunities"])
        t_thr = int(row["hits_at_threshold"])
        t_top = int(row["hits_at_top_k"])
        thr_rate = (t_thr / t_opp) if t_opp else None
        top_rate = (t_top / t_opp) if t_opp else None
        if thr_rate is not None:
            macro_thr.append(thr_rate)
        if top_rate is not None:
            macro_top.append(top_rate)
        per_tag_out[tag] = {
            "opportunities": t_opp,
            "hits_at_threshold": t_thr,
            "hits_at_top_k": t_top,
            "recall_at_threshold": thr_rate,
            "recall_at_top_k": top_rate,
        }
    return {
        "opportunities": opp,
        "hits_at_threshold": thr,
        "hits_at_top_k": top,
        "micro_recall_at_threshold": (thr / opp) if opp else None,
        "micro_recall_at_top_k": (top / opp) if opp else None,
        "macro_recall_at_threshold": (
            sum(macro_thr) / len(macro_thr) if macro_thr else None
        ),
        "macro_recall_at_top_k": (
            sum(macro_top) / len(macro_top) if macro_top else None
        ),
        "per_tag": per_tag_out,
    }


def evaluate_model_on_samples(
    samples: list[dict[str, Any]],
    *,
    tagger_model: str,
    threshold: float,
    top_k: int,
    wd_general_threshold: float,
) -> dict[str, Any]:
    overall = _empty_counters()
    by_bucket: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    errors: list[str] = []

    for sample in samples:
        bucket_id = sample["bucket_id"]
        bucket_counters = by_bucket.setdefault(bucket_id, _empty_counters())
        key = f"{sample['source']}:{sample['post_id']}"
        try:
            scores = _score_image(
                sample["path"],
                tagger_model=tagger_model,
                wd_general_threshold=wd_general_threshold,
            )
        except Exception as err:
            logger.exception("tag_recall_infer_failed sample=%s", key)
            errors.append(f"{key}: inference failed ({err})")
            continue

        top_norm = set(_top_k_tags(scores, top_k))
        tag_rows: list[dict[str, Any]] = []
        for tag in sample["desired_tags"]:
            score = _tag_score(scores, tag)
            thr_hit = score is not None and float(score) >= float(threshold)
            top_hit = normalize_tag_name(tag) in top_norm
            _bump_tag(overall, tag, thr=thr_hit, top=top_hit)
            _bump_tag(bucket_counters, tag, thr=thr_hit, top=top_hit)
            tag_rows.append(
                {
                    "tag": tag,
                    "score": score,
                    "hit_at_threshold": thr_hit,
                    "hit_at_top_k": top_hit,
                }
            )

        items.append(
            {
                "bucket_id": bucket_id,
                "source": sample["source"],
                "post_id": sample["post_id"],
                "file_name": sample["file_name"],
                "rating": sample.get("rating") or "",
                "desired_tags": list(sample["desired_tags"]),
                "tag_results": tag_rows,
            }
        )

    return {
        "tagger_model": tagger_model,
        "threshold": float(threshold),
        "top_k": int(top_k),
        "count_evaluated": len(items),
        "summary": _finalize_counters(overall),
        "by_bucket": {
            bid: _finalize_counters(c) for bid, c in sorted(by_bucket.items())
        },
        "items": items,
        "errors": errors,
    }


def _compare_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        best = reports[0]["tagger_model"] if reports else None
        return {
            "primary_metric": "micro_recall_at_threshold",
            "best_model": best,
            "pairwise": [],
        }

    def micro(report: dict[str, Any]) -> float:
        val = (report.get("summary") or {}).get("micro_recall_at_threshold")
        return float(val) if val is not None else -1.0

    ranked = sorted(reports, key=lambda r: (-micro(r), r["tagger_model"]))
    best = ranked[0]["tagger_model"]
    pairwise: list[dict[str, Any]] = []
    # Item-level win/tie/loss between first two models when both present.
    a, b = reports[0], reports[1]
    a_items = {
        (i["source"], i["post_id"], tr["tag"]): tr
        for i in a.get("items") or []
        for tr in i.get("tag_results") or []
    }
    b_items = {
        (i["source"], i["post_id"], tr["tag"]): tr
        for i in b.get("items") or []
        for tr in i.get("tag_results") or []
    }
    keys = sorted(set(a_items) & set(b_items))
    a_wins = b_wins = ties = 0
    for key in keys:
        ah = bool(a_items[key].get("hit_at_threshold"))
        bh = bool(b_items[key].get("hit_at_threshold"))
        if ah and not bh:
            a_wins += 1
        elif bh and not ah:
            b_wins += 1
        else:
            ties += 1
    pairwise.append(
        {
            "model_a": a["tagger_model"],
            "model_b": b["tagger_model"],
            "metric": "hit_at_threshold",
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "compared": len(keys),
        }
    )
    return {
        "primary_metric": "micro_recall_at_threshold",
        "best_model": best,
        "ranking": [
            {
                "tagger_model": r["tagger_model"],
                "micro_recall_at_threshold": (r.get("summary") or {}).get(
                    "micro_recall_at_threshold"
                ),
                "micro_recall_at_top_k": (r.get("summary") or {}).get(
                    "micro_recall_at_top_k"
                ),
            }
            for r in ranked
        ],
        "pairwise": pairwise,
    }


def run_tag_recall_eval(
    *,
    models: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    refresh_cache: bool = False,
    suite_path: Path | None = None,
    wd_general_threshold: float | None = None,
    include_items: bool = True,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    defaults = suite.get("defaults") if isinstance(suite.get("defaults"), dict) else {}
    thr = float(threshold if threshold is not None else defaults.get("threshold", DEFAULT_THRESHOLD))
    k = int(top_k if top_k is not None else defaults.get("top_k", DEFAULT_TOP_K))
    wd_thr = float(
        wd_general_threshold
        if wd_general_threshold is not None
        else defaults.get("threshold", DEFAULT_THRESHOLD)
    )

    model_list = [str(m).strip() for m in (models or list(DEFAULT_MODELS)) if str(m).strip()]
    if not model_list:
        raise ValueError("At least one model is required")
    for model in model_list:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported tagger_model: {model}")
    if thr <= 0 or thr > 1:
        raise ValueError("threshold must be in (0, 1]")
    if k < 1 or k > 200:
        raise ValueError("top_k must be between 1 and 200")

    samples, cache_errors = ensure_suite_cache(suite, refresh=refresh_cache)
    if not samples:
        raise ValueError(
            "No suite samples available (empty suite or all downloads failed). "
            "Run scripts/fetch_tag_recall_suite.py first."
        )

    reports: list[dict[str, Any]] = []
    for model in model_list:
        report = evaluate_model_on_samples(
            samples,
            tagger_model=model,
            threshold=thr,
            top_k=k,
            wd_general_threshold=wd_thr,
        )
        if not include_items:
            report = {**report, "items": []}
        reports.append(report)

    comparison = _compare_reports(reports)
    return {
        "suite_version": suite.get("version"),
        "suite_path": str(suite_path or DEFAULT_SUITE_PATH),
        "threshold": thr,
        "top_k": k,
        "models": model_list,
        "count_samples_cached": len(samples),
        "cache_errors": cache_errors,
        "reports": reports,
        "comparison": comparison,
        "errors": list(cache_errors)
        + [e for r in reports for e in (r.get("errors") or [])],
    }


def search_posts_for_tag(
    *,
    source_id: str,
    tag: str,
    limit: int,
    allow_nsfw: bool = True,
) -> list[SfwPost]:
    """Discover posts for suite building (one content tag)."""
    tag = str(tag).strip().replace(" ", "_")
    if not tag:
        return []
    source_key = str(source_id).strip().lower()
    if source_key == "safebooru":
        src = SafebooruSource()
        posts = src.fetch_posts([tag], limit * 2)
        return [p for p in posts if _is_image_url(p.file_url)][:limit]
    if source_key == "danbooru":
        if allow_nsfw:
            # Bypass SFW-only DanbooruSource wrapper for NSFW buckets.
            params = urllib.parse.urlencode(
                {"tags": tag, "limit": str(max(limit * 2, limit))}
            )
            payload = _http_json(f"https://danbooru.donmai.us/posts.json?{params}")
            if not isinstance(payload, list):
                raise RuntimeError("Danbooru returned unexpected payload")
            posts: list[SfwPost] = []
            for row in payload:
                if not isinstance(row, dict) or row.get("is_deleted"):
                    continue
                file_url = (
                    row.get("file_url")
                    or row.get("large_file_url")
                    or row.get("preview_file_url")
                )
                if not file_url or not _is_image_url(str(file_url)):
                    continue
                post_id = str(row.get("id") or "")
                if not post_id:
                    continue
                posts.append(
                    SfwPost(
                        source_id="danbooru",
                        post_id=post_id,
                        file_url=str(file_url),
                        rating=str(row.get("rating") or ""),
                        tags=str(row.get("tag_string") or "").split(),
                    )
                )
                if len(posts) >= limit:
                    break
            return posts
        posts = DanbooruSource().fetch_posts([tag], limit * 2)
        return [p for p in posts if _is_image_url(p.file_url)][:limit]
    raise ValueError(f"Unsupported source: {source_id}")


def rebuild_suite_samples(
    suite: dict[str, Any],
    *,
    samples_per_bucket: int = 6,
    replace_all: bool = False,
) -> dict[str, Any]:
    """Fill empty/missing samples via search; optionally replace all pinned IDs."""
    buckets_out: list[dict[str, Any]] = []
    for bucket in suite.get("buckets") or []:
        if not isinstance(bucket, dict):
            continue
        bucket = dict(bucket)
        desired = [
            str(t).strip().replace(" ", "_")
            for t in (bucket.get("desired_tags") or [])
            if str(t).strip()
        ]
        existing = (
            []
            if replace_all
            else [
                s
                for s in (bucket.get("samples") or [])
                if isinstance(s, dict) and s.get("post_id")
            ]
        )
        if len(existing) >= samples_per_bucket:
            bucket["samples"] = existing[:samples_per_bucket]
            buckets_out.append(bucket)
            continue

        source = str(bucket.get("source") or "danbooru").strip().lower()
        search_tag = str(bucket.get("search_tag") or (desired[0] if desired else "")).strip()
        allow_nsfw = bool(bucket.get("allow_nsfw", source == "danbooru"))
        found: list[SfwPost] = []
        if search_tag:
            try:
                found = search_posts_for_tag(
                    source_id=source,
                    tag=search_tag,
                    limit=max(samples_per_bucket * 3, 12),
                    allow_nsfw=allow_nsfw,
                )
            except Exception as err:
                logger.warning(
                    "suite_search_failed bucket=%s tag=%s err=%s",
                    bucket.get("id"),
                    search_tag,
                    err,
                )

        seen = {str(s.get("post_id")) for s in existing}
        samples = list(existing)
        for post in found:
            if post.post_id in seen:
                continue
            if not _is_image_url(post.file_url):
                continue
            # Require at least one desired tag on the post when possible.
            post_norms = {normalize_tag_name(t) for t in post.tags}
            if desired and not any(normalize_tag_name(t) in post_norms for t in desired):
                continue
            samples.append(
                {
                    "source": post.source_id,
                    "post_id": post.post_id,
                    "desired_tags": desired,
                }
            )
            seen.add(post.post_id)
            if len(samples) >= samples_per_bucket:
                break
        bucket["samples"] = samples
        buckets_out.append(bucket)

    out = dict(suite)
    out["buckets"] = buckets_out
    return out


def save_suite(suite: dict[str, Any], path: Path | None = None) -> Path:
    suite_path = path or DEFAULT_SUITE_PATH
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(
        json.dumps(suite, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return suite_path
