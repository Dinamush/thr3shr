#!/usr/bin/env python3
"""Empirical probe: can WD taggers separate real photos from anime/illustration?

Does NOT modify taxonomy.json or production routing. Writes a JSON report under
scripts/out/realism_probe_report.json.

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/debug_realism_probe.py
  ../.venv/Scripts/python.exe scripts/debug_realism_probe.py --photos-dir PATH --anime-dir PATH
  ../.venv/Scripts/python.exe scripts/debug_realism_probe.py --include-ml --limit 12 --threshold 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.inference_engine import (  # noqa: E402
    TAGGER_MODEL_ML,
    TAGGER_MODEL_WD_EVA02,
    TAGGER_MODEL_WD_SWINV2,
    WD_MODEL_NAMES,
    InferenceEngine,
    _normalize_scores,
    _normalize_tag,
    preprocess_mldanbooru,
    preprocess_wd14,
    reset_engine,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".tiff"}

CANDIDATE_TAGS = (
    # Planned taxonomy evidence (may be absent from WD v3 vocab).
    "realistic",
    "photorealistic",
    "photo_(medium)",
    "3d",
    # WD-v3-adjacent tags actually present in selected_tags.csv.
    "photo_(object)",
    "photo_background",
    "cellphone_photo",
    "3d_background",
    "holding_photo",
    "semi-realistic",
    "semi_realistic",
    "realistic_proportions",
    "depth_of_field",
    "blurry",
    "bokeh",
    "film_grain",
    "scan",
    "cover_page",
    "traditional_media",
    "painting_(medium)",
    "colored_pencil_(medium)",
    "graphite_(medium)",
)

# Tags used for precision/recall sweeps + combo policy.
PRIMARY_EVIDENCE_TAGS = (
    "realistic",
    "photorealistic",
    "photo_(medium)",
    "3d",
    "photo_background",
    "cellphone_photo",
    "3d_background",
)

# Optional local dirs via env (colon/semicolon-separated). Empty = Wikimedia fallbacks.
def _env_dirs(name: str) -> list[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    sep = ";" if ";" in raw else os.pathsep
    return [Path(p.strip()) for p in raw.split(sep) if p.strip()]


DEFAULT_PHOTO_DIRS = _env_dirs("THR3SHR_PROBE_PHOTO_DIRS")
DEFAULT_ANIME_DIRS = _env_dirs("THR3SHR_PROBE_ANIME_DIRS") + [
    Path(__file__).resolve().parents[2] / "sample_data" / "sfw_safebooru",
]

WIKIMEDIA_FALLBACK_PHOTOS = [
    # Clearly photographic public-domain / CC-ish Wikimedia files (thumbnails).
    (
        "wm_eiffel.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/640px-Tour_Eiffel_Wikimedia_Commons.jpg",
    ),
    (
        "wm_sunset.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Sunset_2007.jpg/640px-Sunset_2007.jpg",
    ),
    (
        "wm_cat.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg",
    ),
    (
        "wm_dog.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg",
    ),
    (
        "wm_flower.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/640px-Sunflower_sky_backdrop.jpg",
    ),
    (
        "wm_bridge.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/GoldenGateBridge-001.jpg/640px-GoldenGateBridge-001.jpg",
    ),
    (
        "wm_mountain.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/640px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg",
    ),
    (
        "wm_city.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/640px-New_york_times_square-terabass.jpg",
    ),
]


def _list_images(directory: Path, *, recursive: bool = True) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    out: list[Path] = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            out.append(path)
    return sorted(out)


def _take_diverse(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    # Prefer spreading across subdirs.
    by_parent: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_parent[str(path.parent)].append(path)
    selected: list[Path] = []
    buckets = [list(v) for v in by_parent.values()]
    while len(selected) < limit and buckets:
        next_buckets: list[list[Path]] = []
        for bucket in buckets:
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= limit:
                break
            if bucket:
                next_buckets.append(bucket)
        buckets = next_buckets
    return selected[:limit]


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "thr3shr-realism-probe/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as resp:
        dest.write_bytes(resp.read())


def ensure_fallback_photos(dest_dir: Path, needed: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, url in WIKIMEDIA_FALLBACK_PHOTOS:
        if len(paths) >= needed:
            break
        dest = dest_dir / name
        if not dest.exists():
            print(f"downloading fallback photo {name} …", flush=True)
            try:
                _download(url, dest)
            except Exception as exc:  # noqa: BLE001
                print(f"  failed: {exc}", flush=True)
                continue
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(dest)
    return paths


def discover_from_db(limit: int = 20) -> list[Path]:
    db = ROOT / "app.db"
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT file_path FROM items ORDER BY id DESC LIMIT ?",
            (limit * 3,),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    found: list[Path] = []
    for (raw,) in rows:
        path = Path(str(raw))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            found.append(path)
        if len(found) >= limit:
            break
    return found


def collect_labeled_sets(
    *,
    photos_dir: Path | None,
    anime_dir: Path | None,
    limit: int,
    allow_download: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    samples: list[dict[str, Any]] = []

    photo_paths: list[Path] = []
    if photos_dir:
        photo_paths = _list_images(photos_dir)
        notes.append(f"photos-dir={photos_dir} count={len(photo_paths)}")
    else:
        for d in DEFAULT_PHOTO_DIRS:
            found = _list_images(d)
            if found:
                photo_paths.extend(found)
                notes.append(f"auto photo dir {d} (+{len(found)})")
        # Optional: scan Pictures for WhatsApp / camera dumps when enabled
        if os.environ.get("THR3SHR_PROBE_SCAN_PICTURES", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            pics_root = Path.home() / "Pictures"
            if pics_root.exists():
                extras = []
                for p in pics_root.iterdir():
                    if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                        continue
                    name = p.name.lower()
                    if "whatsapp" in name or name.startswith("img_"):
                        extras.append(p)
                if extras:
                    photo_paths.extend(extras)
                    notes.append(f"auto Pictures root camera-like (+{len(extras)})")

    anime_paths: list[Path] = []
    if anime_dir:
        anime_paths = _list_images(anime_dir)
        notes.append(f"anime-dir={anime_dir} count={len(anime_paths)}")
    else:
        # Pull a slice from several anime dirs so semi-real / style diversity appears.
        per_dir = max(limit, 8)
        for d in DEFAULT_ANIME_DIRS:
            found = _list_images(d)
            if not found:
                continue
            slice_n = min(len(found), per_dir)
            anime_paths.extend(_take_diverse(found, slice_n))
            notes.append(f"auto anime dir {d} (+{slice_n}/{len(found)})")
            if len(anime_paths) >= limit * 5:
                break

    if len(photo_paths) < max(4, limit // 2) and allow_download:
        needed = max(limit, 8) - len(photo_paths)
        fallback = ensure_fallback_photos(
            ROOT / "scripts" / "out" / "realism_probe_tmp_photos",
            needed=max(needed, 8),
        )
        photo_paths.extend(fallback)
        notes.append(f"wikimedia fallback photos (+{len(fallback)})")

    if not anime_paths:
        db_paths = discover_from_db(limit * 2)
        anime_paths.extend(db_paths)
        notes.append(f"db recent items as anime-pool (+{len(db_paths)})")

    photo_sel = _take_diverse(photo_paths, limit)
    anime_sel = _take_diverse(anime_paths, limit)
    if len(photo_sel) < 8:
        notes.append(f"SHORTFALL photos: got {len(photo_sel)} (aim ≥8)")
    if len(anime_sel) < 8:
        notes.append(f"SHORTFALL anime: got {len(anime_sel)} (aim ≥8)")

    for path in photo_sel:
        samples.append({"path": str(path), "label": "photo", "source": "photos"})
    for path in anime_sel:
        samples.append({"path": str(path), "label": "anime", "source": "anime"})
    return samples, notes


def score_wd_raw(
    engine: InferenceEngine,
    images: list[Path],
    *,
    tagger_model: str,
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Return full general + rating scores (no threshold filter)."""
    wd_name = WD_MODEL_NAMES[tagger_model]
    session = engine._get_session(tagger_model)
    target = engine._wd_target_size[wd_name]
    tensors = [preprocess_wd14(image, target) for image in images]
    preds_list: list[np.ndarray] = []
    input_name = session.get_inputs()[0].name
    out0 = session.get_outputs()[0].name
    out1 = session.get_outputs()[1].name
    for start in range(0, len(tensors), batch_size):
        chunk = tensors[start : start + batch_size]
        feed = chunk[0] if len(chunk) == 1 else np.concatenate(chunk, axis=0)
        with engine._run_lock:
            preds, _emb = session.run([out0, out1], {input_name: feed})
        for row in preds:
            preds_list.append(row)

    tag_names, rating_idx, general_idx, _char_idx = engine._get_wd_labels(wd_name)
    results: list[dict[str, Any]] = []
    for pred in preds_list:
        labels = list(zip(tag_names, pred.astype(float)))
        general = {
            name: float(score)
            for i in general_idx
            for name, score in [labels[i]]
        }
        rating = {
            name: float(score)
            for i in rating_idx
            for name, score in [labels[i]]
        }
        general_n = _normalize_scores(general)
        rating_n = _normalize_scores(rating)
        results.append(
            {
                "general": general_n,
                "rating": rating_n,
                "all": {**general_n, **{f"rating:{k}": v for k, v in rating_n.items()}},
            }
        )
    return results


def score_ml_raw(engine: InferenceEngine, images: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for image in images:
        tensor = preprocess_mldanbooru(image, size=448, keep_ratio=True)
        session = engine._get_session(TAGGER_MODEL_ML)
        with engine._run_lock:
            (native_output,) = session.run(["output"], {"input": tensor})
        probs = 1.0 / (1.0 + np.exp(-native_output.reshape(-1)))
        labels = engine._get_ml_labels()
        scores = {
            labels[i]: float(probs[i])
            for i in range(min(len(labels), len(probs)))
        }
        scores_n = _normalize_scores(scores)
        results.append({"general": scores_n, "rating": {}, "all": scores_n})
    return results


def top_n(scores: dict[str, float], n: int = 15) -> list[dict[str, float | str]]:
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [{"tag": t, "score": round(float(s), 6)} for t, s in items]


def candidate_scores(scores: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for tag in CANDIDATE_TAGS:
        ntag = _normalize_tag(tag)
        best = float(scores.get(ntag, 0.0) or 0.0)
        # Accept raw/unnormalized keys if present.
        if tag in scores:
            best = max(best, float(scores[tag]))
        out[tag] = round(best, 6)
    return out


def separation_metrics(
    rows: list[dict[str, Any]],
    *,
    model: str,
    tag: str,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    photos = [r for r in rows if r["label"] == "photo"]
    anime = [r for r in rows if r["label"] == "anime"]
    out: list[dict[str, Any]] = []
    for thr in thresholds:
        tp = sum(1 for r in photos if r["models"][model]["candidates"].get(tag, 0) >= thr)
        fn = len(photos) - tp
        fp = sum(1 for r in anime if r["models"][model]["candidates"].get(tag, 0) >= thr)
        tn = len(anime) - fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        # Treat "predict photo" as positive class.
        out.append(
            {
                "tag": tag,
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "anime_false_positive_rate": round(fp / len(anime), 4) if anime else None,
            }
        )
    return out


def single_tag_policy_metrics(
    rows: list[dict[str, Any]],
    *,
    model: str,
    tag: str,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Convenience wrapper naming single-tag threshold sweeps as a policy."""
    rows_out = separation_metrics(rows, model=model, tag=tag, thresholds=thresholds)
    for row in rows_out:
        row["policy"] = f"single_tag:{tag}"
    return rows_out


def combo_metrics(
    rows: list[dict[str, Any]],
    *,
    model: str,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Hard evidence OR soft corroboration policy matching the planned fixture."""
    photos = [r for r in rows if r["label"] == "photo"]
    anime = [r for r in rows if r["label"] == "anime"]
    out: list[dict[str, Any]] = []

    def predicts_photo(cands: dict[str, float], thr: float) -> bool:
        hard = max(
            cands.get("photorealistic", 0.0),
            cands.get("photo_(medium)", 0.0),
            cands.get("photo_background", 0.0),
            cands.get("cellphone_photo", 0.0),
        )
        if hard >= thr:
            return True
        soft_tags = ("realistic", "3d", "3d_background")
        soft_hits = [cands.get(t, 0.0) for t in soft_tags if cands.get(t, 0.0) >= thr]
        # Corroboration: ≥2 soft tags above thr, or one soft + any other realism ≥0.15
        if len(soft_hits) >= 2:
            return True
        if soft_hits:
            others = [
                cands.get(t, 0.0)
                for t in (
                    "photorealistic",
                    "photo_(medium)",
                    "photo_background",
                    "cellphone_photo",
                    "realistic",
                    "3d",
                    "3d_background",
                )
                if cands.get(t, 0.0) >= 0.15
            ]
            if len(others) >= 2:
                return True
        return False

    for thr in thresholds:
        tp = sum(1 for r in photos if predicts_photo(r["models"][model]["candidates"], thr))
        fn = len(photos) - tp
        fp = sum(1 for r in anime if predicts_photo(r["models"][model]["candidates"], thr))
        tn = len(anime) - fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out.append(
            {
                "policy": "hard_or_soft_corroborated",
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "anime_false_positive_rate": round(fp / len(anime), 4) if anime else None,
            }
        )
    return out


def summarize_tag_distributions(
    rows: list[dict[str, Any]], model: str
) -> dict[str, Any]:
    dist: dict[str, Any] = {}
    for tag in PRIMARY_EVIDENCE_TAGS:
        for label in ("photo", "anime"):
            vals = [
                float(r["models"][model]["candidates"].get(tag, 0.0))
                for r in rows
                if r["label"] == label
            ]
            key = f"{label}:{tag}"
            if not vals:
                dist[key] = None
                continue
            dist[key] = {
                "n": len(vals),
                "mean": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "p90": round(sorted(vals)[max(0, int(0.9 * (len(vals) - 1)))], 4),
            }
    return dist


def best_thresholds(
    metric_rows: list[dict[str, Any]],
    *,
    min_precision: float = 0.85,
) -> dict[str, Any]:
    """Prefer high precision (avoid anime FP), then F1, then recall."""
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_tag[row["tag"]].append(row)
    picks: dict[str, Any] = {}
    for tag, rows in by_tag.items():
        eligible = [r for r in rows if r["precision"] >= min_precision]
        pool = eligible or rows
        best = max(pool, key=lambda r: (r["f1"], r["precision"], r["recall"], -r["threshold"]))
        # Also note max-F1 unrestricted
        max_f1 = max(rows, key=lambda r: (r["f1"], r["precision"], r["recall"]))
        picks[tag] = {
            "suggested_for_high_precision": best,
            "max_f1": max_f1,
        }
    return picks


def recommend(
    analysis: dict[str, Any],
    *,
    n_photo: int,
    n_anime: int,
) -> dict[str, Any]:
    """GO if a model can hit high precision with usable recall on hard tags."""
    go_reasons: list[str] = []
    nogo_reasons: list[str] = []
    if n_photo < 8 or n_anime < 8:
        nogo_reasons.append(
            f"Sample shortfall (photos={n_photo}, anime={n_anime}); treat results as preliminary."
        )

    model_rank: list[tuple[str, float, dict[str, Any]]] = []
    for model, block in analysis.items():
        combo = block.get("combo_policy", [])
        realistic_alone = block.get("realistic_alone_policy", [])
        hard_picks = block.get("suggested_thresholds", {})
        best_combo = max(combo, key=lambda r: (r["f1"], r["precision"], r["recall"])) if combo else None
        best_realistic = (
            max(realistic_alone, key=lambda r: (r["f1"], r["precision"], r["recall"]))
            if realistic_alone
            else None
        )
        photo_pr = block["distributions"].get("photo:photorealistic", {}) or {}
        anime_pr = block["distributions"].get("anime:photorealistic", {}) or {}
        photo_r = block["distributions"].get("photo:realistic", {}) or {}
        anime_r = block["distributions"].get("anime:realistic", {}) or {}
        sep_pr = (photo_pr.get("mean", 0) or 0) - (anime_pr.get("mean", 0) or 0)
        sep_r = (photo_r.get("mean", 0) or 0) - (anime_r.get("mean", 0) or 0)
        # Prefer WD-usable signals; combo may under-recall when photo_(medium)/3d absent.
        score = 0.0
        if best_realistic:
            score = best_realistic["f1"] * 0.55 + best_realistic["precision"] * 0.35
        if best_combo:
            score = max(score, best_combo["f1"] * 0.55 + best_combo["precision"] * 0.35)
        score += max(0.0, sep_r) * 0.1 + max(0.0, sep_pr) * 0.05
        model_rank.append(
            (
                model,
                score,
                {
                    "best_combo": best_combo,
                    "best_realistic_alone": best_realistic,
                    "hard": hard_picks,
                },
            )
        )

    model_rank.sort(key=lambda x: x[1], reverse=True)
    best_model = model_rank[0][0] if model_rank else None
    best_combo = model_rank[0][2]["best_combo"] if model_rank else None
    best_realistic = model_rank[0][2]["best_realistic_alone"] if model_rank else None

    decision = "NO-GO"
    # WD path: realistic alone is the strongest discriminator on this corpus.
    primary = best_realistic if (best_realistic and best_realistic["f1"] >= (best_combo or {}).get("f1", 0)) else best_combo
    if primary and primary["precision"] >= 0.9 and primary["recall"] >= 0.5 and primary["f1"] >= 0.6:
        decision = "GO"
        go_reasons.append(
            f"{best_model} policy={primary.get('policy', primary.get('tag'))} "
            f"thr={primary['threshold']}: P={primary['precision']} R={primary['recall']} F1={primary['f1']}"
        )
    elif primary and primary["precision"] >= 0.85 and primary["recall"] >= 0.35:
        decision = "CONDITIONAL-GO"
        go_reasons.append(
            f"{best_model} usable but limited "
            f"(P={primary['precision']} R={primary['recall']} @ {primary['threshold']})"
        )
    else:
        nogo_reasons.append(
            "No model/threshold achieved high-precision photo detection with acceptable recall."
        )

    go_reasons.append(
        "WD v3 vocab lacks photo_(medium) and plain 3d — planned fixture tags are dead for WD; "
        "use photorealistic/realistic (and optionally photo_background)."
    )
    if best_realistic and best_combo and best_realistic["f1"] > best_combo["f1"]:
        go_reasons.append(
            "Fixture-style hard/soft corroboration under-recalls on WD because soft realistic "
            "alone is discarded; consider treating high realistic as hard for WD, or lower "
            "soft_alone_weight."
        )

    failure_modes = [
        "WD selected_tags lack photo_(medium) and 3d — those evidence rows never fire on WD.",
        "Non-person photos (moon/sky/landscape) often under-score photorealistic; realistic still helps on EVA02.",
        "AI-art / photoreal anime may raise photorealistic/realistic (not stressed in this anime sample).",
        "ml_danbooru fires photo_(medium)/3d but has higher anime false-positive risk below ~0.35.",
        "3d / CGI game renders: WD has only 3d_background (weak here); ml_danbooru 3d is broader.",
        "Semi-realistic anime: treat realistic as soft unless threshold is high or corroborated.",
        "Topaz/AI-enhanced portraits can suppress WD realism tags (see IMG_1796 miss on SwinV2).",
    ]

    return {
        "decision": decision,
        "best_model": best_model,
        "model_ranking": [
            {
                "model": m,
                "score": round(s, 4),
                "best_combo": info["best_combo"],
                "best_realistic_alone": info["best_realistic_alone"],
            }
            for m, s, info in model_rank
        ],
        "go_reasons": go_reasons,
        "nogo_reasons": nogo_reasons,
        "failure_modes": failure_modes,
        "suggested_bucket_name": "real_life",
        "suggested_thresholds_by_model": {
            m: {
                "realistic": (info["best_realistic_alone"] or {}).get("threshold"),
                "combo": (info["best_combo"] or {}).get("threshold"),
                "photorealistic_hip": (
                    (info["hard"].get("photorealistic") or {})
                    .get("suggested_for_high_precision", {})
                    .get("threshold")
                ),
            }
            for m, _s, info in model_rank
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--photos-dir", type=Path, default=None)
    p.add_argument("--anime-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=12, help="Max images per class")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="general_threshold used when reporting filtered top tags (raw always kept)",
    )
    p.add_argument("--include-ml", action="store_true", help="Also score ml_danbooru")
    p.add_argument("--no-download", action="store_true", help="Do not fetch Wikimedia fallbacks")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "scripts" / "out" / "realism_probe_report.json",
    )
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    samples, notes = collect_labeled_sets(
        photos_dir=args.photos_dir,
        anime_dir=args.anime_dir,
        limit=args.limit,
        allow_download=not args.no_download,
    )
    n_photo = sum(1 for s in samples if s["label"] == "photo")
    n_anime = sum(1 for s in samples if s["label"] == "anime")
    print(f"samples photos={n_photo} anime={n_anime}", flush=True)
    for note in notes:
        print(f"  note: {note}", flush=True)
    if not samples:
        print("No samples found.", flush=True)
        return 1

    models = [TAGGER_MODEL_WD_SWINV2, TAGGER_MODEL_WD_EVA02]
    if args.include_ml:
        models.append(TAGGER_MODEL_ML)

    reset_engine()
    engine = InferenceEngine()
    paths = [Path(s["path"]) for s in samples]

    # Warm + raw score per model
    per_model_raw: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        print(f"scoring {model} …", flush=True)
        engine.warm(model)
        if model == TAGGER_MODEL_ML:
            per_model_raw[model] = score_ml_raw(engine, paths)
        else:
            per_model_raw[model] = score_wd_raw(
                engine,
                paths,
                tagger_model=model,
                batch_size=max(1, args.batch_size),
            )

    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        entry: dict[str, Any] = {
            "path": sample["path"],
            "label": sample["label"],
            "source": sample["source"],
            "models": {},
        }
        for model in models:
            raw = per_model_raw[model][i]
            general = raw["general"]
            cands = candidate_scores(general)
            filtered = {
                k: float(v)
                for k, v in general.items()
                if float(v) > args.threshold
            }
            entry["models"][model] = {
                "candidates": cands,
                "rating": {k: round(float(v), 6) for k, v in raw["rating"].items()},
                "top_tags": top_n(filtered if filtered else general, n=15),
                "top_tags_raw_unfiltered": top_n(general, n=10),
            }
        rows.append(entry)

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    analysis: dict[str, Any] = {}
    # Vocab presence check (WD models omit photo_(medium)/3d in v3 selected_tags).
    vocab_notes: list[str] = []
    for model in models:
        if model == TAGGER_MODEL_ML:
            labels = engine._get_ml_labels()
            present = sorted({t for t in PRIMARY_EVIDENCE_TAGS if _normalize_tag(t) in {_normalize_tag(x) for x in labels} or t in labels})
            missing = [t for t in PRIMARY_EVIDENCE_TAGS if t not in present and _normalize_tag(t) not in {_normalize_tag(x) for x in labels}]
        else:
            wd_name = WD_MODEL_NAMES[model]
            names, _, _, _ = engine._get_wd_labels(wd_name)
            name_set = set(names)
            norm_map = {_normalize_tag(n): n for n in names}
            present = []
            missing = []
            for t in PRIMARY_EVIDENCE_TAGS:
                if t in name_set or _normalize_tag(t) in norm_map:
                    present.append(t)
                else:
                    missing.append(t)
        vocab_notes.append(f"{model} evidence present={present} missing={missing}")
        notes.append(vocab_notes[-1])

    for model in models:
        tag_metrics: list[dict[str, Any]] = []
        for tag in PRIMARY_EVIDENCE_TAGS:
            tag_metrics.extend(
                separation_metrics(rows, model=model, tag=tag, thresholds=thresholds)
            )
        combo = combo_metrics(rows, model=model, thresholds=thresholds)
        realistic_alone = single_tag_policy_metrics(
            rows, model=model, tag="realistic", thresholds=thresholds
        )
        analysis[model] = {
            "distributions": summarize_tag_distributions(rows, model),
            "per_tag_metrics": tag_metrics,
            "combo_policy": combo,
            "realistic_alone_policy": realistic_alone,
            "suggested_thresholds": best_thresholds(tag_metrics, min_precision=0.85),
        }

    rec = recommend(analysis, n_photo=n_photo, n_anime=n_anime)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "general_threshold_for_top_tags": args.threshold,
        "models": models,
        "candidate_tags": list(CANDIDATE_TAGS),
        "primary_evidence_tags": list(PRIMARY_EVIDENCE_TAGS),
        "vocab_notes": vocab_notes,
        "notes": notes,
        "counts": {"photos": n_photo, "anime": n_anime, "total": len(rows)},
        "recommendation": rec,
        "analysis": analysis,
        "images": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}", flush=True)
    print(f"Decision: {rec['decision']} best_model={rec['best_model']}", flush=True)
    for line in rec["go_reasons"] + rec["nogo_reasons"]:
        print(f"  - {line}", flush=True)

    # Compact console table
    for model in models:
        print(f"\n=== {model} distributions (mean) ===")
        dist = analysis[model]["distributions"]
        for tag in PRIMARY_EVIDENCE_TAGS:
            p = dist.get(f"photo:{tag}") or {}
            a = dist.get(f"anime:{tag}") or {}
            print(
                f"  {tag:18} photo_mean={p.get('mean')} anime_mean={a.get('mean')} "
                f"photo_max={p.get('max')} anime_max={a.get('max')}"
            )
        print("  best combo rows:")
        for row in analysis[model]["combo_policy"]:
            if row["threshold"] in {0.15, 0.25, 0.35, 0.5}:
                print(
                    f"    thr={row['threshold']}: P={row['precision']} R={row['recall']} "
                    f"F1={row['f1']} fp_rate={row['anime_false_positive_rate']}"
                )
        print("  realistic-alone rows:")
        for row in analysis[model]["realistic_alone_policy"]:
            if row["threshold"] in {0.15, 0.25, 0.35, 0.5}:
                print(
                    f"    thr={row['threshold']}: P={row['precision']} R={row['recall']} "
                    f"F1={row['f1']} fp_rate={row['anime_false_positive_rate']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
