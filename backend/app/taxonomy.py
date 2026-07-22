"""Single-winner destination funnel for multi-folder classification.

Bucket definitions live in ``data/taxonomy.json``. This module loads them and
scores destinations. Ignore tags are never evidence and never win a folder
alone; they are not vetoes.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent / "data" / "taxonomy.json"


def _normalize_tag_name(value: str) -> str:
    text = value.strip().lower()
    normalized = []
    for ch in text:
        if ch.isalnum():
            normalized.append(ch)
        elif ch in {" ", "-", ".", "/", "_"}:
            normalized.append("_")
    return "".join(normalized).strip("_")


@dataclass(frozen=True)
class EvidenceTag:
    tag: str
    weight: float


@dataclass(frozen=True)
class DestinationBucket:
    """One destination folder and its scoring rules."""

    id: str
    folder: str
    priority: int  # lower wins on score ties
    evidence: tuple[EvidenceTag, ...]
    ignore: frozenset[str] = field(default_factory=frozenset)
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaxonomyConfig:
    buckets: tuple[DestinationBucket, ...]
    soft_alone_weight: float = 0.6
    soft_corroboration_raw: float = 0.15
    by_alias: dict[str, DestinationBucket] = field(default_factory=dict)


_LOCK = threading.Lock()
_LOADED: TaxonomyConfig | None = None
_LOADED_PATH: Path | None = None


def _parse_bucket(raw: dict[str, Any]) -> DestinationBucket:
    evidence_raw = raw.get("evidence") or []
    evidence = tuple(
        EvidenceTag(tag=str(item["tag"]).strip(), weight=float(item["weight"]))
        for item in evidence_raw
        if str(item.get("tag", "")).strip()
    )
    ignore = frozenset(str(t).strip() for t in (raw.get("ignore") or []) if str(t).strip())
    aliases = tuple(str(a).strip() for a in (raw.get("aliases") or []) if str(a).strip())
    bucket_id = str(raw["id"]).strip()
    folder = str(raw.get("folder") or bucket_id).strip()
    priority = int(raw["priority"])
    if not bucket_id or not folder:
        raise ValueError("taxonomy bucket requires non-empty id and folder")
    if not evidence:
        raise ValueError(f"taxonomy bucket {bucket_id!r} requires at least one evidence tag")
    return DestinationBucket(
        id=bucket_id,
        folder=folder,
        priority=priority,
        evidence=evidence,
        ignore=ignore,
        aliases=aliases,
    )


def _bucket_aliases(bucket: DestinationBucket) -> set[str]:
    aliases = {
        _normalize_tag_name(bucket.id),
        _normalize_tag_name(bucket.folder),
    }
    for ev in bucket.evidence:
        if ev.weight >= 1.0:
            aliases.add(_normalize_tag_name(ev.tag))
    for alias in bucket.aliases:
        aliases.add(_normalize_tag_name(alias))
    return aliases


def load_taxonomy(path: Path | None = None) -> TaxonomyConfig:
    """Load and validate taxonomy JSON into an immutable config."""
    taxonomy_path = Path(path) if path is not None else DEFAULT_TAXONOMY_PATH
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("taxonomy root must be an object")
    buckets_raw = payload.get("buckets")
    if not isinstance(buckets_raw, list) or not buckets_raw:
        raise ValueError("taxonomy.buckets must be a non-empty list")

    buckets = tuple(_parse_bucket(item) for item in buckets_raw)
    by_alias: dict[str, DestinationBucket] = {}
    for bucket in buckets:
        for alias in _bucket_aliases(bucket):
            existing = by_alias.get(alias)
            if existing is not None and existing.id != bucket.id:
                raise ValueError(
                    f"taxonomy alias {alias!r} maps to both {existing.id!r} and {bucket.id!r}"
                )
            by_alias.setdefault(alias, bucket)

    return TaxonomyConfig(
        buckets=buckets,
        soft_alone_weight=float(payload.get("soft_alone_weight", 0.6)),
        soft_corroboration_raw=float(payload.get("soft_corroboration_raw", 0.15)),
        by_alias=by_alias,
    )


def get_taxonomy(path: Path | None = None) -> TaxonomyConfig:
    """Return cached default taxonomy, or load from an explicit path (uncached)."""
    global _LOADED, _LOADED_PATH
    if path is not None:
        return load_taxonomy(path)
    with _LOCK:
        if _LOADED is None or _LOADED_PATH != DEFAULT_TAXONOMY_PATH:
            _LOADED = load_taxonomy(DEFAULT_TAXONOMY_PATH)
            _LOADED_PATH = DEFAULT_TAXONOMY_PATH
        return _LOADED


def reload_taxonomy(path: Path | None = None) -> TaxonomyConfig:
    """Force-reload the default cached taxonomy (tests / hot edit)."""
    global _LOADED, _LOADED_PATH
    taxonomy_path = Path(path) if path is not None else DEFAULT_TAXONOMY_PATH
    config = load_taxonomy(taxonomy_path)
    with _LOCK:
        _LOADED = config
        _LOADED_PATH = taxonomy_path
    return config


def __getattr__(name: str) -> Any:
    if name == "BUCKETS":
        return get_taxonomy().buckets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def resolve_taxonomy_folder(
    name: str, taxonomy: TaxonomyConfig | None = None
) -> DestinationBucket | None:
    """Resolve a selected folder/tag name to a taxonomy bucket, if any."""
    text = (name or "").strip()
    if not text:
        return None
    cfg = taxonomy or get_taxonomy()
    return cfg.by_alias.get(_normalize_tag_name(text))


def is_taxonomy_folder(name: str, taxonomy: TaxonomyConfig | None = None) -> bool:
    return resolve_taxonomy_folder(name, taxonomy=taxonomy) is not None


def taxonomy_folder_names(taxonomy: TaxonomyConfig | None = None) -> list[str]:
    cfg = taxonomy or get_taxonomy()
    return [b.folder for b in cfg.buckets]


def _lookup_score(scores: dict[str, float], tag: str) -> float | None:
    if tag in scores:
        return float(scores[tag])
    wanted = _normalize_tag_name(tag)
    best: float | None = None
    for key, value in scores.items():
        if _normalize_tag_name(key) == wanted:
            score = float(value)
            if best is None or score > best:
                best = score
    return best


def score_bucket(
    scores: dict[str, float],
    bucket: DestinationBucket,
    taxonomy: TaxonomyConfig | None = None,
) -> float | None:
    """Evidence-weighted score for one bucket. Ignore tags never contribute."""
    cfg = taxonomy or get_taxonomy()
    ignore = {_normalize_tag_name(t) for t in bucket.ignore}
    contribs: list[tuple[str, float, float, float]] = []
    for ev in bucket.evidence:
        if _normalize_tag_name(ev.tag) in ignore:
            continue
        raw = _lookup_score(scores, ev.tag)
        if raw is None:
            continue
        contribs.append((ev.tag, float(raw) * float(ev.weight), float(ev.weight), float(raw)))
    if not contribs:
        return None
    contribs.sort(key=lambda item: item[1], reverse=True)
    _tag, best_score, best_weight, _raw = contribs[0]
    if best_weight < cfg.soft_alone_weight:
        corroborated = any(
            raw >= cfg.soft_corroboration_raw for _t, _s, _w, raw in contribs[1:]
        )
        if not corroborated:
            return None
    return best_score


def choose_best_destination(
    scores: dict[str, float],
    selected: set[str],
    max_secondary: int = 3,
    taxonomy: TaxonomyConfig | None = None,
) -> tuple[str | None, float | None, list[dict[str, float]]]:
    """Pick a single destination folder from selected folders/tags.

    Taxonomy folders use evidence weights. Non-taxonomy selections keep exact
    tag matching (legacy). Winner is highest score; ties break by bucket
    priority (and name for non-taxonomy).
    """
    cfg = taxonomy or get_taxonomy()
    candidates: list[tuple[str, float, int]] = []
    seen_folders: set[str] = set()

    for sel in selected:
        name = (sel or "").strip()
        if not name:
            continue
        bucket = resolve_taxonomy_folder(name, taxonomy=cfg)
        if bucket is not None:
            folder_key = _normalize_tag_name(bucket.folder)
            if folder_key in seen_folders:
                continue
            seen_folders.add(folder_key)
            bucket_score = score_bucket(scores, bucket, taxonomy=cfg)
            if bucket_score is None:
                continue
            candidates.append((bucket.folder, bucket_score, bucket.priority))
            continue

        # Legacy exact selected-tag match.
        raw = _lookup_score(scores, name)
        if raw is None:
            continue
        folder_key = _normalize_tag_name(name)
        if folder_key in seen_folders:
            continue
        seen_folders.add(folder_key)
        candidates.append((name, float(raw), 10_000))

    if not candidates:
        return None, None, []

    candidates.sort(key=lambda item: (-item[1], item[2], _normalize_tag_name(item[0])))
    primary_folder, primary_score, _priority = candidates[0]
    secondary = [
        {"tag": folder, "score": float(score)}
        for folder, score, _p in candidates[1 : 1 + max_secondary]
    ]
    return primary_folder, float(primary_score), secondary
