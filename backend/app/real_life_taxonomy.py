"""Isolated real-life adult taxonomy — never mixed with anime taxonomy.json."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent / "data" / "real_life_taxonomy.json"
REAL_LIFE_ROOT_FOLDER = "Real Life"


def _normalize(value: str) -> str:
    text = value.strip().lower()
    normalized = []
    for ch in text:
        if ch.isalnum():
            normalized.append(ch)
        elif ch in {" ", "-", ".", "/", "_"}:
            normalized.append("_")
    return "".join(normalized).strip("_")


@dataclass(frozen=True)
class RealLifeBucket:
    id: str
    folder: str
    priority: int
    role: str  # act | theme | contextual | sensitive
    aliases: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RealLifeTaxonomy:
    buckets: tuple[RealLifeBucket, ...]
    accept_threshold: float = 0.45
    secondary_threshold: float = 0.35
    contextual_threshold: float = 0.7
    sensitive_threshold: float = 0.55
    root_folder: str = REAL_LIFE_ROOT_FOLDER
    sensitive_tags: frozenset[str] = field(default_factory=frozenset)
    contextual_tags: frozenset[str] = field(default_factory=frozenset)
    by_alias: dict[str, RealLifeBucket] = field(default_factory=dict)

    @property
    def closed_vocabulary(self) -> frozenset[str]:
        labels: set[str] = set()
        for bucket in self.buckets:
            labels.add(bucket.folder)
            labels.update(bucket.aliases)
            labels.update(bucket.evidence)
        return frozenset(labels)


_LOCK = threading.Lock()
_LOADED: RealLifeTaxonomy | None = None
_LOADED_PATH: Path | None = None


def _parse_bucket(raw: dict[str, Any]) -> RealLifeBucket:
    return RealLifeBucket(
        id=str(raw.get("id") or raw.get("folder") or "").strip(),
        folder=str(raw.get("folder") or "").strip(),
        priority=int(raw.get("priority") or 100),
        role=str(raw.get("role") or "act").strip().lower(),
        aliases=tuple(str(a).strip() for a in (raw.get("aliases") or []) if str(a).strip()),
        evidence=tuple(str(t).strip() for t in (raw.get("evidence") or []) if str(t).strip()),
    )


def load_real_life_taxonomy(path: Path | None = None) -> RealLifeTaxonomy:
    taxonomy_path = Path(path) if path is not None else DEFAULT_TAXONOMY_PATH
    raw = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    buckets = tuple(_parse_bucket(item) for item in (raw.get("buckets") or []))
    by_alias: dict[str, RealLifeBucket] = {}
    # Prefer exact folder/id matches before shared evidence aliases (e.g. blowjob
    # is both its own folder and oral evidence).
    for bucket in buckets:
        for key in (bucket.folder, bucket.id):
            norm = _normalize(key)
            if norm and norm not in by_alias:
                by_alias[norm] = bucket
    for bucket in buckets:
        for key in (*bucket.aliases, *bucket.evidence):
            norm = _normalize(key)
            if norm and norm not in by_alias:
                by_alias[norm] = bucket
    sensitive = frozenset(
        str(t).strip() for t in (raw.get("sensitive_tags") or []) if str(t).strip()
    )
    contextual = frozenset(
        str(t).strip() for t in (raw.get("contextual_tags") or []) if str(t).strip()
    )
    return RealLifeTaxonomy(
        buckets=buckets,
        accept_threshold=float(raw.get("accept_threshold") or 0.45),
        secondary_threshold=float(raw.get("secondary_threshold") or 0.35),
        contextual_threshold=float(raw.get("contextual_threshold") or 0.7),
        sensitive_threshold=float(raw.get("sensitive_threshold") or 0.55),
        root_folder=str(raw.get("root_folder") or REAL_LIFE_ROOT_FOLDER).strip()
        or REAL_LIFE_ROOT_FOLDER,
        sensitive_tags=sensitive,
        contextual_tags=contextual,
        by_alias=by_alias,
    )


def get_real_life_taxonomy(path: Path | None = None) -> RealLifeTaxonomy:
    global _LOADED, _LOADED_PATH
    taxonomy_path = Path(path) if path is not None else DEFAULT_TAXONOMY_PATH
    with _LOCK:
        if _LOADED is None or _LOADED_PATH != taxonomy_path:
            _LOADED = load_real_life_taxonomy(taxonomy_path)
            _LOADED_PATH = taxonomy_path
        return _LOADED


def reload_real_life_taxonomy(path: Path | None = None) -> RealLifeTaxonomy:
    global _LOADED, _LOADED_PATH
    with _LOCK:
        _LOADED = None
        _LOADED_PATH = None
    return get_real_life_taxonomy(path)


def resolve_real_life_folder(name: str) -> RealLifeBucket | None:
    if not name or not str(name).strip():
        return None
    tax = get_real_life_taxonomy()
    return tax.by_alias.get(_normalize(name))


def real_life_folder_names() -> list[str]:
    return [b.folder for b in get_real_life_taxonomy().buckets]


def is_sensitive_tag(tag: str, tax: RealLifeTaxonomy | None = None) -> bool:
    taxonomy = tax or get_real_life_taxonomy()
    bucket = taxonomy.by_alias.get(_normalize(tag))
    if bucket is None:
        return False
    if bucket.role == "sensitive":
        return True
    return bucket.folder in taxonomy.sensitive_tags or any(
        a in taxonomy.sensitive_tags for a in bucket.aliases
    )


def is_contextual_tag(tag: str, tax: RealLifeTaxonomy | None = None) -> bool:
    taxonomy = tax or get_real_life_taxonomy()
    bucket = taxonomy.by_alias.get(_normalize(tag))
    if bucket is None:
        return False
    if bucket.role == "contextual":
        return True
    return bucket.folder in taxonomy.contextual_tags


def real_life_destination(categories_root: Path, folder: str) -> Path:
    """categories_root / Real Life / <primary>."""
    tax = get_real_life_taxonomy()
    bucket = resolve_real_life_folder(folder)
    leaf = bucket.folder if bucket is not None else str(folder).strip()
    # Keep nested segments safe.
    parts = [p for p in leaf.replace("\\", "/").split("/") if p.strip()]
    safe = ["".join(c if c.isalnum() or c in "-_ ." else "_" for c in part).strip() for part in parts]
    safe = [p for p in safe if p]
    if not safe:
        safe = ["unsorted"]
    return Path(categories_root) / tax.root_folder / Path(*safe)


def canonicalize_score_key(tag: str, tax: RealLifeTaxonomy | None = None) -> str | None:
    """Map any alias/evidence label onto the destination folder name."""
    taxonomy = tax or get_real_life_taxonomy()
    bucket = taxonomy.by_alias.get(_normalize(tag))
    return bucket.folder if bucket is not None else None


def fuse_real_life_scores(
    raw_scores: dict[str, float],
    *,
    selected_folders: set[str] | None = None,
    tax: RealLifeTaxonomy | None = None,
) -> tuple[str | None, float | None, list[dict[str, float]], dict[str, float], list[str]]:
    """Return (primary, primary_score, secondary, folder_scores, review_flags).

    Sensitive tags are never chosen as primary. Contextual tags need a higher
    threshold. All real-life results carry a calibration review flag by default.
    """
    taxonomy = tax or get_real_life_taxonomy()
    folder_scores: dict[str, float] = {}
    for key, score in (raw_scores or {}).items():
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        folder = canonicalize_score_key(str(key), taxonomy)
        if folder is None:
            continue
        if selected_folders and folder not in selected_folders:
            # Always keep sensitive/contextual suggestions visible even if not selected.
            if not (is_sensitive_tag(folder, taxonomy) or is_contextual_tag(folder, taxonomy)):
                continue
        folder_scores[folder] = max(folder_scores.get(folder, 0.0), value)

    def _rank_key(item: tuple[str, float]) -> tuple[float, int, str]:
        folder, score = item
        bucket = taxonomy.by_alias.get(_normalize(folder))
        priority = bucket.priority if bucket is not None else 999
        return (-score, priority, folder.lower())

    ranked = sorted(folder_scores.items(), key=_rank_key)

    review_flags: list[str] = ["calibration_review"]
    for folder, score in ranked:
        if is_sensitive_tag(folder, taxonomy) and score >= taxonomy.sensitive_threshold:
            review_flags.append(f"sensitive:{folder}")
        if is_contextual_tag(folder, taxonomy):
            if score < taxonomy.contextual_threshold:
                review_flags.append(f"contextual_low:{folder}")
            else:
                review_flags.append(f"contextual:{folder}")

    primary: str | None = None
    primary_score: float | None = None
    for folder, score in ranked:
        if is_sensitive_tag(folder, taxonomy):
            continue
        if is_contextual_tag(folder, taxonomy) and score < taxonomy.contextual_threshold:
            continue
        if score < taxonomy.accept_threshold:
            continue
        primary = folder
        primary_score = score
        break

    secondary: list[dict[str, float]] = []
    for folder, score in ranked:
        if primary and folder == primary:
            continue
        if score < taxonomy.secondary_threshold:
            continue
        if is_sensitive_tag(folder, taxonomy) and score < taxonomy.sensitive_threshold:
            continue
        if is_contextual_tag(folder, taxonomy) and score < taxonomy.contextual_threshold:
            continue
        secondary.append({"tag": folder, "score": round(float(score), 4)})

    if primary is None:
        review_flags.append("no_confident_primary")

    return primary, primary_score, secondary, folder_scores, review_flags


def parse_closed_vocabulary_tags(
    payload: Any,
    *,
    tax: RealLifeTaxonomy | None = None,
) -> dict[str, float]:
    """Parse VLM JSON / free-form tag lists into folder-canonical scores."""
    taxonomy = tax or get_real_life_taxonomy()
    scores: dict[str, float] = {}

    def _add(label: Any, score: float = 0.75) -> None:
        if label is None:
            return
        folder = canonicalize_score_key(str(label), taxonomy)
        if folder is None:
            return
        scores[folder] = max(scores.get(folder, 0.0), float(score))

    if payload is None:
        return scores
    if isinstance(payload, dict):
        # Preferred schema: {"tags":[{"tag":"creampie","score":0.9}, ...], "primary": "..."}
        tags = payload.get("tags")
        if isinstance(tags, list):
            for item in tags:
                if isinstance(item, dict):
                    _add(item.get("tag") or item.get("name"), float(item.get("score") or 0.75))
                else:
                    _add(item)
        for key in ("primary", "primary_tag", "main_tag"):
            if payload.get(key):
                _add(payload.get(key), float(payload.get("primary_score") or 0.85))
        for key in ("secondary", "secondary_tags", "categories"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _add(item.get("tag") or item.get("name"), float(item.get("score") or 0.65))
                    else:
                        _add(item, 0.65)
        # Flat {tag: score} map
        for key, value in payload.items():
            if key in {
                "tags",
                "primary",
                "primary_tag",
                "main_tag",
                "primary_score",
                "secondary",
                "secondary_tags",
                "categories",
                "notes",
                "confidence",
            }:
                continue
            try:
                _add(key, float(value))
            except (TypeError, ValueError):
                continue
        return scores
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                _add(item.get("tag") or item.get("name"), float(item.get("score") or 0.75))
            else:
                _add(item)
        return scores
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return scores
        try:
            return parse_closed_vocabulary_tags(json.loads(text), tax=taxonomy)
        except json.JSONDecodeError:
            for piece in text.replace(",", " ").split():
                _add(piece.strip("[]\"'"), 0.6)
            return scores
    return scores
