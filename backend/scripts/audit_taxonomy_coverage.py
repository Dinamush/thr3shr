"""Audit taxonomy coverage against real run data.

Reports, for the most recent runs:
  * how many items auto-file, land as a low-confidence suggestion, or fall to
    review — using the same noise floor / confidence gate as the API
  * the most common tags on unrouted items (candidate evidence)
  * which of those tags exist in tags.csv (so they are actually predictable)

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/audit_taxonomy_coverage.py [run_id ...]
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.taxonomy import (  # noqa: E402
    choose_best_destination,
    get_taxonomy,
    reload_taxonomy,
)

DB_PATH = Path(__file__).resolve().parents[1] / "app.db"
TAGS_CSV = Path(__file__).resolve().parents[2] / "tags.csv"


def load_csv_tags() -> set[str]:
    names: set[str] = set()
    with TAGS_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = (row[0] or "").strip()
            if value and value.lower() != "name":
                names.add(value.lower())
    return names


def selected_folders(conn: sqlite3.Connection) -> set[str]:
    row = conn.execute("SELECT selected_tags_json FROM settings WHERE id = 1").fetchone()
    if not row or not row[0]:
        return set()
    try:
        payload = json.loads(row[0])
    except json.JSONDecodeError:
        return set()
    return {str(t).strip() for t in payload if str(t).strip()}


def main() -> None:
    reload_taxonomy()
    cfg = get_taxonomy()
    csv_tags = load_csv_tags()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    run_ids = [int(a) for a in sys.argv[1:]]
    if not run_ids:
        run_ids = [
            r[0]
            for r in conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 3").fetchall()
        ]

    selected = selected_folders(conn)
    print(f"selected folders in settings: {sorted(selected) or '(none)'}")
    print(f"taxonomy buckets: {len(cfg.buckets)}")
    print()

    # Score every bucket regardless of user selection so we see full potential.
    all_folders = {b.folder for b in cfg.buckets}

    settings = conn.execute("SELECT confidence_threshold FROM settings WHERE id = 1").fetchone()
    confidence = float(settings[0]) if settings and settings[0] is not None else 0.6
    noise_floor = max(0.15, confidence * 0.5)
    print(f"confidence threshold {confidence:.2f}, noise floor {noise_floor:.2f}")
    print()

    unrouted_tags: Counter[str] = Counter()
    routed_counts: Counter[str] = Counter()
    auto_counts: Counter[str] = Counter()
    weak_counts: Counter[str] = Counter()
    total = 0
    unrouted = 0
    below_floor = 0
    unrouted_samples: list[tuple[int, list[tuple[str, float]]]] = []

    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"SELECT id, run_id, full_scores_json FROM items WHERE run_id IN ({placeholders})",
        run_ids,
    ).fetchall()

    for row in rows:
        raw = row["full_scores_json"]
        if not raw:
            continue
        try:
            scores = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(scores, dict) or not scores:
            continue
        total += 1
        folder, score, _secondary = choose_best_destination(scores, set(all_folders))
        if folder and score is not None and score >= noise_floor:
            routed_counts[folder] += 1
            if score >= confidence:
                auto_counts[folder] += 1
            else:
                weak_counts[folder] += 1
            continue
        if folder:
            below_floor += 1
        unrouted += 1
        for tag, value in scores.items():
            if float(value) >= 0.5:
                unrouted_tags[tag.lower()] += 1
        if len(unrouted_samples) < 12:
            top = sorted(scores.items(), key=lambda kv: -float(kv[1]))[:14]
            unrouted_samples.append((row["id"], [(k, round(float(v), 2)) for k, v in top]))

    auto_total = sum(auto_counts.values())
    weak_total = sum(weak_counts.values())
    print(f"runs analysed: {run_ids}")
    print(f"items with scores: {total}")
    print(f"auto-filed (>= {confidence:.2f}): {auto_total}")
    print(f"low-confidence suggestion: {weak_total}")
    print(f"review: {unrouted}  (of which {below_floor} scored below the noise floor)")
    print()
    print("--- routed by folder (auto-filed / suggestion) ---")
    for folder, count in routed_counts.most_common():
        print(f"  {count:5d}  ({auto_counts[folder]:4d} / {weak_counts[folder]:4d})  {folder}")
    print()

    known_evidence = set()
    for bucket in cfg.buckets:
        for ev in bucket.evidence:
            known_evidence.add(ev.tag.lower())
        for ev in bucket.gated_evidence:
            known_evidence.add(ev.tag.lower())

    print("--- top tags on UNROUTED items (not already evidence) ---")
    shown = 0
    for tag, count in unrouted_tags.most_common(400):
        if tag in known_evidence:
            continue
        in_csv = "csv" if tag in csv_tags else "   "
        pct = 100.0 * count / max(unrouted, 1)
        print(f"  {count:5d} ({pct:5.1f}%) [{in_csv}] {tag}")
        shown += 1
        if shown >= 120:
            break
    print()

    print("--- sample unrouted items ---")
    for item_id, top in unrouted_samples:
        print(f"  item {item_id}: {top}")


if __name__ == "__main__":
    main()
