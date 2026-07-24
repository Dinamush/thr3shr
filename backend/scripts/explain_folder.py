"""Explain why run items land in a folder: which evidence tag wins, how often.

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/explain_folder.py Voyeur/see_through
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.taxonomy import (  # noqa: E402
    _lookup_score,
    choose_best_destination,
    get_taxonomy,
    reload_taxonomy,
    resolve_taxonomy_folder,
)

DB_PATH = Path(__file__).resolve().parents[1] / "app.db"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: explain_folder.py <folder> [run_id ...]")
    target = sys.argv[1]

    reload_taxonomy()
    cfg = get_taxonomy()
    bucket = resolve_taxonomy_folder(target, taxonomy=cfg)
    if bucket is None:
        raise SystemExit(f"no taxonomy bucket named {target!r}")
    all_folders = {b.folder for b in cfg.buckets}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    run_ids = [int(a) for a in sys.argv[2:]]
    if not run_ids:
        run_ids = [
            r[0]
            for r in conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 3").fetchall()
        ]

    winners: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    scores_by_winner: dict[str, list[float]] = {}

    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"SELECT id, full_scores_json FROM items WHERE run_id IN ({placeholders})",
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
        folder, score, _sec = choose_best_destination(scores, set(all_folders))
        if folder != bucket.folder:
            continue

        best_tag, best_val = None, 0.0
        for ev in list(bucket.evidence) + list(bucket.gated_evidence):
            hit = _lookup_score(scores, ev.tag)
            if hit is None:
                continue
            weighted = float(hit) * ev.weight
            if weighted > best_val:
                best_tag, best_val = f"{ev.tag} (raw {float(hit):.2f})", weighted
        key = best_tag.split(" (")[0] if best_tag else "(none)"
        winners[key] += 1
        scores_by_winner.setdefault(key, []).append(float(score or 0.0))
        bucket_samples = samples.setdefault(key, [])
        if len(bucket_samples) < 2:
            top = sorted(scores.items(), key=lambda kv: -float(kv[1]))[:10]
            bucket_samples.append(
                f"item {row['id']}: " + ", ".join(f"{k}:{float(v):.2f}" for k, v in top)
            )

    total = sum(winners.values())
    print(f"{total} items route to {bucket.folder} across runs {run_ids}")
    print()
    for tag, count in winners.most_common(30):
        vals = scores_by_winner.get(tag, [0.0])
        avg = sum(vals) / len(vals)
        print(f"  {count:5d}  avg score {avg:.2f}  won by {tag}")
        for sample in samples.get(tag, []):
            print(f"           {sample}")


if __name__ == "__main__":
    main()
