"""Diff two taxonomy revisions over recorded run scores.

Shows which folder each item moves from and to, so a taxonomy edit can be
reviewed for regressions (content leaving a specific folder for a catch-all)
before it is applied to a real library.

The baseline is either a git revision (``HEAD``, a branch, a SHA) or a path to
a taxonomy JSON file.

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/diff_taxonomy_routing.py HEAD
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.taxonomy import (  # noqa: E402
    DEFAULT_TAXONOMY_PATH,
    choose_best_destination,
    load_taxonomy,
)

DB_PATH = Path(__file__).resolve().parents[1] / "app.db"
MAX_EXAMPLES = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_PATH = "backend/app/data/taxonomy.json"


def baseline_path(ref: str) -> Path:
    """Resolve a taxonomy baseline given a file path or a git revision."""
    candidate = Path(ref)
    if candidate.is_file():
        return candidate
    blob = subprocess.run(
        ["git", "show", f"{ref}:{TRACKED_PATH}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "taxonomy_baseline.json"
    tmp.write_bytes(blob)
    return tmp


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: diff_taxonomy_routing.py <git-ref|path> [run_id ...]")

    before = load_taxonomy(baseline_path(sys.argv[1]))
    after = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    before_folders = {b.folder for b in before.buckets}
    after_folders = {b.folder for b in after.buckets}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    run_ids = [int(a) for a in sys.argv[2:]]
    if not run_ids:
        run_ids = [
            r[0]
            for r in conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 3").fetchall()
        ]

    moves: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    total = 0

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
        total += 1
        old = choose_best_destination(scores, set(before_folders), taxonomy=before)[0]
        new = choose_best_destination(scores, set(after_folders), taxonomy=after)[0]
        key = (old or "(review)", new or "(review)")
        moves[key] += 1
        if old != new and len(examples[key]) < MAX_EXAMPLES:
            top = sorted(scores.items(), key=lambda kv: -float(kv[1]))[:8]
            examples[key].append(", ".join(f"{k}:{float(v):.2f}" for k, v in top))

    unchanged = sum(count for (old, new), count in moves.items() if old == new)
    print(f"runs {run_ids}: {total} scored items")
    print(f"unchanged: {unchanged}   changed: {total - unchanged}")
    print()

    print("--- items LEAVING each folder (possible regressions) ---")
    losses: Counter[str] = Counter()
    for (old, new), count in moves.items():
        if old != new and old != "(review)":
            losses[old] += count
    for old, count in losses.most_common():
        print(f"  {old} loses {count}:")
        for (o, n), c in sorted(moves.items(), key=lambda kv: -kv[1]):
            if o != old or o == n:
                continue
            print(f"      -> {n:22s} {c}")
            for sample in examples[(o, n)]:
                print(f"           {sample}")
    print()

    print("--- newly routed (was review) ---")
    for (old, new), count in sorted(moves.items(), key=lambda kv: -kv[1]):
        if old == "(review)" and new != "(review)":
            print(f"  {count:5d}  -> {new}")
    print()

    print("--- newly unrouted (now review) ---")
    for (old, new), count in sorted(moves.items(), key=lambda kv: -kv[1]):
        if new == "(review)" and old != "(review)":
            print(f"  {count:5d}  {old} -> review")
            for sample in examples[(old, new)]:
                print(f"           {sample}")


if __name__ == "__main__":
    main()
