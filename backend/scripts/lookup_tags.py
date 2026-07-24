"""Look up exact tagger-vocabulary spellings for candidate tag names.

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/lookup_tags.py panties nipple poke_ball
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TAGS_CSV = REPO_ROOT / "tags.csv"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"


def load_wd() -> dict[str, int]:
    vocab: dict[str, int] = {}
    for path in HF_HUB.rglob("selected_tags.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                name = (row.get("name") or "").strip().lower()
                if not name:
                    continue
                try:
                    count = int(row.get("count") or 0)
                except ValueError:
                    count = 0
                vocab[name] = max(vocab.get(name, 0), count)
    return vocab


def load_ml() -> set[str]:
    names: set[str] = set()
    if TAGS_CSV.is_file():
        with TAGS_CSV.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                name = (row.get("tag") or "").strip().lower()
                if name:
                    names.add(name)
    return names


def main() -> None:
    wd = load_wd()
    ml = load_ml()
    for needle in sys.argv[1:]:
        key = needle.lower()
        hits = sorted(
            ((t, c) for t, c in wd.items() if key in t),
            key=lambda item: -item[1],
        )
        print(f"=== '{needle}' -> {len(hits)} WD matches ===")
        for tag, count in hits[:35]:
            ml_flag = "ml" if tag in ml else "  "
            print(f"  {count:>9,} [{ml_flag}] {tag}")
        only_ml = sorted(t for t in ml if key in t and t not in wd)
        if only_ml:
            print(f"  (ML-only: {', '.join(only_ml[:20])})")
        print()


if __name__ == "__main__":
    main()
