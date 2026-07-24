"""Explore the tagger vocabulary and validate taxonomy evidence against it.

The active taggers (WD SwinV2 / EVA02) publish their own ``selected_tags.csv``;
that vocabulary - not the repo's ``tags.csv`` - decides which evidence tags can
ever fire. This script lists candidate tags per keyword family and flags
taxonomy evidence that no tagger can emit.

Usage (from backend/):
  ../.venv/Scripts/python.exe scripts/probe_tag_families.py --validate
  ../.venv/Scripts/python.exe scripts/probe_tag_families.py swimsuit legwear
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[2]
TAGS_CSV = REPO_ROOT / "tags.csv"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

FAMILIES: dict[str, tuple[str, ...]] = {
    "swimsuit": ("swimsuit", "bikini", "one-piece", "school_swimsuit", "rash_guard"),
    "legwear": (
        "pantyhose", "thighhigh", "fishnet", "garter", "zettai", "legwear",
        "bodystocking", "kneehigh",
    ),
    "lingerie": (
        "lingerie", "bra", "underwear", "babydoll", "negligee", "camisole",
        "chemise", "corset",
    ),
    "nude": ("nude", "topless", "bottomless", "naked", "nipple", "areola"),
    "leotard": ("leotard", "bodysuit", "playboy", "bunnysuit", "gym_uniform", "unitard"),
    "undress": (
        "clothes_lift", "shirt_lift", "clothes_pull", "open_clothes", "undressing",
        "unbutton", "unzip", "strap_slip", "wardrobe", "unworn", "clothing_aside",
    ),
    "seethrough": ("see-through", "sheer", "wet_clothes", "wet_shirt", "wet_swimsuit"),
    "public": (
        "exhibitionism", "public", "flashing", "voyeur", "peeping", "changing_room",
        "locker", "onsen", "bath", "shower", "towel", "indecency",
    ),
    "sex": (
        "sex", "vaginal", "missionary", "cowgirl", "doggystyle", "girl_on_top",
        "straddling", "penetration", "insertion", "grinding", "humping",
    ),
    "oral": ("fellatio", "irrumatio", "deepthroat", "cunnilingus", "oral", "licking"),
    "paizuri": ("paizuri", "breast_press", "titjob", "breast_sex"),
    "handjob": ("handjob", "masturbat", "penis_grab", "stroking", "fingering"),
    "footjob": ("footjob", "foot", "feet", "sole", "toe"),
    "bdsm": (
        "bdsm", "bondage", "bound", "rope", "shibari", "chain", "handcuff",
        "collar", "leash", "gag", "blindfold", "restrain", "spanking", "whip",
    ),
    "yaoi": ("yaoi", "bara", "male_focus", "2boys", "multiple_boys", "crossdress"),
    "yuri": ("yuri", "tribadism", "2girls", "multiple_girls", "scissoring"),
    "futanari": ("futanari", "newhalf", "dickgirl", "futa"),
    "tentacle": ("tentacle", "slime", "vore"),
    "pregnancy": ("pregnan", "lactation", "breast_milk", "birth", "womb"),
    "toys": (
        "sex_toy", "vibrator", "dildo", "butt_plug", "anal_beads", "onahole",
        "sex_machine",
    ),
    "censor": ("censor", "uncensored", "mosaic", "steam", "convenient"),
    "pose": (
        "spread_legs", "presenting", "arched_back", "bent_over", "all_fours",
        "top-down", "on_back", "on_stomach", "kneeling", "squatting", "m_legs",
        "legs_up", "leg_lift", "cameltoe", "suggestive",
    ),
    "loli_shota": ("loli", "shota", "child", "young", "flat_chest", "petite"),
    "furry": ("furry", "kemono", "anthro", "fur", "snout", "paw", "digitigrade"),
    "monster_girl": (
        "monster_girl", "lamia", "harpy", "slime_girl", "mermaid", "centaur",
        "succubus", "demon", "elf", "orc", "goblin", "oni", "vampire",
    ),
    "real": ("photo", "realistic", "3d", "render", "cosplay"),
    "comic": ("comic", "speech_bubble", "text", "subtitled", "translated", "4koma"),
}


def load_wd_vocab() -> dict[str, int]:
    """Tag -> danbooru post count, merged across cached WD taggers."""
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


def load_ml_vocab() -> set[str]:
    names: set[str] = set()
    if not TAGS_CSV.is_file():
        return names
    with TAGS_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("tag") or "").strip().lower()
            if name:
                names.add(name)
    return names


def validate(wd: dict[str, int], ml: set[str]) -> None:
    # Scoring matches tags through _normalize_tag_name, so "one-piece_swimsuit"
    # and "one_piece_swimsuit" are the same tag. Compare on normalized form.
    from app.taxonomy import _normalize_tag_name, get_taxonomy, reload_taxonomy

    known = {_normalize_tag_name(t) for t in wd} | {_normalize_tag_name(t) for t in ml}

    reload_taxonomy()
    cfg = get_taxonomy()
    print("=== taxonomy evidence no tagger can ever emit ===")
    dead = 0
    for bucket in cfg.buckets:
        for ev in list(bucket.evidence) + list(bucket.gated_evidence):
            if _normalize_tag_name(ev.tag) in known:
                continue
            print(f"  {bucket.folder:22s} {ev.tag}  (weight {ev.weight})")
            dead += 1
    print(f"  total dead evidence: {dead}")
    print()

    print("=== duplicate evidence within a bucket (same normalized tag) ===")
    for bucket in cfg.buckets:
        seen_norm: dict[str, str] = {}
        for ev in list(bucket.evidence) + list(bucket.gated_evidence):
            norm = _normalize_tag_name(ev.tag)
            if norm in seen_norm:
                print(f"  {bucket.folder:22s} {seen_norm[norm]} == {ev.tag}")
            seen_norm[norm] = ev.tag
    print()

    print("=== veto / gate tags no tagger can ever emit ===")
    reported: set[str] = set()
    for bucket in cfg.buckets:
        for tag in sorted(set(bucket.veto) | set(bucket.gate_tags)):
            norm = _normalize_tag_name(tag)
            if norm in reported or norm in known:
                continue
            reported.add(norm)
            print(f"  {tag}")
    print()


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--validate"]
    wd = load_wd_vocab()
    ml = load_ml_vocab()
    print(f"WD vocabulary: {len(wd)} tags   ML-danbooru vocabulary: {len(ml)} tags")
    print()

    if "--validate" in sys.argv:
        validate(wd, ml)
        if not args:
            return

    for family in args or sorted(FAMILIES):
        needles = FAMILIES.get(family)
        if not needles:
            print(f"!! unknown family {family}")
            continue
        hits = [(t, c) for t, c in wd.items() if any(n in t for n in needles)]
        hits.sort(key=lambda item: -item[1])
        print(f"=== {family} ({len(hits)} WD tags) ===")
        for tag, count in hits[:60]:
            print(f"  {count:>9,}  {tag}")
        print()


if __name__ == "__main__":
    main()
