"""WD-primary + ML allowlist rescue on needs_review.

Benchmark-driven: ML wins recall on character/act cues (loli, fellatio,
tentacles, monster_girl, …) but has higher Voyeur soft-tag FPR. Never merge
those soft tags from ML into WD scores.
"""

from __future__ import annotations

# Tags where ML recall@0.35 beat WD and FP rates stayed low on the curated suite.
HYBRID_ML_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Character / age
        "loli",
        "shota",
        # Oral
        "fellatio",
        "irrumatio",
        "deepthroat",
        "licking_penis",
        "cooperative_fellatio",
        "after_fellatio",
        # Creampie / nakadashi evidence
        "nakadashi",
        "cum_in_pussy",
        "internal_cumshot",
        "after_vaginal",
        # Tentacle act (not bare "tentacles")
        "tentacle_sex",
        "consensual_tentacles",
        "tentacles_on_male",
        "tentacle_pit",
        "tentacles_under_clothes",
        # Monster-girl catch-all identity
        "monster_girl",
        "monster_boy",
        "slime_girl",
        "slime_(creature)",
        "lamia",
        "harpy",
        "scylla",
        "spider_girl",
        "arthropod_girl",
        "dragon_girl",
        "plant_girl",
        "fish_girl",
        "shark_girl",
        "frog_girl",
        "bird_girl",
        "moth_girl",
        "centaur",
        "mermaid",
        "female_goblin",
        "minotaur",
        "orc",
        "cyclops",
        "oni",
        "traditional_youkai",
        "demon_girl",
        "vampire",
        "werewolf",
        "monsterification",
        # Android catch-all identity
        "android",
        "robot_girl",
        "humanoid_robot",
        "cyborg",
        "mecha_musume",
        "mechanization",
        # Other preferred folders with strong ML recall
        "furry",
        "pokemon_(creature)",
        "bestiality",
        "animal_penis",
        "incest",
        "twincest",
        "netorare",
        "cheating_(relationship)",
        "impregnation",
        "fertilization",
        "ovum",
    }
)

# Never trust ML for these even if someone expands the allowlist later.
HYBRID_ML_DENYLIST: frozenset[str] = frozenset(
    {
        "cleavage",
        "nude",
        "completely_nude",
        "pussy",
        "ass",
        "breasts",
        "large_breasts",
        "huge_breasts",
        "medium_breasts",
        "small_breasts",
        "nipples",
        "areolae",
        "panties",
        "underwear",
        "bra",
        "bikini",
        "swimsuit",
        "lingerie",
        "upskirt",
        "see-through",
        "see_through",
        "cameltoe",
        "underboob",
        "sideboob",
        "topless",
        "bottomless",
        "no_panties",
        "spread_legs",
        "from_below",
        "looking_at_viewer",
    }
)


def should_run_hybrid_ml(
    *,
    enabled: bool,
    tagger_model: str,
    needs_review: bool,
    inference_failed: bool,
) -> bool:
    if not enabled or not needs_review or inference_failed:
        return False
    return str(tagger_model or "").startswith("wd_")


def merge_ml_allowlist_scores(
    wd_scores: dict[str, float],
    ml_scores: dict[str, float],
    allowlist: frozenset[str] | set[str] | None = None,
    denylist: frozenset[str] | set[str] | None = None,
) -> dict[str, float]:
    """Copy WD scores; for allowlisted tags take max(WD, ML). Deny Voyeur soft tags."""
    allow = allowlist if allowlist is not None else HYBRID_ML_ALLOWLIST
    deny = denylist if denylist is not None else HYBRID_ML_DENYLIST
    merged = {str(k): float(v) for k, v in wd_scores.items()}
    for tag, score in ml_scores.items():
        key = str(tag)
        if key in deny or key not in allow:
            continue
        try:
            ml_val = float(score)
        except (TypeError, ValueError):
            continue
        prev = float(merged.get(key, 0.0))
        if ml_val > prev:
            merged[key] = ml_val
    return merged
