import csv
import json
from pathlib import Path

import pytest

from app.services import discover_tag_folders
from app.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    _normalize_tag_name,
    choose_best_destination,
    load_taxonomy,
    resolve_taxonomy_folder,
    score_bucket,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_taxonomy_json_loads() -> None:
    cfg = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    assert DEFAULT_TAXONOMY_PATH.is_file()
    assert len(cfg.buckets) >= 10
    assert resolve_taxonomy_folder("Pokemon", taxonomy=cfg).folder == "Pokemon"
    assert resolve_taxonomy_folder("NTR", taxonomy=cfg).folder == "NTR"
    assert resolve_taxonomy_folder("nakadashi", taxonomy=cfg).folder == "nakadashi"


def _tagger_vocabulary() -> set[str]:
    """Normalized tag names any installed tagger can emit."""
    names: set[str] = set()
    tags_csv = REPO_ROOT / "tags.csv"
    if tags_csv.is_file():
        with tags_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                tag = (row.get("tag") or "").strip()
                if tag:
                    names.add(_normalize_tag_name(tag))
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hub.is_dir():
        for path in hub.rglob("selected_tags.csv"):
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    name = (row.get("name") or "").strip()
                    if name:
                        names.add(_normalize_tag_name(name))
    return names


def test_every_evidence_tag_exists_in_a_tagger_vocabulary() -> None:
    """A typo'd evidence tag silently never fires, so fail loudly on one."""
    vocabulary = _tagger_vocabulary()
    if len(vocabulary) < 1000:
        pytest.skip("no tagger vocabulary available to validate against")

    cfg = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    unknown = [
        f"{bucket.folder}:{ev.tag}"
        for bucket in cfg.buckets
        for ev in list(bucket.evidence) + list(bucket.gated_evidence)
        if _normalize_tag_name(ev.tag) not in vocabulary
    ]
    assert not unknown, f"evidence tags no tagger can emit: {unknown}"


def test_no_duplicate_evidence_within_a_bucket() -> None:
    cfg = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    for bucket in cfg.buckets:
        seen: set[str] = set()
        for ev in list(bucket.evidence) + list(bucket.gated_evidence):
            norm = _normalize_tag_name(ev.tag)
            assert norm not in seen, f"{bucket.folder} lists {ev.tag} twice"
            seen.add(norm)


def test_custom_taxonomy_json_is_used(tmp_path: Path) -> None:
    path = tmp_path / "taxonomy.json"
    path.write_text(
        json.dumps(
            {
                "soft_alone_weight": 0.6,
                "soft_corroboration_raw": 0.15,
                "buckets": [
                    {
                        "id": "demo",
                        "folder": "DemoFolder",
                        "priority": 1,
                        "aliases": ["demo_alias"],
                        "ignore": ["noise_tag"],
                        "evidence": [
                            {"tag": "demo_signal", "weight": 1.0},
                            {"tag": "demo_soft", "weight": 0.5},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_taxonomy(path)
    folder, score, _ = choose_best_destination(
        {"demo_signal": 0.8, "noise_tag": 0.99},
        {"demo_alias"},
        taxonomy=cfg,
    )
    assert folder == "DemoFolder"
    assert score == 0.8
    assert choose_best_destination({"noise_tag": 0.99}, {"DemoFolder"}, taxonomy=cfg)[0] is None


def test_resolve_pokemon_folder_aliases() -> None:
    assert resolve_taxonomy_folder("Pokemon").folder == "Pokemon"
    assert resolve_taxonomy_folder("pokemon").folder == "Pokemon"
    assert resolve_taxonomy_folder("pokemon_(creature)").folder == "Pokemon"


def test_loli_ignore_not_evidence_and_not_veto() -> None:
    selected = {"loli", "shota", "furry"}
    # Fashion alone must not route to loli.
    folder, score, _ = choose_best_destination(
        {"lolita_fashion": 0.99, "gothic_lolita": 0.98, "flat_chest": 0.95},
        selected,
    )
    assert folder is None

    # Co-occurring ignore tags must not veto a real loli hit.
    folder, score, _ = choose_best_destination(
        {"loli": 0.91, "flat_chest": 0.95, "petite": 0.9, "lolita_fashion": 0.8},
        selected,
    )
    assert folder == "loli"
    assert score == 0.91


def test_shota_ignores_common_male_false_positives() -> None:
    selected = {"shota", "loli"}
    folder, score, _ = choose_best_destination(
        {"1boy": 0.99, "male_focus": 0.98, "child": 0.9, "otoko_no_ko": 0.88},
        selected,
    )
    assert folder is None

    folder, score, _ = choose_best_destination(
        {"shota": 0.87, "1boy": 0.99, "child": 0.8},
        selected,
    )
    assert folder == "shota"
    assert score == 0.87


def test_fellatio_implication_children_and_ignore_oral() -> None:
    selected = {"fellatio", "loli"}
    folder, score, _ = choose_best_destination({"oral": 0.99, "cunnilingus": 0.9}, selected)
    assert folder is None

    folder, score, _ = choose_best_destination({"irrumatio": 0.9}, selected)
    assert folder == "fellatio"
    assert abs(score - 0.9 * 0.95) < 1e-9


def test_fertilization_ignores_pregnant_and_creampie_alone() -> None:
    selected = {"fertilization", "nakadashi", "fellatio"}
    folder, score, _ = choose_best_destination({"pregnant": 0.99}, selected)
    assert folder is None

    # Creampie tags no longer count as fertilization evidence.
    folder, score, _ = choose_best_destination({"cum_in_pussy": 0.99}, selected)
    assert folder == "nakadashi"

    folder, score, _ = choose_best_destination(
        {"fertilization": 0.72, "pregnant": 0.99, "cum_in_pussy": 0.95},
        selected,
    )
    assert folder == "fertilization"
    assert score == 0.72


def test_impregnation_alias_routes_to_fertilization() -> None:
    folder, score, _ = choose_best_destination(
        {"impregnation": 0.88},
        {"impregnation"},
    )
    assert folder == "fertilization"
    assert score == 0.88


def test_pregnancy_test_routes_to_fertilization() -> None:
    selected = {"fertilization", "nakadashi", "sex", "Voyeur"}
    folder, score, _ = choose_best_destination({"pregnancy_test": 0.9}, selected)
    assert folder == "fertilization"
    assert abs(score - 0.9 * 0.95) < 1e-9

    # Bare pregnant still does not count; pregnancy_test does.
    assert choose_best_destination({"pregnant": 0.99}, selected)[0] is None


def test_ntr_requires_netorare_or_cheating() -> None:
    selected = {"NTR", "incest"}
    folder, score, _ = choose_best_destination(
        {"voyeurism": 0.99, "caught": 0.9, "rape": 0.95},
        selected,
    )
    assert folder is None

    folder, score, _ = choose_best_destination({"netorare": 0.8, "caught": 0.99}, selected)
    assert folder == "NTR"
    assert score == 0.8

    folder, score, _ = choose_best_destination(
        {"cheating_(relationship)": 0.9, "caught": 0.99},
        selected,
    )
    assert folder == "NTR"
    assert abs(score - 0.9 * 0.85) < 1e-9


def test_mesugaki_and_onee_shota_route() -> None:
    folder, score, _ = choose_best_destination({"mesugaki": 0.88}, {"loli", "shota"})
    assert folder == "loli"
    assert abs(score - 0.88 * 0.8) < 1e-9

    folder, score, _ = choose_best_destination({"onee-shota": 0.91}, {"loli", "shota"})
    assert folder == "shota"
    assert abs(score - 0.91 * 0.9) < 1e-9


def test_incest_ignores_siblings_alone() -> None:
    selected = {"incest", "loli"}
    folder, score, _ = choose_best_destination({"siblings": 0.99}, selected)
    assert folder is None

    folder, score, _ = choose_best_destination(
        {"incest": 0.77, "siblings": 0.99, "brother_and_sister": 0.9},
        selected,
    )
    assert folder == "incest"
    assert score == 0.77


def test_monster_girl_ignores_parts_alone() -> None:
    selected = {"monster_girl", "furry"}
    folder, score, _ = choose_best_destination(
        {"horns": 0.99, "wings": 0.98, "tail": 0.97, "pointy_ears": 0.96},
        selected,
    )
    assert folder is None

    folder, score, _ = choose_best_destination({"slime_girl": 0.84}, selected)
    assert folder == "monster_girl"
    assert abs(score - 0.84 * 0.95) < 1e-9


def test_monster_girl_catch_all_species() -> None:
    """Monster Musume / classic species tags should all land in the catch-all."""
    selected = {"monster_girl", "Voyeur", "furry", "sex"}
    cases = [
        ("lamia", 0.9),
        ("harpy", 0.88),
        ("scylla", 0.87),
        ("spider_girl", 0.86),
        ("dragon_girl", 0.85),
        ("plant_girl", 0.84),
        ("fish_girl", 0.83),
        ("shark_girl", 0.82),
        ("frog_girl", 0.81),
        ("bird_girl", 0.8),
        ("moth_girl", 0.8),
        ("arthropod_girl", 0.79),
        ("centaur", 0.78),
        ("mermaid", 0.77),
        ("cyclops", 0.76),
        ("traditional_youkai", 0.85),
        ("oni", 0.8),
        ("werewolf", 0.8),
        ("monsterification", 0.75),
        ("slime_(creature)", 0.8),
    ]
    for tag, score in cases:
        folder, _score, _ = choose_best_destination({tag: score}, selected)
        assert folder == "monster_girl", f"{tag} routed to {folder}"

    # Kemonomimi / mammal-girl noise must not become the catch-all.
    for tag in ("fox_girl", "cat_girl", "dog_girl", "wolf_girl", "cow_girl", "rabbit_girl"):
        assert choose_best_destination({tag: 0.99}, selected)[0] is None, tag

    # Costume-only vampire must not route; real vampire tag does.
    assert choose_best_destination({"vampire_costume": 0.99}, selected)[0] is None
    folder, _score, _ = choose_best_destination({"vampire": 0.85}, selected)
    assert folder == "monster_girl"

    # Body-feature gated hits need a species/identity cue.
    assert choose_best_destination({"extra_eyes": 0.95, "multiple_legs": 0.9}, selected)[
        0
    ] is None
    folder, _score, _ = choose_best_destination(
        {"extra_eyes": 0.9, "monster_girl": 0.4}, selected
    )
    assert folder == "monster_girl"


def test_android_catch_all() -> None:
    selected = {"android", "monster_girl", "Voyeur", "sex"}
    cases = [
        ("android", 0.9),
        ("robot_girl", 0.88),
        ("humanoid_robot", 0.87),
        ("cyborg", 0.86),
        ("mecha_musume", 0.85),
        ("mechanization", 0.8),
        ("robot", 0.82),
    ]
    for tag, score in cases:
        folder, _score, _ = choose_best_destination({tag: score}, selected)
        assert folder == "android", f"{tag} routed to {folder}"

    # Giant robots / props / weapons must not become the catch-all.
    for tag in (
        "mecha",
        "mecha_focus",
        "non-humanoid_robot",
        "robot_animal",
        "mechanical_pencil",
        "machine_gun",
        "vending_machine",
        "science_fiction",
        "cyberpunk",
    ):
        assert choose_best_destination({tag: 0.99}, selected)[0] is None, tag

    # sex_machine belongs to the sex act folder, not android.
    folder, _score, _ = choose_best_destination({"sex_machine": 0.99}, selected)
    assert folder == "sex"

    # Mechanical body parts alone are noise; need an android/robot identity cue.
    assert choose_best_destination(
        {"mechanical_arms": 0.95, "robot_joints": 0.9}, selected
    )[0] is None
    folder, _score, _ = choose_best_destination(
        {"mechanical_arms": 0.9, "android": 0.4}, selected
    )
    assert folder == "android"

    # Theme beats soft Voyeur when both fire.
    folder, _score, _ = choose_best_destination(
        {"robot_girl": 0.7, "cleavage": 0.95}, selected
    )
    assert folder == "android"


def test_nakadashi_folder_alias() -> None:
    folder, score, _ = choose_best_destination(
        {"internal_cumshot": 0.9},
        {"creampie"},
    )
    assert folder == "nakadashi"
    assert score == 0.9


def test_furry_ignores_kemonomimi() -> None:
    selected = {"furry", "Pokemon"}
    folder, score, _ = choose_best_destination(
        {"animal_ears": 0.99, "fake_animal_ears": 0.9},
        selected,
    )
    assert folder is None

    folder, score, _ = choose_best_destination({"furry_female": 0.8}, selected)
    assert folder == "furry"
    assert abs(score - 0.8 * 0.95) < 1e-9


def test_pokemon_routes_to_folder_name() -> None:
    selected = {"Pokemon", "furry"}
    folder, score, _ = choose_best_destination(
        {"pokemon_(creature)": 0.88, "pokemon_ears": 0.99},
        selected,
    )
    assert folder == "Pokemon"
    assert score == 0.88


def test_soft_evidence_alone_does_not_win() -> None:
    selected = {"fellatio", "loli"}
    folder, score, _ = choose_best_destination({"implied_fellatio": 0.99}, selected)
    assert folder is None

    folder, score, _ = choose_best_destination(
        {"implied_fellatio": 0.9, "imminent_fellatio": 0.7},
        selected,
    )
    assert folder == "fellatio"
    assert abs(score - 0.9 * 0.5) < 1e-9


def test_priority_breaks_score_ties() -> None:
    selected = {"fertilization", "NTR", "fellatio", "loli"}
    # Equal weighted scores → loli wins among these (incest not selected).
    folder, score, secondary = choose_best_destination(
        {"fertilization": 0.8, "netorare": 0.8, "fellatio": 0.8, "loli": 0.8},
        selected,
    )
    assert folder == "loli"
    assert score == 0.8
    assert [row["tag"] for row in secondary][:3] == ["fertilization", "NTR", "fellatio"]


def test_incest_priority_beats_loli_and_shota_on_tie() -> None:
    selected = {"incest", "loli", "shota"}
    folder, score, secondary = choose_best_destination(
        {"incest": 0.85, "loli": 0.85, "shota": 0.85},
        selected,
    )
    assert folder == "incest"
    assert score == 0.85
    assert [row["tag"] for row in secondary][:2] == ["loli", "shota"]


def test_loli_priority_beats_shota_on_tie() -> None:
    selected = {"loli", "shota"}
    folder, score, secondary = choose_best_destination(
        {"loli": 0.85, "shota": 0.85},
        selected,
    )
    assert folder == "loli"
    assert score == 0.85
    assert secondary[0]["tag"] == "shota"


def test_legacy_non_taxonomy_still_exact_match() -> None:
    folder, score, secondary = choose_best_destination(
        {"1girl": 0.99, "monster_girl": 0.81, "slime_girl": 0.77},
        {"monster_girl", "slime_girl"},
    )
    assert folder == "monster_girl"
    assert score == 0.81
    assert secondary == [{"tag": "slime_girl", "score": 0.77}]


def test_discover_maps_pokemon_via_taxonomy(tmp_path: Path) -> None:
    mappings = discover_tag_folders(
        tmp_path,
        known_tags={"loli", "furry"},
        selected_folders=["loli", "Pokemon", "unknown_x"],
    )
    mapped = {m.folder_name: (m.matched, m.matched_tag) for m in mappings}
    assert mapped["loli"] == (True, "loli")
    assert mapped["Pokemon"] == (True, "Pokemon")
    assert mapped["unknown_x"] == (False, None)


def test_score_bucket_skips_ignore_even_if_listed() -> None:
    bucket = resolve_taxonomy_folder("loli")
    assert bucket is not None
    # Only ignore-like body tags present → no score.
    assert score_bucket({"flat_chest": 0.99, "petite": 0.98}, bucket) is None
    assert score_bucket({"loli": 0.7, "flat_chest": 0.99}, bucket) == 0.7


def test_character_beats_higher_scoring_act() -> None:
    selected = {"loli", "NTR", "fellatio", "shota"}
    folder, score, secondary = choose_best_destination(
        {"netorare": 0.79, "loli": 0.41, "fellatio": 0.55},
        selected,
    )
    assert folder == "loli"
    assert score == 0.41
    assert "NTR" in {row["tag"] for row in secondary}


def test_higher_theme_score_still_beats_act() -> None:
    """Theme vs act stays score-based (no theme-over-act override)."""
    selected = {"Pokemon", "bestiality", "furry"}
    folder, score, _ = choose_best_destination(
        {"bestiality": 0.95, "pokemon_(creature)": 0.97, "dog": 0.9},
        selected,
    )
    assert folder == "Pokemon"
    assert score == 0.97


def test_act_still_wins_without_character() -> None:
    selected = {"NTR", "fellatio", "fertilization"}
    folder, score, _ = choose_best_destination(
        {"netorare": 0.8, "fellatio": 0.5},
        selected,
    )
    assert folder == "NTR"
    assert score == 0.8


def test_bestiality_bucket_from_animal_sex_cues() -> None:
    selected = {"bestiality", "furry", "fellatio"}
    # Bare animal must not route.
    assert choose_best_destination({"dog": 0.95, "animal": 0.8}, selected)[0] is None

    folder, score, _ = choose_best_destination(
        {"dog": 0.92, "sex": 0.9, "penis": 0.85, "1girl": 0.9},
        selected,
    )
    assert folder == "bestiality"
    assert score is not None and score >= 0.4

    folder, score, _ = choose_best_destination({"bestiality": 0.88, "dog": 0.5}, selected)
    assert folder == "bestiality"
    assert score == 0.88


def test_pokemon_pokephilia_and_creature_alias() -> None:
    selected = {"Pokemon", "furry"}
    folder, score, _ = choose_best_destination({"pokephilia": 0.9}, selected)
    assert folder == "Pokemon"
    assert abs(score - 0.9 * 0.9) < 1e-9

    folder, score, _ = choose_best_destination({"pokemon_creature": 0.7}, selected)
    assert folder == "Pokemon"
    assert score == 0.7


def test_fertilization_accepts_cross_section_underscore() -> None:
    selected = {"fertilization", "nakadashi"}
    folder, score, _ = choose_best_destination(
        {"cross_section": 0.9, "ovum": 0.4},
        selected,
    )
    assert folder == "fertilization"
    assert abs(score - 0.9 * 0.45) < 1e-9


GROUP_SEX_SELECTED = {
    "group_sex",
    "sex",
    "milf",
    "fellatio",
    "nakadashi",
    "paizuri",
    "footjob",
    "fertilization",
    "loli",
    "shota",
    "incest",
    "bestiality",
    "NTR",
    "Pokemon",
    "furry",
    "monster_girl",
    "android",
    "tentacles",
}


def test_group_sex_bucket_resolves() -> None:
    bucket = resolve_taxonomy_folder("group_sex")
    assert bucket is not None
    assert bucket.folder == "group_sex"
    assert resolve_taxonomy_folder("gangbang").folder == "group_sex"


def test_group_sex_beats_sex_and_vanilla_acts() -> None:
    folder, score, _ = choose_best_destination(
        {"gangbang": 0.92, "sex": 0.9, "vaginal": 0.88}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"
    assert score is not None and score > 0.5

    folder, _, _ = choose_best_destination(
        {"group_sex": 0.9, "fellatio": 0.95}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"

    folder, _, _ = choose_best_destination(
        {"threesome": 0.9, "nakadashi": 0.95, "cum_in_pussy": 0.9},
        GROUP_SEX_SELECTED,
    )
    assert folder == "group_sex"

    folder, _, _ = choose_best_destination(
        {"mmf_threesome": 0.88, "paizuri": 0.95}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"

    folder, _, _ = choose_best_destination(
        {"orgy": 0.9, "footjob": 0.95}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"

    folder, _, _ = choose_best_destination(
        {"spitroast": 0.9, "fertilization": 0.95, "impregnation": 0.9},
        GROUP_SEX_SELECTED,
    )
    assert folder == "group_sex"


def test_sex_still_wins_without_group_markers() -> None:
    folder, score, _ = choose_best_destination(
        {"sex": 0.9, "vaginal": 0.88, "missionary": 0.85}, GROUP_SEX_SELECTED
    )
    assert folder == "sex"
    assert score is not None and score > 0.5


def test_group_sex_loses_to_overrides() -> None:
    cases = [
        ({"gangbang": 0.95, "loli": 0.9}, "loli"),
        ({"gangbang": 0.95, "shota": 0.9}, "shota"),
        ({"gangbang": 0.95, "incest": 0.9}, "incest"),
        ({"gangbang": 0.95, "netorare": 0.85}, "NTR"),
        ({"gangbang": 0.95, "bestiality": 0.9}, "bestiality"),
        ({"gangbang": 0.95, "pokemon_(creature)": 0.9}, "Pokemon"),
        ({"gangbang": 0.95, "furry": 0.9}, "furry"),
        ({"gangbang": 0.95, "monster_girl": 0.9}, "monster_girl"),
        ({"gangbang": 0.95, "android": 0.9}, "android"),
        ({"gangbang": 0.95, "tentacle_sex": 0.9}, "tentacles"),
    ]
    for scores, expected in cases:
        folder, _, _ = choose_best_destination(scores, GROUP_SEX_SELECTED)
        assert folder == expected, (scores, folder)


def test_group_sex_beats_milf_even_with_strong_mature_female() -> None:
    folder, _, _ = choose_best_destination(
        {"gangbang": 0.85, "mature_female": 0.99}, GROUP_SEX_SELECTED
    )
    assert folder == "group_sex"


def test_fellatio_still_wins_without_group() -> None:
    folder, score, _ = choose_best_destination(
        {"fellatio": 0.95}, GROUP_SEX_SELECTED
    )
    assert folder == "fellatio"
    assert score is not None and score > 0.5


def test_voyeur_unchanged_by_group_sex_work() -> None:
    # Soft Voyeur path must still work; selecting group_sex must not break Voyeur group expand.
    folder, score, _ = choose_best_destination({"upskirt": 0.88}, {"Voyeur"})
    assert folder == "Voyeur/upskirt"
    assert score is not None and score > 0.5


def test_voyeur_subfolder_routes_and_group_expands() -> None:
    # Selecting only parent Voyeur still scores nested buckets.
    folder, score, _ = choose_best_destination(
        {"pantyshot": 0.91, "1girl": 0.99},
        {"Voyeur"},
    )
    assert folder == "Voyeur/panties"
    assert score == 0.91

    folder, score, _ = choose_best_destination({"upskirt": 0.88}, {"Voyeur"})
    assert folder == "Voyeur/upskirt"
    assert score == 0.88

    folder, score, _ = choose_best_destination({"ass_focus": 0.9}, {"Voyeur"})
    assert folder == "Voyeur/ass"

    folder, score, _ = choose_best_destination({"cleavage": 0.86}, {"Voyeur"})
    assert folder == "Voyeur/cleavage"

    folder, score, _ = choose_best_destination({"voyeurism": 0.8}, {"Voyeur"})
    assert folder == "Voyeur/caught"

    folder, score, _ = choose_best_destination({"flashing": 0.77}, {"Voyeur"})
    assert folder == "Voyeur/public"
    assert abs(score - 0.77 * 0.95) < 1e-9


def test_character_and_act_beat_voyeur_soft() -> None:
    selected = {"Voyeur", "loli", "fellatio"}
    folder, score, secondary = choose_best_destination(
        {"pantyshot": 0.95, "loli": 0.55},
        selected,
    )
    assert folder == "loli"
    assert score == 0.55
    assert "Voyeur/panties" in {row["tag"] for row in secondary}

    folder, score, _ = choose_best_destination(
        {"ass_focus": 0.99, "fellatio": 0.7},
        selected,
    )
    assert folder == "fellatio"
    assert score == 0.7


def test_theme_beats_voyeur_soft() -> None:
    folder, score, secondary = choose_best_destination(
        {"highleg_leotard": 0.95, "poke_ball_basic": 0.7},
        {"Voyeur", "Pokemon"},
    )
    assert folder == "Pokemon"
    assert abs(score - 0.7 * 0.85) < 1e-9
    assert "Voyeur/leotard" in {row["tag"] for row in secondary}

    folder, score, _ = choose_best_destination(
        {"ass_focus": 0.99, "furry": 0.5},
        {"Voyeur", "furry"},
    )
    assert folder == "furry"
    assert score == 0.5

    folder, score, _ = choose_best_destination(
        {"cleavage": 0.9, "monster_girl": 0.55},
        {"Voyeur", "monster_girl"},
    )
    assert folder == "monster_girl"
    assert score == 0.55


def test_voyeur_catch_all_when_no_subfolder() -> None:
    # Generic tease cues with no sub-folder home land in the catch-all.
    folder, score, _ = choose_best_destination(
        {"cameltoe": 0.9, "sexually_suggestive": 0.5},
        {"Voyeur", "loli"},
    )
    assert folder == "Voyeur"
    assert abs(score - 0.9 * 0.75) < 1e-9

    # Lingerie now has its own sub-folder rather than falling through.
    folder, score, _ = choose_best_destination({"lingerie": 0.9}, {"Voyeur"})
    assert folder == "Voyeur/lingerie"
    assert score == 0.9


def test_voyeur_fellatio_gesture_is_soft_not_act() -> None:
    folder, score, _ = choose_best_destination(
        {
            "fellatio_gesture": 0.97,
            "sexually_suggestive": 0.55,
            "tongue_out": 0.89,
            "pov": 0.79,
        },
        {"Voyeur", "fellatio"},
    )
    assert folder == "Voyeur"
    assert abs(score - 0.97 * 0.95) < 1e-9

    # Real oral still prefers the act folder.
    folder, score, _ = choose_best_destination(
        {"fellatio_gesture": 0.9, "fellatio": 0.8},
        {"Voyeur", "fellatio"},
    )
    assert folder == "fellatio"


def test_voyeur_condom_pose_without_penetration() -> None:
    folder, score, _ = choose_best_destination(
        {
            "condom": 0.99,
            "used_condom": 0.9,
            "condom_in_mouth": 0.92,
            "mouth_hold": 0.94,
        },
        {"Voyeur", "fellatio"},
    )
    assert folder == "Voyeur"
    assert abs(score - 0.92 * 0.95) < 1e-9

    # Penetration / insertion still vetoes soft.
    folder, score, _ = choose_best_destination(
        {"condom_in_mouth": 0.95, "anal": 0.8, "object_insertion": 0.7},
        {"Voyeur"},
    )
    assert folder is None
    assert score is None

    # Oral aftermath still prefers act over soft condom pose.
    folder, score, _ = choose_best_destination(
        {"condom_in_mouth": 0.9, "after_fellatio": 0.7},
        {"Voyeur", "fellatio"},
    )
    assert folder == "fellatio"


def test_voyeur_sexual_presentation_needs_corroboration() -> None:
    # Low-weight midriff/crop cues must not win alone (soft_alone_weight).
    folder, score, _ = choose_best_destination({"crop_top": 0.9}, {"Voyeur"})
    assert folder is None
    assert score is None

    folder, score, _ = choose_best_destination(
        {"crop_top": 0.68, "midriff": 0.68},
        {"Voyeur"},
    )
    assert folder == "Voyeur"
    assert abs(score - 0.68 * 0.55) < 1e-9

    folder, score, _ = choose_best_destination(
        {"navel": 0.8, "bare_shoulders": 0.7, "thighs": 0.6},
        {"Voyeur"},
    )
    assert folder == "Voyeur"

    # Fashion portrait without sexual-presentation cues stays unmatched.
    folder, score, _ = choose_best_destination(
        {
            "jirai_kei": 0.88,
            "mouth_mask": 0.92,
            "lipstick_tube": 0.78,
            "shirt": 0.89,
            "skirt": 0.9,
            "looking_at_viewer": 0.87,
        },
        {"Voyeur"},
    )
    assert folder is None


def test_voyeur_bikini_pantyhose_and_feet_route() -> None:
    folder, score, _ = choose_best_destination(
        {"bikini": 0.95, "swimsuit": 0.9, "navel": 0.85},
        {"Voyeur"},
    )
    assert folder == "Voyeur/swimsuit"
    assert abs(score - 0.95 * 0.95) < 1e-9

    folder, score, _ = choose_best_destination(
        {"pantyhose": 0.92, "fishnet_pantyhose": 0.88},
        {"Voyeur"},
    )
    assert folder == "Voyeur/legwear"
    assert abs(score - 0.88 * 0.95) < 1e-9

    folder, score, _ = choose_best_destination(
        {"soles": 0.92, "feet": 0.9, "toes": 0.8},
        {"Voyeur"},
    )
    assert folder == "Voyeur/feet"
    assert score == 0.92

    folder, score, _ = choose_best_destination(
        {"one_breast_out": 0.9, "bra": 0.8},
        {"Voyeur"},
    )
    assert folder == "Voyeur/cleavage"


def test_voyeur_costume_tease_routes_leotard() -> None:
    folder, score, _ = choose_best_destination(
        {
            "highleg_leotard": 0.89,
            "leotard": 0.87,
            "covered_nipples": 0.74,
        },
        {"Voyeur"},
    )
    assert folder == "Voyeur/leotard"
    assert score == 0.89


def test_voyeur_funnel_splits_catch_all_clusters() -> None:
    """Each big cluster that used to pile into Voyeur now has its own folder."""
    cases = {
        "Voyeur/nude": {"nude": 0.93, "nipples": 0.9},
        "Voyeur/lingerie": {"lingerie": 0.88, "garter_belt": 0.7},
        "Voyeur/swimsuit": {"school_swimsuit": 0.94},
        "Voyeur/legwear": {"zettai_ryouiki": 0.9, "thighhighs": 0.99},
        "Voyeur/undressing": {"clothes_lift": 0.91, "open_clothes": 0.8},
        "Voyeur/see_through": {"see-through": 0.87, "no_bra": 0.5},
        "Voyeur/public": {"exhibitionism": 0.82},
        "Voyeur/leotard": {"bodysuit": 0.9},
    }
    for expected, scores in cases.items():
        folder, _score, _ = choose_best_destination(scores, {"Voyeur"})
        assert folder == expected, f"{scores} routed to {folder}, expected {expected}"

    # Thighhighs alone are too common to justify a legwear destination.
    assert choose_best_destination({"thighhighs": 0.99}, {"Voyeur"})[0] is None


def test_see_through_requires_visible_skin() -> None:
    """The tagger calls crystalline characters see-through; that is not a tease."""
    gems = {
        "see-through": 0.85,
        "androgynous": 0.93,
        "other_focus": 0.95,
        "crystal_hair": 0.68,
        "necktie": 0.86,
    }
    assert choose_best_destination(gems, {"Voyeur"})[0] is None

    folder, score, _ = choose_best_destination({**gems, "nipples": 0.4}, {"Voyeur"})
    assert folder == "Voyeur/see_through"
    assert abs(score - 0.85 * 0.95) < 1e-9


def test_voyeur_pussy_routes_and_cum_vetoes() -> None:
    folder, score, _ = choose_best_destination(
        {"spread_pussy": 0.9},
        {"Voyeur"},
    )
    assert folder == "Voyeur/pussy"
    assert score == 0.9

    folder, score, _ = choose_best_destination(
        {"pussy": 0.88, "close_up": 0.7},
        {"Voyeur"},
    )
    assert folder == "Voyeur/pussy"
    assert abs(score - 0.88 * 0.85) < 1e-9

    # Surface cum is allowed for soft tease; internal/oral cum still vetoes.
    folder, score, _ = choose_best_destination(
        {"pussy": 0.88, "cum": 0.61, "cum_on_body": 0.55},
        {"Voyeur"},
    )
    assert folder == "Voyeur/pussy"
    assert abs(score - 0.88 * 0.85) < 1e-9

    folder, score, _ = choose_best_destination(
        {
            "pasties": 0.93,
            "bandaids_on_nipples": 0.92,
            "cum": 0.77,
            "cum_on_body": 0.61,
        },
        {"Voyeur"},
    )
    # pasties is cleavage evidence (higher weight than catch-all).
    assert folder == "Voyeur/cleavage"
    assert abs(score - 0.93 * 0.85) < 1e-9

    folder, score, _ = choose_best_destination(
        {"pussy": 0.88, "cum_in_mouth": 0.7},
        {"Voyeur"},
    )
    assert folder is None
    assert score is None

    # Outdoor clothed masturbation / public presenting → Voyeur (not vetoed).
    folder, score, _ = choose_best_destination(
        {
            "spread_pussy": 0.9,
            "pussy": 0.96,
            "masturbation": 0.87,
            "female_masturbation": 0.86,
            "clothed_masturbation": 0.76,
            "public_indecency": 0.77,
            "presenting": 0.69,
        },
        {"Voyeur"},
    )
    assert folder == "Voyeur/pussy"
    assert score == 0.9

    # Active fingering / anal play is hard, not soft Voyeur.
    folder, score, _ = choose_best_destination(
        {
            "pussy": 0.96,
            "spread_pussy": 0.5,
            "ass": 0.93,
            "fingering": 0.97,
            "anal_fingering": 0.94,
            "masturbation": 0.84,
        },
        {"Voyeur"},
    )
    assert folder is None


def test_voyeur_vetoed_by_cum_or_penetration() -> None:
    # Soft tease must not claim hard sex / creampie even if pantyshot fires.
    folder, score, _ = choose_best_destination(
        {"pantyshot": 0.95, "cum_in_pussy": 0.8},
        {"Voyeur"},
    )
    assert folder is None
    assert score is None

    folder, score, _ = choose_best_destination(
        {"upskirt": 0.9, "sex": 0.55},
        {"Voyeur"},
    )
    assert folder is None

    folder, score, _ = choose_best_destination(
        {"ass_focus": 0.92, "deep_penetration": 0.4},
        {"Voyeur", "nakadashi"},
    )
    assert folder is None or folder == "nakadashi"
    assert folder != "Voyeur/ass"


ALL_ACTS = {
    "sex",
    "nakadashi",
    "fellatio",
    "paizuri",
    "footjob",
    "tentacles",
    "fertilization",
    "NTR",
}


def test_sex_catches_vanilla_penetration() -> None:
    folder, score, _ = choose_best_destination(
        {"sex": 0.97, "vaginal": 0.94, "hetero": 0.99, "penis": 0.98, "1girl": 1.0},
        ALL_ACTS,
    )
    assert folder == "sex"
    assert abs(score - 0.97 * 0.9) < 1e-9

    # A bare penis is not enough on its own; it needs corroboration.
    assert choose_best_destination({"penis": 0.98}, ALL_ACTS)[0] is None

    folder, _score, _ = choose_best_destination(
        {"penis": 0.98, "hetero": 0.9}, ALL_ACTS
    )
    assert folder == "sex"


def test_sex_defers_to_more_specific_acts() -> None:
    """The catch-all act folder is vetoed whenever a specific act fires."""
    specific = {
        "nakadashi": {"sex": 0.97, "cum_in_pussy": 0.42},
        "fellatio": {"sex": 0.97, "fellatio": 0.55},
        "paizuri": {"sex": 0.9, "paizuri": 0.61},
        "footjob": {"sex": 0.9, "footjob": 0.58},
        "tentacles": {"sex": 0.9, "tentacle_sex": 0.55},
        "fertilization": {"sex": 0.95, "impregnation": 0.4},
        "NTR": {"sex": 0.95, "netorare": 0.45},
    }
    for expected, scores in specific.items():
        folder, _score, _ = choose_best_destination(scores, ALL_ACTS)
        assert folder == expected, f"{scores} routed to {folder}, expected {expected}"

    # Below the veto threshold the specific tag is treated as tagger noise.
    folder, _score, _ = choose_best_destination(
        {"sex": 0.97, "fellatio": 0.12}, ALL_ACTS
    )
    assert folder == "sex"


def test_sex_still_loses_to_character() -> None:
    folder, score, _ = choose_best_destination(
        {"sex": 0.98, "vaginal": 0.95, "loli": 0.44},
        ALL_ACTS | {"loli", "shota"},
    )
    assert folder == "loli"
    assert score == 0.44


def test_tentacles_catch_all_and_veto_voyeur() -> None:
    folder, score, _ = choose_best_destination(
        {"tentacle_sex": 0.9, "tentacles": 0.85}, ALL_ACTS
    )
    assert folder == "tentacles"
    assert score == 0.9

    folder, _score, _ = choose_best_destination(
        {"consensual_tentacles": 0.88, "nude": 0.9, "pussy": 0.8},
        ALL_ACTS | {"Voyeur"},
    )
    assert folder == "tentacles"

    # Bare tentacles alone is character noise; needs a sexual gate.
    assert choose_best_destination({"tentacles": 0.95}, ALL_ACTS)[0] is None
    folder, _score, _ = choose_best_destination(
        {"tentacles": 0.9, "sex": 0.5}, ALL_ACTS
    )
    assert folder == "tentacles"

    # tentacle_hair is a hairstyle, not the act.
    assert choose_best_destination({"tentacle_hair": 0.99}, ALL_ACTS)[0] is None


def test_paizuri_and_footjob_ignore_bare_body_parts() -> None:
    assert choose_best_destination(
        {"breasts": 0.99, "large_breasts": 0.98, "breast_press": 0.9}, ALL_ACTS
    )[0] is None
    assert choose_best_destination(
        {"feet": 0.99, "soles": 0.98, "barefoot": 0.97}, ALL_ACTS
    )[0] is None

    folder, score, _ = choose_best_destination({"paizuri": 0.86}, ALL_ACTS)
    assert folder == "paizuri"
    assert score == 0.86

    folder, score, _ = choose_best_destination(
        {"footjob": 0.9, "two_footed_footjob": 0.85, "feet": 0.99},
        ALL_ACTS | {"Voyeur"},
    )
    assert folder == "footjob"
    assert score == 0.9


FALLBACKS = {"SFW", "comic", "scenery"}


def test_fallback_folders_yield_to_every_other_role() -> None:
    folder, score, _ = choose_best_destination({"1girl": 0.99, "solo": 0.98}, FALLBACKS)
    assert folder == "SFW"
    assert score == 0.99

    # Character, act, theme and soft all outrank a fallback home.
    for other, scores in (
        ("loli", {"1girl": 0.99, "loli": 0.3}),
        ("sex", {"1girl": 0.99, "sex": 0.4}),
        ("Pokemon", {"1girl": 0.99, "pokemon_(creature)": 0.35}),
        ("Voyeur/panties", {"1girl": 0.99, "pantyshot": 0.3}),
    ):
        folder, _score, _ = choose_best_destination(
            scores, FALLBACKS | {"loli", "sex", "Pokemon", "Voyeur"}
        )
        assert folder == other, f"{scores} routed to {folder}, expected {other}"


def test_comic_and_scenery_beat_generic_sfw() -> None:
    folder, _score, _ = choose_best_destination(
        {"1girl": 0.79, "comic": 0.9, "speech_bubble": 0.7}, FALLBACKS
    )
    assert folder == "comic"

    folder, _score, _ = choose_best_destination(
        {"scenery": 0.72, "no_humans": 0.9, "sky": 0.77}, FALLBACKS
    )
    assert folder == "scenery"


def test_sfw_refuses_explicit_content() -> None:
    """Unrouted explicit art must go to review rather than be filed as safe."""
    for scores in (
        {"1girl": 0.99, "nipples": 0.8},
        {"1girl": 0.99, "censored": 0.6, "penis": 0.7},
        {"1girl": 0.99, "cameltoe": 0.4},
    ):
        assert choose_best_destination(scores, FALLBACKS)[0] is None
