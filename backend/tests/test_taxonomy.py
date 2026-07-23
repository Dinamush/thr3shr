import json
from pathlib import Path

from app.services import discover_tag_folders
from app.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    choose_best_destination,
    load_taxonomy,
    resolve_taxonomy_folder,
    score_bucket,
)


def test_default_taxonomy_json_loads() -> None:
    cfg = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    assert DEFAULT_TAXONOMY_PATH.is_file()
    assert len(cfg.buckets) >= 10
    assert resolve_taxonomy_folder("Pokemon", taxonomy=cfg).folder == "Pokemon"
    assert resolve_taxonomy_folder("NTR", taxonomy=cfg).folder == "NTR"
    assert resolve_taxonomy_folder("nakadashi", taxonomy=cfg).folder == "nakadashi"


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
    assert abs(score - 0.84 * 0.9) < 1e-9


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
