"""Isolated real-life taxonomy: fusion, sensitive policy, destinations."""

from __future__ import annotations

from pathlib import Path

from app.real_life_taxonomy import (
    canonicalize_score_key,
    fuse_real_life_scores,
    get_real_life_taxonomy,
    is_sensitive_tag,
    parse_closed_vocabulary_tags,
    real_life_destination,
    real_life_folder_names,
    resolve_real_life_folder,
)


def test_real_life_folders_seeded():
    folders = set(real_life_folder_names())
    for name in (
        "creampie",
        "cumshot",
        "facial",
        "oral",
        "anal",
        "vaginal",
        "blowjob",
        "handjob",
        "masturbation",
        "threesome",
        "group",
        "lesbian",
        "POV",
        "interracial",
        "cuck",
        "hotwife",
        "BBC",
        "Ebony",
        "Asian",
    ):
        assert name in folders


def test_sensitive_tags_never_auto_primary():
    primary, score, secondary, folder_scores, flags = fuse_real_life_scores(
        {"BBC": 0.99, "creampie": 0.8, "Ebony": 0.95}
    )
    assert primary == "creampie"
    assert score == 0.8
    assert any(s["tag"] == "BBC" for s in secondary)
    assert any(f.startswith("sensitive:") for f in flags)
    assert "calibration_review" in flags


def test_contextual_tags_need_higher_threshold():
    primary, _, secondary, _, flags = fuse_real_life_scores(
        {"cuck": 0.5, "vaginal": 0.8}
    )
    assert primary == "vaginal"
    assert not any(s["tag"] == "cuck" for s in secondary)
    assert any("contextual_low:cuck" in f for f in flags)

    primary2, _, secondary2, _, flags2 = fuse_real_life_scores(
        {"cuck": 0.85, "vaginal": 0.5}
    )
    assert primary2 == "cuck"
    assert any("contextual:cuck" in f for f in flags2)


def test_closed_vocabulary_parser():
    scores = parse_closed_vocabulary_tags(
        {
            "primary": "blowjob",
            "tags": [{"tag": "oral", "score": 0.9}, {"tag": "Asian", "score": 0.7}],
        }
    )
    assert scores["blowjob"] >= 0.85
    assert scores["oral"] == 0.9
    assert scores["Asian"] == 0.7
    assert canonicalize_score_key("fellatio") == "oral"
    assert resolve_real_life_folder("Hotwife") is not None
    assert is_sensitive_tag("Asian")


def test_real_life_destination_under_root(tmp_path: Path):
    dest = real_life_destination(tmp_path, "creampie")
    assert dest == tmp_path / "Real Life" / "creampie"
    tax = get_real_life_taxonomy()
    assert tax.root_folder == "Real Life"


def test_taxonomy_isolated_from_anime_taxonomy():
    """Real-life BBC/Ebony/Asian folders must not collide with anime routing."""
    assert resolve_real_life_folder("BBC") is not None
    from app.taxonomy import resolve_taxonomy_folder

    anime = resolve_taxonomy_folder("BBC")
    assert anime is None or anime.folder != "BBC"
