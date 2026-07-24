"""Accuracy audit for taxonomy routing + classify gates (score fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.api import _classify_from_scores
from app.taxonomy import choose_best_destination, reload_taxonomy


SELECTED_ALL = {
    "fertilization",
    "NTR",
    "incest",
    "nakadashi",
    "fellatio",
    "loli",
    "shota",
    "monster_girl",
    "furry",
    "Pokemon",
    "bestiality",
}


@pytest.fixture(autouse=True)
def _reload_default_taxonomy() -> None:
    reload_taxonomy()


@pytest.mark.parametrize(
    "scores,expected",
    [
        ({"loli": 0.92, "flat_chest": 0.99, "lolita_fashion": 0.95}, "loli"),
        ({"shota": 0.88, "1boy": 0.99, "child": 0.9}, "shota"),
        ({"netorare": 0.81, "caught": 0.99}, "NTR"),
        ({"incest": 0.77, "siblings": 0.99}, "incest"),
        ({"internal_cumshot": 0.9, "cum_in_pussy": 0.95}, "nakadashi"),
        ({"fertilization": 0.8, "cum_in_pussy": 0.99}, "fertilization"),
        ({"impregnation": 0.85, "pregnant": 0.99}, "fertilization"),
        ({"irrumatio": 0.9}, "fellatio"),
        ({"monster_girl": 0.9, "horns": 0.99, "wings": 0.98}, "monster_girl"),
        ({"slime_girl": 0.86}, "monster_girl"),
        ({"furry": 0.9, "animal_ears": 0.99}, "furry"),
        ({"pokemon_(creature)": 0.91, "pokemon_ears": 0.99}, "Pokemon"),
        # False-positive clusters must not route
        ({"lolita_fashion": 0.99, "gothic_lolita": 0.98, "flat_chest": 0.97}, None),
        ({"1boy": 0.99, "male_focus": 0.98, "otoko_no_ko": 0.9}, None),
        ({"animal_ears": 0.99, "fake_animal_ears": 0.9}, None),
        ({"siblings": 0.99}, None),
        ({"pregnant": 0.99}, None),
        ({"voyeurism": 0.99, "caught": 0.9}, None),
        ({"horns": 0.99, "wings": 0.98, "tail": 0.97}, None),
        ({"oral": 0.99}, None),
    ],
)
def test_taxonomy_routing_matrix(scores: dict[str, float], expected: str | None) -> None:
    folder, _score, _secondary = choose_best_destination(scores, SELECTED_ALL)
    assert folder == expected


def test_classify_gate_clears_weak_taxonomy_winner(tmp_path: Path) -> None:
    result = _classify_from_scores(
        tmp_path / "x.jpg",
        {"loli": 0.2, "1girl": 0.99},
        {"loli", "shota"},
        confidence_threshold=0.6,
    )
    assert result.primary_tag is None
    assert result.needs_review is True
    assert result.secondary[0]["tag"] == "loli"


def test_classify_keeps_mid_character_primary_for_review(tmp_path: Path) -> None:
    """Character hits below confidence stay as primary (needs_review), not cleared."""
    result = _classify_from_scores(
        tmp_path / "x.jpg",
        {"loli": 0.42, "1girl": 0.99},
        {"loli", "shota", "NTR", "fellatio"},
        confidence_threshold=0.6,
    )
    assert result.primary_tag == "loli"
    assert result.primary_score == 0.42
    assert result.needs_review is True
    assert "threshold" in (result.reason or "").lower()


def test_classify_still_clears_mid_act_primary(tmp_path: Path) -> None:
    result = _classify_from_scores(
        tmp_path / "x.jpg",
        {"fellatio": 0.42, "1girl": 0.99},
        {"loli", "fellatio"},
        confidence_threshold=0.6,
    )
    assert result.primary_tag is None
    assert result.needs_review is True
    assert result.secondary[0]["tag"] == "fellatio"


def test_classify_assigns_strong_taxonomy_winner(tmp_path: Path) -> None:
    result = _classify_from_scores(
        tmp_path / "x.jpg",
        {"loli": 0.91, "flat_chest": 0.95, "shota": 0.4},
        {"loli", "shota", "furry"},
        confidence_threshold=0.6,
    )
    assert result.primary_tag == "loli"
    assert result.primary_score == 0.91
    assert result.needs_review is False


def test_priority_fertilization_over_nakadashi_on_tie(tmp_path: Path) -> None:
    result = _classify_from_scores(
        tmp_path / "x.jpg",
        {"fertilization": 0.8, "internal_cumshot": 0.8},
        {"fertilization", "nakadashi"},
        confidence_threshold=0.6,
    )
    assert result.primary_tag == "fertilization"


def test_model_smoke_nonempty_scores_all_taggers(tmp_path: Path) -> None:
    """Smoke: each tagger returns a non-empty score dict on a tiny RGB image.

    This is not semantic accuracy (no labels); it catches broken sessions /
    empty-output regressions across models.
    """
    pytest.importorskip("imgutils")
    from app.services import extract_scores

    img = tmp_path / "probe.png"
    Image.new("RGB", (448, 448), color=(180, 120, 160)).save(img)

    models = ["ml_danbooru", "wd_swinv2_v3", "wd_eva02_large"]
    for model in models:
        try:
            scores = extract_scores(img, tagger_model=model, wd_general_threshold=0.35)
        except Exception as err:  # pragma: no cover - environment-specific
            pytest.skip(f"tagger {model} unavailable: {err}")
        assert isinstance(scores, dict)
        assert len(scores) > 0, f"{model} returned empty scores"
        assert all(0.0 <= float(v) <= 1.0 for v in scores.values())
