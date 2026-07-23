"""Tests for the ``real_life`` destination bucket (WD-aware policy).

Policy (from probe + label audit):
- Priority 0; aliases real_life / photo / Real Life.
- Hard evidence: ``photorealistic``, ``realistic``, ``photo_(medium)`` (ML).
- Soft: ``3d`` (ML-only) needs corroboration.
- WD v3 has no ``photo_(medium)`` / ``3d``; routing relies on realistic/photorealistic.
- Semi-real anime: moderate ``realistic`` can still lose to stronger content on score.
- WD inference also always-includes realistic/photorealistic above floor 0.10
  (see ``inference_engine.WD_REALISM_FLOOR``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    choose_best_destination,
    get_taxonomy,
    load_taxonomy,
    resolve_taxonomy_folder,
    score_bucket,
)

REAL_LIFE_FIXTURE: dict = {
    "soft_alone_weight": 0.6,
    "soft_corroboration_raw": 0.15,
    "buckets": [
        {
            "id": "real_life",
            "folder": "real_life",
            "priority": 0,
            "aliases": ["photo", "Real Life", "real life", "real-life"],
            "ignore": [
                "depth_of_field",
                "blurry",
                "bokeh",
                "film_grain",
                "photo_(object)",
                "photo_background",
                "selfie",
                "semi-realistic",
                "semi_realistic",
            ],
            "evidence": [
                {"tag": "photorealistic", "weight": 1.0},
                {"tag": "realistic", "weight": 1.0},
                {"tag": "photo_(medium)", "weight": 1.0},
                {"tag": "3d", "weight": 0.45},
            ],
        },
        {
            "id": "loli",
            "folder": "loli",
            "priority": 6,
            "ignore": ["lolita_fashion", "flat_chest", "petite"],
            "evidence": [
                {"tag": "loli", "weight": 1.0},
                {"tag": "oppai_loli", "weight": 0.85},
            ],
        },
        {
            "id": "fellatio",
            "folder": "fellatio",
            "priority": 5,
            "ignore": ["oral", "cunnilingus"],
            "evidence": [
                {"tag": "fellatio", "weight": 1.0},
                {"tag": "irrumatio", "weight": 0.95},
                {"tag": "implied_fellatio", "weight": 0.5},
            ],
        },
        {
            "id": "nakadashi",
            "folder": "nakadashi",
            "priority": 4,
            "aliases": ["creampie"],
            "ignore": ["cumdrip", "pregnant"],
            "evidence": [
                {"tag": "cum_in_pussy", "weight": 0.72},
                {"tag": "internal_cumshot", "weight": 1.0},
            ],
        },
    ],
}


@pytest.fixture()
def real_life_taxonomy(tmp_path: Path):
    path = tmp_path / "taxonomy_real_life_fixture.json"
    path.write_text(json.dumps(REAL_LIFE_FIXTURE), encoding="utf-8")
    return load_taxonomy(path)


def test_resolve_real_life_folder_aliases(real_life_taxonomy) -> None:
    cfg = real_life_taxonomy
    for name in ("real_life", "photo", "Real Life", "real life", "real-life"):
        bucket = resolve_taxonomy_folder(name, taxonomy=cfg)
        assert bucket is not None, f"alias {name!r} should resolve"
        assert bucket.folder == "real_life"
        assert bucket.priority == 0


def test_get_taxonomy_loads_fixture_path(tmp_path: Path) -> None:
    path = tmp_path / "taxonomy_real_life_fixture.json"
    path.write_text(json.dumps(REAL_LIFE_FIXTURE), encoding="utf-8")
    cfg = get_taxonomy(path)
    assert resolve_taxonomy_folder("photo", taxonomy=cfg).folder == "real_life"


def test_real_life_priority_beats_content_on_tie(real_life_taxonomy) -> None:
    cfg = real_life_taxonomy
    folder, score, secondary = choose_best_destination(
        {"photorealistic": 0.8, "loli": 0.8, "fellatio": 0.8},
        {"real_life", "loli", "fellatio"},
        taxonomy=cfg,
    )
    assert folder == "real_life"
    assert score == 0.8
    assert [row["tag"] for row in secondary][:2] == ["fellatio", "loli"]


def test_hard_evidence_photorealistic_realistic_photo_medium(real_life_taxonomy) -> None:
    cfg = real_life_taxonomy
    selected = {"real_life", "loli"}

    folder, score, _ = choose_best_destination(
        {"photorealistic": 0.91}, selected, taxonomy=cfg
    )
    assert folder == "real_life"
    assert score == 0.91

    # WD path: realistic is hard (probe: anime max ≈0.004).
    folder, score, _ = choose_best_destination(
        {"realistic": 0.18}, selected, taxonomy=cfg
    )
    assert folder == "real_life"
    assert score == 0.18

    folder, score, _ = choose_best_destination(
        {"photo_(medium)": 0.87}, selected, taxonomy=cfg
    )
    assert folder == "real_life"
    assert score == 0.87


def test_weak_3d_alone_does_not_route(real_life_taxonomy) -> None:
    cfg = real_life_taxonomy
    bucket = resolve_taxonomy_folder("real_life", taxonomy=cfg)
    assert score_bucket({"3d": 0.99}, bucket, taxonomy=cfg) is None
    folder, _, _ = choose_best_destination(
        {"3d": 0.99}, {"real_life", "fellatio"}, taxonomy=cfg
    )
    assert folder is None


def test_soft_3d_plus_realistic_corroboration_routes(real_life_taxonomy) -> None:
    """ML soft ``3d`` can corroborate; with hard realistic the score is realistic*1.0."""
    cfg = real_life_taxonomy
    folder, score, _ = choose_best_destination(
        {"realistic": 0.2, "3d": 0.4},
        {"real_life", "loli"},
        taxonomy=cfg,
    )
    assert folder == "real_life"
    assert abs(score - 0.2) < 1e-9


def test_ignore_tags_never_score_real_life(real_life_taxonomy) -> None:
    cfg = real_life_taxonomy
    bucket = resolve_taxonomy_folder("real_life", taxonomy=cfg)
    ignore_scores = {
        "depth_of_field": 0.99,
        "blurry": 0.98,
        "bokeh": 0.97,
        "film_grain": 0.96,
        "photo_(object)": 0.95,
        "photo_background": 0.94,
        "selfie": 0.93,
        "semi-realistic": 0.92,
    }
    assert score_bucket(ignore_scores, bucket, taxonomy=cfg) is None
    folder, _, _ = choose_best_destination(
        ignore_scores, {"real_life", "loli"}, taxonomy=cfg
    )
    assert folder is None


def test_ignore_cooccurrence_does_not_veto_hard_evidence(real_life_taxonomy) -> None:
    cfg = real_life_taxonomy
    folder, score, _ = choose_best_destination(
        {
            "photorealistic": 0.88,
            "depth_of_field": 0.99,
            "blurry": 0.97,
            "semi-realistic": 0.9,
        },
        {"real_life", "loli"},
        taxonomy=cfg,
    )
    assert folder == "real_life"
    assert score == 0.88


def test_high_photorealistic_beats_medium_content_scores(real_life_taxonomy) -> None:
    cfg = real_life_taxonomy
    folder, score, secondary = choose_best_destination(
        {
            "photorealistic": 0.93,
            "loli": 0.72,
            "fellatio": 0.68,
            "cum_in_pussy": 0.7,
        },
        {"real_life", "loli", "fellatio", "nakadashi"},
        taxonomy=cfg,
    )
    assert folder == "real_life"
    assert score == 0.93
    assert "loli" in {row["tag"] for row in secondary}


def test_pure_anime_content_tags_do_not_route_to_real_life(real_life_taxonomy) -> None:
    cfg = real_life_taxonomy
    folder, score, _ = choose_best_destination(
        {"loli": 0.91, "fellatio": 0.88, "1girl": 0.99, "anime_coloring": 0.8},
        {"real_life", "loli", "fellatio"},
        taxonomy=cfg,
    )
    assert folder == "loli"
    assert score == 0.91


def test_semi_real_moderate_realistic_loses_to_stronger_content(
    real_life_taxonomy,
) -> None:
    """Score-first: realistic 0.65 loses to fellatio 0.9 even though realistic is hard."""
    cfg = real_life_taxonomy
    folder, score, _ = choose_best_destination(
        {"realistic": 0.65, "fellatio": 0.9, "loli": 0.85},
        {"real_life", "fellatio", "loli"},
        taxonomy=cfg,
    )
    assert folder == "fellatio"
    assert score == 0.9


def test_hard_realism_vs_content_score_first_priority_only_on_ties(
    real_life_taxonomy,
) -> None:
    cfg = real_life_taxonomy
    folder, score, _ = choose_best_destination(
        {"photo_(medium)": 0.86, "fellatio": 0.95, "loli": 0.9},
        {"real_life", "fellatio", "loli"},
        taxonomy=cfg,
    )
    assert folder == "fellatio"
    assert score == 0.95

    folder, score, _ = choose_best_destination(
        {"photo_(medium)": 0.96, "fellatio": 0.95, "loli": 0.9},
        {"real_life", "fellatio", "loli"},
        taxonomy=cfg,
    )
    assert folder == "real_life"
    assert score == 0.96


def test_production_taxonomy_includes_real_life() -> None:
    cfg = load_taxonomy(DEFAULT_TAXONOMY_PATH)
    for name in ("real_life", "photo", "Real Life"):
        bucket = resolve_taxonomy_folder(name, taxonomy=cfg)
        assert bucket is not None
        assert bucket.folder == "real_life"
        assert bucket.priority == 0
    evidence = {ev.tag: ev.weight for ev in bucket.evidence}
    assert evidence["realistic"] >= 0.6
    assert evidence["photorealistic"] >= 0.6
