"""Hybrid ML-on-review: allowlist merge must raise preferred tags without Voyeur noise."""

from __future__ import annotations

import pytest

from app.hybrid_ml import (
    HYBRID_ML_ALLOWLIST,
    merge_ml_allowlist_scores,
    should_run_hybrid_ml,
)
from app.taxonomy import choose_best_destination


def test_merge_raises_allowlisted_ml_score_with_max() -> None:
    wd = {"1girl": 0.99, "loli": 0.2}
    ml = {"1girl": 0.5, "loli": 0.88}
    merged = merge_ml_allowlist_scores(wd, ml)
    assert merged["loli"] == pytest.approx(0.88)
    assert merged["1girl"] == pytest.approx(0.99)


def test_merge_ignores_voyeur_soft_tags_from_ml() -> None:
    wd = {"1girl": 0.9, "cleavage": 0.4}
    ml = {"cleavage": 0.99, "nude": 0.95, "pussy": 0.9, "fellatio": 0.8}
    merged = merge_ml_allowlist_scores(wd, ml)
    assert merged["cleavage"] == pytest.approx(0.4)
    assert "nude" not in merged
    assert "pussy" not in merged
    assert merged["fellatio"] == pytest.approx(0.8)


def test_merge_does_not_inject_non_allowlist_tags() -> None:
    wd = {"solo": 0.9}
    ml = {"blue_eyes": 0.99, "monster_girl": 0.7}
    merged = merge_ml_allowlist_scores(wd, ml)
    assert "blue_eyes" not in merged
    assert merged["monster_girl"] == pytest.approx(0.7)
    assert merged["solo"] == pytest.approx(0.9)


def test_allowlist_covers_preferred_recall_tags() -> None:
    for tag in (
        "loli",
        "shota",
        "fellatio",
        "irrumatio",
        "cum_in_pussy",
        "tentacle_sex",
        "monster_girl",
        "slime_girl",
        "android",
        "robot_girl",
    ):
        assert tag in HYBRID_ML_ALLOWLIST, tag


def test_merged_scores_can_rescue_routing() -> None:
    """WD miss + ML hit on allowlisted act should route after merge."""
    selected = {"fellatio", "Voyeur", "sex"}
    wd = {"sex": 0.4, "cleavage": 0.9}
    ml = {"fellatio": 0.85, "cleavage": 0.99}
    folder_before, _, _ = choose_best_destination(wd, selected)
    assert folder_before != "fellatio"
    merged = merge_ml_allowlist_scores(wd, ml)
    folder_after, score, _ = choose_best_destination(merged, selected)
    assert folder_after == "fellatio"
    assert score >= 0.85


def test_should_run_hybrid_ml_only_for_wd_review() -> None:
    assert should_run_hybrid_ml(
        enabled=True, tagger_model="wd_swinv2_v3", needs_review=True, inference_failed=False
    )
    assert not should_run_hybrid_ml(
        enabled=True, tagger_model="ml_danbooru", needs_review=True, inference_failed=False
    )
    assert not should_run_hybrid_ml(
        enabled=True, tagger_model="wd_swinv2_v3", needs_review=False, inference_failed=False
    )
    assert not should_run_hybrid_ml(
        enabled=False, tagger_model="wd_swinv2_v3", needs_review=True, inference_failed=False
    )
    assert not should_run_hybrid_ml(
        enabled=True, tagger_model="wd_swinv2_v3", needs_review=True, inference_failed=True
    )


def test_maybe_hybrid_ml_rescue_reclassifies(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from pathlib import Path

    from app.api import _ImageInferenceResult, _maybe_hybrid_ml_rescue

    path = tmp_path / "x.jpg"
    path.write_bytes(b"not-an-image")
    base = _ImageInferenceResult(
        image_path=path,
        scores={"sex": 0.4, "cleavage": 0.9},
        primary_tag=None,
        primary_score=None,
        secondary=[],
        needs_review=True,
        reason="No matching tags found among selected tags.",
        inference_failed=False,
    )

    monkeypatch.setattr(
        "app.api.extract_scores",
        lambda *a, **k: {"fellatio": 0.9, "cleavage": 0.99},
    )
    out = _maybe_hybrid_ml_rescue(
        base,
        {"fellatio", "Voyeur", "sex"},
        0.6,
        False,
        "wd_swinv2_v3",
        0.35,
        False,
        True,
    )
    assert out.primary_tag == "fellatio"
    assert out.needs_review is False
    assert out.scores["fellatio"] == pytest.approx(0.9)
    # ML Voyeur soft tag must not overwrite WD.
    assert out.scores["cleavage"] == pytest.approx(0.9)
