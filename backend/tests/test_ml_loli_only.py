"""ML-Danbooru primary runs are loli-only (strong recall on that folder)."""

from __future__ import annotations

from app.services import destination_folders_for_tagger


def test_ml_danbooru_forces_loli_only_destinations() -> None:
    selected = ["Voyeur", "fellatio", "loli", "monster_girl"]
    assert destination_folders_for_tagger(selected, "ml_danbooru") == ["loli"]
    assert destination_folders_for_tagger(selected, "wd_swinv2_v3") == selected
    assert destination_folders_for_tagger(
        ["real_life"], "ml_danbooru", real_life_filter=True
    ) == ["real_life"]
