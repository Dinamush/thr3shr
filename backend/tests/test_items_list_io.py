"""Items list must not pull full_scores_json (disk-read spike on large runs)."""

from __future__ import annotations

from app.api import _ITEMS_LIST_COLUMNS, _item_from_row


def test_item_list_columns_exclude_full_scores() -> None:
    assert "full_scores_json" not in _ITEMS_LIST_COLUMNS
    assert "primary_tag" in _ITEMS_LIST_COLUMNS
    assert "secondary_json" in _ITEMS_LIST_COLUMNS


def test_item_from_row_skips_scores_unless_requested() -> None:
    row = {
        "id": 1,
        "run_id": 9,
        "file_path": "x.jpg",
        "relative_path": "x.jpg",
        "primary_tag": "loli",
        "primary_score": 0.9,
        "secondary_json": "[]",
        "full_scores_json": '{"loli": 0.9, "1girl": 0.99}',
        "suggested_destination": None,
        "final_tag": "loli",
        "final_destination": None,
        "status": "proposed",
        "needs_review": 0,
        "review_reason": None,
        "migrated_to": None,
    }
    slim = _item_from_row(row, include_full_scores=False)
    assert slim.full_scores is None
    assert slim.global_top_tags == []
    assert slim.primary_tag == "loli"

    full = _item_from_row(row, include_full_scores=True)
    assert full.full_scores is not None
    assert full.full_scores["loli"] == 0.9
    assert any(t.tag == "loli" for t in full.global_top_tags)
