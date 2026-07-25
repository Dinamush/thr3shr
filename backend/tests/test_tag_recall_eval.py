"""Unit tests for curated tag-recall scoring (no network / no ONNX)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tag_recall_eval import (
    _compare_reports,
    _finalize_counters,
    evaluate_model_on_samples,
    iter_suite_samples,
    load_suite,
)


def test_load_default_suite_has_buckets() -> None:
    suite = load_suite()
    assert suite.get("version") == 1
    assert len(suite.get("buckets") or []) >= 8
    samples = iter_suite_samples(suite)
    # Suite may be empty until fetch --rebuild; still valid structure.
    assert isinstance(samples, list)


def test_evaluate_model_threshold_and_topk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img = tmp_path / "1.jpg"
    img.write_bytes(b"not-a-real-image")

    def fake_score(path: Path, *, tagger_model: str, wd_general_threshold: float):
        assert path == img
        # High monster_girl; android buried below top-3.
        return {
            "1girl": 0.99,
            "solo": 0.95,
            "monster_girl": 0.9,
            "horns": 0.8,
            "wings": 0.7,
            "android": 0.2,
            "noise_a": 0.6,
            "noise_b": 0.55,
        }

    monkeypatch.setattr("app.tag_recall_eval._score_image", fake_score)

    samples = [
        {
            "bucket_id": "monster_girl",
            "source": "danbooru",
            "post_id": "1",
            "desired_tags": ["monster_girl", "android"],
            "path": img,
            "file_name": img.name,
            "rating": "e",
        }
    ]
    report = evaluate_model_on_samples(
        samples,
        tagger_model="ml_danbooru",
        threshold=0.35,
        top_k=3,
        wd_general_threshold=0.35,
    )
    summary = report["summary"]
    assert summary["opportunities"] == 2
    assert summary["hits_at_threshold"] == 1  # monster_girl only
    assert summary["hits_at_top_k"] == 1  # android not in top-3
    assert summary["micro_recall_at_threshold"] == 0.5
    assert summary["per_tag"]["monster_girl"]["recall_at_threshold"] == 1.0
    assert summary["per_tag"]["android"]["recall_at_top_k"] == 0.0

    item = report["items"][0]
    by_tag = {r["tag"]: r for r in item["tag_results"]}
    assert by_tag["monster_girl"]["hit_at_threshold"] is True
    assert by_tag["monster_girl"]["hit_at_top_k"] is True
    assert by_tag["android"]["hit_at_threshold"] is False
    assert by_tag["android"]["hit_at_top_k"] is False


def test_compare_reports_pairwise() -> None:
    reports = [
        {
            "tagger_model": "ml_danbooru",
            "summary": {"micro_recall_at_threshold": 0.5, "micro_recall_at_top_k": 0.6},
            "items": [
                {
                    "source": "danbooru",
                    "post_id": "1",
                    "tag_results": [
                        {"tag": "a", "hit_at_threshold": True},
                        {"tag": "b", "hit_at_threshold": False},
                    ],
                }
            ],
        },
        {
            "tagger_model": "wd_swinv2_v3",
            "summary": {"micro_recall_at_threshold": 1.0, "micro_recall_at_top_k": 1.0},
            "items": [
                {
                    "source": "danbooru",
                    "post_id": "1",
                    "tag_results": [
                        {"tag": "a", "hit_at_threshold": True},
                        {"tag": "b", "hit_at_threshold": True},
                    ],
                }
            ],
        },
    ]
    cmp = _compare_reports(reports)
    assert cmp["best_model"] == "wd_swinv2_v3"
    pair = cmp["pairwise"][0]
    assert pair["a_wins"] == 0
    assert pair["b_wins"] == 1
    assert pair["ties"] == 1


def test_finalize_counters_empty() -> None:
    out = _finalize_counters(
        {
            "opportunities": 0,
            "hits_at_threshold": 0,
            "hits_at_top_k": 0,
            "per_tag": {},
        }
    )
    assert out["micro_recall_at_threshold"] is None
    assert out["macro_recall_at_top_k"] is None
