"""Unit tests for preferred-tag FP probe selection (no ONNX)."""

from __future__ import annotations

from app.tag_fp_eval import preferred_probe_tags
from app.taxonomy import reload_taxonomy


def test_preferred_probe_tags_respects_min_weight_and_selection() -> None:
    reload_taxonomy()
    selected = {"fellatio", "tentacles", "android"}
    probe = preferred_probe_tags(selected, min_weight=0.95)
    assert "fellatio" in probe
    assert "tentacle_sex" in probe
    assert "android" in probe
    # Low-weight / other-folder tags stay out.
    assert "cleavage" not in probe
    assert "loli" not in probe

    loose = preferred_probe_tags({"voyeur"}, min_weight=0.95)
    assert "cleavage" in loose or "nude" in loose


def test_run_tag_fp_eval_requires_selected(monkeypatch) -> None:
    from app import tag_fp_eval

    def boom(*_a, **_k):
        raise AssertionError("should not score without selected tags")

    monkeypatch.setattr(tag_fp_eval, "ensure_suite_cache", boom)
    try:
        tag_fp_eval.run_tag_fp_eval(selected_tags=[])
        assert False, "expected ValueError"
    except ValueError as err:
        assert "selected_tags" in str(err)
