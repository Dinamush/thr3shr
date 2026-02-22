from pathlib import Path

from PIL import Image

from app.services import (
    choose_best_tags,
    discover_tag_folders,
    ensure_collision_free_destination,
    migrate_file,
    normalize_tag_name,
    scan_images,
)


def test_normalize_tag_name() -> None:
    assert normalize_tag_name("Black Hair") == "black_hair"
    assert normalize_tag_name("  blue-eyes ") == "blue_eyes"


def test_discover_tag_folders_mapping(tmp_path: Path) -> None:
    tags = {"black_hair", "blue_eyes", "1girl"}
    (tmp_path / "black hair").mkdir()
    (tmp_path / "Blue-Eyes").mkdir()
    (tmp_path / "Unknown Folder").mkdir()

    mappings = discover_tag_folders(tmp_path, tags)
    mapped = {m.folder_name: m.matched_tag for m in mappings}

    assert mapped["black hair"] == "black_hair"
    assert mapped["Blue-Eyes"] == "blue_eyes"
    assert mapped["Unknown Folder"] is None


def test_scan_images_filters_types(tmp_path: Path) -> None:
    img_path = tmp_path / "ok.jpg"
    Image.new("RGB", (16, 16), color="red").save(img_path)

    (tmp_path / "video.mp4").write_text("not-video", encoding="utf-8")
    (tmp_path / "anim.gif").write_text("gif", encoding="utf-8")
    (tmp_path / "doc.txt").write_text("txt", encoding="utf-8")
    (tmp_path / "broken.png").write_text("broken", encoding="utf-8")

    result = scan_images(tmp_path)
    assert len(result.image_paths) == 1
    assert result.stats.eligible_images == 1
    assert result.stats.ignored_gif == 1
    assert result.stats.ignored_unsupported >= 2
    assert result.stats.failed_to_read == 1


def test_choose_best_tags_single_primary_and_secondary() -> None:
    scores = {"black_hair": 0.9, "blue_eyes": 0.8, "solo": 0.7}
    primary_tag, primary_score, secondary = choose_best_tags(scores, {"black_hair", "solo"})
    assert primary_tag == "black_hair"
    assert primary_score == 0.9
    assert secondary[0]["tag"] == "solo"


def test_choose_best_tags_ignores_non_selected_high_score() -> None:
    scores = {"1girl": 0.99, "monster_girl": 0.81, "slime_girl": 0.77}
    primary_tag, primary_score, secondary = choose_best_tags(scores, {"monster_girl", "slime_girl"})
    assert primary_tag == "monster_girl"
    assert primary_score == 0.81
    assert secondary == [{"tag": "slime_girl", "score": 0.77}]


def test_choose_best_tags_matches_normalized_selected_tags() -> None:
    scores = {"monster girl": 0.88, "slime-girl": 0.84}
    primary_tag, primary_score, secondary = choose_best_tags(scores, {"monster_girl", "slime_girl"})
    assert primary_tag == "monster_girl"
    assert primary_score == 0.88
    assert secondary == [{"tag": "slime_girl", "score": 0.84}]


def test_migrate_file_copy_with_collision_suffix(tmp_path: Path) -> None:
    src = tmp_path / "sample.jpg"
    src.write_text("abc", encoding="utf-8")
    dst = tmp_path / "out" / "sample.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("existing", encoding="utf-8")

    result = migrate_file(src, dst, "copy")
    assert result.success is True
    assert result.destination is not None
    assert result.destination.endswith("sample_1.jpg")
    assert src.exists()


def test_migrate_file_fails_when_source_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    dst = tmp_path / "out" / "missing.jpg"
    result = migrate_file(missing, dst, "copy")
    assert result.success is False
    assert "not found" in (result.error or "").lower()


def test_collision_resolution_has_max_attempts(tmp_path: Path) -> None:
    dst = tmp_path / "sample.jpg"
    dst.write_text("x", encoding="utf-8")
    try:
        ensure_collision_free_destination(dst, max_attempts=0)
    except RuntimeError as err:
        assert "collision" in str(err).lower()
    else:
        raise AssertionError("Expected RuntimeError for collision exhaustion")
