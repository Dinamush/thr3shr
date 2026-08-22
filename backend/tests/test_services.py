from pathlib import Path

from PIL import Image

from app.services import (
    TAGGER_MODEL_ML,
    TAGGER_MODEL_WD_EVA02,
    TAGGER_MODEL_WD_SWINV2,
    _even_frame_indices,
    categories_exclude_dirs,
    choose_best_tags,
    discover_tag_folders,
    ensure_collision_free_destination,
    extract_scores,
    extract_scores_with_experimental_media,
    global_top_tags,
    migrate_file,
    normalize_tag_name,
    pool_frame_scores,
    sample_gif_frames,
    sanitize_folder_name,
    scaled_media_sample_count,
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


def test_discover_tag_folders_selected_tag_exact_match_with_special_chars(tmp_path: Path) -> None:
    tags = {"remodel_(kantai_collection)", "monster_girl"}
    mappings = discover_tag_folders(
        tmp_path,
        tags,
        selected_folders=["remodel_(kantai_collection)", "monster_girl"],
    )
    mapped = {m.folder_name: m.matched_tag for m in mappings}
    assert mapped["remodel_(kantai_collection)"] == "remodel_(kantai_collection)"
    assert mapped["monster_girl"] == "monster_girl"


def test_sanitize_folder_name_replaces_forbidden_chars() -> None:
    assert sanitize_folder_name('remodel_(kantai_collection)') == "remodel_(kantai_collection)"
    assert sanitize_folder_name('tag:with*bad|chars?') == "tag_with_bad_chars_"


def test_scan_images_filters_types(tmp_path: Path) -> None:
    img_path = tmp_path / "ok.jpg"
    Image.new("RGB", (16, 16), color="red").save(img_path)

    (tmp_path / "video.mp4").write_text("not-video", encoding="utf-8")
    (tmp_path / "anim.gif").write_text("gif", encoding="utf-8")
    (tmp_path / "doc.txt").write_text("txt", encoding="utf-8")
    # broken.png has a known image extension so it is accepted at scan time;
    # corrupt files are caught later during inference (not at discovery).
    (tmp_path / "broken.png").write_text("broken", encoding="utf-8")

    result = scan_images(tmp_path)
    names = {p.name for p in result.image_paths}
    assert "ok.jpg" in names
    assert "broken.png" in names  # accepted at scan; inference will handle corruption
    assert result.stats.eligible_images == 2
    assert result.stats.ignored_gif == 1
    assert result.stats.ignored_unsupported >= 1  # doc.txt
    assert result.stats.failed_to_read == 0  # no OSError-level failures


def test_scan_images_includes_gif_and_video_when_experimental_enabled(tmp_path: Path) -> None:
    gif_path = tmp_path / "anim.gif"
    Image.new("RGB", (16, 16), color="orange").save(gif_path, format="GIF")
    video_path = tmp_path / "clip.mp4"
    video_path.write_text("fake-video", encoding="utf-8")

    result = scan_images(tmp_path, experimental_media_enabled=True)
    names = {p.name for p in result.image_paths}
    assert "anim.gif" in names
    assert "clip.mp4" in names
    assert result.stats.ignored_gif == 0


def test_scan_images_includes_extensionless_valid_image(tmp_path: Path) -> None:
    extless = tmp_path / "no_extension_image"
    Image.new("RGB", (16, 16), color="purple").save(extless, format="PNG")

    result = scan_images(tmp_path)
    names = {p.name for p in result.image_paths}
    assert "no_extension_image" in names
    assert result.stats.eligible_images == 1


def test_scan_images_includes_unknown_extension_if_decodable(tmp_path: Path) -> None:
    odd_ext = tmp_path / "odd_format.weird"
    Image.new("RGB", (16, 16), color="yellow").save(odd_ext, format="PNG")

    result = scan_images(tmp_path)
    names = {p.name for p in result.image_paths}
    assert "odd_format.weird" in names
    assert result.stats.eligible_images == 1
    assert result.stats.ignored_unsupported == 0


def test_scan_images_rejects_unknown_extension_when_not_image(tmp_path: Path) -> None:
    non_image = tmp_path / "not_image.weird"
    non_image.write_text("plain text", encoding="utf-8")

    result = scan_images(tmp_path)
    names = {p.name for p in result.image_paths}
    assert "not_image.weird" not in names
    assert result.stats.eligible_images == 0
    assert result.stats.ignored_unsupported == 1


def test_scan_images_descends_into_subdirectories(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    nested_img = nested / "nested.jpg"
    Image.new("RGB", (16, 16), color="blue").save(nested_img)

    top_img = tmp_path / "top.jpg"
    Image.new("RGB", (16, 16), color="green").save(top_img)

    result = scan_images(tmp_path)
    names = {p.name for p in result.image_paths}
    assert "top.jpg" in names
    assert "nested.jpg" in names


def test_scan_images_excludes_specified_directory(tmp_path: Path) -> None:
    included_dir = tmp_path / "included"
    included_dir.mkdir()
    Image.new("RGB", (16, 16), color="red").save(included_dir / "in.jpg")

    excluded_dir = tmp_path / "excluded"
    excluded_dir.mkdir()
    Image.new("RGB", (16, 16), color="blue").save(excluded_dir / "out.jpg")

    result = scan_images(tmp_path, exclude_dirs={excluded_dir})
    names = {p.name for p in result.image_paths}
    assert "in.jpg" in names
    assert "out.jpg" not in names


def test_categories_exclude_dirs_skips_parent_categories_root(tmp_path: Path) -> None:
    """Inbox under categories_root must not exclude the whole tree (0-image runs)."""
    art = tmp_path / "library"
    inbox = art / "inbox" / "unsorted"
    dest = art / "loli"
    inbox.mkdir(parents=True)
    dest.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="red").save(inbox / "a.jpg")
    Image.new("RGB", (8, 8), color="blue").save(dest / "b.jpg")

    # Parent categories root: do not exclude (would wipe the inbox scan).
    assert categories_exclude_dirs(inbox, art) == set()
    kept = scan_images(inbox, exclude_dirs=categories_exclude_dirs(inbox, art))
    assert {p.name for p in kept.image_paths} == {"a.jpg"}

    # Nested destination under scan root: still excluded.
    scan_root = tmp_path / "inbox"
    nested_cats = scan_root / "organised"
    nested_cats.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="green").save(nested_cats / "c.jpg")
    Image.new("RGB", (8, 8), color="yellow").save(scan_root / "d.jpg")
    excl = categories_exclude_dirs(scan_root, nested_cats)
    assert excl == {nested_cats.resolve()}
    nested = scan_images(scan_root, exclude_dirs=excl)
    assert {p.name for p in nested.image_paths} == {"d.jpg"}


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


def test_global_top_tags_orders_by_score() -> None:
    tops = global_top_tags({"a": 0.1, "b": 0.9, "c": 0.5}, limit=2)
    assert tops == [{"tag": "b", "score": 0.9}, {"tag": "c", "score": 0.5}]


def test_extract_scores_routes_ml_danbooru(monkeypatch, tmp_path: Path) -> None:
    import app.services as services

    calls: list[str] = []

    monkeypatch.setattr("app.providers.ensure_nvidia_dll_search_path", lambda: [])
    monkeypatch.setattr("app.providers.preload_onnx_runtime_dlls", lambda: None)
    monkeypatch.setattr(
        services,
        "_run_mldanbooru",
        lambda _image: calls.append("ml") or {"1girl": 0.9, "black_hair": 0.8},
    )
    monkeypatch.setattr(
        services,
        "_run_wd14",
        lambda *_a, **_k: calls.append("wd") or {"solo": 0.7},
    )

    img = tmp_path / "x.jpg"
    img.write_text("x", encoding="utf-8")
    scores = extract_scores(img, tagger_model=TAGGER_MODEL_ML)
    assert calls == ["ml"]
    assert scores["1girl"] == 0.9
    assert scores["black_hair"] == 0.8


def test_extract_scores_routes_wd_models(monkeypatch, tmp_path: Path) -> None:
    import app.services as services

    seen: list[tuple[str, float]] = []

    def fake_wd(image, *, model_name: str, general_threshold: float):
        seen.append((model_name, general_threshold))
        return {"solo": 0.66, "long_hair": 0.55}

    monkeypatch.setattr("app.providers.ensure_nvidia_dll_search_path", lambda: [])
    monkeypatch.setattr("app.providers.preload_onnx_runtime_dlls", lambda: None)
    monkeypatch.setattr(services, "_run_mldanbooru", lambda _image: {"nope": 1.0})
    monkeypatch.setattr(services, "_run_wd14", fake_wd)

    img = tmp_path / "y.jpg"
    img.write_text("y", encoding="utf-8")

    scores = extract_scores(
        img, tagger_model=TAGGER_MODEL_WD_SWINV2, wd_general_threshold=0.41
    )
    assert seen[-1] == ("SwinV2_v3", 0.41)
    assert scores["solo"] == 0.66
    assert scores["long_hair"] == 0.55

    extract_scores(img, tagger_model=TAGGER_MODEL_WD_EVA02, wd_general_threshold=0.2)
    assert seen[-1] == ("EVA02_Large", 0.2)


def test_normalize_score_tags_collapses_space_forms() -> None:
    from app.services import _normalize_score_tags

    scores = _normalize_score_tags({"Black Hair": 0.8, "black_hair": 0.9, "solo": 0.5})
    assert scores["black_hair"] == 0.9
    assert scores["solo"] == 0.5


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


def test_destination_path_supports_nested_taxonomy_folders(tmp_path: Path) -> None:
    from app.services import destination_path

    nested = destination_path(tmp_path, "Voyeur/panties")
    assert nested == tmp_path / "Voyeur" / "panties"
    flat = destination_path(tmp_path, "loli")
    assert flat == tmp_path / "loli"
    # Path separators are preserved as nesting; only illegal chars are sanitized.
    weird = destination_path(tmp_path, "Voyeur/panty:shot")
    assert weird == tmp_path / "Voyeur" / "panty_shot"


def test_even_frame_indices_covers_span() -> None:
    assert _even_frame_indices(1, 8) == [0]
    assert _even_frame_indices(8, 8) == list(range(8))
    idxs = _even_frame_indices(100, 4)
    assert len(idxs) == 4
    assert idxs[0] < idxs[-1]
    assert idxs[0] >= 0 and idxs[-1] <= 99


def test_scaled_media_sample_count_grows_with_length(monkeypatch) -> None:
    monkeypatch.delenv("MEDIA_SAMPLE_FRAMES", raising=False)
    monkeypatch.setenv("MEDIA_SAMPLE_FRAMES_MIN", "4")
    monkeypatch.setenv("MEDIA_SAMPLE_FRAMES_MAX", "48")

    short = scaled_media_sample_count(duration_seconds=2.0)
    medium = scaled_media_sample_count(duration_seconds=30.0)
    long = scaled_media_sample_count(duration_seconds=4000.0)
    assert short == 8
    assert medium == 12
    assert long == 48
    assert long >= medium >= short

    gif_short = scaled_media_sample_count(total_frames=6)
    gif_long = scaled_media_sample_count(total_frames=200)
    assert gif_short == 4
    assert gif_long == 48


def test_pool_frame_scores_uses_presence_corroboration() -> None:
    pooled = pool_frame_scores(
        [
            {"loli": 0.9, "1girl": 0.8},
            {"1girl": 0.7},
            {"loli": 0.0, "1girl": 0.6},
            {},
        ]
    )
    # Single-frame loli spike suppressed (omitted from pooled map).
    assert "loli" not in pooled
    # 1girl hits on 3 frames → top-3 mean.
    assert abs(pooled["1girl"] - ((0.8 + 0.7 + 0.6) / 3)) < 1e-6


def test_sample_gif_frames_evenly(tmp_path: Path) -> None:
    path = tmp_path / "anim.gif"
    frames = [Image.new("RGB", (8, 8), color=c) for c in ("red", "green", "blue", "yellow", "purple", "cyan")]
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
    )
    sampled = sample_gif_frames(path, sample_count=3)
    assert len(sampled) == 3
    assert all(frame.mode == "RGB" for frame in sampled)


def test_experimental_media_gif_pools_frame_scores(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "multi.gif"
    frames = [Image.new("RGB", (16, 16), color=(50, 20, 20)) for _ in range(4)]
    for img in frames:
        px = img.load()
        for x in range(16):
            px[x, 2] = (255, 255, 255)
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
    )

    def fake_score_many(images, **kwargs):
        assert kwargs.get("raw_general") is True
        out = []
        for idx, _img in enumerate(images):
            out.append({"loli": 0.85 if idx % 2 == 0 else 0.05, "1girl": 0.5})
        return out

    class _Engine:
        def score_many(self, images, **kwargs):
            return fake_score_many(images, **kwargs)

    monkeypatch.setattr("app.inference_engine.get_engine", lambda: _Engine())

    scores = extract_scores_with_experimental_media(
        path,
        experimental_media_enabled=True,
        tagger_model=TAGGER_MODEL_WD_SWINV2,
        sample_count=4,
    )
    assert "loli" in scores
    assert "1girl" in scores
    assert scores["loli"] > 0.5
    assert scores["1girl"] >= 0.5