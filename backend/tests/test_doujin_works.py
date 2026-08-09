"""Doujin works: scan, sample, route, migrate junctions."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

from app.doujin_works import (
    DOUJIN_FAVOURITE_FOLDERS,
    classify_doujin_work,
    create_tag_link,
    doujin_destination,
    doujin_destination_folder,
    even_page_indices,
    load_sample_images,
    scan_doujin_works,
    write_tags_sidecar,
)
from app.main import app
from app.storage import execute, fetch_one, to_json
from app.taxonomy import choose_best_destination, resolve_taxonomy_folder


def _rgb(path: Path, color: tuple[int, int, int] = (20, 40, 60)) -> None:
    Image.new("RGB", (32, 32), color).save(path)


def test_milf_and_inseki_taxonomy_routing() -> None:
    assert resolve_taxonomy_folder("milf").folder == "milf"
    assert resolve_taxonomy_folder("inseki").folder == "incest"
    folder, score, _ = choose_best_destination(
        {"mature_female": 0.88}, set(DOUJIN_FAVOURITE_FOLDERS)
    )
    assert folder == "milf"
    assert score is not None and score > 0.5
    folder, _, _ = choose_best_destination(
        {"incest": 0.95}, set(DOUJIN_FAVOURITE_FOLDERS)
    )
    assert folder == "incest"


def test_doujin_favourites_include_bestiality_and_pokemon() -> None:
    assert "bestiality" in DOUJIN_FAVOURITE_FOLDERS
    assert "Pokemon" in DOUJIN_FAVOURITE_FOLDERS
    folder, score, _ = choose_best_destination(
        {"bestiality": 0.89, "animal_penis": 0.7}, set(DOUJIN_FAVOURITE_FOLDERS)
    )
    assert folder == "bestiality"
    assert score is not None and score > 0.5
    folder, score, _ = choose_best_destination(
        {"pokephilia": 0.88, "pokemon_(creature)": 0.96},
        set(DOUJIN_FAVOURITE_FOLDERS),
    )
    assert folder == "Pokemon"
    assert score is not None and score > 0.5


def test_doujin_favourites_include_ntr_tentacles_furry_android() -> None:
    for name in ("NTR", "tentacles", "furry", "android"):
        assert name in DOUJIN_FAVOURITE_FOLDERS
    folder, score, _ = choose_best_destination(
        {"netorare": 0.85, "cheating_(relationship)": 0.7},
        set(DOUJIN_FAVOURITE_FOLDERS),
    )
    assert folder == "NTR"
    assert score is not None and score > 0.5
    folder, score, _ = choose_best_destination(
        {"tentacles": 0.9, "tentacle_sex": 0.8}, set(DOUJIN_FAVOURITE_FOLDERS)
    )
    assert folder == "tentacles"


def test_doujin_favourites_include_group_sex() -> None:
    assert "group_sex" in DOUJIN_FAVOURITE_FOLDERS
    folder, score, _ = choose_best_destination(
        {"gangbang": 0.9, "sex": 0.85}, set(DOUJIN_FAVOURITE_FOLDERS)
    )
    assert folder == "group_sex"
    assert score is not None and score > 0.5


def test_scan_doujin_works_folders_and_archives(tmp_path: Path) -> None:
    work = tmp_path / "Title A"
    work.mkdir()
    _rgb(work / "001.jpg")
    _rgb(work / "002.png")
    empty = tmp_path / "Empty"
    empty.mkdir()
    cbz = tmp_path / "Loose.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (1, 2, 3)).save(buf, format="JPEG")
        zf.writestr("page1.jpg", buf.getvalue())

    works = scan_doujin_works(tmp_path)
    names = {w.name for w in works}
    assert names == {"Title A", "Loose.cbz"}
    kinds = {w.name: w.kind for w in works}
    assert kinds["Title A"] == "folder"
    assert kinds["Loose.cbz"] == "archive"


def test_even_page_indices_includes_cover_and_spreads() -> None:
    idxs = even_page_indices(20, 5)
    assert idxs[0] == 0
    assert idxs[-1] == 19
    assert len(idxs) == 5
    assert idxs == sorted(set(idxs))


def test_load_sample_images_folder_and_cbz(tmp_path: Path) -> None:
    folder = tmp_path / "Vol1"
    folder.mkdir()
    for i in range(8):
        _rgb(folder / f"{i:03d}.jpg", (i * 10, 0, 0))
    from app.doujin_works import DoujinWork

    folder_work = DoujinWork(path=folder, name=folder.name, kind="folder")
    images = load_sample_images(folder_work, sample_count=4)
    assert 4 <= len(images) <= 8

    cbz = tmp_path / "vol.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        for i in range(6):
            buf = io.BytesIO()
            Image.new("RGB", (12, 12), (i, i, i)).save(buf, format="JPEG")
            zf.writestr(f"p{i}.jpg", buf.getvalue())
    archive_work = DoujinWork(path=cbz, name=cbz.name, kind="archive")
    images = load_sample_images(archive_work, sample_count=4)
    assert len(images) >= 4


def test_classify_doujin_work_pools_and_sets_primary(tmp_path: Path) -> None:
    folder = tmp_path / "IncestTitle"
    folder.mkdir()
    for i in range(6):
        _rgb(folder / f"{i}.jpg")
    from app.doujin_works import DoujinWork

    work = DoujinWork(path=folder, name=folder.name, kind="folder")

    def fake_extract(image, **kwargs):
        return {"incest": 0.9, "loli": 0.4, "milf": 0.2}

    with patch("app.doujin_works.extract_scores", side_effect=fake_extract):
        result = classify_doujin_work(
            work,
            matched_tags=set(DOUJIN_FAVOURITE_FOLDERS),
            confidence_threshold=0.55,
        )
    assert result.primary_tag == "incest"
    assert result.needs_review is False
    assert any(row["tag"] == "incest" for row in result.category_tags)


def test_doujin_destination_layout(tmp_path: Path) -> None:
    dest = doujin_destination(tmp_path, "loli", "My Work")
    assert dest == tmp_path / "Doujins" / "loli" / "My Work"
    assert doujin_destination_folder(tmp_path, "monster_girl") == (
        tmp_path / "Doujins" / "monster_girl"
    )


def test_create_tag_link_and_sidecar(tmp_path: Path) -> None:
    target = tmp_path / "Doujins" / "loli" / "WorkA"
    target.mkdir(parents=True)
    _rgb(target / "cover.jpg")
    write_tags_sidecar(target, ["loli", "incest"])
    assert (target / "tags.json").is_file()

    link = tmp_path / "Doujins" / "incest" / "WorkA"
    create_tag_link(link, target)
    assert link.exists()
    assert link.is_dir()


def test_migrate_doujin_folder_moves_and_junctions(tmp_path: Path) -> None:
    root = tmp_path / "DoujinsInbox"
    cats = tmp_path / "Art"
    root.mkdir()
    cats.mkdir()
    work = root / "BigTitle"
    work.mkdir()
    _rgb(work / "01.jpg")

    dest_folder = cats / "Doujins" / "loli"
    run_id = execute(
        """
        INSERT INTO runs (
            root_repo, categories_root, confidence_threshold, status,
            total_images, processed_images, failed_images, cancel_requested, tagger_model
        ) VALUES (?, ?, 0.55, 'completed', 1, 1, 0, 0, 'wd_swinv2_v3')
        """,
        (str(root), str(cats)),
    )
    execute(
        """
        INSERT INTO items (
            run_id, file_path, relative_path, primary_tag, primary_score, secondary_json,
            full_scores_json, suggested_destination, final_tag, final_destination,
            status, needs_review, review_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'approved', 0, NULL)
        """,
        (
            run_id,
            str(work),
            work.name,
            "loli",
            0.9,
            to_json([{"tag": "incest", "score": 0.7}]),
            to_json({}),
            str(dest_folder),
            "loli",
            str(dest_folder),
        ),
    )

    with TestClient(app) as client:
        resp = client.post(f"/api/runs/{run_id}/migrate", json={"mode": "copy"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["migrated_count"] == 1
    assert body["failed_count"] == 0

    moved = cats / "Doujins" / "loli" / "BigTitle"
    assert moved.is_dir()
    assert not work.exists()
    junction = cats / "Doujins" / "incest" / "BigTitle"
    assert junction.exists()
    assert (moved / "tags.json").is_file()
    row = fetch_one("SELECT status, migrated_to FROM items WHERE run_id = ?", (run_id,))
    assert row["status"] == "migrated"
    assert Path(row["migrated_to"]) == moved
