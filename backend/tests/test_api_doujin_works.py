"""API smoke for doujin_works run mode."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

from app.doujin_works import DoujinClassifyResult, DoujinWork
from app.main import app
from app.storage import fetch_all, fetch_one


def _rgb(path: Path) -> None:
    Image.new("RGB", (24, 24), (10, 20, 30)).save(path)


def test_start_doujin_works_run_inserts_work_items(tmp_path: Path) -> None:
    root = tmp_path / "Doujins"
    cats = tmp_path / "Art"
    root.mkdir()
    cats.mkdir()
    work = root / "Sample Title"
    work.mkdir()
    _rgb(work / "01.jpg")
    _rgb(work / "02.jpg")

    fake = DoujinClassifyResult(
        work=DoujinWork(path=work, name=work.name, kind="folder"),
        scores={"loli": 0.9, "incest": 0.6},
        primary_tag="loli",
        primary_score=0.9,
        category_tags=[
            {"tag": "loli", "score": 0.9},
            {"tag": "incest", "score": 0.6},
        ],
        needs_review=False,
    )

    with patch("app.api.classify_doujin_work", return_value=fake):
        with TestClient(app) as client:
            start = client.post(
                "/api/runs/start",
                json={
                    "root_repo": str(root),
                    "categories_root": str(cats),
                    "confidence_threshold": 0.55,
                    "run_mode": "doujin_works",
                },
            )
            assert start.status_code == 200, start.text
            run_id = start.json()["run_id"]
            assert "Doujin works" in start.json()["message"]

            # Worker is a daemon thread; wait via status polling.
            for _ in range(40):
                status = client.get(f"/api/runs/{run_id}/status")
                assert status.status_code == 200
                if status.json()["status"] in {"completed", "failed", "cancelled"}:
                    break
                import time

                time.sleep(0.05)

    run = fetch_one("SELECT status, tagger_model, total_images FROM runs WHERE id = ?", (run_id,))
    assert run["status"] == "completed"
    assert str(run["tagger_model"]).startswith("wd_")
    assert run["total_images"] == 1
    items = fetch_all("SELECT * FROM items WHERE run_id = ?", (run_id,))
    assert len(items) == 1
    item = items[0]
    assert item["relative_path"] == "Sample Title"
    assert item["primary_tag"] == "loli"
    assert "Doujins" in Path(item["suggested_destination"]).parts
    assert item["status"] == "approved"
