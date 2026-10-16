"""Run history list: finished runs must stay reachable from the UI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.storage import execute


def _seed_run(status: str, root: str, item_statuses: list[str]) -> int:
    run_id = execute(
        """
        INSERT INTO runs (
            root_repo, categories_root, confidence_threshold, status,
            total_images, processed_images, failed_images, cancel_requested, tagger_model
        ) VALUES (?, '/cats', 0.6, ?, ?, ?, 0, 0, 'wd_swinv2_v3')
        """,
        (root, status, len(item_statuses), len(item_statuses)),
    )
    for idx, item_status in enumerate(item_statuses):
        execute(
            """
            INSERT INTO items (run_id, file_path, relative_path, status, needs_review)
            VALUES (?, ?, ?, ?, 0)
            """,
            (run_id, f"{root}/{idx}.jpg", f"{idx}.jpg", item_status),
        )
    return run_id


def test_runs_list_is_newest_first_with_item_counts() -> None:
    first = _seed_run("completed", "/inbox/a", ["approved", "approved", "migrated"])
    second = _seed_run("cancelled", "/inbox/b", ["proposed"])
    empty = _seed_run("failed", "/inbox/c", [])

    with TestClient(app) as client:
        resp = client.get("/api/runs")
    resp.raise_for_status()
    runs = resp.json()

    assert [r["run_id"] for r in runs] == [empty, second, first]
    by_id = {r["run_id"]: r for r in runs}
    assert by_id[first]["item_counts"] == {"approved": 2, "migrated": 1}
    assert by_id[first]["status"] == "completed"
    assert by_id[first]["root_repo"] == "/inbox/a"
    assert by_id[second]["item_counts"] == {"proposed": 1}
    assert by_id[empty]["item_counts"] == {}
    assert by_id[first]["tagger_model"] == "wd_swinv2_v3"


def test_runs_list_respects_limit() -> None:
    ids = [_seed_run("completed", f"/inbox/{n}", []) for n in range(3)]

    with TestClient(app) as client:
        resp = client.get("/api/runs", params={"limit": 2})
        assert client.get("/api/runs", params={"limit": 0}).status_code == 422
    resp.raise_for_status()
    assert [r["run_id"] for r in resp.json()] == [ids[2], ids[1]]
