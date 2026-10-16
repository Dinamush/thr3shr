"""Only the local UI may drive the unauthenticated file-moving API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import DEFAULT_ALLOWED_HOSTS, DEFAULT_CORS_ORIGINS, _env_list, app
from app.storage import execute


def _seed_approved_move(tmp_path: Path) -> tuple[int, Path, Path]:
    root = tmp_path / "root"
    dest = tmp_path / "cats" / "tag"
    root.mkdir()
    src = root / "a.jpg"
    src.write_bytes(b"aaa")
    run_id = execute(
        """
        INSERT INTO runs (
            root_repo, categories_root, confidence_threshold, status,
            total_images, processed_images, failed_images, cancel_requested, tagger_model
        ) VALUES (?, ?, 0.6, 'completed', 1, 1, 0, 0, 'wd_swinv2_v3')
        """,
        (str(root), str(dest.parent)),
    )
    execute(
        """
        INSERT INTO items (
            run_id, file_path, relative_path, primary_tag, final_tag, final_destination, status
        ) VALUES (?, ?, 'a.jpg', 'tag', 'tag', ?, 'approved')
        """,
        (run_id, str(src), str(dest)),
    )
    return run_id, src, dest / "a.jpg"


def _simple_move_post(client: TestClient, run_id: int, headers: dict[str, str]):
    # No Content-Type: a browser sends this cross-site without a CORS preflight.
    return client.post(f"/api/runs/{run_id}/migrate", content=b'{"mode":"move"}', headers=headers)


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_cross_site_write_is_refused_before_the_handler(tmp_path: Path, headers) -> None:
    run_id, src, moved = _seed_approved_move(tmp_path)
    with TestClient(app) as client:
        resp = _simple_move_post(client, run_id, headers)
    assert resp.status_code == 403
    assert src.read_bytes() == b"aaa"
    assert not moved.exists()


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://localhost:5173"},
        {"Origin": "http://127.0.0.1:4173", "Sec-Fetch-Site": "same-site"},
        {},
    ],
)
def test_local_ui_and_script_writes_are_allowed(tmp_path: Path, headers) -> None:
    run_id, src, moved = _seed_approved_move(tmp_path)
    with TestClient(app) as client:
        resp = _simple_move_post(client, run_id, headers)
    assert resp.status_code == 200
    assert resp.json()["migrated_count"] == 1
    assert not src.exists()
    assert moved.read_bytes() == b"aaa"


def test_local_ui_origin_gets_cors_access() -> None:
    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_foreign_origin_gets_no_cors_access() -> None:
    with TestClient(app) as client:
        simple = client.get("/api/settings", headers={"Origin": "https://evil.example"})
        preflight = client.options(
            "/api/runs/1/migrate",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert "access-control-allow-origin" not in simple.headers
    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers


def test_unknown_host_header_is_rejected() -> None:
    # DNS rebinding: a foreign name resolving to 127.0.0.1 arrives with its own Host.
    with TestClient(app, base_url="http://evil.example:8000") as client:
        resp = client.get("/api/settings")
    assert resp.status_code == 400


def test_env_lists_override_defaults(monkeypatch) -> None:
    assert _env_list("THR3SHR_TEST_UNSET_LIST", DEFAULT_CORS_ORIGINS) == list(DEFAULT_CORS_ORIGINS)
    monkeypatch.setenv("THR3SHR_TEST_LIST", " http://nas.lan:5173 , http://10.0.0.5:5173 ,")
    assert _env_list("THR3SHR_TEST_LIST", DEFAULT_ALLOWED_HOSTS) == [
        "http://nas.lan:5173",
        "http://10.0.0.5:5173",
    ]


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://lan-ui.example"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_wildcard_cors_origins_allow_any_write(tmp_path: Path, monkeypatch, headers) -> None:
    monkeypatch.setattr(main, "cors_origins", ["*"])
    run_id, src, moved = _seed_approved_move(tmp_path)
    with TestClient(app) as client:
        resp = _simple_move_post(client, run_id, headers)
    assert resp.status_code == 200
    assert not src.exists()
    assert moved.read_bytes() == b"aaa"
