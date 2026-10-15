"""Only the local UI may drive the unauthenticated file-moving API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import DEFAULT_ALLOWED_HOSTS, DEFAULT_CORS_ORIGINS, _env_list, app


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
