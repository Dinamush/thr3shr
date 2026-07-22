from __future__ import annotations

from pathlib import Path

import pytest

from app.storage import init_db


@pytest.fixture(autouse=True)
def _isolated_app_db(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep API/integration tests from clobbering the developer's app.db.

    Uses a dedicated temp directory (not the test's tmp_path) so scan/filesystem
    tests are not polluted by the SQLite file.
    """
    db_dir = tmp_path_factory.mktemp("app_db")
    db_path = db_dir / "test_app.db"
    monkeypatch.setattr("app.storage.DB_PATH", db_path)
    # Deterministic single-image tasks for API tests (batch path bypasses many mocks).
    monkeypatch.setenv("INFERENCE_MODE", "single")
    monkeypatch.setenv("INFERENCE_BATCH_SIZE", "1")
    monkeypatch.setenv("MAX_INFERENCE_WORKERS", "1")
    init_db()
