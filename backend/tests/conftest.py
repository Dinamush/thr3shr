from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator

import pytest

from app.storage import execute, init_db

# TestClient sends Host: testserver; allow it alongside the local defaults
# before any test imports app.main and builds its middleware.
os.environ.setdefault("THR3SHR_ALLOWED_HOSTS", "testserver,127.0.0.1,localhost")

# Run/reclassify workers are daemon threads that outlive the request that
# started them. A test that returns early must not leave one writing after
# DB_PATH is restored: it would land in the next test's DB or the real app.db.
_WORKER_JOIN_DEADLINE_S = 60.0


def _join_new_threads(before: set[threading.Thread]) -> None:
    deadline = time.monotonic() + _WORKER_JOIN_DEADLINE_S
    for thread in threading.enumerate():
        if thread in before or thread is threading.current_thread():
            continue
        thread.join(timeout=max(0.0, deadline - time.monotonic()))


@pytest.fixture(autouse=True)
def _isolated_app_db(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Keep API/integration tests from clobbering the developer's app.db.

    Uses a dedicated temp directory (not the test's tmp_path) so scan/filesystem
    tests are not polluted by the SQLite file.
    """
    threads_before = set(threading.enumerate())
    db_dir = tmp_path_factory.mktemp("app_db")
    db_path = db_dir / "test_app.db"
    monkeypatch.setattr("app.storage.DB_PATH", db_path)
    # Deterministic single-image tasks for API tests (batch path bypasses many mocks).
    monkeypatch.setenv("INFERENCE_MODE", "single")
    monkeypatch.setenv("INFERENCE_BATCH_SIZE", "1")
    monkeypatch.setenv("MAX_INFERENCE_WORKERS", "1")
    init_db()
    # Settings batch size is authoritative over env; keep tests on single-image mode.
    execute(
        """
        UPDATE settings
        SET inference_batch_size = 1, max_inference_workers = 1
        WHERE id = 1
        """
    )
    yield
    # Teardown runs before monkeypatch restores DB_PATH.
    _join_new_threads(threads_before)
