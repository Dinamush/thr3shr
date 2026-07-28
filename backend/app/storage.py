from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "app.db"
logger = logging.getLogger(__name__)

# Writers (classify workers) + readers (UI poll) contend on a multi‑GB DB.
# WAL lets readers proceed during writes; busy_timeout retries instead of
# immediate "database is locked" failures that abort whole runs.
_SQLITE_TIMEOUT_S = 60.0


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=_SQLITE_TIMEOUT_S, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db() -> None:
    try:
        with get_connection() as conn:
            # Ensure WAL is sticky for this DB file (also set per-connection above).
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                root_repo TEXT NOT NULL DEFAULT '',
                categories_root TEXT NOT NULL DEFAULT '',
                confidence_threshold REAL NOT NULL DEFAULT 0.6,
                default_migrate_mode TEXT NOT NULL DEFAULT 'copy',
                scan_recursive INTEGER NOT NULL DEFAULT 1,
                experimental_media_enabled INTEGER NOT NULL DEFAULT 0
            );

            INSERT OR IGNORE INTO settings (id) VALUES (1);

            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                root_repo TEXT NOT NULL,
                categories_root TEXT NOT NULL,
                confidence_threshold REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                total_images INTEGER NOT NULL DEFAULT 0,
                processed_images INTEGER NOT NULL DEFAULT 0,
                failed_images INTEGER NOT NULL DEFAULT 0,
                started_at TEXT,
                finished_at TEXT,
                last_error TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                primary_tag TEXT,
                primary_score REAL,
                secondary_json TEXT NOT NULL DEFAULT '[]',
                suggested_destination TEXT,
                final_tag TEXT,
                final_destination TEXT,
                status TEXT NOT NULL,
                needs_review INTEGER NOT NULL DEFAULT 0,
                review_reason TEXT,
                migrated_to TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """
            )
            _ensure_settings_columns(conn)
            _ensure_runs_columns(conn)
            _ensure_items_columns(conn)
    except sqlite3.DatabaseError:
        logger.exception("failed to initialize database")
        raise


def _ensure_settings_columns(conn: sqlite3.Connection) -> None:
    expected_columns = {
        "scan_recursive": "INTEGER NOT NULL DEFAULT 1",
        "experimental_media_enabled": "INTEGER NOT NULL DEFAULT 0",
        "experimental_style_detector_enabled": "INTEGER NOT NULL DEFAULT 0",
        "hybrid_ml_on_review": "INTEGER NOT NULL DEFAULT 1",
        "tagging_domain": "TEXT NOT NULL DEFAULT 'drawn'",
        "selected_tags_json": "TEXT NOT NULL DEFAULT '[]'",
        "max_inference_workers": "INTEGER NOT NULL DEFAULT 2",
        "inference_batch_size": "INTEGER NOT NULL DEFAULT 4",
        "force_cpu_inference": "INTEGER NOT NULL DEFAULT 0",
        "tagger_model": "TEXT NOT NULL DEFAULT 'wd_swinv2_v3'",
        "wd_general_threshold": "REAL NOT NULL DEFAULT 0.35",
    }
    rows = conn.execute("PRAGMA table_info(settings)").fetchall()
    existing = {row[1] for row in rows}
    for name, definition in expected_columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE settings ADD COLUMN {name} {definition}")


def _ensure_runs_columns(conn: sqlite3.Connection) -> None:
    expected_columns = {
        "status": "TEXT NOT NULL DEFAULT 'pending'",
        "total_images": "INTEGER NOT NULL DEFAULT 0",
        "processed_images": "INTEGER NOT NULL DEFAULT 0",
        "failed_images": "INTEGER NOT NULL DEFAULT 0",
        "started_at": "TEXT",
        "finished_at": "TEXT",
        "last_error": "TEXT",
        "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
        "tagger_model": "TEXT NOT NULL DEFAULT 'wd_swinv2_v3'",
        "tagging_domain": "TEXT NOT NULL DEFAULT 'drawn'",
    }
    rows = conn.execute("PRAGMA table_info(runs)").fetchall()
    existing = {row[1] for row in rows}
    for name, definition in expected_columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")


def _ensure_items_columns(conn: sqlite3.Connection) -> None:
    expected_columns = {
        "full_scores_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    rows = conn.execute("PRAGMA table_info(items)").fetchall()
    existing = {row[1] for row in rows}
    for name, definition in expected_columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE items ADD COLUMN {name} {definition}")


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    try:
        with get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None
    except sqlite3.DatabaseError:
        logger.exception("fetch_one failed")
        raise


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    try:
        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.DatabaseError:
        logger.exception("fetch_all failed")
        raise


def execute(query: str, params: tuple[Any, ...] = ()) -> int:
    try:
        with get_connection() as conn:
            cur = conn.execute(query, params)
            conn.commit()
            return cur.lastrowid
    except sqlite3.DatabaseError:
        logger.exception("execute failed")
        raise


def execute_many(query: str, params: list[tuple[Any, ...]]) -> None:
    try:
        with get_connection() as conn:
            conn.executemany(query, params)
            conn.commit()
    except sqlite3.DatabaseError:
        logger.exception("execute_many failed")
        raise


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True)


def from_json(value: str, default: Any = None) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        logger.warning("invalid json value encountered; returning default")
        return [] if default is None else default


@contextmanager
def transaction():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except sqlite3.DatabaseError:
        conn.rollback()
        logger.exception("transaction failed; rolled back")
        raise
    finally:
        conn.close()
