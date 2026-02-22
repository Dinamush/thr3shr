from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "app.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                root_repo TEXT NOT NULL DEFAULT '',
                categories_root TEXT NOT NULL DEFAULT '',
                confidence_threshold REAL NOT NULL DEFAULT 0.6,
                default_migrate_mode TEXT NOT NULL DEFAULT 'copy'
            );

            INSERT OR IGNORE INTO settings (id) VALUES (1);

            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                root_repo TEXT NOT NULL,
                categories_root TEXT NOT NULL,
                confidence_threshold REAL NOT NULL
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


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def execute(query: str, params: tuple[Any, ...] = ()) -> int:
    with get_connection() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.lastrowid


def execute_many(query: str, params: list[tuple[Any, ...]]) -> None:
    with get_connection() as conn:
        conn.executemany(query, params)
        conn.commit()


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True)


def from_json(value: str) -> Any:
    return json.loads(value)
