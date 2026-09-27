"""SQLite persistence for invite counts. Survives restarts."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inviters (
    inviter_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    total INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS joins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id TEXT NOT NULL,
    inviter_id TEXT,
    joined_at TEXT NOT NULL
);
"""


class InviteStore:
    def __init__(self, path: str = "invites.db") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def add_invite(self, inviter_id: str, name: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO inviters (inviter_id, name, total, updated_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(inviter_id) DO UPDATE SET
                    name = excluded.name,
                    total = inviters.total + 1,
                    updated_at = excluded.updated_at
                """,
                (inviter_id, name, now),
            )
            self._conn.commit()

    def record_join(self, member_id: str, inviter_id: str | None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO joins (member_id, inviter_id, joined_at) VALUES (?, ?, ?)",
                (member_id, inviter_id, now),
            )
            self._conn.commit()

    def get_count(self, inviter_id: str) -> tuple[int, str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT total, name FROM inviters WHERE inviter_id = ?", (inviter_id,)
            ).fetchone()
        if row:
            return int(row[0]), str(row[1])
        return 0, ""

    def leaderboard(self, limit: int = 10) -> list[tuple[str, str, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT inviter_id, name, total FROM inviters "
                "ORDER BY total DESC, updated_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]
