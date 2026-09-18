"""Bounded cross-process single-flight for expensive league context builds.

Postgres advisory locks are preferred in production.  A flock in the system
temporary directory is the fallback for local development and deployments
without Postgres.  The JSON generation marker is intentionally tiny: it never
contains a context, provider response, or credentials.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time


class LeagueBuildBusy(TimeoutError):
    """The bounded wait expired while another worker owned this league."""


def stable_lock_key(platform, season, league_id):
    raw = f"{str(platform).lower()}\0{int(season)}\0{league_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=True)


def _stem(platform, season, league_id):
    digest = hashlib.sha256(
        f"{str(platform).lower()}\0{int(season)}\0{league_id}".encode()
    ).hexdigest()[:32]
    return os.path.join(tempfile.gettempdir(), f"br_league_build_{digest}")


def read_generation(platform, season, league_id):
    try:
        with open(_stem(platform, season, league_id) + ".json", encoding="utf-8") as fh:
            value = json.load(fh)
        return {"generation": int(value.get("generation") or 0),
                "completed_at": float(value.get("completed_at") or 0)}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"generation": 0, "completed_at": 0.0}


def mark_success(platform, season, league_id):
    path = _stem(platform, season, league_id) + ".json"
    old = read_generation(platform, season, league_id)
    value = {"generation": old["generation"] + 1, "completed_at": time.time()}
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(value, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return value


@contextlib.contextmanager
def league_build_lock(platform, season, league_id, timeout=20.0):
    """Yield lock wait seconds, or raise :class:`LeagueBuildBusy` on timeout."""
    started = time.monotonic()
    conn = None
    key = stable_lock_key(platform, season, league_id)
    if os.getenv("DATABASE_URL", "").lower().startswith(("postgres://", "postgresql://")):
        acquired = False
        database_available = False
        try:
            from dashboard_services.db import get_conn
            conn = get_conn()
            database_available = True
            while time.monotonic() - started < timeout:
                row = conn.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()
                acquired = bool((row.get("pg_try_advisory_lock") if hasattr(row, "get") else row[0]))
                if acquired:
                    break
                time.sleep(0.1)
        except Exception:
            # Database outages must not disable local mutual exclusion.
            acquired = False
            database_available = False
        if acquired:
            try:
                yield time.monotonic() - started
            finally:
                try:
                    conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
                finally:
                    conn.close()
            return
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        # A healthy Postgres connection that could not acquire the advisory
        # lock means another worker still owns it.  Falling through to flock
        # here would create a second, independent lock domain and permit the
        # exact concurrent build this helper exists to prevent.
        if database_available:
            raise LeagueBuildBusy("league build already in progress")

    import fcntl
    fd = os.open(_stem(platform, season, league_id) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        while time.monotonic() - started < timeout:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    yield time.monotonic() - started
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                return
            except BlockingIOError:
                time.sleep(0.05)
        raise LeagueBuildBusy("league build already in progress")
    finally:
        os.close(fd)
