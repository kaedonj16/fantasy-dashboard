"""Regression tests for the 2026-09-27 hourly-crash incident fixes.

1. Gunicorn threads default to 4 (was 2): threads share a worker's address
   space, so this doubles concurrent request capacity (4 -> 8) with no
   meaningful memory increase. The DB pool default follows WEB_THREADS.
2. RedZone polling moved out of the gunicorn master thread into the
   ``redzone-store-poll`` Render cron (``scripts/redzone_poll.py``): one
   ``poll_once()`` per run, with the same Postgres advisory lock skipping
   overlapping runs cleanly.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from data_building.updates.startup import resolve_gunicorn_config


def _clean_gunicorn_env(monkeypatch):
    for var in ("PORT", "WEB_WORKERS", "WEB_THREADS", "DB_POOL_MAX"):
        monkeypatch.delenv(var, raising=False)


# --- Change 1: thread default --------------------------------------------------


def test_gunicorn_defaults_to_four_threads(monkeypatch):
    _clean_gunicorn_env(monkeypatch)
    assert resolve_gunicorn_config() == (5000, 2, 4)


def test_gunicorn_honors_env_overrides(monkeypatch):
    _clean_gunicorn_env(monkeypatch)
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("WEB_WORKERS", "3")
    monkeypatch.setenv("WEB_THREADS", "8")
    assert resolve_gunicorn_config() == (9000, 3, 8)


def test_db_pool_default_keeps_pace_with_threads():
    src = Path("dashboard_services/db.py").read_text()
    # Pool default must track the new WEB_THREADS default (4) with headroom
    # for background threads -- 8 per worker process.
    assert 'os.getenv("WEB_THREADS", "4")' in src


# --- Change 2: standalone redzone poll script ----------------------------------


def _load_poll_module():
    path = Path("scripts/redzone_poll.py")
    assert path.exists(), "scripts/redzone_poll.py missing"
    spec = importlib.util.spec_from_file_location("redzone_poll_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeConn:
    """Stand-in psycopg connection; lock_held=True simulates a live leader."""

    def __init__(self, lock_held: bool):
        self._lock_held = lock_held
        self.queries: list[str] = []
        self.closed = False
        self.autocommit = False

    def execute(self, sql, params=None):
        self.queries.append(str(sql))

        class _Cur:
            def __init__(self, outer):
                self._outer = outer

            def fetchone(self):
                if "pg_try_advisory_lock" in str(sql):
                    return {"pg_try_advisory_lock": not self._outer._lock_held}
                return None

        return _Cur(self)

    def close(self):
        self.closed = True


@pytest.fixture()
def _poll_harness(monkeypatch):
    """Install fake psycopg + dummy DATABASE_URL; return (module, install_conn)."""
    import utils.redzone_store as store_mod

    mod = _load_poll_module()
    conns: list[_FakeConn] = []

    def install_conn(lock_held: bool) -> _FakeConn:
        conn = _FakeConn(lock_held)
        conns.append(conn)
        fake_psycopg = types.ModuleType("psycopg")
        fake_psycopg.connect = lambda *a, **k: conn
        fake_rows = types.ModuleType("psycopg.rows")
        fake_rows.dict_row = object()
        # monkeypatch auto-undoes: never leaks the stub into other tests.
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
        monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)
        return conn

    monkeypatch.setenv("DATABASE_URL", "postgresql://dummy/dummy")
    return mod, install_conn, store_mod


def test_poll_skips_cleanly_when_lock_held(monkeypatch, _poll_harness):
    mod, install_conn, store_mod = _poll_harness
    conn = install_conn(lock_held=True)
    calls: list = []
    monkeypatch.setattr(store_mod, "poll_once", lambda: calls.append(1) or {})

    assert mod.main() == 0
    assert calls == [], "poll_once must not run when another poller holds the lock"
    assert conn.closed
    assert not any("pg_advisory_unlock" in q for q in conn.queries), (
        "must not unlock a lock it never held"
    )


def test_poll_runs_once_and_unlocks_when_lock_free(monkeypatch, _poll_harness):
    mod, install_conn, store_mod = _poll_harness
    conn = install_conn(lock_held=False)
    calls: list = []
    monkeypatch.setattr(
        store_mod, "poll_once", lambda: calls.append(1) or {"games": 2, "plays": 5}
    )

    assert mod.main() == 0
    assert len(calls) == 1
    assert conn.closed
    assert any("pg_advisory_unlock" in q for q in conn.queries), (
        "must release the advisory lock after polling"
    )


def test_poll_returns_one_and_unlocks_on_failure(monkeypatch, _poll_harness):
    mod, install_conn, store_mod = _poll_harness
    conn = install_conn(lock_held=False)

    def _boom():
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(store_mod, "poll_once", _boom)

    assert mod.main() == 1
    assert conn.closed
    assert any("pg_advisory_unlock" in q for q in conn.queries), (
        "lock must be released even when poll_once raises"
    )


def test_in_app_poller_defaults_to_off():
    src = Path("app.py").read_text()
    assert "REDZONE_STORE_THREAD" in src
    # The cron owns polling now; the import-time thread must not start
    # unless explicitly re-enabled.
    assert 'os.environ.get("REDZONE_STORE_THREAD", "")' in src
