"""Timeout hardening for request-thread blockers (2026-10-02 incident).

Incident: trade package endpoints hung with zero-byte responses
(intermittently, then site-wide 502/524). Root cause class: gunicorn gthread
(2 workers x 2 threads) combined with blocking calls that have no timeout --
a wedged call permanently eats a request thread, and gunicorn never reaps
stuck request threads (--timeout only heartbeats the worker main thread).

Hardening (every blocker must now fail fast instead of hanging forever):
1. app._acquire_context_lock: bounded wait, ContextLockBusy on timeout, and
   the refcount slot is released so the entry stays prunable.
2. extensions._limiter_storage_uri: redis-py socket timeouts on the
   Flask-Limiter storage URI (every other Redis client already had them).
3. dashboard_services.db._configure_pooled_conn: PG statement_timeout on
   every pooled connection.
"""
import threading
import time

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import app as app_module
import dashboard_services.db as db
import extensions


# --- 1. league context lock: bounded wait ---------------------------------

def test_ctx_lock_timeout_raises_and_releases_refcount():
    key = "test-ctx-timeout-key"
    state, _ = app_module._acquire_context_lock(key)  # hold the lock
    try:
        with app_module._CTX_LOCKS_LOCK:
            users_before = app_module._CTX_LOCKS[key].users
        with pytest.raises(app_module.ContextLockBusy):
            app_module._acquire_context_lock(key, timeout=0.2)
        with app_module._CTX_LOCKS_LOCK:
            users_after = app_module._CTX_LOCKS[key].users
        # The timed-out waiter must not leak its refcount slot.
        assert users_after == users_before
    finally:
        app_module._release_context_lock(key, state)


def test_ctx_lock_none_timeout_waits_until_released():
    key = "test-ctx-legacy-key"
    state, _ = app_module._acquire_context_lock(key)
    acquired = []

    def waiter():
        s2, _ = app_module._acquire_context_lock(key)  # timeout=None: legacy wait
        acquired.append(True)
        app_module._release_context_lock(key, s2)

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    try:
        time.sleep(0.3)
        assert not acquired  # still blocked on the held lock
    finally:
        app_module._release_context_lock(key, state)
    t.join(timeout=5)
    assert acquired  # proceeded once the holder released


def test_ctx_lock_busy_is_a_timeout_error():
    assert issubclass(app_module.ContextLockBusy, TimeoutError)


# --- 2. limiter Redis storage: socket timeouts ------------------------------

def test_limiter_storage_uri_memory_when_no_redis(monkeypatch):
    monkeypatch.setattr(extensions, "_redis_url", "")
    assert extensions._limiter_storage_uri() == "memory://"


def test_limiter_storage_uri_carries_socket_timeouts(monkeypatch):
    monkeypatch.setattr(extensions, "_redis_url", "redis://:pw@redis-host:6379/0")
    uri = extensions._limiter_storage_uri()
    assert uri.startswith("redis://:pw@redis-host:6379/0?")
    assert "socket_timeout=" in uri
    assert "socket_connect_timeout=" in uri


def test_limiter_storage_uri_preserves_existing_query_string(monkeypatch):
    monkeypatch.setattr(extensions, "_redis_url", "redis://h:6379/0?db=0")
    uri = extensions._limiter_storage_uri()
    assert "?db=0&socket_timeout=" in uri


# --- 3. Postgres: statement_timeout on pooled connections -------------------

class _FakeConn:
    def __init__(self):
        self.executed = []

    def execute(self, sql, *args, **kwargs):
        self.executed.append(sql)

    def commit(self):
        pass

    def rollback(self):
        pass


def test_statement_timeout_set_by_default(monkeypatch):
    monkeypatch.delenv("PG_STATEMENT_TIMEOUT_MS", raising=False)
    conn = _FakeConn()
    db._configure_pooled_conn(conn)
    assert any("statement_timeout" in s and "60000" in s for s in conn.executed)


def test_statement_timeout_env_override(monkeypatch):
    monkeypatch.setenv("PG_STATEMENT_TIMEOUT_MS", "15000")
    conn = _FakeConn()
    db._configure_pooled_conn(conn)
    assert any("statement_timeout" in s and "15000" in s for s in conn.executed)


def test_statement_timeout_zero_disables(monkeypatch):
    monkeypatch.setenv("PG_STATEMENT_TIMEOUT_MS", "0")
    conn = _FakeConn()
    db._configure_pooled_conn(conn)
    assert not any("statement_timeout" in s for s in conn.executed)
