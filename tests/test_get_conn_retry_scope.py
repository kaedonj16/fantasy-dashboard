"""Regression tests for get_conn retry scope + compute_adp statement timeout.

Production incident 2026-10-10 / 2026-10-12: draft_adp_crawl's compute_adp
hit Render's statement timeout (psycopg.errors.QueryCanceled, an
OperationalError subclass). get_conn's retry loop wrapped the yield, so the
body's QueryCanceled was caught as a checkout failure and the loop
re-yielded -- CPython raised "RuntimeError: generator didn't stop after
throw()", masking the real error behind a confusing traceback.

Retry must cover ONLY pool acquisition: the caller's body runs exactly once
and its exceptions propagate unwrapped. Separately, compute_adp now raises
the statement timeout for its own transaction (SET LOCAL) because the
full-table recompute outgrows the pooled 60s default as draft_adp_picks
grows.

Stubs are scoped with monkeypatch (auto-undone); nothing is left behind in
sys.modules.
"""
import types
from contextlib import contextmanager

import pytest

import dashboard_services.db as db


class FakeConn:
    """Mimics the psycopg3 bits get_conn touches."""

    def __init__(self):
        self._autocommit = False

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        self._autocommit = value


class FakeCheckout:
    """Mimics psycopg_pool's pool.connection() context manager."""

    def __init__(self, conn):
        self._conn = conn
        self.exit_exc_type = "not-exited"

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        self.exit_exc_type = exc_type
        return False


class FakePool:
    """Scripted pool: each entry is a FakeCheckout or an Exception to raise."""

    def __init__(self, script):
        self._script = list(script)
        self.acquisitions = 0

    def connection(self):
        self.acquisitions += 1
        outcome = self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _patch_db(monkeypatch):
    """Point get_conn at fakes; return (QueryCanceled, PoolTimeout) classes."""
    operational_error = type("OperationalError", (Exception,), {})
    # Mirror the real psycopg3 hierarchy: QueryCanceled IS an OperationalError.
    query_canceled = type("QueryCanceled", (operational_error,), {})
    pool_timeout = type("PoolTimeout", (Exception,), {})
    fake_psycopg = types.SimpleNamespace(
        OperationalError=operational_error,
        Connection=object,
    )
    monkeypatch.setattr(db, "psycopg", fake_psycopg)
    monkeypatch.setattr(db, "PoolTimeout", pool_timeout)
    monkeypatch.setattr(db, "_POOL_AVAILABLE", True)
    monkeypatch.setattr(db.time, "sleep", lambda s: None)
    return query_canceled, pool_timeout


def test_body_query_canceled_propagates_unmasked(monkeypatch):
    """The exact production failure: the body's QueryCanceled must surface
    as QueryCanceled -- not RuntimeError -- with no retry of the body."""
    conn = FakeConn()
    checkout = FakeCheckout(conn)
    pool = FakePool([checkout])
    query_canceled, _ = _patch_db(monkeypatch)
    monkeypatch.setattr(db, "_get_pool", lambda: pool)

    with pytest.raises(query_canceled, match="statement timeout"):
        with db.get_conn() as c:
            assert c is conn
            raise query_canceled("canceling statement due to statement timeout")

    assert pool.acquisitions == 1  # body ran exactly once, never retried
    assert checkout.exit_exc_type is query_canceled  # checkout returned w/ exc


def test_body_non_operational_error_not_retried(monkeypatch):
    """A plain body bug (ValueError) also runs exactly once and propagates."""
    conn = FakeConn()
    checkout = FakeCheckout(conn)
    pool = FakePool([checkout])
    _patch_db(monkeypatch)
    monkeypatch.setattr(db, "_get_pool", lambda: pool)

    with pytest.raises(ValueError, match="boom"):
        with db.get_conn():
            raise ValueError("boom")

    assert pool.acquisitions == 1
    assert checkout.exit_exc_type is ValueError


def test_acquisition_pool_timeout_still_retries(monkeypatch):
    """Transient checkout failures retry; the body still runs exactly once."""
    _, pool_timeout = _patch_db(monkeypatch)
    conn = FakeConn()
    checkout = FakeCheckout(conn)
    pool = FakePool([pool_timeout("t1"), pool_timeout("t2"), checkout])
    monkeypatch.setattr(db, "_get_pool", lambda: pool)

    with db.get_conn() as c:
        assert c is conn

    assert pool.acquisitions == 3
    assert checkout.exit_exc_type is None  # clean return to the pool


def test_acquisition_failure_exhausts_retries(monkeypatch):
    """All checkouts failing raises the last error after `retries` tries."""
    _, pool_timeout = _patch_db(monkeypatch)
    pool = FakePool([pool_timeout("x")] * 3)
    monkeypatch.setattr(db, "_get_pool", lambda: pool)

    with pytest.raises(pool_timeout):
        with db.get_conn():
            pass  # pragma: no cover - acquisition never succeeds

    assert pool.acquisitions == 3


def test_compute_adp_raises_statement_timeout(monkeypatch):
    """compute_adp must SET LOCAL statement_timeout before the big upsert so
    the recompute is not killed by the pooled 60s default."""
    import data_building.trade_intel.draft_adp_crawler as crawler

    executed = []

    class FakeC:
        def execute(self, sql, params=None):
            executed.append(sql)
            return self

        def fetchone(self):
            return {"n": 7}

    @contextmanager
    def fake_get_conn():
        yield FakeC()

    monkeypatch.setattr(crawler, "get_conn", fake_get_conn)
    assert crawler.compute_adp() == 7

    assert executed[0].strip().upper().startswith("SET LOCAL STATEMENT_TIMEOUT")
    assert "10MIN" in executed[0].upper().replace(" ", "")
    assert "INSERT INTO draft_adp" in executed[1]
