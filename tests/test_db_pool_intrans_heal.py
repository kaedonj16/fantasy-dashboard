"""Regression tests for the pooled get_conn INTRANS autocommit bug.

Production symptom (Render cron logs, 2026-09-28/29): every _save_users /
_save_leagues batch failed with
    "can't change 'autocommit' now: connection in transaction status INTRANS"
because a recycled pooled connection carried a stray open transaction, and the
autocommit flip raised before the pool checkout was cleanly returned -- so the
poisoned connection kept getting recycled into the next checkout.

get_conn must now roll back a stray transaction before flipping autocommit,
and the pool's context manager must stay nested so a failed flip still goes
through the pool exit (rollback + putconn).

These tests patch attributes on the already-imported dashboard_services.db
module (monkeypatch auto-undoes); they never reimport the module, so other
test files that hold a reference to it are unaffected.
"""
import types
from contextlib import contextmanager

import dashboard_services.db as db


class FakeTransactionStatus:
    IDLE = 0
    INTRANS = 2


class FakeProgrammingError(Exception):
    pass


class FakeConn:
    """Mimics the psycopg3 bits get_conn touches, including the real
    refusal to change autocommit while a transaction is open."""

    def __init__(self, intrans=False):
        self._autocommit = False
        self.transaction_status = (
            FakeTransactionStatus.INTRANS if intrans else FakeTransactionStatus.IDLE
        )
        self.rolled_back = 0

    @property
    def info(self):
        return self

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        if value != self._autocommit and self.transaction_status != FakeTransactionStatus.IDLE:
            raise FakeProgrammingError(
                "can't change 'autocommit' now: connection in transaction status INTRANS"
            )
        self._autocommit = value

    def rollback(self):
        self.rolled_back += 1
        self.transaction_status = FakeTransactionStatus.IDLE

    def commit(self):
        self.transaction_status = FakeTransactionStatus.IDLE


class FakePool:
    """Yields a recycled connection (optionally INTRANS); pool exit rolls back
    stray state and records the return, like psycopg_pool."""

    def __init__(self, intrans=True):
        self.intrans = intrans
        self.returned = []

    @contextmanager
    def connection(self):
        conn = FakeConn(intrans=self.intrans)
        try:
            yield conn
        finally:
            try:
                if conn.transaction_status != FakeTransactionStatus.IDLE:
                    conn.rollback()
            except Exception:
                pass
            self.returned.append(conn)


def _patch_db(monkeypatch, intrans=True):
    pool = FakePool(intrans=intrans)
    fake_psycopg = types.SimpleNamespace(
        OperationalError=type("OperationalError", (Exception,), {}),
        Connection=object,
    )
    monkeypatch.setattr(db, "psycopg", fake_psycopg)
    monkeypatch.setattr(db, "TransactionStatus", FakeTransactionStatus)
    monkeypatch.setattr(db, "_POOL_AVAILABLE", True)
    monkeypatch.setattr(db, "_get_pool", lambda: pool)
    return pool


def test_get_conn_autocommit_heals_intrans_checkout(monkeypatch):
    """A poisoned INTRANS checkout must not raise: rollback, then flip."""
    pool = _patch_db(monkeypatch, intrans=True)

    with db.get_conn(autocommit=True) as conn:
        assert conn.autocommit is True
        assert conn.transaction_status == FakeTransactionStatus.IDLE

    # stray transaction was rolled back (heal) and the connection restored
    # to the pool default before return
    assert conn.rolled_back >= 1
    assert conn.autocommit is False
    assert pool.returned == [conn]


def test_get_conn_autocommit_idle_checkout_no_rollback(monkeypatch):
    """Normal path unchanged: IDLE checkout flips without needing a rollback."""
    pool = _patch_db(monkeypatch, intrans=False)

    with db.get_conn(autocommit=True) as conn:
        assert conn.autocommit is True

    assert conn.rolled_back == 0
    assert conn.autocommit is False
    assert pool.returned == [conn]


def test_get_conn_plain_checkout_never_flips(monkeypatch):
    """autocommit=False checkouts never touch the autocommit flag."""
    pool = _patch_db(monkeypatch, intrans=True)

    with db.get_conn() as conn:
        assert conn.autocommit is False

    assert conn.autocommit is False
    assert pool.returned == [conn]
