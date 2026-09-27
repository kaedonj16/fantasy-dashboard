"""Regression tests for DB connection resilience.

Incident 2026-09-26: ``/auth/google/callback`` 500'd with
``psycopg.OperationalError: SSL error: unexpected eof while reading`` when
the pool handed out a connection the server had already closed. Two fixes:

1. The pool validates connections at checkout (``check=_check_pooled_conn``)
   so dead ones are discarded before reaching request code.
2. ``upsert_google_account`` retries once on ``OperationalError`` -- the
   upsert is idempotent (dead transaction rolls back), so a transient DB
   blip becomes a successful sign-in instead of a 500.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import dashboard_services.accounts as accounts
import dashboard_services.db as dbmod


def test_pool_registers_checkout_health_check():
    src = Path("dashboard_services/db.py").read_text()
    assert "check=_check_pooled_conn" in src


def test_check_hook_raises_on_dead_connection(monkeypatch):
    monkeypatch.setattr(dbmod, "is_connection_healthy", lambda conn: False)
    with pytest.raises(Exception, match="checkout health check"):
        dbmod._check_pooled_conn(object())


def test_check_hook_passes_healthy_connection(monkeypatch):
    monkeypatch.setattr(dbmod, "is_connection_healthy", lambda conn: True)
    dbmod._check_pooled_conn(object())  # must not raise


def test_upsert_retries_once_on_operational_error(monkeypatch):
    calls = []

    def flaky(sub, email, first_name=None):
        calls.append(sub)
        if len(calls) == 1:
            raise accounts.OperationalError("SSL error: unexpected eof")
        return (42, False)

    monkeypatch.setattr(accounts, "_upsert_google_account_once", flaky)
    monkeypatch.setattr(accounts.time, "sleep", lambda s: None)
    assert accounts.upsert_google_account("sub123", "a@b.c") == (42, False)
    assert calls == ["sub123", "sub123"]


def test_upsert_raises_after_two_failures(monkeypatch):
    calls = []

    def always_dead(sub, email, first_name=None):
        calls.append(sub)
        raise accounts.OperationalError("SSL error: unexpected eof")

    monkeypatch.setattr(accounts, "_upsert_google_account_once", always_dead)
    monkeypatch.setattr(accounts.time, "sleep", lambda s: None)
    with pytest.raises(accounts.OperationalError):
        accounts.upsert_google_account("sub123", "a@b.c")
    assert len(calls) == 2  # exactly one retry, then give up


def test_upsert_empty_sub_short_circuits(monkeypatch):
    called = []
    monkeypatch.setattr(
        accounts, "_upsert_google_account_once", lambda *a: called.append(a)
    )
    assert accounts.upsert_google_account("", "a@b.c") == (None, False)
    assert called == []
