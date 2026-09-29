"""Tests for redzone poller auto-unlink of dead leagues.

Regression context: two deleted Sleeper league IDs sat in push_subscriptions
and made the redzone TD poller log a 404 warning every poll cycle. Now a
provider 404 deletes that league's subscription rows so polling stops.
"""
from __future__ import annotations

import sys
import types

import pytest

from utils.push_notifications import _is_provider_not_found, _unlink_dead_league


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeHTTPError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.response = _FakeResponse(status_code)


def test_404_response_is_not_found():
    exc = _FakeHTTPError(
        "404 Client Error: Not Found for url: https://api.sleeper.app/v1/league/123/rosters",
        404,
    )
    assert _is_provider_not_found(exc) is True


def test_500_response_is_not_not_found():
    assert _is_provider_not_found(_FakeHTTPError("500 Server Error", 500)) is False


def test_plain_timeout_is_not_not_found():
    assert _is_provider_not_found(Exception("read timed out")) is False


def test_404_in_message_without_response_falls_back_true():
    assert _is_provider_not_found(Exception("404 Client Error: Not Found")) is True


class _FakeConn:
    def __init__(self):
        self.executed = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def commit(self):
        self.committed = True


def _install_fake_db(monkeypatch):
    conn = _FakeConn()
    fake_db = types.ModuleType("dashboard_services.db")
    fake_db.get_conn = lambda: conn
    monkeypatch.setitem(sys.modules, "dashboard_services.db", fake_db)
    return conn


def test_unlink_deletes_league_subscriptions_and_commits(monkeypatch):
    conn = _install_fake_db(monkeypatch)
    _unlink_dead_league("1387197724763357184", "sleeper")
    assert len(conn.executed) == 1
    sql, params = conn.executed[0]
    assert "DELETE FROM push_subscriptions" in sql
    assert params == ("1387197724763357184", "sleeper")
    assert conn.committed is True


def test_unlink_scopes_delete_to_platform(monkeypatch):
    conn = _install_fake_db(monkeypatch)
    _unlink_dead_league("999", "espn")
    _sql, params = conn.executed[0]
    assert params == ("999", "espn")


def test_unlink_db_failure_is_swallowed(monkeypatch):
    fake_db = types.ModuleType("dashboard_services.db")

    def _boom():
        raise RuntimeError("db down")

    fake_db.get_conn = _boom
    monkeypatch.setitem(sys.modules, "dashboard_services.db", fake_db)
    # Must not raise: the poller keeps going for the remaining leagues.
    _unlink_dead_league("123", "sleeper")
