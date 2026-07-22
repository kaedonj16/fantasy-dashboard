"""Yahoo Phase 1 — token lifecycle & league→owner resolution.

These exercise the plumbing that lets Yahoo data be fetched outside a single
short-lived browser session: the DB-backed, auto-refreshing token accessor, the
league→owner mapping, and the priority order in platform_api._yahoo_token.

The modules under test import the full app stack (flask + pandas + bs4), so the
whole module skips cleanly when that stack isn't installed — matching the other
integration-style tests in this suite.
"""
import types

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("bs4")
yahoo_api = pytest.importorskip("dashboard_services.providers.yahoo_api")
platform_api = pytest.importorskip("dashboard_services.platform_api")


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    """Minimal get_conn() context manager. execute() returns a cursor whose
    fetchall() yields the pre-seeded rows (used for the SELECT); DDL/insert
    execute() calls simply return an empty cursor."""
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if "SELECT" in sql.upper():
            return _FakeCursor(self._rows)
        return _FakeCursor([])


def _patch_conn(monkeypatch, rows):
    monkeypatch.setattr(yahoo_api, "get_conn", lambda *a, **k: _FakeConn(rows), raising=False)
    # get_conn is imported lazily inside the functions via
    # `from dashboard_services.db import get_conn`, so patch it at the source too.
    import dashboard_services.db as db
    monkeypatch.setattr(db, "get_conn", lambda *a, **k: _FakeConn(rows), raising=False)


# ── get_league_token ────────────────────────────────────────────────────────

def test_get_league_token_returns_first_owner_with_valid_token(monkeypatch):
    _patch_conn(monkeypatch, rows=[{"guid": "G_old"}, {"guid": "G_new"}])
    seen = []

    def fake_valid(guid):
        seen.append(guid)
        return None if guid == "G_old" else "TOK_NEW"

    monkeypatch.setattr(yahoo_api, "get_valid_access_token", fake_valid)
    tok = yahoo_api.get_league_token("L1", 2026)
    assert tok == "TOK_NEW"
    # It should have tried the first owner, found no token, then the second.
    assert seen == ["G_old", "G_new"]


def test_get_league_token_none_when_no_owners(monkeypatch):
    _patch_conn(monkeypatch, rows=[])
    monkeypatch.setattr(yahoo_api, "get_valid_access_token", lambda g: "SHOULD_NOT_BE_USED")
    assert yahoo_api.get_league_token("L1", 2026) is None


def test_get_league_token_empty_league_id_short_circuits(monkeypatch):
    # No DB access at all for an empty league id.
    def boom(*a, **k):
        raise AssertionError("get_conn should not be called for empty league_id")
    monkeypatch.setattr(yahoo_api, "get_conn", boom, raising=False)
    assert yahoo_api.get_league_token("", 2026) is None


# ── save_league_owner ───────────────────────────────────────────────────────

def test_save_league_owner_noops_on_missing_args(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("get_conn should not be called when args are missing")
    monkeypatch.setattr(yahoo_api, "get_conn", boom, raising=False)
    import dashboard_services.db as db
    monkeypatch.setattr(db, "get_conn", boom, raising=False)
    # Neither of these should touch the DB.
    yahoo_api.save_league_owner("", 2026, "G1")
    yahoo_api.save_league_owner("L1", 2026, "")


# ── _yahoo_token priority ───────────────────────────────────────────────────

def test_yahoo_token_prefers_session_guid(monkeypatch):
    from flask import Flask
    app = Flask(__name__)
    app.secret_key = "test"
    monkeypatch.setattr(yahoo_api, "get_valid_access_token",
                        lambda g: "TOK_SESSION" if g == "G1" else None)
    # Should never reach the league-owner path when the session yields a token.
    monkeypatch.setattr(yahoo_api, "get_league_token",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("used league path")))
    with app.test_request_context("/"):
        from flask import session
        session["yahoo_guid"] = "G1"
        assert platform_api._yahoo_token("L1", 2026) == "TOK_SESSION"


def test_yahoo_token_falls_back_to_league_owner_without_session(monkeypatch):
    # Outside any request context, flask.session access raises RuntimeError, so
    # _yahoo_token must fall through to the league-owner token (the background path).
    monkeypatch.setattr(yahoo_api, "get_valid_access_token", lambda g: None)
    monkeypatch.setattr(yahoo_api, "get_league_token",
                        lambda lid, season: "TOK_OWNER" if lid == "L1" else "")
    assert platform_api._yahoo_token("L1", 2026) == "TOK_OWNER"


def test_yahoo_token_empty_when_nothing_available(monkeypatch):
    monkeypatch.setattr(yahoo_api, "get_valid_access_token", lambda g: None)
    monkeypatch.setattr(yahoo_api, "get_league_token", lambda lid, season: None)
    assert platform_api._yahoo_token("L1", 2026) == ""
