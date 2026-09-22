"""Regression guard: a push subscription must store the *platform* roster owner
id (Sleeper user id / ESPN SWID) as ``owner_id``, never the signed-in Google
``account_id``.

Owner-targeted notifications (``_broadcast_owner``) match
``push_subscriptions.owner_id`` against the roster's ``owner_id``, which is the
platform user id. When subscribe stored the Google ``account_id`` instead (a
different id namespace), that WHERE clause matched zero rows and every
owner-targeted push — value drops, playoff odds, breakouts, matchup preview,
standings, close game, injury, RedZone, per-owner lineup lock — was silently
sent to nobody while the league-wide/broadcast notifications still arrived. That
is the "not getting most of the notifications" bug this pins shut.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")

from flask import Flask, session

from extensions import limiter
import routes.push_bp as push_bp


class _Result:
    def __init__(self, rows=None, one=None):
        self._rows = rows or []
        self._one = one

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._one


class _FakeConn:
    """Records every execute() so the test can inspect the subscribe INSERT."""

    def __init__(self, calls):
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self._calls.append((query, params))
        return _Result()

    def commit(self):
        pass


def _make_client(monkeypatch, calls):
    monkeypatch.setattr(push_bp, "_init_push_table", lambda: None)
    monkeypatch.setattr(
        "dashboard_services.db.get_conn", lambda *a, **k: _FakeConn(calls)
    )
    app = Flask(__name__)
    app.secret_key = "test"
    limiter.init_app(app)
    app.register_blueprint(push_bp.push_bp)
    return app


def _subscribe_insert(calls):
    for query, params in calls:
        if "INSERT INTO push_subscriptions" in query:
            return params
    return None


def test_subscribe_stores_platform_viewer_id_not_account_id(monkeypatch):
    calls: list = []
    app = _make_client(monkeypatch, calls)

    with app.test_client() as client:
        with client.session_transaction() as sess:
            # A signed-in Google user viewing their Sleeper league: account_id is
            # the internal PK, viewer_user_id is the Sleeper roster owner id.
            sess["account_id"] = 42
            sess["viewer_user_id"] = "876543210987654321"
        resp = client.post(
            "/api/push/subscribe",
            json={
                "endpoint": "https://push.example/ep1",
                "keys": {"p256dh": "k", "auth": "a"},
                "league_ids": ["L1"],
                "platform": "sleeper",
            },
        )
        assert resp.status_code == 200

    params = _subscribe_insert(calls)
    assert params is not None, "subscribe never issued an INSERT"
    # INSERT columns order: (endpoint, p256dh, auth, league_id, platform, owner_id)
    owner_id = params[5]
    assert owner_id == "876543210987654321", (
        "owner_id must be the platform viewer id so _broadcast_owner matches the "
        f"roster owner_id; got {owner_id!r} (the account_id leak regression)"
    )
    assert owner_id != "42"


def test_subscribe_prefers_client_owner_id_when_no_session_viewer(monkeypatch):
    """Anonymous/username-only devices still link via the client-supplied
    owner_id (the settings-modal league toggle path sends window._viewerUid)."""
    calls: list = []
    app = _make_client(monkeypatch, calls)

    with app.test_client() as client:
        resp = client.post(
            "/api/push/subscribe",
            json={
                "endpoint": "https://push.example/ep2",
                "keys": {"p256dh": "k", "auth": "a"},
                "league_id": "L1",
                "platform": "sleeper",
                "owner_id": "111222333444555666",
            },
        )
        assert resp.status_code == 200

    params = _subscribe_insert(calls)
    assert params is not None
    assert params[5] == "111222333444555666"
