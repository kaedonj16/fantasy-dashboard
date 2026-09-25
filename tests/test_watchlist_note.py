"""POST /api/watchlist/note writes the note verbatim (empty string clears it),
so the watchlist page and player-modal editors have a real backend.

The Postgres layer is faked with an in-memory dict (monkeypatched), so these
run without a database.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

pytest.importorskip("flask")


@pytest.fixture
def client(monkeypatch):
    notes = {}  # (user_key, player_id) -> note

    class Conn:
        def execute(self, sql, args=()):
            s = str(sql)
            if "INSERT INTO user_watchlist" in s and "note" in s and len(args) == 6:
                notes[(args[0], args[1])] = args[5]
            return self

        def fetchall(self):
            return []

        def fetchone(self):
            return None

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    @contextmanager
    def fake_conn():
        yield Conn()

    monkeypatch.setattr("dashboard_services.db.get_conn", fake_conn)
    import routes.watchlist_bp as WLB
    WLB._TABLES_READY = False

    from flask import Flask
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(WLB.watchlist_bp)
    app.config.update(TESTING=True)
    app.notes = notes
    return app.test_client()


def _sign_in(client):
    with client.session_transaction() as sess:
        sess["account_id"] = "test-acct-1"


def test_note_set_then_cleared(client):
    _sign_in(client)
    r = client.post("/api/watchlist/note",
                    json={"player_id": "p1", "name": "Test Player",
                          "note": "Stash him for the playoff run"})
    assert r.get_json() == {"synced": True, "ok": True}
    assert client.application.notes[("acct:test-acct-1", "p1")] == "Stash him for the playoff run"

    # Empty string clears the note verbatim (not a no-op).
    r = client.post("/api/watchlist/note",
                    json={"player_id": "p1", "note": ""})
    assert r.get_json() == {"synced": True, "ok": True}
    assert client.application.notes[("acct:test-acct-1", "p1")] == ""


def test_note_capped_at_500_chars(client):
    _sign_in(client)
    client.post("/api/watchlist/note",
                json={"player_id": "p1", "note": "x" * 600})
    assert len(client.application.notes[("acct:test-acct-1", "p1")]) == 500


def test_note_requires_player_id(client):
    _sign_in(client)
    r = client.post("/api/watchlist/note", json={"note": "hi"})
    assert r.status_code == 400


def test_note_not_signed_in_is_noop(client):
    r = client.post("/api/watchlist/note",
                    json={"player_id": "p1", "note": "hi"})
    assert r.get_json() == {"synced": False}
    assert client.application.notes == {}
