"""HTTP tests for /api/portfolio-move-done (v25 board Done states):
guest handling, scope validation, and the persisted roundtrip behind a
fake connection (same pattern as tests/test_action_dismissals.py)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _FakeConn:
    """Implements exactly the statements portfolio_board issues."""

    def __init__(self, store):
        self.store = store
        self._last = ("", ())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=()):
        s = " ".join(str(sql).split()).upper()
        self._last = (s, params)
        if s.startswith("INSERT INTO PORTFOLIO_MOVE_STATES"):
            (uk, plat, lid, sn, wk, mk, verified, label) = params
            self.store[(uk, plat, lid, sn, wk, mk)] = {
                "verified": verified, "label": label}
        elif s.startswith("DELETE FROM PORTFOLIO_MOVE_STATES"):
            self.store.pop(tuple(params), None)
        return self

    def fetchall(self):
        s, p = self._last
        if s.startswith("SELECT PLATFORM, LEAGUE_ID, MOVE_KEY"):
            user_key, season, week = p
            return [
                {"platform": plat, "league_id": lid, "move_key": mk,
                 "verified": v["verified"], "verify_label": v["label"],
                 "done_at": "2026-10-25T13:41:00+00:00"}
                for (uk, plat, lid, sn, wk, mk), v in self.store.items()
                if uk == user_key and sn == season and wk == week
            ]
        return []

    def fetchone(self):
        return None


@pytest.fixture
def done_env(offline_client, monkeypatch):
    import dashboard_services.db as db

    store = {}
    monkeypatch.setattr(db, "get_conn", lambda *a, **k: _FakeConn(store))
    from dashboard_services import portfolio_board as pb
    monkeypatch.setattr(pb, "_DONE_TABLE_READY", True)
    return offline_client, store


def _sign_in(client, account_id=42):
    with client.session_transaction() as sess:
        sess["account_id"] = account_id


_BODY = {
    "platform": "sleeper", "league_id": "1312", "season": 2026, "week": 8,
    # Board move keys are sanitized to the dismissal-store alphabet:
    # the swap factory's "swap:2>1" travels as "swap:2-1".
    "move_key": "swap:2-1",
    "action": {"kind": "lineup", "verify": {"type": "swap", "in": "2", "out": "1"}},
}


def test_guest_done_is_not_synced(done_env):
    client, _ = done_env
    res = client.post("/api/portfolio-move-done", json=_BODY)
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True and data["synced"] is False
    res = client.get("/api/portfolio-move-done",
                     query_string={"season": 2026, "week": 8})
    assert res.get_json()["states"] == []


def test_invalid_scope_rejected(done_env):
    client, _ = done_env
    _sign_in(client)
    bad = dict(_BODY, platform="bad platform!")
    res = client.post("/api/portfolio-move-done", json=bad)
    assert res.status_code == 400


def test_done_roundtrip_is_marked_without_readback(done_env):
    client, store = done_env
    _sign_in(client)
    res = client.post("/api/portfolio-move-done", json=_BODY)
    assert res.status_code == 200
    data = res.get_json()
    # No warm league context in the test env, so there is no read-back:
    # the move must land as marked, never verified.
    assert data["synced"] is True
    assert data["verified"] is False
    assert data["label"]
    assert store, "done state must persist server-side"

    res = client.get("/api/portfolio-move-done",
                     query_string={"season": 2026, "week": 8})
    states = res.get_json()["states"]
    assert len(states) == 1
    assert states[0]["move_key"] == "swap:2-1"
    assert states[0]["verified"] is False
    assert states[0]["time_label"]  # rendered from done_at

    res = client.delete("/api/portfolio-move-done", json={
        "platform": "sleeper", "league_id": "1312", "season": 2026,
        "week": 8, "move_key": "swap:2-1",
    })
    assert res.get_json()["ok"] is True
    assert store == {}
