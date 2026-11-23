"""Close-game alert gate: a raw gap under 20 is not "close" when the game
is statistically over.

notify_close_game used to fire on the score gap alone, so a Monday-night
alert went out for matchups the trailing team could no longer win: nobody
left to play, or one long-shot player against a gap they cannot plausibly
close. The gate reuses the matchup pages' win-probability model: suppress
when the trailing team's win chance is under 5%, and fail open (alert as
before) whenever the gate cannot judge.
"""
from __future__ import annotations

import datetime as _dt

import pytest

pytest.importorskip("flask")

import utils.push_notifications as pn

SEASON = 2026
WEEK = 12
FINAL = "final"
NOT_STARTED = "not_started"


def _ctx(status_by_pid, proj_map, players_index):
    return {
        "status_by_pid": status_by_pid,
        "proj_map": proj_map,
        "frac_lookup": None,
        "players_index": players_index,
    }


def _row(roster_id, points, starters, starters_points):
    return {
        "roster_id": roster_id,
        "matchup_id": 1,
        "points": points,
        "starters": starters,
        "starters_points": starters_points,
    }


# ── The gate itself ─────────────────────────────────────────────────────────

def test_trailing_team_with_nobody_left_is_over():
    # Down 8, but every trailing starter's game is final: no path back.
    players = {"a1": {"pos": "RB", "team": "BUF"}, "b1": {"pos": "WR", "team": "KC"}}
    ctx = _ctx({"a1": FINAL, "b1": FINAL}, {"a1": 15.0, "b1": 15.0}, players)
    a = _row(1, 100.0, ["a1"], [100.0])
    b = _row(2, 92.0, ["b1"], [92.0])
    assert pn._close_game_is_over(a, b, ctx) is True


def test_both_teams_done_is_over_even_when_tied():
    players = {"a1": {"pos": "RB", "team": "BUF"}, "b1": {"pos": "WR", "team": "KC"}}
    ctx = _ctx({"a1": FINAL, "b1": FINAL}, {"a1": 15.0, "b1": 15.0}, players)
    a = _row(1, 95.0, ["a1"], [95.0])
    b = _row(2, 95.0, ["b1"], [95.0])
    assert pn._close_game_is_over(a, b, ctx) is True


def test_long_shot_only_player_is_over():
    # Down 18 with only a kicker (proj 8) still to play on Monday night.
    players = {"a1": {"pos": "RB", "team": "BUF"},
               "b0": {"pos": "QB", "team": "DEN"}, "b1": {"pos": "K", "team": "KC"}}
    ctx = _ctx({"a1": FINAL, "b0": FINAL, "b1": NOT_STARTED},
               {"a1": 15.0, "b0": 20.0, "b1": 8.0}, players)
    a = _row(1, 100.0, ["a1"], [100.0])
    b = _row(2, 82.0, ["b0", "b1"], [82.0, 0.0])
    assert pn._close_game_is_over(a, b, ctx) is True


def test_real_mnf_swing_player_stays_live():
    # Down 10 with a WR (proj 12) still to play: a genuine Monday sweat.
    players = {"a1": {"pos": "RB", "team": "BUF"},
               "b0": {"pos": "QB", "team": "DEN"}, "b1": {"pos": "WR", "team": "KC"}}
    ctx = _ctx({"a1": FINAL, "b0": FINAL, "b1": NOT_STARTED},
               {"a1": 15.0, "b0": 20.0, "b1": 12.0}, players)
    a = _row(1, 100.0, ["a1"], [100.0])
    b = _row(2, 90.0, ["b0", "b1"], [90.0, 0.0])
    assert pn._close_game_is_over(a, b, ctx) is False


def test_gate_cannot_judge_without_context_or_starters():
    players = {"a1": {"pos": "RB", "team": "BUF"}}
    a = _row(1, 100.0, ["a1"], [100.0])
    b = _row(2, 92.0, ["b1"], [92.0])
    assert pn._close_game_is_over(a, b, None) is None
    ctx = _ctx({"a1": FINAL}, {"a1": 15.0}, players)
    bare_a = {"roster_id": 1, "matchup_id": 1, "points": 100.0}
    bare_b = {"roster_id": 2, "matchup_id": 1, "points": 92.0}
    assert pn._close_game_is_over(bare_a, bare_b, ctx) is None


# ── notify_close_game integration ───────────────────────────────────────────

class _R:
    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self):
        self.app_state = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def execute(self, q, params=None):
        if "SELECT value FROM app_state" in q:
            v = self.app_state.get(params[0])
            return _R({"value": v} if v is not None else None)
        if "INSERT INTO app_state" in q:
            self.app_state[params[0]] = params[1]
            return _R(None)
        return _R(None)


class _MondayNightDateTime(_dt.datetime):
    _now = _dt.datetime(2026, 11, 23, 23, 30, tzinfo=_dt.timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls._now.astimezone(tz) if tz else cls._now


def _install(monkeypatch, matchups, ctx):
    import dashboard_services.api as api
    import dashboard_services.db as db
    import dashboard_services.platform_api as papi

    conn = _FakeConn()
    sends = []
    monkeypatch.setattr(db, "get_conn", lambda: conn)
    monkeypatch.setattr(api, "get_nfl_state", lambda: {
        "season": SEASON, "week": WEEK, "season_type": "reg"})
    monkeypatch.setattr(pn, "datetime", _MondayNightDateTime)
    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: [("L1", "sleeper")])
    monkeypatch.setattr(pn, "_league_display_name", lambda *a: "League One")
    monkeypatch.setattr(pn, "_close_game_context", lambda *a: ctx)
    monkeypatch.setattr(papi, "get_matchups", lambda *a: matchups)
    monkeypatch.setattr(papi, "get_rosters", lambda *a: [
        {"roster_id": 1, "owner_id": "o1"},
        {"roster_id": 2, "owner_id": "o2"},
    ])
    monkeypatch.setattr(papi, "get_users", lambda *a: [
        {"user_id": "o1", "display_name": "Alice"},
        {"user_id": "o2", "display_name": "Bob"},
    ])

    def _capture(league_id, owner_id, title=None, body=None, **kw):
        sends.append({"owner_id": owner_id, "title": title, "body": body})
        return 1

    monkeypatch.setattr(pn, "_broadcast_owner", _capture)
    return conn, sends


def test_notify_suppresses_statistically_over_matchup(monkeypatch):
    players = {"a1": {"pos": "RB", "team": "BUF"}, "b1": {"pos": "WR", "team": "KC"}}
    ctx = _ctx({"a1": FINAL, "b1": FINAL}, {"a1": 15.0, "b1": 15.0}, players)
    matchups = [_row(1, 100.0, ["a1"], [100.0]), _row(2, 92.0, ["b1"], [92.0])]
    conn, sends = _install(monkeypatch, matchups, ctx)

    assert pn.notify_close_game() in (0, None)
    assert sends == []
    # Not marked notified: a later check may still alert if the game
    # somehow comes back to life.
    assert conn.app_state.get(f"close_game_notified_{SEASON}_{WEEK}") in (None, "")


def test_notify_still_alerts_live_matchup(monkeypatch):
    players = {"a1": {"pos": "RB", "team": "BUF"},
               "b0": {"pos": "QB", "team": "DEN"}, "b1": {"pos": "WR", "team": "KC"}}
    ctx = _ctx({"a1": FINAL, "b0": FINAL, "b1": NOT_STARTED},
               {"a1": 15.0, "b0": 20.0, "b1": 12.0}, players)
    matchups = [_row(1, 100.0, ["a1"], [100.0]),
                _row(2, 90.0, ["b0", "b1"], [90.0, 0.0])]
    conn, sends = _install(monkeypatch, matchups, ctx)

    assert pn.notify_close_game() == 2
    assert {s["owner_id"] for s in sends} == {"o1", "o2"}
    assert all(s["title"] == "Close matchup tonight" for s in sends)


def test_notify_fails_open_when_gate_cannot_judge(monkeypatch):
    # No starter data on the rows and no gate context: legacy behaviour.
    matchups = [
        {"roster_id": 1, "matchup_id": 1, "points": 100.0},
        {"roster_id": 2, "matchup_id": 1, "points": 95.0},
    ]
    conn, sends = _install(monkeypatch, matchups, None)

    assert pn.notify_close_game() == 2
    assert len(sends) == 2
