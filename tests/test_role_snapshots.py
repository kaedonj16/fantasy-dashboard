"""Frontier 1: week-over-week role snapshots (utils/role_snapshots.py).

Wave 2 v1 derived role within a week only; this store is what lets a
surface say "role changed since last week". Storage is stubbed at the
module's ``_conn`` seam (same convention as test_league_rules), so the
tests pin the diff/capture logic without a database.
"""
import json

import pytest

from utils import role_snapshots as rs
from utils.role_state import get_role_states


# ── fake storage ────────────────────────────────────────────────────

class _Cursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    """In-memory stand-in for the psycopg connection the module uses."""

    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if sql.startswith("CREATE TABLE"):
            return _Cursor(None)
        if sql.startswith("INSERT INTO role_snapshots"):
            platform, league_id, season, week, states = params
            self.store[(platform, league_id, season, week)] = (
                json.loads(states))
            return _Cursor(None)
        if "ORDER BY week DESC" in sql:
            platform, league_id, season, week = params
            prior = [k for k in self.store
                     if k[:3] == (platform, league_id, season)
                     and k[3] < week]
            if not prior:
                return _Cursor(None)
            wk = max(k[3] for k in prior)
            return _Cursor({"week": wk,
                            "states": self.store[(platform, league_id,
                                                   season, wk)]})
        if sql.startswith("SELECT states FROM role_snapshots"):
            platform, league_id, season, week = params
            states = self.store.get((platform, league_id, season, week))
            return _Cursor({"states": states} if states else None)
        raise AssertionError(f"unexpected SQL: {sql[:60]}")


@pytest.fixture()
def store(monkeypatch):
    holder = {}
    monkeypatch.setattr(rs, "_conn", lambda: _FakeConn(holder))
    monkeypatch.setattr(rs, "_TABLE_READY", True)
    return holder


def _ctx(players=None, **kw):
    ctx = {"platform": "sleeper", "league_id": "L1", "season": 2026,
           "current_week": 9}
    if players is not None:
        ctx["players"] = players
    ctx.update(kw)
    return ctx


# ── identity ────────────────────────────────────────────────────────

def test_league_key_requires_full_identity():
    assert rs.league_key(_ctx()) == ("sleeper", "L1", 2026)
    assert rs.league_key({"platform": "sleeper"}) is None
    assert rs.league_key({"league_id": "L1", "season": 2026}) is None
    assert rs.league_key({"platform": "sleeper", "league_id": "L1"}) is None
    assert rs.league_key({}) is None
    assert rs.league_key(None) is None


def test_league_key_reads_nested_league_dict():
    ctx = {"platform": "Fleaflicker",
           "league": {"league_id": "92916", "season": 2026}}
    assert rs.league_key(ctx) == ("fleaflicker", "92916", 2026)


# ── compact map ─────────────────────────────────────────────────────

def test_compact_states_drops_signal_free_unknowns():
    players = {
        "a": {"team": "bal", "position": "QB", "depth_chart_order": 2},
        "b": {"team": "BAL", "position": "QB"},
        "c": {"team": "KC", "position": "RB", "depth_chart_order": 1},
    }
    states = {
        "a": {"state": "promoted"},
        "b": {"state": "unknown"},
        "c": {"state": "unknown"},
    }
    out = rs.compact_states({"players": players}, states)
    assert out["a"] == {"state": "promoted", "team": "BAL",
                        "pos": "QB", "depth_order": 2}
    assert "b" not in out          # unknown + no depth order: no signal
    assert out["c"]["depth_order"] == 1   # unknown but on the depth line


# ── diff ────────────────────────────────────────────────────────────

def _entry(state, order=None):
    return {"state": state, "team": "BAL", "pos": "QB",
            "depth_order": order}


def test_diff_promotion_is_a_gain():
    prior = {"qb2": _entry("unknown", 2)}
    current = {"qb2": _entry("promoted", 2)}
    out = rs.diff_states(prior, current)
    assert out["qb2"]["kind"] == "gained_starting_role"
    assert out["qb2"]["from_state"] == "unknown"
    assert out["qb2"]["to_state"] == "promoted"


def test_diff_benching_is_a_loss_and_return_is_neutral_change():
    prior = {"a": _entry("starter", 1), "b": _entry("starter_out", 1)}
    current = {"a": _entry("benched", 2), "b": _entry("returning", 1)}
    out = rs.diff_states(prior, current)
    assert out["a"]["kind"] == "lost_starting_role"
    assert out["b"]["kind"] == "gained_starting_role"


def test_diff_unknown_on_either_side_needs_the_depth_line():
    prior = {"a": _entry("starter", 1), "b": _entry("unknown", 2),
             "c": _entry("unknown", 1)}
    current = {"a": _entry("unknown", 1), "b": _entry("unknown", 1),
               "c": _entry("unknown", 3)}
    out = rs.diff_states(prior, current)
    assert "a" not in out   # starter -> unknown alone proves nothing
    assert out["b"]["kind"] == "gained_starting_role"   # QB2 -> QB1
    assert out["c"]["kind"] == "lost_starting_role"     # QB1 -> QB3


def test_diff_ignores_players_missing_from_either_week():
    prior = {"a": _entry("starter", 1)}
    current = {"b": _entry("promoted", 1)}
    assert rs.diff_states(prior, current) == {}


# ── storage round trip ──────────────────────────────────────────────

def test_capture_and_prior_round_trip(store):
    ctx = _ctx(players={
        "qb2": {"team": "BAL", "position": "QB", "depth_chart_order": 2},
    })
    wk8 = {"qb2": {"state": "unknown", "reason": "", "signal": "none",
                   "week": 8}}
    assert rs.capture_snapshot(ctx, 8, wk8) is True
    loaded = rs.load_snapshot("sleeper", "L1", 2026, 8)
    assert loaded["qb2"]["state"] == "unknown"
    prior = rs.latest_prior_snapshot(ctx, 9)
    assert prior[0] == 8
    changes = rs.role_changes(ctx, 9, states={
        "qb2": {"state": "promoted", "reason": "x", "signal": "depth",
                "week": 9},
    })
    assert changes["qb2"]["kind"] == "gained_starting_role"


def test_capture_without_identity_writes_nothing(store):
    assert rs.capture_snapshot({"players": {}}, 9,
                               {"a": {"state": "starter"}}) is False
    assert store == {}


def test_first_week_is_silent(store):
    ctx = _ctx(players={})
    assert rs.latest_prior_snapshot(ctx, 9) is None
    assert rs.role_changes(ctx, 9, states={}) == {}


# ── serve-path hook ─────────────────────────────────────────────────

def _player(team, pos, order, status="", name=""):
    return {"team": team, "position": pos, "depth_chart_order": order,
            "depth_chart_position": pos, "injury_status": status,
            "status": "Active" if not status else "Inactive",
            "full_name": name}


def test_get_role_states_annotates_change_and_captures(store):
    players = {
        "qb1": _player("BAL", "QB", 1, "IR", "Lamar Jackson"),
        "qb2": _player("BAL", "QB", 2, "", "Jacoby Brissett"),
    }
    ctx = _ctx(players=players,
               proj_by_week={9: {"projections": {"qb2": 19.5}}})
    # Week 8 snapshot: Brissett a plain unknown QB2.
    store[("sleeper", "L1", 2026, 8)] = {
        "qb2": {"state": "unknown", "team": "BAL", "pos": "QB",
                "depth_order": 2},
    }
    states = get_role_states(ctx, 9)
    change = states["qb2"].get("change_since_last_week")
    assert change and change["kind"] == "gained_starting_role"
    # And week 9 was captured for next week's diff.
    assert ("sleeper", "L1", 2026, 9) in store


def test_get_role_states_without_identity_never_touches_store(store):
    players = {"qb1": _player("BAL", "QB", 1, "", "Lamar Jackson")}
    ctx = {"players": players, "current_week": 9,
           "proj_by_week": {9: {"projections": {"qb1": 22.0}}}}
    states = get_role_states(ctx, 9)
    assert states["qb1"]["state"] in ("starter", "unknown")
    assert store == {}


def test_history_weeks_are_not_snapshotted(store):
    players = {"qb1": _player("BAL", "QB", 1, "", "Lamar Jackson")}
    ctx = _ctx(players=players,
               proj_by_week={7: {"projections": {"qb1": 22.0}}})
    get_role_states(ctx, 7)   # ctx current week is 9, not 7
    assert store == {}
