"""Tests for locked-player handling in swap suggestions.

Regression coverage for the Week 4 report: the hub card suggested
"Start DK Metcalf over Jalen Coker" on Sunday evening, but Metcalf's
Steelers had already played Thursday, so the swap was impossible to make.
``projection_upgrades`` must never suggest a swap involving a player whose
NFL game already kicked off, on either side.
"""
from utils.lineup_issues import (
    locked_teams_for_week,
    locked_teams_from_games,
    projection_upgrades,
)

SLOTS = ["WR", "WR"]
POS = {"w_out": "WR", "w_ok": "WR", "w_metcalf": "WR", "w_alt": "WR"}
PROJ = {"w_out": 0.0, "w_ok": 9.0, "w_metcalf": 8.0, "w_alt": 5.0}
STARTERS = ["w_out", "w_ok"]
ELIGIBLE = list(POS)


def test_locked_bench_player_is_never_suggested_as_swap_in():
    # Without lock info the optimizer picks Metcalf (8.0) over Coker (0.0):
    # this is the exact unactionable suggestion from the bug report.
    swaps = projection_upgrades(STARTERS, ELIGIBLE, PROJ, POS, SLOTS)
    assert swaps and swaps[0]["in"] == "w_metcalf" and swaps[0]["out"] == "w_out"

    # With Metcalf's game over, he must not appear; the best *unlocked*
    # bench WR is suggested instead.
    swaps = projection_upgrades(
        STARTERS, ELIGIBLE, PROJ, POS, SLOTS, locked_pids={"w_metcalf"}
    )
    assert swaps
    assert all(s["in"] != "w_metcalf" for s in swaps)
    assert swaps[0]["in"] == "w_alt" and swaps[0]["out"] == "w_out"
    assert swaps[0]["gain"] == 5.0


def test_locked_starter_is_never_suggested_as_swap_out():
    # A starter whose game already kicked off cannot be benched, so no swap
    # may name them as the "out" side.
    swaps = projection_upgrades(
        STARTERS, ELIGIBLE, PROJ, POS, SLOTS, locked_pids={"w_out"}
    )
    assert all(s["out"] != "w_out" for s in swaps)


def test_every_suggested_swap_is_actionable():
    # Fuzz-ish sweep: whatever the lock set, no suggestion may involve a
    # locked player on either side.
    import itertools

    for r in range(len(ELIGIBLE) + 1):
        for locked in itertools.combinations(ELIGIBLE, r):
            locked_set = set(locked)
            swaps = projection_upgrades(
                STARTERS, ELIGIBLE, PROJ, POS, SLOTS, locked_pids=locked_set
            )
            for s in swaps:
                assert s["in"] not in locked_set
                assert s["out"] not in locked_set


def test_no_locks_keeps_previous_behavior():
    assert projection_upgrades(STARTERS, ELIGIBLE, PROJ, POS, SLOTS, locked_pids=None) == \
        projection_upgrades(STARTERS, ELIGIBLE, PROJ, POS, SLOTS, locked_pids=set())


def test_locked_teams_from_games():
    games = [
        {"home": "CLE", "away": "PIT", "gameTime_epoch": 1000},      # kicked off
        {"home": "NYG", "away": "ARI", "gameTime_epoch": 999999},    # future
        {"home": "BUF", "away": "NE", "gameTime_epoch": "bogus"},    # ignored
        {"home": "", "away": None},                                  # ignored
    ]
    locked = locked_teams_from_games(games, now_ts=2000)
    assert locked == {"CLE", "PIT"}


def test_locked_teams_boundary_is_inclusive():
    games = [{"home": "KC", "away": "LV", "gameTime_epoch": 5000}]
    assert locked_teams_from_games(games, now_ts=5000) == {"KC", "LV"}
    assert locked_teams_from_games(games, now_ts=4999) == set()


def test_locked_teams_for_week_fail_open(monkeypatch):
    # A broken schedule loader must not nuke every swap suggestion.
    import sys
    import types

    stub = types.ModuleType("utils.data_cache")

    def _boom(season, week):
        raise RuntimeError("schedule unavailable")

    stub.load_week_schedule = _boom
    monkeypatch.setitem(sys.modules, "utils.data_cache", stub)
    # locked_teams_for_week does a lazy `from utils.data_cache import ...`;
    # make sure the already-imported real module also sees the stub.
    import utils.data_cache as dc
    monkeypatch.setattr(dc, "load_week_schedule", _boom)
    assert locked_teams_for_week(2026, 4) == set()
