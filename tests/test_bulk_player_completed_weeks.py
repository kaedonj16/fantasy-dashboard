"""Tests for bulk_player_completed_weeks (per-player PPG on rankings page)."""
import sys
import types

import pytest

from utils.season_qualification import bulk_player_completed_weeks


def _game(team_a, team_b, final=True):
    return {
        "away": team_a,
        "home": team_b,
        "gameStatus": "Final" if final else "Scheduled",
        "gameStatusCode": "2" if final else "0",
    }


@pytest.fixture
def _mocks(monkeypatch):
    """Mock get_nfl_state, team maps, and schedule loads."""
    # get_nfl_state from dashboard_services.api
    api_mod = types.ModuleType("dashboard_services.api")
    api_mod.get_nfl_state = lambda: {"season": 2026, "week": 4}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", api_mod)
    # dashboard_services package stub (import system needs the parent)
    pkg = types.ModuleType("dashboard_services")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "dashboard_services", pkg)

    # Team maps: p1 on TB all 4 weeks, p2 on DAL weeks 1-3 then traded to NYG
    # week 4, p3 has no team data.
    pth_mod = types.ModuleType("data_building.external_data.player_team_history")
    pth_mod.load_weekly_team_map = lambda season: {
        "p1": {1: "TB", 2: "TB", 3: "TB", 4: "TB"},
        "p2": {1: "DAL", 2: "DAL", 3: "DAL", 4: "NYG"},
    }
    pth_mod.season_team_map = lambda season: {}
    monkeypatch.setitem(
        sys.modules, "data_building.external_data.player_team_history", pth_mod)
    ed_pkg = types.ModuleType("data_building.external_data")
    ed_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "data_building.external_data", ed_pkg)
    db_pkg = types.ModuleType("data_building")
    db_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "data_building", db_pkg)

    # Schedules: weeks 1-3 final for everyone; week 4: TB final, DAL/NYG not.
    schedules = {
        1: [_game("TB", "KC"), _game("DAL", "PHI"), _game("NYG", "WAS")],
        2: [_game("TB", "ATL"), _game("DAL", "NYJ"), _game("NYG", "SEA")],
        3: [_game("KC", "TB"), _game("PHI", "DAL"), _game("WAS", "NYG")],
        4: [_game("TB", "NO"),  # final
            _game("DAL", "GB", final=False),  # MNF, not final
            _game("NYG", "MIN", final=False)],  # MNF, not final
    }
    utils_mod = types.ModuleType("utils.utils")
    utils_mod.load_week_schedule = lambda season, w: schedules.get(w, [])
    monkeypatch.setitem(sys.modules, "utils.utils", utils_mod)
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)


def test_bulk_matches_per_player_semantics(_mocks):
    result = bulk_player_completed_weeks(["p1", "p2", "p3"], 2026)
    # p1 (TB): all 4 weeks final.
    assert result["p1"] == [1, 2, 3, 4]
    # p2: DAL weeks 1-3 final, then traded to NYG whose week 4 is not final.
    # Chronological break at week 4 (NYG game scheduled but not final).
    assert result["p2"] == [1, 2, 3]
    # p3: no team data -> [].
    assert result["p3"] == []


def test_bulk_bye_weeks_skipped(_mocks):
    # p4 on BUF; BUF has no game in week 2 (bye) but weeks 1,3,4 final.
    import data_building.external_data.player_team_history as pth
    orig = pth.load_weekly_team_map
    pth.load_weekly_team_map = lambda season: {
        "p4": {1: "BUF", 3: "BUF", 4: "BUF"},
    }
    try:
        import utils.utils as uu
        orig_sched = uu.load_week_schedule
        uu.load_week_schedule = lambda season, w: {
            1: [_game("BUF", "MIA")],
            3: [_game("BUF", "NE")],
            4: [_game("MIA", "BUF")],
        }.get(w, [])
        try:
            result = bulk_player_completed_weeks(["p4"], 2026)
            # Bye in week 2 is skipped, not a break.
            assert result["p4"] == [1, 3, 4]
        finally:
            uu.load_week_schedule = orig_sched
    finally:
        pth.load_weekly_team_map = orig


def test_bulk_future_season_returns_empty(_mocks):
    import dashboard_services.api as api
    orig = api.get_nfl_state
    api.get_nfl_state = lambda: {"season": 2026, "week": 4}
    try:
        result = bulk_player_completed_weeks(["p1"], 2027)
        assert result["p1"] == []
    finally:
        api.get_nfl_state = orig


def test_bulk_empty_input():
    assert bulk_player_completed_weeks([], 2026) == {}
