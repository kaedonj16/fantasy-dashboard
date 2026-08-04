"""Trade/departure detection compares a player's CURRENT team to the team they
played for LAST season. That only works if last season's usage row carries the
historical team — if a trade rewrites the prior-season row to the new team, the
comparison sees no move and the vacated opportunity silently disappears.

These guard the detection contract that the player_history team-stamping fix feeds.
"""
import pytest

prc = pytest.importorskip("data_building.populate_roster_changes")

# Sleeper id for Michael Pittman Jr. (the real regression: IND -> PIT, 111 targets).
_PID = "6819"


def _patch(monkeypatch, current_team, prev_team, targets=111, games=17):
    monkeypatch.setattr(prc, "load_players_index", lambda: {
        _PID: {"name": "Michael Pittman Jr.", "team": current_team, "pos": "WR"},
    })
    monkeypatch.setattr(prc, "load_usage_table_for_season", lambda season: [
        {"player_id": _PID, "team": prev_team, "position": "WR",
         "usage": {"games": games, "total_targets": targets}},
    ])


def test_trade_detected_when_prev_team_is_historical(monkeypatch):
    # Prior-season row correctly shows IND; current index shows PIT -> trade found,
    # vacating IND's targets (the behavior the fix restores).
    _patch(monkeypatch, current_team="PIT", prev_team="IND")
    changes = prc.detect_roster_changes_between_seasons(2026, compare_to_season=2025)
    assert len(changes) == 1
    c = changes[0]
    assert c["old_team"] == "IND"
    assert c["new_team"] == "PIT"
    assert c["change_type"] == "trade"
    assert c["stats"]["targets"] == 111


def test_no_departure_when_prev_team_overwritten_to_current(monkeypatch):
    # The bug: last-season row stamped with the CURRENT team (PIT). current == prev,
    # so no move is detected and the vacated targets vanish. Guards the failure mode.
    _patch(monkeypatch, current_team="PIT", prev_team="PIT")
    changes = prc.detect_roster_changes_between_seasons(2026, compare_to_season=2025)
    assert changes == []


def test_low_usage_departure_ignored(monkeypatch):
    # A player with negligible prior usage doesn't vacate meaningful opportunity.
    _patch(monkeypatch, current_team="PIT", prev_team="IND", targets=2)
    changes = prc.detect_roster_changes_between_seasons(2026, compare_to_season=2025)
    assert changes == []
