"""Tests for injury rows in the league activity feed."""
import pytest

pytest.importorskip("pandas")
import pandas as pd
from datetime import datetime, timezone

from dashboard_services.service import _append_injury_rows, _INJURY_FEED_STATUSES


def _injury_df(rows):
    return pd.DataFrame(rows)


def test_injury_rows_appended_for_serious_statuses():
    rows = []
    df = _injury_df([
        {
            "RosterID": "1",
            "PlayerID": "1234",
            "Player": "Jalen Coker",
            "Pos": "WR",
            "NFL": "CAR",
            "Status": "Out",
            "Injury": "Out",
            "Body": "ankle",
            "Last Updated": datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
            "Team": "Caleb's Casting Couch",
        },
        {
            "RosterID": "2",
            "PlayerID": "5678",
            "Player": "Some Guy",
            "Pos": "RB",
            "NFL": "DAL",
            "Status": "Questionable",
            "Injury": "Questionable",
            "Body": "knee",
            "Last Updated": datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
            "Team": "Other Team",
        },
    ])
    roster_name = {"1": "Caleb's Casting Couch", "2": "Other Team"}

    # Mock get_nfl_state to return current week
    import dashboard_services.service as svc
    orig = svc.get_nfl_state
    svc.get_nfl_state = lambda: {"week": 4, "season": 2026, "season_type": "reg"}
    try:
        _append_injury_rows(rows, df, roster_name, 2026)
    finally:
        svc.get_nfl_state = orig

    # Only the Out player should be included (Questionable is not serious)
    assert len(rows) == 1
    assert rows[0]["kind"] == "injury"
    assert rows[0]["week"] == 4
    assert rows[0]["data"]["player"]["name"] == "Jalen Coker"
    assert rows[0]["data"]["status"] == "OUT"
    assert rows[0]["data"]["team_name"] == "Caleb's Casting Couch"


def test_injury_rows_skip_free_agents():
    rows = []
    df = _injury_df([
        {
            "RosterID": "",
            "PlayerID": "9999",
            "Player": "Free Agent",
            "Pos": "WR",
            "NFL": "FA",
            "Status": "Out",
            "Injury": "Out",
            "Body": "",
            "Last Updated": datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
            "Team": "Free Agent",
        },
    ])
    import dashboard_services.service as svc
    orig = svc.get_nfl_state
    svc.get_nfl_state = lambda: {"week": 4, "season": 2026, "season_type": "reg"}
    try:
        _append_injury_rows(rows, df, {}, 2026)
    finally:
        svc.get_nfl_state = orig

    assert len(rows) == 0


def test_injury_rows_skip_missing_timestamp():
    rows = []
    df = _injury_df([
        {
            "RosterID": "1",
            "PlayerID": "1234",
            "Player": "No Time",
            "Pos": "WR",
            "NFL": "CAR",
            "Status": "IR",
            "Injury": "IR",
            "Body": "",
            "Last Updated": None,
            "Team": "Some Team",
        },
    ])
    import dashboard_services.service as svc
    orig = svc.get_nfl_state
    svc.get_nfl_state = lambda: {"week": 4, "season": 2026, "season_type": "reg"}
    try:
        _append_injury_rows(rows, df, {"1": "Some Team"}, 2026)
    finally:
        svc.get_nfl_state = orig

    assert len(rows) == 0


def test_injury_feed_statuses_match_serious():
    # Ensure our local set stays in sync with the canonical serious set
    from utils.lineup_issues import SERIOUS_INJURY_STATUSES
    assert _INJURY_FEED_STATUSES == SERIOUS_INJURY_STATUSES
