"""Washington is WAS everywhere; WSH is only an incoming alias."""
import pytest

# utils.utils pulls requests + bs4 + Flask. Gate so the slim lint job skips.
pytest.importorskip("requests")
pytest.importorskip("bs4")
pytest.importorskip("flask")

from utils.utils import (
    canon_team,
    canonical_teams_index,
    canonicalize_game_teams,
    canonicalize_schedule,
    load_teams_index,
)


def test_canon_team_maps_wsh_to_was():
    assert canon_team("WSH") == "WAS"
    assert canon_team("wsh") == "WAS"
    assert canon_team("WAS") == "WAS"


def test_canonical_teams_index_merges_wsh_into_was():
    merged = canonical_teams_index({
        "WAS": {"teamId": "32", "byeWeek": 12, "rush_yds_pg": None, "Logo": "was.png"},
        "WSH": {"rush_yds_pg": 134.7, "pass_yds_pg": 184.1},
        "KC": {"teamId": "12"},
    })
    assert "WSH" not in merged
    assert set(merged) == {"WAS", "KC"}
    assert merged["WAS"]["teamId"] == "32"
    assert merged["WAS"]["byeWeek"] == 12
    assert merged["WAS"]["rush_yds_pg"] == 134.7
    assert merged["WAS"]["pass_yds_pg"] == 184.1
    assert merged["WAS"]["Logo"] == "was.png"


def test_canonicalize_game_and_schedule_use_was():
    game = canonicalize_game_teams({"home": "PHI", "away": "WSH", "gameID": "20260913_WSH@PHI"})
    assert game["away"] == "WAS"
    assert game["home"] == "PHI"
    assert game["gameID"] == "20260913_WSH@PHI"

    week = canonicalize_schedule([
        {"home": "WSH", "away": "DAL"},
        {"home": "KC", "away": "LV"},
    ])
    assert week[0]["home"] == "WAS"
    assert week[1]["home"] == "KC"


def test_load_teams_index_has_was_not_wsh():
    index = load_teams_index() or {}
    if not index:
        return
    assert "WAS" in index
    assert "WSH" not in index
    assert len([k for k in index if k in ("WAS", "WSH")]) == 1
