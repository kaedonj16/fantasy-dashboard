"""Unit tests for utils.player_team_schedule helpers."""
from __future__ import annotations

from utils.player_team_schedule import (
    build_team_schedule,
    shape_boxscore_payload,
    tank_boxscore_game_id,
)


def test_tank_boxscore_game_id_prefers_at_format():
    assert tank_boxscore_game_id({
        "gameID": "20250905_KC@LAC",
        "gameDate": "20250905",
        "away": "KC",
        "home": "LAC",
    }) == "20250905_KC@LAC"


def test_tank_boxscore_game_id_synthesizes_from_date():
    assert tank_boxscore_game_id({
        "gameID": "2025_01_KC_LAC",
        "gameDate": "20250905",
        "away": "KC",
        "home": "LAC",
    }) == "20250905_KC@LAC"


def test_build_team_schedule_bye_and_order():
    rows = build_team_schedule(
        "KC", 2025, bye_week=10, enrich_scores=False, include_postseason=False,
    )
    assert len(rows) == 18
    assert rows[0]["week"] == 1
    assert rows[0]["opponent"] == "LAC"
    assert rows[0]["ha"] == "@"
    assert rows[0]["game_id"].endswith("KC@LAC")
    bye = next(r for r in rows if r.get("bye"))
    assert bye["week"] == 10
    assert bye["expandable"] is False
    # Chronological: bye sits among weeks.
    weeks = [r["week"] for r in rows]
    assert weeks == sorted(weeks)


def test_shape_boxscore_missing_vs_zero():
    box = {
        "home": "KC",
        "away": "BAL",
        "homePts": "10",
        "awayPts": "7",
        "gameStatusCode": "2",
        "playerStats": {
            "1": {
                "longName": "Only Pass",
                "teamAbv": "KC",
                "Passing": {"passCompletions": "0", "passAttempts": "0", "passYds": "0", "passTD": "0", "int": "0"},
                # No Rushing block → rush cells unavailable (–), not 0.
            },
        },
    }
    payload = shape_boxscore_payload(
        box,
        game_id="20240101_BAL@KC",
        view_team="KC",
        players_index={"9": {"name": "Only Pass", "team": "KC", "pos": "QB", "tankId": "1"}},
    )
    qb = payload["teams"]["KC"]["groups"][0]["players"][0]
    assert qb["cells"]["pass_yds"] == 0
    assert qb["cells"]["cmp_att"] == "0/0"
    assert qb["cells"]["rush_yds"] is None
