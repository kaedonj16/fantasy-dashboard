from datetime import datetime, timedelta, timezone

from dashboard_services.market_intelligence.season import team_environment_input
from dashboard_services.market_intelligence.team import build_team_environments


def _event(event_id, home, away, total, home_spread, books=("a", "b")):
    odds = []
    for book in books:
        odds.extend([
            {"bookmakerID": book, "periodID": "game", "statID": "points",
             "betTypeID": "ou", "bookOverUnder": total},
            {"bookmakerID": book, "periodID": "game", "statID": "spread",
             "betTypeID": "sp", "statEntityID": home, "bookSpread": home_spread},
            {"bookmakerID": book, "periodID": "game", "statID": "spread",
             "betTypeID": "sp", "statEntityID": away, "bookSpread": -home_spread},
        ])
    return {"eventID": event_id, "homeTeamID": home, "awayTeamID": away,
            "startTime": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
            "teams": [{"teamID": home}, {"teamID": away}], "odds": odds}


def test_team_environment_aggregates_relative_sgo_implied_points_and_coverage():
    events = [_event("1", "BUF", "NYJ", 48, -6), _event("2", "BUF", "MIA", 50, -4),
              _event("3", "NYJ", "MIA", 40, 3), _event("4", "BUF", "NE", 46, -7)]
    env = build_team_environments(events)
    assert env["BUF"]["implied_points"] > env["NYJ"]["implied_points"]
    assert env["BUF"]["score"] > 0 > env["NYJ"]["score"]
    assert env["BUF"]["coverage"] > env["NYJ"]["coverage"]
    assert env["BUF"]["source"] == "sportsgameodds"


def test_team_environment_rejects_player_and_partial_game_markets():
    event = _event("1", "BUF", "NYJ", 48, -6)
    for odd in event["odds"]:
        odd["periodID"] = "1h"
    event["odds"].append({"playerID": "p", "statID": "points", "bookOverUnder": 45})
    assert build_team_environments([event]) == {}


def test_team_environment_missing_spread_fails_closed():
    event = _event("1", "BUF", "NYJ", 48, -6)
    event["odds"] = [row for row in event["odds"] if row["statID"] != "spread"]
    assert build_team_environments([event]) == {}


def test_team_environment_reads_v2_nested_bookmaker_lines():
    now = datetime.now(timezone.utc)
    event = {
        "eventID": "v2", "homeTeamID": "team-buf", "awayTeamID": "team-nyj",
        "status": {"startsAt": (now + timedelta(days=1)).isoformat()},
        "teams": {
            "team-buf": {"teamID": "team-buf", "abbreviation": "BUF"},
            "team-nyj": {"teamID": "team-nyj", "abbreviation": "NYJ"},
        },
        "odds": [
            {"periodID": "game", "statID": "points", "statEntityID": "v2", "betTypeID": "ou",
             "sideID": "over", "bookOverUnder": 48,
             "byBookmaker": {"a": {"overUnder": 47, "available": True},
                              "b": {"overUnder": 49, "available": True},
                              "closed": {"overUnder": 80, "available": False}}},
            {"periodID": "game", "statID": "points", "statEntityID": "v2", "betTypeID": "ou",
             "sideID": "under", "byBookmaker": {"a": {"overUnder": 47, "available": True},
                                                   "b": {"overUnder": 49, "available": True}}},
            {"periodID": "game", "statID": "points", "betTypeID": "sp", "sideID": "home",
             "byBookmaker": {"a": {"overUnder": -5, "available": True},
                              "b": {"overUnder": -7, "available": True}}},
            {"periodID": "game", "statID": "points", "betTypeID": "sp", "sideID": "away",
             "byBookmaker": {"a": {"spread": 5, "available": True},
                              "b": {"spread": 7, "available": True}}},
        ],
    }

    diagnostics = {}
    env = build_team_environments([event], diagnostics, now)
    assert env["BUF"]["implied_points"] == 27
    assert env["NYJ"]["implied_points"] == 21
    assert env["BUF"]["book_count"] == 2
    assert diagnostics["team_market_odds_identified"] == 4
    assert diagnostics["full_game_totals_accepted"] == 4
    assert diagnostics["full_game_spreads_accepted"] == 4
    assert diagnostics["games_with_usable_total_spread"] == 1
    assert diagnostics["unavailable"] == 1


def test_team_environment_rejects_live_stale_team_total_and_unaligned_games():
    now = datetime.now(timezone.utc)
    live = _event("live", "BUF", "NYJ", 48, -6)
    live["status"] = {"startsAt": (now + timedelta(days=1)).isoformat(), "live": True}
    unaligned = _event("unaligned", "BUF", "NYJ", 48, -6)
    unaligned.pop("homeTeamID")
    unaligned.pop("awayTeamID")
    bad = _event("bad", "BUF", "NYJ", 48, -6)
    bad["odds"].extend([
        {"periodID": "1h", "statID": "points", "betTypeID": "ou", "bookOverUnder": 24},
        {"periodID": "game", "statID": "points", "statEntityID": "BUF",
         "betTypeID": "ou", "bookOverUnder": 27},
        {"periodID": "game", "statID": "points", "playerID": "p",
         "betTypeID": "ou", "bookOverUnder": 1.5},
        {"periodID": "game", "statID": "points", "betTypeID": "ou",
         "bookOverUnder": 50, "updatedAt": (now - timedelta(days=1)).isoformat()},
    ])
    # Remove the valid rows so only rejected market shapes remain.
    bad["odds"] = bad["odds"][-4:]
    diagnostics = {}
    assert build_team_environments([live, unaligned, bad], diagnostics, now) == {}
    assert diagnostics["wrong_period"] == 1
    assert diagnostics["unsupported"] >= 2
    assert diagnostics["unavailable"] == 2
    assert diagnostics["missing_team"] >= 1


def test_team_environment_confidence_and_projection_effect_remain_capped():
    env = build_team_environments([_event("1", "BUF", "NYJ", 48, -6)])["BUF"]
    assert 0 < env["confidence"] <= 0.68
    item = team_environment_input("p", "QB", env)
    assert item is not None
    assert 0 < item.value <= 0.03


def test_v2_team_mapping_uses_explicit_home_away_keys_without_top_level_ids():
    event = _event("aligned", "BUF", "NYJ", 44, -2, books=("a",))
    event.pop("homeTeamID")
    event.pop("awayTeamID")
    event["teams"] = {
        "home": {"teamID": "buf-id", "abbreviation": "BUF"},
        "away": {"teamID": "nyj-id", "abbreviation": "NYJ"},
    }
    for odd in event["odds"]:
        if odd.get("statEntityID") == "BUF":
            odd["statEntityID"] = "buf-id"
        elif odd.get("statEntityID") == "NYJ":
            odd["statEntityID"] = "nyj-id"
    env = build_team_environments([event])
    assert env["BUF"]["implied_points"] == 23
    assert env["NYJ"]["implied_points"] == 21
