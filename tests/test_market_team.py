from dashboard_services.market_intelligence.team import build_team_environments


def _event(event_id, home, away, total, home_spread, books=("a", "b")):
    odds = []
    for book in books:
        odds.extend([
            {"bookmakerID": book, "periodID": "game", "statID": "points",
             "betTypeID": "over", "bookOverUnder": total},
            {"bookmakerID": book, "periodID": "game", "statID": "spread",
             "statEntityID": home, "bookSpread": home_spread},
            {"bookmakerID": book, "periodID": "game", "statID": "spread",
             "statEntityID": away, "bookSpread": -home_spread},
        ])
    return {"eventID": event_id, "teams": [{"teamID": home}, {"teamID": away}], "odds": odds}


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
