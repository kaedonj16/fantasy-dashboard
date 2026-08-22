from datetime import datetime, timedelta, timezone

from dashboard_services.market_intelligence.season import (
    build_adjusted_season_projection, map_team_environment_inputs, team_environment_input,
)
from dashboard_services.market_intelligence.team import _event_teams, build_team_environments


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


def test_provider_team_variants_are_canonical_environment_keys():
    env = build_team_environments([_event("aliases", "KAN", "NWE", 46, -4)])
    assert set(env) == {"KC", "NE"}
    assert env["KC"]["implied_points"] == 25
    assert env["NE"]["implied_points"] == 21


def _production_v2_event(home_id="CLEVELAND_BROWNS_NFL", home_short="CLE",
                         home_long="Cleveland Browns", away_id="BUFFALO_BILLS_NFL",
                         away_short="BUF", away_long="Buffalo Bills"):
    event = _event("production-v2", home_id, away_id, 46, -3, books=("book",))
    event.pop("homeTeamID")
    event.pop("awayTeamID")
    event["teams"] = {
        "home": {"names": {"long": home_long, "medium": home_long.split()[-1],
                            "short": home_short},
                 "statEntityID": "home", "teamID": home_id},
        "away": {"names": {"long": away_long, "medium": away_long.split()[-1],
                            "short": away_short},
                 "statEntityID": "away", "teamID": away_id},
    }
    return event


def test_real_v2_nested_names_resolve_home_away_and_provider_aliases():
    event = _production_v2_event()
    home, away, aliases, details = _event_teams(event)
    assert (home, away) == ("CLE", "BUF")
    assert aliases["CLEVELAND_BROWNS_NFL"] == "CLE"
    assert aliases["BUFFALO_BILLS_NFL"] == "BUF"
    assert details["resolved_home"] == "CLE"
    assert details["resolved_away"] == "BUF"


def test_real_v2_nested_names_normalize_additional_provider_teams():
    event = _production_v2_event(
        "MINNESOTA_VIKINGS_NFL", "", "Minnesota Vikings",
        "BALTIMORE_RAVENS_NFL", "", "Baltimore Ravens",
    )
    home, away, aliases, _ = _event_teams(event)
    assert (home, away) == ("MIN", "BAL")
    assert aliases == {"MINNESOTA_VIKINGS_NFL": "MIN", "BALTIMORE_RAVENS_NFL": "BAL"}


def test_real_v2_nested_names_flow_through_team_environment_pipeline():
    event = _production_v2_event(
        "WASHINGTON_COMMANDERS_NFL", "WSH", "Washington Commanders",
        "NEW_ORLEANS_SAINTS_NFL", "NOR", "New Orleans Saints",
    )
    diagnostics = {}
    environments = build_team_environments([event], diagnostics)
    assert set(environments) == {"WAS", "NO"}
    assert environments["WAS"]["implied_points"] == 24.5
    assert environments["NO"]["implied_points"] == 21.5
    assert diagnostics["events_team_identity_resolved"] == 1
    assert diagnostics["team_market_odds_identified"] == 3
    assert diagnostics["games_with_usable_total_spread"] == 1


def test_real_v2_team_evidence_attaches_and_can_clear_existing_threshold():
    events = []
    for index in range(4):
        event = _production_v2_event(
            "WASHINGTON_COMMANDERS_NFL", "WSH", "Washington Commanders",
            "NEW_ORLEANS_SAINTS_NFL", "NOR", "New Orleans Saints",
        )
        event["eventID"] = f"production-v2-{index}"
        events.append(event)
    environments = build_team_environments(events)
    inputs, mapping = map_team_environment_inputs({
        "qb": {"team": "WAS", "pos": "QB"},
        "rb": {"team": "NO", "pos": "RB"},
    }, environments)
    assert mapping["matched_players"] == mapping["input_players"] == 2
    bullish = build_adjusted_season_projection(300, "QB", {}, [inputs["qb"]])
    bearish = build_adjusted_season_projection(200, "RB", {}, [inputs["rb"]])
    assert bullish["basis"] == bearish["basis"] == "team_environment"
    assert bullish["points"] > 300 and bearish["points"] < 200
    assert bullish["confidence"] >= 0.35
