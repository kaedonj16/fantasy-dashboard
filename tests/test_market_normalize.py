from datetime import datetime, timedelta, timezone

import pytest

from dashboard_services.market_intelligence.normalize import normalize_event


def _v2_player_event(now):
    return {
        "eventID": "nfl-2026-01",
        "status": {"startsAt": (now + timedelta(days=1)).isoformat()},
        "odds": {
            "passing_yards-QB1-game-ou-over": {
                "statID": "passing_yards", "statEntityID": "QB1",
                "periodID": "game", "sideID": "over",
                "byBookmaker": {
                    "draftkings": {"odds": -115, "overUnder": 275.5, "available": True},
                    "fanduel": {"odds": -110, "overUnder": 275.5, "available": True},
                    "closed": {"odds": -105, "overUnder": 275.5, "available": False},
                },
            },
            "passing_yards-QB1-game-ou-under": {
                "statID": "passing_yards", "statEntityID": "QB1",
                "periodID": "game", "sideID": "under",
                "byBookmaker": {
                    "draftkings": {"odds": -105, "overUnder": 275.5, "available": True},
                    "fanduel": {"odds": -110, "overUnder": 275.5, "available": True},
                },
            },
            "game_total": {
                "statID": "points", "statEntityID": "BUF", "periodID": "game",
                "byBookmaker": {"draftkings": {"odds": -110, "overUnder": 47.5, "available": True}},
            },
        },
    }


def test_v2_nested_books_and_sides_form_one_market_per_sportsbook():
    now = datetime.now(timezone.utc)
    diagnostics = {}
    records = normalize_event(_v2_player_event(now), now, diagnostics)

    assert len(records) == 2
    assert {row.sportsbook for row in records} == {"draftkings", "fanduel"}
    assert all(row.provider_player_id == "QB1" for row in records)
    assert all(row.stat_type == "passing_yards" and row.line == 275.5 for row in records)
    assert {(row.over_price, row.under_price) for row in records} == {(-115.0, -105.0), (-110.0, -110.0)}
    assert all(row.side is None and row.period == "game" for row in records)
    assert diagnostics == {
        "odds_inspected": 3, "player_props_identified": 2,
        "bookmaker_entries_inspected": 5, "missing_player": 0, "missing_book": 0,
        "missing_stat": 1, "missing_line": 0, "unavailable": 1,
    }


def test_v2_missing_start_and_malformed_payload_fail_closed():
    now = datetime.now(timezone.utc)
    event = _v2_player_event(now)
    event.pop("status")
    assert normalize_event(event, now) == []
    assert normalize_event({"eventID": "x", "startTime": "not-a-date", "odds": "bad"}, now) == []


def test_player_identity_precedence_keeps_legacy_player_id():
    now = datetime.now(timezone.utc)
    event = _v2_player_event(now)
    odd = next(iter(event["odds"].values()))
    odd.update({"playerID": "legacy", "statEntityID": "entity", "player": {"playerID": "nested"}})
    records = normalize_event(event, now)
    assert records[0].provider_player_id == "legacy"


@pytest.mark.parametrize(("provider_stat", "canonical_stat"), [
    ("passing_yards", "passing_yards"),
    ("passing_touchdowns", "passing_touchdowns"),
    ("passing_interceptions", "interceptions"),
    ("rushing_yards", "rushing_yards"),
    ("rushing_attempts", "rushing_attempts"),
    ("receptions", "receptions"),
    ("receiving_yards", "receiving_yards"),
    ("receiving_touchdowns", "receiving_touchdowns"),
    ("anytime_touchdown", "touchdowns"),
])
def test_common_v2_nfl_stat_ids_map_to_canonical_stats(provider_stat, canonical_stat):
    now = datetime.now(timezone.utc)
    event = {
        "eventID": "stats", "status": {"startsAt": (now + timedelta(days=1)).isoformat()},
        "odds": [{
            "statID": provider_stat, "statEntityID": "player", "periodID": "game",
            "byBookmaker": {"book": {"overUnder": 1.5, "odds": -110, "available": True}},
        }],
    }
    records = normalize_event(event, now)
    assert len(records) == 1
    assert records[0].stat_type == canonical_stat


def test_season_market_is_kept_separate_from_weekly_context():
    now = datetime.now(timezone.utc)
    event = {
        "eventID": "future-1",
        "startTime": (now + timedelta(days=30)).isoformat(),
        "eventType": "NFL Futures",
        "odds": [{
            "playerID": "player-1", "bookmakerID": "book-1",
            "marketName": "Regular Season Receiving Yards",
            "line": "1099.5", "overOdds": -110, "underOdds": -110,
        }],
    }

    records = normalize_event(event, now)

    assert len(records) == 1
    assert records[0].context == "season"
    assert records[0].stat_type == "receiving_yards"
