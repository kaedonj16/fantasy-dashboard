from datetime import datetime, timedelta, timezone

from dashboard_services.market_intelligence.normalize import normalize_event


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
