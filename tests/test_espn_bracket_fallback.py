"""Focused tests for the ESPN bracket fallback fix."""
import pytest

pytest.importorskip("pandas")

import sys
from unittest.mock import patch

sys.path.insert(0, ".")

from dashboard_services.providers import espn_api


def _game(mid, mp, tier, h_id, a_id, h_pts, a_pts, winner=None):
    g = {
        "id": mid,
        "matchupPeriodId": mp,
        "playoffTierType": tier,
        "home": {"teamId": h_id, "totalPoints": h_pts},
        "away": {"teamId": a_id, "totalPoints": a_pts},
    }
    if winner:
        g["winner"] = winner
    return g


class FakeSettings:
    reg_season_count = 14


class FakeLeague:
    settings = FakeSettings()


def test_fallback_uses_matchup_period_when_tier_empty():
    """Past seasons with empty playoffTierType: filter by matchupPeriodId."""
    schedule = [
        _game(1, 14, "", 1, 2, 100.0, 90.0),  # regular season, excluded
        _game(2, 15, "", 1, 4, 110.0, 95.0),  # playoff, included
        _game(3, 16, "", 1, 3, 120.0, 115.0),  # playoff, included
    ]
    with patch.object(espn_api, "_playoff_schedule_cached", return_value=schedule), \
         patch.object(espn_api, "_league", return_value=FakeLeague()):
        result = espn_api.espn_get_bracket_like("123", 2025, "winners")
    assert len(result) == 2
    assert all(m["r"] in (1, 2) for m in result)


def test_winner_extracted_from_explicit_field():
    """ESPN's winner field (HOME/AWAY) determines w/l."""
    schedule = [
        _game(1, 15, "WINNERS", 1, 2, 100.0, 90.0, winner="HOME"),
        _game(2, 15, "WINNERS", 3, 4, 80.0, 95.0, winner="AWAY"),
    ]
    with patch.object(espn_api, "_playoff_schedule_cached", return_value=schedule), \
         patch.object(espn_api, "_league", return_value=FakeLeague()):
        result = espn_api.espn_get_bracket_like("123", 2025, "winners")
    by_mid = {m["m"]: m for m in result}
    assert by_mid[1]["w"] == 1
    assert by_mid[1]["l"] == 2
    assert by_mid[2]["w"] == 4
    assert by_mid[2]["l"] == 3


def test_winner_falls_back_to_score_comparison():
    """When winner field is missing, higher score wins."""
    schedule = [
        _game(1, 15, "", 1, 2, 100.0, 90.0),
    ]
    with patch.object(espn_api, "_playoff_schedule_cached", return_value=schedule), \
         patch.object(espn_api, "_league", return_value=FakeLeague()):
        result = espn_api.espn_get_bracket_like("123", 2025, "winners")
    assert result[0]["w"] == 1
    assert result[0]["l"] == 2


def test_tier_type_still_preferred_when_present():
    """When playoffTierType is populated, use it (not the fallback)."""
    schedule = [
        _game(1, 15, "WINNERS", 1, 2, 100.0, 90.0),
        _game(2, 15, "CONSOLATION", 3, 4, 80.0, 95.0),
    ]
    with patch.object(espn_api, "_playoff_schedule_cached", return_value=schedule), \
         patch.object(espn_api, "_league", return_value=FakeLeague()):
        winners = espn_api.espn_get_bracket_like("123", 2025, "winners")
        losers = espn_api.espn_get_bracket_like("123", 2025, "losers")
    assert len(winners) == 1
    assert winners[0]["m"] == 1
    assert len(losers) == 1
    assert losers[0]["m"] == 2


def test_losers_empty_when_no_tier_type():
    """Cannot distinguish consolation without tier type; return empty."""
    schedule = [
        _game(1, 15, "", 1, 2, 100.0, 90.0),
    ]
    with patch.object(espn_api, "_playoff_schedule_cached", return_value=schedule), \
         patch.object(espn_api, "_league", return_value=FakeLeague()):
        result = espn_api.espn_get_bracket_like("123", 2025, "losers")
    assert result == []
