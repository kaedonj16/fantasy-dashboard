"""ESPN schedule requests must not silently fall back to the current week."""
import json
from types import SimpleNamespace

from dashboard_services.providers import espn_api


class FakeRequest:
    def __init__(self):
        self.calls = []

    def league_get(self, **kwargs):
        self.calls.append(kwargs)
        return {"schedule": []}


def test_explicit_scoring_and_multiweek_matchup_period(monkeypatch):
    request = FakeRequest()
    league = SimpleNamespace(
        settings=SimpleNamespace(matchup_periods={1: [1], 2: [2], 15: [15, 16]}),
        espn_request=request, teams=[],
        _get_pro_schedule=lambda week: {}, _get_positional_ratings=lambda week: {},
    )
    monkeypatch.setattr(espn_api, "_league", lambda season, league_id: league)
    espn_api._box_score_cache.clear()

    espn_api._box_scores_cached(2026, "league-a", 16)
    call = request.calls[-1]
    assert call["params"]["scoringPeriodId"] == 16
    assert json.loads(call["headers"]["x-fantasy-filter"])["schedule"]["filterMatchupPeriodIds"]["value"] == [15]


def test_box_score_cache_isolated_by_week_league_and_season(monkeypatch):
    leagues = {}
    def fake_league(season, league_id):
        key = (season, league_id)
        if key not in leagues:
            leagues[key] = SimpleNamespace(
                settings=SimpleNamespace(matchup_periods={1: [1], 2: [2]}),
                espn_request=FakeRequest(), teams=[],
                _get_pro_schedule=lambda week: {}, _get_positional_ratings=lambda week: {},
            )
        return leagues[key]
    monkeypatch.setattr(espn_api, "_league", fake_league)
    espn_api._box_score_cache.clear()
    for args in ((2025, "a", 1), (2025, "a", 2), (2025, "b", 1), (2026, "a", 1)):
        espn_api._box_scores_cached(*args)
    assert sum(len(x.espn_request.calls) for x in leagues.values()) == 4
