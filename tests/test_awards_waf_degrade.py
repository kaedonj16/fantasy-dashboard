"""Regression: the dashboard's League Awards must degrade (not 503) when
weekly matchup fetches fail.

Production incident 2026-09-26: Fleaflicker's edge/WAF intermittently
returned HTML 403s for FetchLeagueScoreboard. highest_single_game_points()
looped weeks 1-17 calling get_matchups() with no guard, so the
ProviderUnavailableError propagated out of build_dashboard_body and the
dashboard 503'd (twice: the retry hit the cached per-key failure).
"""
import pytest

pd = pytest.importorskip("pandas")

from dashboard_services import awards
from dashboard_services.providers.base import ProviderUnavailableError


def _boom(platform, league_id, week, season):
    raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")


def _stable_roster_map(*args, **kwargs):
    return {"1": "Team One"}


def _players_map():
    return {"101": {"name": "Test Player", "position": "WR", "team": "DAL"}}


def test_all_weeks_failing_returns_default_instead_of_raising(monkeypatch):
    monkeypatch.setattr(awards, "get_matchups", _boom)
    monkeypatch.setattr(awards, "build_roster_map", _stable_roster_map)
    result = awards.highest_single_game_points(
        "92916", _players_map(), "fleaflicker", 2026, [], [], 18,
    )
    # Same "nothing found" default as an empty season: must not raise.
    assert result[1] == 0.0


def test_partial_failure_still_scores_loaded_weeks(monkeypatch):
    calls = []

    def _flaky(platform, league_id, week, season):
        calls.append(week)
        if week < 3:
            raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")
        return [{"roster_id": "1", "week": week, "players_points": {"101": 25.5}}]

    monkeypatch.setattr(awards, "get_matchups", _flaky)
    monkeypatch.setattr(awards, "build_roster_map", _stable_roster_map)
    result = awards.highest_single_game_points(
        "92916", _players_map(), "fleaflicker", 2026, [], [], 4,
    )
    assert calls == [1, 2, 3]  # kept going after the failed weeks
    assert result[0] == 3
    assert result[1] == 25.5
    assert result[2] == "Test Player"
    assert result[5] == "Team One"


def test_compute_awards_season_survives_matchup_outage(monkeypatch):
    # The dashboard calls compute_awards_season unguarded; the weekly
    # superlatives come from df_weekly while highest_player degrades.
    monkeypatch.setattr(awards, "get_matchups", _boom)
    monkeypatch.setattr(awards, "build_roster_map", _stable_roster_map)
    df = pd.DataFrame([
        {"owner": "Team One", "week": 1, "points": 100.0, "points_against": 90.0},
        {"owner": "Team One", "week": 2, "points": 110.0, "points_against": 95.0},
    ])
    result = awards.compute_awards_season(df, {}, "92916", "fleaflicker", 2026, [], [])
    assert result["highest_single_week"][2] == 110.0
    assert result["highest_player"][1] == 0.0  # degraded, not raised
