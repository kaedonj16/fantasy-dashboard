"""Playoff odds in median leagues: the weekly median game must be simulated.

Regression coverage: the Monte Carlo used to tally only head-to-head games
in the remaining schedule, so in Sleeper ``league_average_match`` leagues
every team could gain at most one win per remaining week instead of two —
completed weeks (which fold the median game in) were right, the sim was not.

These tests pin the tally directly with deterministic score maps (no RNG
flakiness): with well-separated fixed scores the median outcome per team is
exact, so expected win totals are exact too.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from data_building.simulate_playoff_odds import (
    _median_match_enabled,
    _run_mc,
    _simulate_week_scores,
    _standings_from_score_map,
)


def _teams():
    # Four teams with fixed, well-separated scoring levels. Median of
    # (130, 120, 110, 100) is 115, so T1/T2 beat the median and T3/T4 lose
    # to it, every week, in every sim.
    return [
        {"roster_id": 1, "name": "T1", "wins": 0, "losses": 0, "ties": 0,
         "avg": 130.0, "std": 1.0, "pf": 0.0},
        {"roster_id": 2, "name": "T2", "wins": 0, "losses": 0, "ties": 0,
         "avg": 120.0, "std": 1.0, "pf": 0.0},
        {"roster_id": 3, "name": "T3", "wins": 0, "losses": 0, "ties": 0,
         "avg": 110.0, "std": 1.0, "pf": 0.0},
        {"roster_id": 4, "name": "T4", "wins": 0, "losses": 0, "ties": 0,
         "avg": 100.0, "std": 1.0, "pf": 0.0},
    ]


def _matchups(weeks=(1, 2)):
    return {w: [(1, 2), (3, 4)] for w in weeks}


def _score_map(teams, weeks, n_sims=500):
    """Deterministic: every team scores its avg every week in every sim."""
    return {
        w: {t["roster_id"]: np.full(n_sims, t["avg"], dtype=np.float32)
            for t in teams}
        for w in weeks
    }


def _avg_wins(rows):
    return {r["roster_id"]: r["avg_final_wins"] for r in rows}


def _games(n_teams=4, weeks=(1, 2)):
    zeros = np.zeros(n_teams, dtype=np.float32)
    return zeros


def test_median_match_enabled_flag_forms():
    assert _median_match_enabled({"league_settings": {"league_average_match": 1}})
    assert _median_match_enabled({"league_settings": {"league_average_match": "1"}})
    assert _median_match_enabled({"league_settings": {"league_average_match": True}})
    assert not _median_match_enabled({"league_settings": {"league_average_match": 0}})
    assert not _median_match_enabled({"league_settings": {}})
    assert not _median_match_enabled({})


def test_head_to_head_only_without_flag():
    teams, weeks = _teams(), (1, 2)
    rows = _standings_from_score_map(
        teams, _matchups(weeks), _score_map(teams, weeks), 2, 500,
        _games(), median_match=False,
    )
    wins = _avg_wins(rows)
    # 1 beats 2, 3 beats 4, one game per week.
    assert wins == {1: 2.0, 2: 0.0, 3: 2.0, 4: 0.0}


def test_median_game_adds_second_weekly_game():
    teams, weeks = _teams(), (1, 2)
    rows = _standings_from_score_map(
        teams, _matchups(weeks), _score_map(teams, weeks), 2, 500,
        _games(), median_match=True,
    )
    wins = _avg_wins(rows)
    # H2H: 1>2, 3>4. Median (115): 1 and 2 win, 3 and 4 lose.
    assert wins == {1: 4.0, 2: 2.0, 3: 2.0, 4: 0.0}


def test_median_wins_seed_by_record_then_points():
    # T2 and T3 both finish at 2.0 wins; T2's higher PF must seed it above T3.
    teams, weeks = _teams(), (1, 2)
    rows = _standings_from_score_map(
        teams, _matchups(weeks), _score_map(teams, weeks), 2, 500,
        _games(), median_match=True,
    )
    by_id = {r["roster_id"]: r for r in rows}
    assert by_id[2]["playoff_pct"] > by_id[3]["playoff_pct"]
    assert by_id[1]["playoff_pct"] >= by_id[2]["playoff_pct"]


def test_median_game_tie_is_half_win():
    # All four teams score exactly the median: every median game ties.
    teams = _teams()
    for t in teams:
        t["avg"] = 110.0
    weeks = (1, 2)
    rows = _standings_from_score_map(
        teams, _matchups(weeks), _score_map(teams, weeks), 2, 500,
        _games(), median_match=True,
    )
    by_id = {r["roster_id"]: r for r in rows}
    # H2H games tie too: 0.5 per H2H tie + 0.5 per median tie. The projected
    # record is conventional W-L-T, so ties show as ties, not wins.
    for r in rows:
        assert r["avg_final_wins"] == 0.0
        assert r["avg_final_ties"] == 4.0


def test_run_mc_counts_two_games_per_week_in_median_leagues():
    teams = _teams()
    weeks = (1, 2)
    lost = np.array([4.0] * 9, dtype=np.float32)
    haz = np.zeros(9, dtype=np.float32)  # no injury noise: keep it deterministic
    week_profiles = {
        w: {t["roster_id"]: {"mean": t["avg"], "std": t["std"],
                             "lost": lost, "haz": haz}
            for t in teams}
        for w in weeks
    }
    plain = _run_mc(teams, _matchups(weeks), week_profiles, 2, 400, 7,
                    median_match=False)
    median = _run_mc(teams, _matchups(weeks), week_profiles, 2, 400, 7,
                     median_match=True)
    plain_w = _avg_wins(plain)[1]
    median_w = _avg_wins(median)[1]
    # Top team nearly doubles its projected wins: ~1/week -> ~2/week.
    assert median_w > plain_w * 1.5


def test_games_per_team_doubles_in_median_leagues():
    teams = _teams()
    weeks = (1, 2)
    lost = np.array([4.0] * 9, dtype=np.float32)
    haz = np.array([0.05] * 9, dtype=np.float32)
    week_profiles = {
        w: {t["roster_id"]: {"mean": t["avg"], "std": t["std"],
                             "lost": lost, "haz": haz}
            for t in teams}
        for w in weeks
    }
    _, games_plain, _ = _simulate_week_scores(
        teams, _matchups(weeks), week_profiles, 100, 7, median_match=False)
    _, games_med, _ = _simulate_week_scores(
        teams, _matchups(weeks), week_profiles, 100, 7, median_match=True)
    assert (games_plain == 2.0).all()
    assert (games_med == 4.0).all()
