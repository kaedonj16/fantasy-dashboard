"""Weekly-win Impact numbers must compare remaining-week PPG to remaining-week
PPG. Mixing that with season-lineup PPG made every contending acquisition look
like a weekly-win loss while playoff odds (true before/after Monte Carlo) rose.
"""
from dashboard_services.archetype_engine import (
    _viewer_remaining_week_avg,
    _win_prob,
)


def test_viewer_remaining_week_avg_prefers_week_profiles():
    """The before-side of Impact wk is the remaining-week mean, not team['avg']."""
    state = {
        "week_profiles": {
            1: {7: {"mean": 111.0}},
            2: {7: {"mean": 109.0}},
        },
        "teams": [{"roster_id": 7, "avg": 99.0}],
    }
    # (111 + 109) / 2, rounded to 1 decimal — same as _viewer_week_mean
    assert _viewer_remaining_week_avg(state, 7) == 110.0


def test_viewer_remaining_week_avg_falls_back_to_team_avg():
    state = {"week_profiles": {}, "teams": [{"roster_id": 7, "avg": 104.5}]}
    assert _viewer_remaining_week_avg(state, 7) == 104.5


def test_viewer_remaining_week_avg_missing_viewer_returns_none():
    assert _viewer_remaining_week_avg({"week_profiles": {}, "teams": []}, 1) is None


def test_remaining_week_baseline_keeps_upgrade_positive():
    """A slightly stronger remaining-week lineup must show a weekly-win *gain*.

    The old Impact path did win_prob(new_week_avg) - win_prob(season_lineup).
    Season-lineup PPG sits well above remaining-week PPG, so that subtraction
    went negative for every target — including upgrades like +4 PPG.
    """
    league_avg = 110.0
    remaining_before = 110.0
    remaining_after = 114.0
    season_lineup = 140.0  # typical _ppg_lineup vs remaining-week gap

    correct = _win_prob(remaining_after, league_avg) - _win_prob(remaining_before, league_avg)
    mixed = _win_prob(remaining_after, league_avg) - _win_prob(season_lineup, league_avg)

    assert correct > 0
    assert mixed < 0
    assert abs(correct) < 0.10  # a 4-PPG bump is a modest weekly-win swing
