"""Sample-size-aware preseason-prior decay for the playoff-odds strength model.

Regression guards for the fix to the old fixed ``_WEEKLY_BLEND = 0.30``: one
completed game used to overwrite ~70% of a team's projected future strength
league-wide, causing playoff odds to swing wildly in weeks 1-3. The model now
decays the projection prior per-roster based on that roster's own
games-played count (``_games_played`` / ``_projection_actual_weights`` /
``_strength_weights_for_teams``), and blends variance (not just mean) so a
single noisy game can't collapse a team's modeled volatility.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from data_building.simulate_playoff_odds import (
    _PRESEASON_PRIOR_GAMES,
    _build_teams,
    _games_played,
    _projection_actual_weights,
    _strength_weights_for_teams,
    _team_stats_signature,
    _team_week_profile,
    _MIN_STD,
)


# ---------------------------------------------------------------------------
# _games_played / _projection_actual_weights / _strength_weights_for_teams
# ---------------------------------------------------------------------------

def test_games_played_from_wins_losses_ties():
    assert _games_played({"wins": 0, "losses": 0, "ties": 0}) == 0
    assert _games_played({"wins": 3, "losses": 1, "ties": 1}) == 5
    # Missing/garbage fields degrade to 0 rather than raising.
    assert _games_played({}) == 0
    assert _games_played({"wins": None, "losses": None, "ties": None}) == 0


def test_projection_actual_weights_matches_spec_curve():
    # 0 games -> 100% projection / 0% actual (preseason behavior unchanged).
    pw, aw = _projection_actual_weights(0)
    assert pw == pytest.approx(1.0)
    assert aw == pytest.approx(0.0)

    # 1 game -> ~83.3% / ~16.7%.
    pw, aw = _projection_actual_weights(1)
    assert pw == pytest.approx(5 / 6, abs=1e-6)
    assert aw == pytest.approx(1 / 6, abs=1e-6)

    # 5 games (== _PRESEASON_PRIOR_GAMES) -> exactly 50/50.
    pw, aw = _projection_actual_weights(int(_PRESEASON_PRIOR_GAMES))
    assert pw == pytest.approx(0.5)
    assert aw == pytest.approx(0.5)

    # 10 games -> ~33.3% / ~66.7%.
    pw, aw = _projection_actual_weights(10)
    assert pw == pytest.approx(1 / 3, abs=1e-6)
    assert aw == pytest.approx(2 / 3, abs=1e-6)

    # Weights always sum to 1 and never go negative.
    for gp in range(0, 20):
        pw, aw = _projection_actual_weights(gp)
        assert pw + aw == pytest.approx(1.0)
        assert pw >= 0.0 and aw >= 0.0
    # The prior never fully vanishes, however many games are played.
    pw, _ = _projection_actual_weights(17)
    assert pw > 0.0


def test_strength_weights_are_per_roster_not_global():
    """Two teams with different games-played get independently decayed
    weights -- the old model applied one scalar to the whole league."""
    teams = [
        {"roster_id": 1, "wins": 0, "losses": 0, "ties": 0},   # 0 games (bye/no data)
        {"roster_id": 2, "wins": 1, "losses": 0, "ties": 0},   # 1 game
        {"roster_id": 3, "wins": 2, "losses": 2, "ties": 0},   # 4 games
    ]
    weights = _strength_weights_for_teams(teams)
    assert weights[1]["games_played"] == 0
    assert weights[1]["projection_weight"] == pytest.approx(1.0)
    assert weights[2]["games_played"] == 1
    assert weights[3]["games_played"] == 4
    # More games played -> strictly less projection weight.
    assert weights[1]["projection_weight"] > weights[2]["projection_weight"] > weights[3]["projection_weight"]


def test_preseason_all_teams_get_pure_projection_weight():
    """Every roster at 0 games played -> projection_weight == 1.0 uniformly,
    matching the old (unconditional) preseason behavior exactly."""
    teams = [{"roster_id": i, "wins": 0, "losses": 0, "ties": 0} for i in range(1, 9)]
    weights = _strength_weights_for_teams(teams)
    assert all(w["projection_weight"] == pytest.approx(1.0) for w in weights.values())
    assert all(w["actual_weight"] == pytest.approx(0.0) for w in weights.values())


# ---------------------------------------------------------------------------
# One played game shouldn't dominate the strength estimate
# ---------------------------------------------------------------------------

def test_one_game_does_not_dominate_mean_like_old_fixed_blend():
    """Under the old _WEEKLY_BLEND=0.30, one game got 70% control of a team's
    mean. Under the new model, one game gets ~16.7% control -- a much smaller,
    proportionate nudge."""
    projected = 100.0
    one_game_actual = 160.0  # a big outlier single-game score

    _, actual_weight_1game = _projection_actual_weights(1)
    old_fixed_actual_weight = 0.70  # (1 - old _WEEKLY_BLEND)
    assert actual_weight_1game < old_fixed_actual_weight

    profile = _team_week_profile(
        pids=[], ppg_map={}, pos_map={}, roster_positions=[],
        hist_avg=one_game_actual, hist_std=0.0,
        projection_weight=1.0 - actual_weight_1game,
    )
    # With projection falling back to 0 (empty roster) and hist_avg=160,
    # the blended mean should stay much closer to a small nudge off 0 than
    # to the naive 70%-weighted 112 the old model would have produced.
    old_style_mean = 0.30 * 0 + 0.70 * one_game_actual
    assert profile["mean"] < old_style_mean


def test_variance_blends_instead_of_overwriting():
    """A team's future volatility should not collapse to a tiny single-game
    historical std; it blends proportionally with the projected variance."""
    # hist_std=2.0 (tiny, from a single suspiciously-consistent game) should
    # not fully override the projected std when actual_weight is small.
    _, actual_weight_1game = _projection_actual_weights(1)
    profile = _team_week_profile(
        pids=[], ppg_map={}, pos_map={}, roster_positions=[],
        hist_avg=100.0, hist_std=2.0,
        projection_weight=1.0 - actual_weight_1game,
    )
    # Pure historical (old model, effectively) would floor at hist_std=2 -> _MIN_STD.
    # The blended value should be higher than a naive full-weight historical std,
    # since the projected variance (from an empty roster, floored at _MIN_STD)
    # still carries most of the weight.
    assert profile["std"] > 2.0


def test_nan_std_from_single_game_sample_does_not_propagate():
    """pandas groupby(...).std() is NaN for a 1-game sample (ddof=1 needs >=2
    points). _build_teams must sanitize this rather than let NaN leak into
    the Monte Carlo engine."""
    team_stats = pd.DataFrame([
        {"owner": "Alice", "Wins": 1, "Losses": 0, "Ties": 0, "PF": 120.0,
         "AVG": 120.0, "STD": float("nan")},
        {"owner": "Bob", "Wins": 0, "Losses": 1, "Ties": 0, "PF": 90.0,
         "AVG": 90.0, "STD": float("nan")},
    ])
    roster_map = {1: "Alice", 2: "Bob"}
    teams = _build_teams(team_stats, roster_map)
    for t in teams:
        assert not math.isnan(t["std"])
        assert not math.isnan(t["avg"])

    # And the downstream profile-builder must produce a finite, floored std
    # (not NaN) when handed that sanitized 0.0 "no signal" sentinel.
    profile = _team_week_profile(
        pids=[], ppg_map={}, pos_map={}, roster_positions=[],
        hist_avg=teams[0]["avg"], hist_std=teams[0]["std"],
        projection_weight=0.8333,
    )
    assert not math.isnan(profile["std"])
    assert profile["std"] >= _MIN_STD


# ---------------------------------------------------------------------------
# Cache-invalidation signature
# ---------------------------------------------------------------------------

def test_team_stats_signature_changes_with_score_correction():
    """A mid-week score correction changes AVG/STD/PF without necessarily
    advancing current_week -- the cache signature must reflect that so a
    stale sim_state isn't served for up to the TTL."""
    before = pd.DataFrame([
        {"owner": "Alice", "Wins": 1, "Losses": 0, "Ties": 0, "PF": 120.0,
         "AVG": 120.0, "STD": 10.0},
    ])
    after = pd.DataFrame([
        {"owner": "Alice", "Wins": 1, "Losses": 0, "Ties": 0, "PF": 130.0,
         "AVG": 130.0, "STD": 10.0},
    ])
    assert _team_stats_signature(before) != _team_stats_signature(after)


def test_team_stats_signature_stable_for_identical_frames():
    stats = pd.DataFrame([
        {"owner": "Alice", "Wins": 1, "Losses": 0, "Ties": 0, "PF": 120.0,
         "AVG": 120.0, "STD": 10.0},
    ])
    assert _team_stats_signature(stats) == _team_stats_signature(stats.copy())


def test_team_stats_signature_empty_is_stable_and_falsy():
    assert _team_stats_signature(None) == ""
    assert _team_stats_signature(pd.DataFrame()) == ""
