"""Tests for the draft-grade backtest harness.

Pure logic — no DB/network — so these run anywhere pytest does. They pin the
correlation math and verify that the weight sweep recovers a known signal on
deterministic synthetic data.
"""
import math

from utils.pick_score import PS_WEIGHTS
from data_building.draft_grade_backtest import (
    TeamSample,
    _perturb,
    _synthetic_samples,
    candidate_grid,
    correlate_grades_to_finish,
    outcome_from_rank,
    pearson,
    spearman,
    sweep,
    team_avg_ps,
)


# ---- pure statistics ------------------------------------------------------

def test_pearson_perfect_positive():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0


def test_pearson_perfect_negative():
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == -1.0


def test_pearson_none_on_degenerate():
    assert pearson([1], [1]) is None          # n < 2
    assert pearson([1, 1, 1], [1, 2, 3]) is None  # zero variance in x


def test_pearson_matches_manual():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 1, 4, 3, 6]
    # Reference: sxy=10, sxx=10, syy=14.8 -> 10/sqrt(10*14.8).
    assert math.isclose(pearson(xs, ys), 10.0 / math.sqrt(148.0), rel_tol=1e-9)


def test_spearman_monotonic_but_nonlinear_is_one():
    # Spearman only cares about rank order, so a monotone-nonlinear pair is 1.0.
    xs = [1, 2, 3, 4, 5]
    ys = [1, 4, 9, 16, 25]
    assert math.isclose(spearman(xs, ys), 1.0, rel_tol=1e-9)


def test_spearman_handles_ties():
    # Ties share the average rank; a constant-tie column is undefined (None).
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) is not None
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_outcome_from_rank_inverts():
    assert outcome_from_rank(1, 12) == 12.0   # champion is best
    assert outcome_from_rank(12, 12) == 1.0   # last is worst
    # Out-of-range clamps rather than raising.
    assert outcome_from_rank(0, 12) == 12.0
    assert outcome_from_rank(99, 12) == 1.0


# ---- grading --------------------------------------------------------------

def _pick(**over):
    kw = dict(
        pos="RB", value=5000, vor=2000, tier=2, age=23, rank_change_7d=0,
        avg_pick=10, pick_no=10, max_val=10000, draft_type="startup",
        is_sf=False, need_raw=0.5, qb_count=0, total_picks=180, num_teams=12,
    )
    kw.update(over)
    return kw


def test_team_avg_ps_is_mean_of_picks():
    s = TeamSample(picks=[_pick(value=9000, vor=5000), _pick(value=1000, vor=200)], outcome=1.0)
    g = team_avg_ps(s)
    assert g is not None and 0 <= g <= 100


def test_team_avg_ps_none_when_no_picks():
    assert team_avg_ps(TeamSample(picks=[], outcome=1.0)) is None


def test_weights_override_changes_grade():
    # A weight table that leans entirely on value grades a high-value pick higher
    # than one that leans entirely on momentum (which is neutral here).
    s = TeamSample(picks=[_pick(value=9500, vor=5000, rank_change_7d=0)], outcome=1.0)
    value_heavy = {"vor": 0, "value": 1.0, "adp": 0, "tier": 0, "need": 0, "youth": 0, "mom": 0, "ppg": 0}
    mom_heavy = {"vor": 0, "value": 0, "adp": 0, "tier": 0, "need": 0, "youth": 0, "mom": 1.0, "ppg": 0}
    assert team_avg_ps(s, value_heavy) > team_avg_ps(s, mom_heavy)


# ---- correlation over samples --------------------------------------------

def test_correlate_positive_on_synthetic_signal():
    # The synthetic outcome tracks drafted value, so the shipped weights (which
    # weigh value/adp heavily) should positively predict the outcome.
    samples = _synthetic_samples(seed=3)
    r = correlate_grades_to_finish(samples)
    assert r is not None and r > 0.3


def test_correlate_unknown_method_raises():
    try:
        correlate_grades_to_finish(_synthetic_samples(seed=1), method="kendall")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown method")


# ---- sweep ----------------------------------------------------------------

def test_sweep_prefers_value_over_noise_weighting():
    # Given the value-driven synthetic signal, a value-heavy table must out-predict
    # a momentum-heavy table (momentum is pure noise in the synthetic data).
    samples = _synthetic_samples(seed=5)
    value_heavy = {"vor": 0.2, "value": 0.6, "adp": 0.2, "tier": 0.0, "need": 0.0, "youth": 0.0, "mom": 0.0, "ppg": 0.0}
    mom_heavy = {"vor": 0.0, "value": 0.0, "adp": 0.0, "tier": 0.0, "need": 0.0, "youth": 0.0, "mom": 1.0, "ppg": 0.0}
    ranked = sweep(samples, [("value_heavy", value_heavy), ("mom_heavy", mom_heavy)])
    assert ranked[0][0] == "value_heavy"
    assert ranked[0][2] > ranked[1][2]


def test_sweep_sorts_best_first_and_handles_none():
    samples = _synthetic_samples(seed=9)
    # A degenerate table where every pick scores identically -> undefined corr,
    # which must sort last (after the real, defined candidate).
    flat = {"vor": 0.0, "value": 0.0, "adp": 0.0, "tier": 0.0, "need": 0.0, "youth": 0.0, "mom": 0.0, "ppg": 0.0}
    real = dict(PS_WEIGHTS["startup"])
    ranked = sweep(samples, [("flat", flat), ("real", real)])
    assert ranked[0][0] == "real"
    assert ranked[-1][2] is None


def test_candidate_grid_renormalizes_and_covers_all_levers():
    base = PS_WEIGHTS["startup"]
    cands = candidate_grid(base)
    labels = {c[0] for c in cands}
    assert "base" in labels
    for key in base:
        assert any(lbl.startswith(key) for lbl in labels if lbl != "base")
    # _perturb keeps the total weight budget (renormalized to base sum).
    bumped = _perturb(base, "value", 0.10)
    assert math.isclose(sum(bumped.values()), sum(base.values()), rel_tol=1e-9)
