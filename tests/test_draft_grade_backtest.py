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
    calibration_bins,
    candidate_grid,
    correlate_grades_to_finish,
    detect_sleeper_meta,
    final_ranks,
    letter_calibration,
    multiyear_outcome,
    outcome_from_rank,
    pearson,
    rank_success,
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


def test_detect_sleeper_meta_superflex_and_type():
    sf_league = {
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN"],
        "settings": {"type": 2}, "total_rosters": 12,
    }
    # 12-round draft in a dynasty SF league -> startup, SF true.
    is_sf, dtype, teams = detect_sleeper_meta(sf_league, {"settings": {"rounds": 15}}, 12)
    assert is_sf is True and dtype == "startup" and teams == 12


def test_detect_sleeper_meta_1qb_rookie_and_redraft():
    oneqb = {"roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
             "settings": {"type": 2}, "total_rosters": 10}
    # Short draft in a dynasty 1QB league -> rookie draft.
    is_sf, dtype, teams = detect_sleeper_meta(oneqb, {"settings": {"rounds": 4}}, 10)
    assert is_sf is False and dtype == "rookie" and teams == 10
    # type == 0 is redraft regardless of rounds.
    redraft = {**oneqb, "settings": {"type": 0}}
    _, dtype2, _ = detect_sleeper_meta(redraft, {"settings": {"rounds": 15}}, 10)
    assert dtype2 == "redraft"


def test_detect_sleeper_meta_falls_back_on_garbage():
    is_sf, dtype, teams = detect_sleeper_meta(None, None, 0,
                                              default_type="startup", default_sf=True, default_teams=12)
    assert is_sf is True and dtype == "startup" and teams == 12


def test_rank_success_normalized_across_sizes():
    # Champion 1.0, last 0.0, regardless of league size (comparable across seasons).
    assert rank_success(1, 12) == 1.0
    assert rank_success(12, 12) == 0.0
    assert rank_success(1, 10) == 1.0
    assert rank_success(10, 10) == 0.0
    # Mid-pack is between; clamps out-of-range.
    assert 0.0 < rank_success(6, 12) < 1.0
    assert rank_success(99, 12) == 0.0


def test_multiyear_outcome_weights_draft_season_most():
    # One unit of success in the draft year outweighs the same unit years later.
    draft_year = multiyear_outcome([1.0, 0.0, 0.0])
    later_year = multiyear_outcome([0.0, 0.0, 1.0])
    assert draft_year > later_year
    # Empty -> None; single season -> that season's value.
    assert multiyear_outcome([]) is None
    assert multiyear_outcome([0.7]) == 0.7


def test_final_ranks_bracket_then_regular_season():
    # Champion (w=1,p=1) and runner-up (l=1,p=2) from the bracket; the rest
    # ranked beneath by wins then points-for.
    bracket = [{"p": 1, "w": 5, "l": 8}]
    rosters = [
        {"roster_id": 5, "settings": {"wins": 10, "fpts": 1500}},
        {"roster_id": 8, "settings": {"wins": 9, "fpts": 1400}},
        {"roster_id": 3, "settings": {"wins": 8, "fpts": 1600}},
        {"roster_id": 4, "settings": {"wins": 2, "fpts": 900}},
    ]
    ranks = final_ranks(rosters, bracket)
    assert ranks["5"] == 1 and ranks["8"] == 2   # champion, runner-up
    assert ranks["3"] == 3 and ranks["4"] == 4   # better record ranks ahead


def test_final_ranks_no_bracket_is_regular_season():
    rosters = [
        {"roster_id": 1, "settings": {"wins": 5, "fpts": 1000}},
        {"roster_id": 2, "settings": {"wins": 8, "fpts": 1200}},
    ]
    ranks = final_ranks(rosters, [])
    assert ranks["2"] == 1 and ranks["1"] == 2   # best record first


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


def test_calibration_bins_are_monotonic_on_signal():
    # On the value-driven synthetic signal, higher grade bins should have higher
    # mean outcomes (a well-calibrated scale).
    samples = _synthetic_samples(seed=11)
    bins = calibration_bins(samples, n_bins=5)
    assert len(bins) == 5
    assert all(b["n"] > 0 for b in bins)
    # Bins are ordered low->high grade, and the top bin out-performs the bottom.
    assert bins[0]["grade_mean"] < bins[-1]["grade_mean"]
    assert bins[-1]["outcome_mean"] > bins[0]["outcome_mean"]


def test_calibration_bins_empty_when_too_few():
    assert calibration_bins([TeamSample(picks=[], outcome=1.0)], n_bins=5) == []


def _league_samples(n_leagues=4, teams=12, seed=2):
    import random
    rnd = random.Random(seed)
    out = []
    for lg in range(n_leagues):
        for t in range(teams):
            # value spread across the league; outcome tracks drafted value.
            v = 1000 + t * 700 + rnd.uniform(-300, 300)
            out.append(TeamSample(
                picks=[_pick(value=v, vor=v * 0.5, max_val=10000)],
                outcome=v,
                meta={"league_id": f"L{lg}"},
            ))
    return out


def test_letter_calibration_tracks_outcome_on_signal():
    rows = letter_calibration(_league_samples())
    assert rows and all(r["n"] > 0 for r in rows)
    # Rows are ordered best->worst; the best letter out-performs the worst.
    assert rows[0]["outcome_mean"] > rows[-1]["outcome_mean"]
    # Letters come from the canonical A+..F order.
    order = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
    idxs = [order.index(r["letter"]) for r in rows]
    assert idxs == sorted(idxs)


def test_letter_calibration_skips_uncurvable_leagues():
    # A 2-team league can't be curved against a field -> no letters.
    two = [TeamSample(picks=[_pick(value=v)], outcome=v, meta={"league_id": "X"})
           for v in (2000, 8000)]
    assert letter_calibration(two) == []


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
