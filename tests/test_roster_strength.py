"""Unit tests for utils.roster_strength.weighted_pos_strength.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
import pytest

from utils.roster_strength import (
    average_league_percentiles, fit_roster_component_weights,
    positional_strength_profile, rank_rosters_by_position,
    roster_pos_value_lists, strength_percentile, weighted_pos_strength,
)


def test_empty_values_return_zero():
    assert weighted_pos_strength([], "RB", {}) == 0.0
    assert weighted_pos_strength(None, "RB", {}) == 0.0


def test_single_player_scores_against_the_full_denominator():
    # Dilution fix: unfilled slots score 0 instead of shrinking the
    # denominator, so a lone player scores value / full weight sum.
    # QB 1QB weights [1.0, 0.20, 0.05]; RB no-flex [1.0, 0.85, 0.15, 0.05].
    assert weighted_pos_strength([100.0], "QB", {}) == pytest.approx(100.0 / 1.25)
    assert weighted_pos_strength([100.0], "RB", {}) == pytest.approx(100.0 / 2.05)


def test_values_sorted_descending_before_weighting():
    # Order shouldn't matter; the best value gets the top weight.
    a = weighted_pos_strength([50, 100], "QB", {})
    b = weighted_pos_strength([100, 50], "QB", {})
    assert a == pytest.approx(b)


def test_qb_weighting_favors_starter():
    # QB weights [1.0, 0.20, 0.05]: (100 + 100*0.2) / 1.25 with equal QBs.
    assert weighted_pos_strength([100, 100], "QB", {}) == pytest.approx(120 / 1.25)
    # A weak QB2 barely moves the number.
    val = weighted_pos_strength([100, 0], "QB", {})
    assert val == pytest.approx(100 / 1.25)


def test_superflex_treats_qb2_as_a_starter():
    vals = [100, 20]
    one_qb = weighted_pos_strength(vals, "QB", {"QB": 1})
    superflex = weighted_pos_strength(vals, "QB", {"QB": 1, "SUPER_FLEX": 1})

    assert one_qb == pytest.approx((100 + 20 * 0.20) / 1.25)
    assert superflex == pytest.approx((100 + 20 * 0.90) / 2.15)
    assert superflex < one_qb


@pytest.mark.parametrize("slot_name", ["SUPER_FLEX", "SFLEX", "OP", "QB_RB_WR_TE"])
def test_superflex_provider_aliases_use_the_same_qb_formula(slot_name):
    expected = weighted_pos_strength([100, 50], "QB", {"QB": 1, "SUPER_FLEX": 1})
    actual = weighted_pos_strength([100, 50], "QB", {"QB": 1, slot_name: 1})
    assert actual == pytest.approx(expected)


def test_two_qb_lineup_treats_both_qbs_as_starters():
    assert weighted_pos_strength([100, 50], "QB", {"QB": 2}) == pytest.approx(
        (100 + 50 * 0.90) / 2.15
    )


def test_two_elite_beat_many_mediocre_rb():
    elite = weighted_pos_strength([100, 100], "RB", {})
    depth = weighted_pos_strength([40, 40, 40, 40, 40], "RB", {"FLEX": 2})
    assert elite > depth


def test_flex_slots_add_depth_credit_for_rb():
    vals = [100, 90, 80, 70, 60]
    no_flex = weighted_pos_strength(vals, "RB", {})
    one_flex = weighted_pos_strength(vals, "RB", {"FLEX": 1})
    two_flex = weighted_pos_strength(vals, "RB", {"FLEX": 2})
    # More flex slots use more of the roster's depth -> distinct results.
    assert no_flex != one_flex != two_flex


def test_flex_provider_alias_adds_the_same_depth_credit():
    vals = [100, 90, 80, 70]
    canonical = weighted_pos_strength(vals, "WR", {"FLEX": 1})
    provider_alias = weighted_pos_strength(vals, "WR", {"RB_WR_FLEX": 1})
    assert provider_alias == pytest.approx(canonical)


def test_wrrb_flex_alias_adds_the_same_depth_credit():
    vals = [100, 90, 80, 70]
    canonical = weighted_pos_strength(vals, "RB", {"FLEX": 1})
    assert weighted_pos_strength(vals, "RB", {"WRRB_FLEX": 1}) == pytest.approx(canonical)


def test_wrrb_flex_does_not_give_te_flex_depth():
    vals = [100, 80, 60]
    no_flex = weighted_pos_strength(vals, "TE", {})
    wrrb = weighted_pos_strength(vals, "TE", {"WRRB_FLEX": 1})
    rec = weighted_pos_strength(vals, "TE", {"REC_FLEX": 1})
    full = weighted_pos_strength(vals, "TE", {"FLEX": 1})
    assert wrrb == pytest.approx(no_flex)
    assert rec == pytest.approx(full)
    assert rec != no_flex


def test_unknown_position_uses_top_player_only():
    # Fallback weights [1.0]: only the best value counts.
    assert weighted_pos_strength([80, 200, 50], "K", {}) == pytest.approx(200.0)


def test_none_values_coerced_to_zero():
    # A None in the list must not raise.
    val = weighted_pos_strength([100, None], "QB", {})
    assert val == pytest.approx(100 / 1.25)


def test_te_weighting_uses_flex_variant():
    vals = [100, 100, 100]
    no_flex = weighted_pos_strength(vals, "TE", {})          # [1.0, 0.15, 0.05]
    with_flex = weighted_pos_strength(vals, "TE", {"FLEX": 1})  # [1.0, 0.20, 0.08, 0.05]
    # No-flex fills every weight slot with a 100, so it stays at 100. The
    # flex set has a fourth (tail) slot this room leaves empty, and empty
    # slots score 0 under the fixed denominator, so it lands just under.
    assert no_flex == pytest.approx(100.0)
    assert with_flex == pytest.approx((100 + 20 + 8) / 1.33)


def test_strength_percentile_empty_or_solo_is_median():
    assert strength_percentile(100, []) == pytest.approx(50.0)
    assert strength_percentile(100, [100]) == pytest.approx(50.0)


def test_strength_percentile_best_and_worst_in_twelve():
    field = list(range(12))  # 0 worst … 11 best
    assert strength_percentile(11, field) == pytest.approx(100.0)
    assert strength_percentile(0, field) == pytest.approx(0.0)
    # 6 of 11 others are worse -> ~54.5th, not a signed-percent hole.
    assert strength_percentile(6, field) == pytest.approx(100.0 * 6 / 11)


def test_strength_percentile_splits_ties():
    field = [100, 100, 50, 40]
    # Two tied for first: 2 strictly worse + half of 1 tied other, over 3 others.
    assert strength_percentile(100, field) == pytest.approx(100.0 * 2.5 / 3)


def test_average_league_percentiles_blends_without_ratio_drag():
    # Elite in one league and last in another is a typical 50th, not negative.
    blended = average_league_percentiles([
        {"QB": 100, "RB": 90, "WR": 80, "TE": 70},
        {"QB": 0, "RB": 10, "WR": 20, "TE": 30},
    ])
    assert blended["QB"] == pytest.approx(50.0)
    assert blended["RB"] == pytest.approx(50.0)
    assert blended["WR"] == pytest.approx(50.0)
    assert blended["TE"] == pytest.approx(50.0)


def test_average_league_percentiles_skips_missing_positions():
    blended = average_league_percentiles([
        {"QB": 80, "RB": 40},
        {"QB": 60, "TE": 20},
    ])
    assert blended["QB"] == pytest.approx(70.0)
    assert blended["RB"] == pytest.approx(40.0)
    assert blended["TE"] == pytest.approx(20.0)
    assert blended["WR"] == pytest.approx(50.0)


def test_average_league_percentiles_empty_is_median():
    assert average_league_percentiles([]) == {
        "QB": 50.0, "RB": 50.0, "WR": 50.0, "TE": 50.0,
    }


def test_strength_profile_separates_starters_depth_and_fragility():
    profile = positional_strength_profile([100, 80, 30, 20], "WR", {"WR": 2})
    assert profile["starter"] == pytest.approx(90)
    assert profile["depth"] == pytest.approx(25)
    assert 0 < profile["fragility"] < 1
    assert profile["confidence"]["label"] == "High"


def test_strength_profile_penalizes_missing_required_starter():
    thin = positional_strength_profile([100], "RB", {"RB": 2})
    full = positional_strength_profile([100, 80], "RB", {"RB": 2})
    assert thin["starter"] < full["starter"]
    assert thin["fragility"] > full["fragility"]


def test_component_weight_fitter_learns_outcome_signal():
    samples = [
        {"starter": 1, "depth": 0, "resilience": 0, "outcome": 1},
        {"starter": .8, "depth": 1, "resilience": 1, "outcome": .8},
        {"starter": 0, "depth": 1, "resilience": 1, "outcome": 0},
    ]
    fitted = fit_roster_component_weights(samples, step=.1)
    assert fitted["starter"] > fitted["depth"]
    assert fitted["starter"] > fitted["resilience"]
    assert sum(fitted[k] for k in ("starter", "depth", "resilience")) == pytest.approx(1)


def test_roster_pos_value_lists_skips_zero_and_missing():
    rosters = [
        {"roster_id": 1, "players": ["a", "b", "c", "gone"]},
        {"roster_id": 2, "players": ["d"]},
    ]
    values = {
        "a": {"position": "WR", "value": 100},
        "b": {"position": "WR", "value": 0},
        "c": {"pos": "RB", "value": 50},
    }
    buckets = roster_pos_value_lists(rosters, values, rid_cast=str)
    assert buckets["1"]["WR"] == [100]
    assert buckets["1"]["RB"] == [50]
    assert buckets["2"]["WR"] == []


def test_rank_rosters_by_position_orders_by_weighted_strength():
    # In a league starting 3 WRs (WR/WR/FLEX), the deeper room wins under
    # the fixed denominator: its weighted starters (95/88/70) out-sum the
    # top-heavy room's (110/90/5). Before the dilution fix the top-heavy
    # room won only because its shorter player list shrank its denominator.
    slots = {"WR": 2, "FLEX": 1}
    team_pos = {
        "elite": {"QB": [], "RB": [], "WR": [110, 90, 5], "TE": []},
        "depth": {"QB": [], "RB": [], "WR": [95, 88, 70, 50, 40], "TE": []},
    }
    strengths, ranks = rank_rosters_by_position(team_pos, slots)
    assert strengths["depth"]["WR"] > strengths["elite"]["WR"]
    assert ranks["WR"]["depth"] == 1
    assert ranks["WR"]["elite"] == 2


def test_adding_a_player_never_lowers_strength():
    # The dilution fix, stated as a property: for every position, tacking
    # any positive-value player onto a room must not reduce its strength
    # (dropping Purdy must not raise a QB room, and so on).
    cases = [
        ("QB", {"QB": 1}, [600.0]),
        ("QB", {"QB": 1}, [600.0, 300.0]),
        ("QB", {"QB": 1, "SUPER_FLEX": 1}, [600.0]),
        ("RB", {"RB": 2, "FLEX": 2}, [500.0, 400.0]),
        ("RB", {"RB": 2, "FLEX": 2}, [500.0, 400.0, 200.0, 100.0, 50.0]),
        ("WR", {"WR": 2, "FLEX": 1}, [500.0, 400.0, 300.0]),
        ("TE", {"TE": 1}, [400.0]),
        ("TE", {"TE": 1, "FLEX": 1}, [400.0, 100.0]),
    ]
    for pos, slots, base in cases:
        before = weighted_pos_strength(base, pos, slots)
        for extra in (1.0, 40.0, 250.0):
            after = weighted_pos_strength(base + [extra], pos, slots)
            assert after >= before, (pos, base, extra, before, after)


def test_thin_room_gets_no_undiluted_free_pass():
    # A lone 650 QB no longer scores 650: the empty QB2/tail slots count
    # as 0, so a 600 QB1 with a real 300 QB2 behind him outranks him.
    solo = weighted_pos_strength([650.0], "QB", {"QB": 1})
    room = weighted_pos_strength([600.0, 300.0], "QB", {"QB": 1})
    assert room > solo


def test_tail_credit_counts_one_player_past_the_old_cutoff():
    # QB3 in 1QB used to count exactly zero; at the 0.05 tail weight a
    # 25-value QB3 adds 25 * 0.05 / 1.25 = 1.0.
    without = weighted_pos_strength([100.0, 50.0], "QB", {"QB": 1})
    with_tail = weighted_pos_strength([100.0, 50.0, 25.0], "QB", {"QB": 1})
    assert with_tail == pytest.approx(without + 1.0)
    # RB6 in a 2-flex league likewise adds a little instead of nothing.
    five = weighted_pos_strength([100, 90, 80, 70, 60], "RB", {"FLEX": 2})
    six = weighted_pos_strength([100, 90, 80, 70, 60, 50], "RB", {"FLEX": 2})
    assert six > five


def test_rank_rosters_by_position_tie_breaks_by_roster_id():
    team_pos = {
        "b": {"QB": [10], "RB": [], "WR": [], "TE": []},
        "a": {"QB": [10], "RB": [], "WR": [], "TE": []},
    }
    _strengths, ranks = rank_rosters_by_position(team_pos, {"QB": 1})
    assert ranks["QB"]["a"] == 1
    assert ranks["QB"]["b"] == 2
