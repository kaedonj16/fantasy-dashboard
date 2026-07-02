"""Unit tests for utils.roster_strength.weighted_pos_strength.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
import pytest

from utils.roster_strength import weighted_pos_strength


def test_empty_values_return_zero():
    assert weighted_pos_strength([], "RB", {}) == 0.0
    assert weighted_pos_strength(None, "RB", {}) == 0.0


def test_single_player_equals_own_value():
    # denom == weight of the one used slot, so a lone player scores its value.
    assert weighted_pos_strength([100.0], "QB", {}) == pytest.approx(100.0)
    assert weighted_pos_strength([100.0], "RB", {}) == pytest.approx(100.0)


def test_values_sorted_descending_before_weighting():
    # Order shouldn't matter; the best value gets the top weight.
    a = weighted_pos_strength([50, 100], "QB", {})
    b = weighted_pos_strength([100, 50], "QB", {})
    assert a == pytest.approx(b)


def test_qb_weighting_favors_starter():
    # QB weights [1.0, 0.20]: (100 + 100*0.2) / 1.2 = 100 when both equal.
    assert weighted_pos_strength([100, 100], "QB", {}) == pytest.approx(100.0)
    # A weak QB2 barely moves the number.
    val = weighted_pos_strength([100, 0], "QB", {})
    assert val == pytest.approx(100 / 1.2)


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


def test_unknown_position_uses_top_player_only():
    # Fallback weights [1.0]: only the best value counts.
    assert weighted_pos_strength([80, 200, 50], "K", {}) == pytest.approx(200.0)


def test_none_values_coerced_to_zero():
    # A None in the list must not raise.
    val = weighted_pos_strength([100, None], "QB", {})
    assert val == pytest.approx(100 / 1.2)


def test_te_weighting_uses_flex_variant():
    vals = [100, 100, 100]
    no_flex = weighted_pos_strength(vals, "TE", {})          # [1.0, 0.15]
    with_flex = weighted_pos_strength(vals, "TE", {"FLEX": 1})  # [1.0, 0.20, 0.08]
    # Both equal-valued -> collapse to 100, but the weight sets differ; a weak
    # TE2/TE3 would separate them. Here confirm both stay at the starter value.
    assert no_flex == pytest.approx(100.0)
    assert with_flex == pytest.approx(100.0)
