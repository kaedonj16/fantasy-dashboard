"""Unit tests for the pure Expected Fantasy Points math.

These exercise the buckets, table lookups, per-opportunity accumulation, and the
season/weekly column derivation without any pandas / nfl_data_py dependency.
"""

import math

import pytest

from data_building.external_data import expected_points as xp


# --- buckets ------------------------------------------------------------------

def test_yardline_bucket_monotonic_and_edges():
    # goal-to-go is the closest bucket; backed up is the deepest.
    assert xp.yardline_bucket(1) == 0
    assert xp.yardline_bucket(2.9) == 0
    assert xp.yardline_bucket(3) == 1
    assert xp.yardline_bucket(50) == 5
    assert xp.yardline_bucket(99) == len(xp._YARDLINE_EDGES)
    # None / junk never lands in a scoring bucket.
    assert xp.yardline_bucket(None) == len(xp._YARDLINE_EDGES)
    assert xp.yardline_bucket("nope") == len(xp._YARDLINE_EDGES)


def test_airyards_bucket_and_deep_flag():
    assert xp.airyards_bucket(-3) == 0
    assert xp.airyards_bucket(None) == 1  # unknown treated as short, not behind LOS
    assert xp.airyards_bucket(25) == 5          # 20 <= 25 < 30
    assert xp.airyards_bucket(35) == len(xp._AIRYARDS_EDGES)  # >= 30, top bucket
    assert xp.is_deep(20) is True
    assert xp.is_deep(19.9) is False
    assert xp.is_deep(None) is False


# --- table fallbacks ----------------------------------------------------------

def test_tables_fall_back_to_global_then_zero():
    t = xp.ExpectedPointsTables(
        rec_td_prob={(0, False): 0.30},
        rec_td_prob_global=0.05,
    )
    # populated bucket wins
    assert t.rec_td_prob_for(1, 0) == 0.30
    # missing bucket -> global mean
    assert t.rec_td_prob_for(50, 0) == 0.05
    # empty table with no global -> 0.0 (never a wild rate)
    empty = xp.ExpectedPointsTables()
    assert empty.rush_td_prob_for(1) == 0.0


# --- per-opportunity accumulation --------------------------------------------

def _tables():
    return xp.ExpectedPointsTables(
        rec_td_prob={(1, False): 0.05},
        rush_td_prob={0: 0.5, 6: 0.01},
        rush_yds_mean={0: 0.8, 6: 4.5},
        pass_td_prob={1: 0.06},
        comp_prob={2: 0.65},
        yac_mean={2: 5.0},
        int_rate=0.023,
        comp_prob_global=0.64,
        yac_mean_global=4.8,
    )


def test_add_target_uses_cp_and_xyac_when_present():
    t = _tables()
    comp = xp.new_components()
    # 10-yard target, cp/xyac supplied directly
    xp.add_target(comp, t, air_yards=10.0, yardline_100=35, cp=0.7, xyac=4.0)
    assert comp["x_receptions"] == pytest.approx(0.7)
    # expected yards = cp * (air + xyac) = 0.7 * 14
    assert comp["x_rec_yards"] == pytest.approx(0.7 * 14.0)
    # TD equity from (yardline bucket for 35 -> 4, deep False); not in table -> global 0
    assert comp["x_rec_td"] == pytest.approx(0.0)


def test_add_target_falls_back_to_empirical_tables():
    t = _tables()
    comp = xp.new_components()
    # air_yards 7 -> bucket 2 -> comp 0.65, yac 5.0; no cp/xyac
    xp.add_target(comp, t, air_yards=7.0, yardline_100=3, cp=None, xyac=None)
    assert comp["x_receptions"] == pytest.approx(0.65)
    assert comp["x_rec_yards"] == pytest.approx(0.65 * (7.0 + 5.0))
    # yardline 3 -> bucket 1, not deep -> 0.05
    assert comp["x_rec_td"] == pytest.approx(0.05)


def test_add_carry_goal_line_vs_midfield():
    t = _tables()
    goal = xp.new_components()
    xp.add_carry(goal, t, yardline_100=1)   # bucket 0
    mid = xp.new_components()
    xp.add_carry(mid, t, yardline_100=65)   # bucket 6
    assert goal["x_rush_td"] == pytest.approx(0.5)
    assert goal["x_rush_yards"] == pytest.approx(0.8)
    assert mid["x_rush_td"] == pytest.approx(0.01)
    assert mid["x_rush_yards"] == pytest.approx(4.5)
    # A goal-line carry is worth far more expected points than a midfield one.
    assert xp.expected_points(goal, "ppr") > xp.expected_points(mid, "ppr")


def test_add_pass_attempt_accumulates_td_and_int():
    t = _tables()
    comp = xp.new_components()
    xp.add_pass_attempt(comp, t, air_yards=7.0, yardline_100=3, cp=0.6, xyac=4.0)
    assert comp["x_pass_att"] == 1.0
    assert comp["x_pass_yards"] == pytest.approx(0.6 * 11.0)
    assert comp["x_pass_td"] == pytest.approx(0.06)
    assert comp["x_int"] == pytest.approx(0.023)


# --- format scoring -----------------------------------------------------------

def test_reception_bonus_orders_ppr_half_standard():
    comp = xp.new_components()
    comp["x_receptions"] = 6.0
    comp["x_rec_yards"] = 60.0
    ppr = xp.expected_points(comp, "ppr")
    half = xp.expected_points(comp, "half")
    std = xp.expected_points(comp, "standard")
    # 6 receptions -> +6 / +3 / +0 on top of the same 6.0 yardage points
    assert ppr == pytest.approx(std + 6.0)
    assert half == pytest.approx(std + 3.0)
    assert std == pytest.approx(6.0)


def test_actual_and_expected_share_scoring_basis():
    comp = xp.new_components()
    # identical actual and expected component values -> zero delta in every format
    for base in ("receptions", "rec_yards", "rec_td", "rush_yards", "rush_td",
                 "pass_yards", "pass_td", "int"):
        comp[f"x_{base}"] = 3.0
        comp[f"a_{base}"] = 3.0
    for fmt in xp.FORMATS:
        assert xp.actual_points(comp, fmt) == pytest.approx(xp.expected_points(comp, fmt))


# --- season / weekly column derivation ---------------------------------------

def test_season_columns_are_signed_totals():
    comp = xp.new_components()
    comp["x_rec_yards"] = 20.0   # 2.0 expected pts (standard)
    comp["a_rec_yards"] = 40.0   # 4.0 actual pts
    out = xp.season_columns_from_components(comp)
    # season value is a TOTAL (not per game), so 2.0 expected, +2.0 over expected
    assert out["expected_standard"] == pytest.approx(2.0)
    assert out["standard_over_expected"] == pytest.approx(2.0)
    assert set(out) == set(xp.XFP_COLS)


def test_season_and_weekly_use_the_same_total_columns():
    comp = xp.new_components()
    comp["x_rec_yards"] = 30.0
    comp["a_rec_yards"] = 30.0
    # Same accumulator -> identical columns from both helpers (both are totals).
    assert (xp.season_columns_from_components(comp)
            == xp.weekly_columns_from_components(comp))


def test_weekly_columns_are_totals():
    comp = xp.new_components()
    comp["x_rec_yards"] = 50.0    # 5.0 expected pts
    comp["x_receptions"] = 4.0
    comp["a_rec_yards"] = 30.0    # 3.0 actual pts
    comp["a_receptions"] = 3.0
    out = xp.weekly_columns_from_components(comp)
    # standard: no reception bonus
    assert out["expected_standard"] == pytest.approx(5.0)
    assert out["standard_over_expected"] == pytest.approx(3.0 - 5.0)
    # ppr adds reception bonus to both sides
    assert out["expected_ppr"] == pytest.approx(5.0 + 4.0)
    assert out["ppr_over_expected"] == pytest.approx((3.0 + 3.0) - (5.0 + 4.0))
    assert set(out) == set(xp.XFP_COLS)


def test_column_name_list_lines_up_across_formats():
    # XFP_COLS must stay in sync with the format suffixes (season == weekly).
    for fmt in xp.FORMATS:
        suf = xp._FMT_SUFFIX[fmt]
        assert f"expected_{suf}" in xp.XFP_COLS
        assert f"{suf}_over_expected" in xp.XFP_COLS
    assert len(xp.XFP_COLS) == 2 * len(xp.FORMATS)
