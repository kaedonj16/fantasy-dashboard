"""Family A tests: expected TDs (xTD) from the expected-points model.

expected_tds = x_rec_td (as a target) + x_rush_td (as a rusher) +
x_pass_rec_td (receiving-TD equity of throws, as a passer).
td_over_expected = (a_rec_td + a_rush_td + a_pass_td) - expected_tds.

The passer accumulates the play's rec_td_prob (the receiving table), NOT
the separate yardline-only x_pass_td table behind expected passing points,
so a pass play's TD equity is counted exactly once per player.
"""

import pytest

pytest.importorskip("pandas")

import pandas as pd

from data_building.external_data import expected_points as xp


def _tables():
    return xp.ExpectedPointsTables(
        # yardline 3 -> bucket 1, air 7 -> not deep
        rec_td_prob={(1, False): 0.05},
        rec_td_prob_global=0.05,
        rush_td_prob={0: 0.5},
        rush_td_prob_global=0.02,
        rush_yds_mean={0: 0.8},
        pass_td_prob={1: 0.40},
        comp_prob={2: 0.65},
        yac_mean={2: 5.0},
        int_rate=0.02,
        comp_prob_global=0.64,
        yac_mean_global=4.8,
    )


def _play(**kw):
    base = {
        "play_type": "pass", "week": 1, "yardline_100": 3.0, "air_yards": 7.0,
        "cp": None, "xyac_mean_yardage": None,
        "pass_attempt": 1, "rush_attempt": 0, "complete_pass": 1,
        "interception": 0, "yards_gained": 3, "passing_yards": 3,
        "rushing_yards": 0, "receiving_yards": 3,
        "pass_touchdown": 1, "rush_touchdown": 0,
        "passer_player_id": "QB1", "rusher_player_id": None,
        "receiver_player_id": "WR1",
    }
    base.update(kw)
    return base


def test_total_columns_expected_tds_components():
    comp = xp.new_components()
    comp["x_rec_td"] = 2.0
    comp["x_rush_td"] = 1.5
    comp["x_pass_rec_td"] = 4.0
    comp["x_pass_td"] = 99.0  # must NOT leak into expected_tds
    comp["a_rec_td"] = 3.0
    comp["a_rush_td"] = 1.0
    comp["a_pass_td"] = 5.0
    out = xp.season_columns_from_components(comp)
    assert out["expected_tds"] == pytest.approx(7.5)
    assert out["td_over_expected"] == pytest.approx(9.0 - 7.5)


def test_passer_gets_rec_td_equity_not_pass_td_table():
    pbp = pd.DataFrame([_play()])
    by_player = xp._accumulate_player_weeks(pbp, _tables())
    wr = by_player["WR1"][1]
    qb = by_player["QB1"][1]
    # Receiver: the target's rec TD equity.
    assert wr["x_rec_td"] == pytest.approx(0.05)
    # Passer: the SAME play valued on the receiving table (0.05), not the
    # yardline-only passing table (0.40), and never double-counted.
    assert qb["x_pass_rec_td"] == pytest.approx(0.05)
    assert qb["x_rec_td"] == pytest.approx(0.0)
    assert qb["x_pass_td"] == pytest.approx(0.40)
    qb_cols = xp.season_columns_from_components(qb)
    assert qb_cols["expected_tds"] == pytest.approx(0.05)
    # He threw 1 TD: actual 1 - expected 0.05.
    assert qb_cols["td_over_expected"] == pytest.approx(0.95)
    wr_cols = xp.season_columns_from_components(wr)
    assert wr_cols["expected_tds"] == pytest.approx(0.05)
    assert wr_cols["td_over_expected"] == pytest.approx(0.95)


def test_rusher_expected_tds_from_carries():
    pbp = pd.DataFrame([_play(
        play_type="run", pass_attempt=0, rush_attempt=1, complete_pass=0,
        pass_touchdown=0, rush_touchdown=1, passing_yards=0,
        rushing_yards=3, receiving_yards=0, yardline_100=1.0,
        passer_player_id=None, receiver_player_id=None,
        rusher_player_id="RB1",
    )])
    by_player = xp._accumulate_player_weeks(pbp, _tables())
    rb = by_player["RB1"][1]
    assert rb["x_rush_td"] == pytest.approx(0.5)  # goal-line bucket 0
    cols = xp.season_columns_from_components(rb)
    assert cols["expected_tds"] == pytest.approx(0.5)
    assert cols["td_over_expected"] == pytest.approx(0.5)


def test_xtd_wiring_specs_and_weekly():
    from data_building import advanced_metrics as am

    for key in ("expected_tds", "xtd_per_game", "td_over_expected"):
        spec = am.LEADERBOARD_METRICS[key]
        assert spec["category"] == "Expected Pts"
        assert spec["positions"] == ["QB", "RB", "WR", "TE"]
        assert key not in am.PRO_METRICS
        assert key not in am.PREMIUM_METRICS
    assert "expected_tds" in am.WEEKLY_ADV_METRIC_COLS
    assert "td_over_expected" in am.WEEKLY_ADV_METRIC_COLS
    assert "expected_tds" in am._ADV_WEEKLY_TOTAL_METRICS
    assert "td_over_expected" in am._ADV_WEEKLY_TOTAL_METRICS
    assert "xtd_per_game" in am._ADV_WEEKLY_DERIVED_METRICS
    assert "expected_tds" in am.LEADERBOARD_METRICS["xtd_per_game"]["computed_sql"]
    assert "expected_tds" in xp.XFP_COLS
    assert "td_over_expected" in xp.XFP_COLS
