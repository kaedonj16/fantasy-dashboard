"""Unit tests for utils.waiver_score.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
import pytest

from utils.waiver_score import (
    USAGE_SPIKE_MIN,
    WAIVER_PRIME_MAX,
    build_depth_index,
    depth_chart_vacancy_score,
    injured_ahead,
    injured_ahead_for_player,
    usage_ratio,
    value_component,
    waiver_pickup_score,
    waiver_signal,
)


def _cand(value=500, age=None, position="RB", rank_change_7d=0, player_id="p1",
          usage_stat=None, usage_delta=None):
    return {
        "value": value, "age": age, "position": position,
        "rank_change_7d": rank_change_7d, "player_id": player_id,
        "usage_stat": usage_stat, "usage_delta": usage_delta,
    }


# ---- value_component ------------------------------------------------------

def test_value_component_shape():
    assert value_component(0) == 0.0
    assert value_component(500) == pytest.approx(60.0)
    assert value_component(1500) == pytest.approx(90.0)


def test_value_component_monotonic_and_concave():
    a, b, c = value_component(200), value_component(400), value_component(600)
    assert a < b < c
    # Concave: equal value steps yield shrinking gains.
    assert (b - a) > (c - b)


def test_value_component_bad_input_zero():
    assert value_component(None) == 0.0
    assert value_component("x") == 0.0


# ---- usage_ratio ----------------------------------------------------------

def test_usage_ratio_thresholds():
    assert usage_ratio("snap_pct", 8) == pytest.approx(1.0)
    assert usage_ratio("touches", 6) == pytest.approx(2.0)
    assert usage_ratio("targets", 1) == pytest.approx(0.5)


def test_usage_ratio_unknown_stat_uses_default():
    assert usage_ratio("mystery", 3) == pytest.approx(1.0)


def test_usage_ratio_missing_data_zero():
    assert usage_ratio(None, 5) == 0.0
    assert usage_ratio("snap_pct", None) == 0.0
    assert usage_ratio("snap_pct", "nan") == 0.0


# ---- depth-chart injury vacancy -------------------------------------------

def test_injured_ahead_only_counts_players_above():
    teammates = [
        {"depth_order": 1, "status": "IR"},        # starter, ahead, injured
        {"depth_order": 2, "status": "OUT"},        # ahead, injured
        {"depth_order": 4, "status": "OUT"},        # behind me (order 3) -> ignored
        {"depth_order": 1, "status": "ACTIVE"},     # ahead but healthy -> ignored
    ]
    assert injured_ahead(3, teammates) == ["IR", "OUT"]


def test_injured_ahead_questionable_excluded():
    assert injured_ahead(2, [{"depth_order": 1, "status": "QUESTIONABLE"}]) == []


def test_injured_ahead_missing_order_treated_as_deep():
    # Candidate with no depth order still benefits from an injured starter.
    assert injured_ahead(None, [{"depth_order": 1, "status": "OUT"}]) == ["OUT"]


def test_vacancy_score_scales_with_severity_and_count():
    assert depth_chart_vacancy_score([]) == 0.0
    assert depth_chart_vacancy_score(["IR"]) == pytest.approx(40.0)     # 1.0 * 40
    assert depth_chart_vacancy_score(["OUT"]) == pytest.approx(34.0)    # 0.85 * 40
    # Two injured ahead: top dominates, second adds a little.
    assert depth_chart_vacancy_score(["IR", "OUT"]) == pytest.approx(40.0 + 0.85 * 8)
    # Capped at 55.
    assert depth_chart_vacancy_score(["IR", "IR", "IR", "IR"]) == pytest.approx(55.0)


def test_build_depth_index_and_lookup():
    full = {
        "starter": {"team": "KC", "position": "RB", "depth_chart_order": 1, "injury_status": "IR"},
        "backup":  {"team": "KC", "position": "RB", "depth_chart_order": 2, "injury_status": ""},
        "other":   {"team": "KC", "position": "WR", "depth_chart_order": 1, "injury_status": "OUT"},
    }
    idx = build_depth_index(full)
    assert set(idx) == {("KC", "RB"), ("KC", "WR")}
    # The backup RB has an injured starter (IR) ahead.
    assert injured_ahead_for_player("backup", full, idx) == ["IR"]
    # The starter has nobody ahead.
    assert injured_ahead_for_player("starter", full, idx) == []


def test_injury_vacancy_boosts_score_and_badge():
    healthy = _cand(value=250, position="RB")
    vacated = _cand(value=250, position="RB", player_id="b")
    vacated["injured_ahead"] = ["IR"]
    assert waiver_pickup_score(vacated, {}) > waiver_pickup_score(healthy, {})
    assert waiver_signal(vacated, {})[1] == "Next Man Up"


def test_next_man_up_outranks_usage_spike_badge():
    c = _cand(player_id="x", usage_stat="snap_pct", usage_delta=8)
    c["injured_ahead"] = ["OUT"]
    assert waiver_signal(c, {})[1] == "Next Man Up"


def test_doubtful_ahead_scores_but_no_next_man_up_badge():
    # DOUBTFUL contributes points but is too weak to claim the "Next Man Up" badge.
    c = _cand(value=250, position="RB", rank_change_7d=5)
    c["injured_ahead"] = ["DOUBTFUL"]
    assert depth_chart_vacancy_score(c["injured_ahead"]) > 0
    assert waiver_signal(c, {})[1] != "Next Man Up"


# ---- waiver_pickup_score --------------------------------------------------

def test_value_only_base():
    assert waiver_pickup_score(_cand(value=500), {}) == pytest.approx(60.0)


def test_usage_spike_adds_and_caps():
    assert waiver_pickup_score(
        _cand(value=500, usage_stat="snap_pct", usage_delta=8), {}) == pytest.approx(90.0)
    # 2x threshold -> 60 raw, capped at +50.
    assert waiver_pickup_score(
        _cand(value=500, usage_stat="snap_pct", usage_delta=16), {}) == pytest.approx(110.0)


def test_positive_trend_and_cap():
    assert waiver_pickup_score(_cand(value=500, rank_change_7d=10), {}) == pytest.approx(95.0)
    assert waiver_pickup_score(_cand(value=500, rank_change_7d=20), {}) == pytest.approx(105.0)


def test_negative_trend_penalized_and_floored():
    assert waiver_pickup_score(_cand(value=500, rank_change_7d=-4), {}) == pytest.approx(54.0)
    assert waiver_pickup_score(_cand(value=500, rank_change_7d=-30), {}) == pytest.approx(45.0)


def test_breakout_adds_and_caps():
    assert waiver_pickup_score(_cand(value=500, player_id="x"), {"x": 60}) == pytest.approx(90.0)
    assert waiver_pickup_score(_cand(value=500, player_id="x"), {"x": 200}) == pytest.approx(105.0)


def test_age_curve():
    # RB prime 26. Young (22) rewarded, at-prime modest, past-prime decays/floors.
    assert waiver_pickup_score(_cand(value=500, position="RB", age=22), {}) == pytest.approx(90.0)
    assert waiver_pickup_score(_cand(value=500, position="RB", age=26), {}) == pytest.approx(82.0)
    assert waiver_pickup_score(_cand(value=500, position="RB", age=30), {}) == pytest.approx(54.0)
    assert waiver_pickup_score(_cand(value=500, position="RB", age=40), {}) == pytest.approx(38.0)


def test_age_youth_bonus_capped():
    # Very young still caps the age term at +36 (value 500 -> 60).
    assert waiver_pickup_score(_cand(value=500, position="RB", age=18), {}) == pytest.approx(96.0)


def test_emerging_player_outranks_static_veteran():
    veteran = _cand(value=1500, position="WR", age=28)                    # high static value
    emerging = _cand(value=250, position="RB", age=22, rank_change_7d=6,
                     player_id="e", usage_stat="snap_pct", usage_delta=12)  # opportunity stack
    assert waiver_pickup_score(emerging, {"e": 60}) > waiver_pickup_score(veteran, {})


def test_missing_fields_do_not_raise():
    assert waiver_pickup_score({}, {}) == pytest.approx(0.0)


# ---- waiver_signal --------------------------------------------------------

def test_signal_usage_spike_takes_precedence():
    c = _cand(player_id="x", usage_stat="snap_pct", usage_delta=8)
    # Even with a breakout score, usage spike wins.
    assert waiver_signal(c, {"x": 99})[1] == "Usage Spike"


def test_signal_breakout():
    assert waiver_signal(_cand(player_id="x"), {"x": 55})[1] == "Breakout"


def test_signal_rising_bands():
    assert waiver_signal(_cand(rank_change_7d=8), {})[1] == "Rising Fast"
    assert waiver_signal(_cand(rank_change_7d=4), {})[1] == "Trending Up"


def test_signal_value_play():
    # RB prime 26, age 22 (< prime-2), value >= 300.
    assert waiver_signal(_cand(position="RB", age=22, value=400), {})[1] == "Value Play"


def test_signal_sell_window():
    # RB prime 26, age 29 (> prime+2).
    assert waiver_signal(_cand(position="RB", age=29, value=400), {})[1] == "Sell Window"


def test_signal_default_available():
    assert waiver_signal(_cand(position="RB", age=26, value=100), {})[1] == "Available"


def test_tables_have_expected_positions():
    assert set(WAIVER_PRIME_MAX) == {"QB", "RB", "WR", "TE"}
    assert set(USAGE_SPIKE_MIN) == {"snap_pct", "touches", "targets"}
