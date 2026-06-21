"""Unit tests for the over-market value guardrail (pure, no DB/deps)."""
from data_building.value_guardrails import overmarket_capped


def test_chubb_case_pulled_down():
    # Model rates him RB28 (266) but FC absent (0) and DP ~bottom (~0.5).
    out = overmarket_capped(266.2, fc_val=0.0, dp_val=0.5)
    assert out < 10  # pulled down toward the dead market
    assert out == max(0.5 * 2.5, 0.5)


def test_absent_from_one_source_other_rates_high_is_left_alone():
    # FC absent but DP rates him decently → consensus high → no pull-down,
    # preserving the model's above-market call.
    assert overmarket_capped(800.0, fc_val=0.0, dp_val=400.0) == 800.0


def test_one_source_high_keeps_value_even_if_other_low():
    # max() consensus: FC high, DP low → not both-low → untouched.
    assert overmarket_capped(900.0, fc_val=850.0, dp_val=0.0) == 900.0


def test_no_external_coverage_is_noop():
    # Pure prospect / data gap: neither source covers → never touched.
    assert overmarket_capped(500.0, fc_val=0.0, dp_val=0.0) == 500.0


def test_within_band_not_touched():
    # 600 vs consensus 300 = exactly 2x (< 2.5x trigger) → left alone.
    assert overmarket_capped(600.0, fc_val=300.0, dp_val=200.0) == 600.0


def test_extreme_over_market_capped_to_trigger_multiple():
    # 800 vs consensus 300 = 2.67x (>= 2.5x) and gap 500 (>= 100) → cap to 750.
    assert overmarket_capped(800.0, fc_val=300.0, dp_val=100.0) == 750.0


def test_min_gap_skips_low_stakes():
    # Big ratio but tiny absolute values → don't fuss (gap < 100).
    assert overmarket_capped(20.0, fc_val=2.0, dp_val=0.0) == 20.0


def test_zero_or_negative_value_noop():
    assert overmarket_capped(0.0, fc_val=100.0, dp_val=100.0) == 0.0
