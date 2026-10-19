"""Unit tests for the over-market value guardrail (pure, no DB/deps)."""
from data_building.value_guardrails import blend_wls_value, overmarket_capped


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


def test_no_external_coverage_phantom_is_zeroed():
    # Neither source covers the player and the value is at/above the
    # min-gap stake: a phantom with no market behind it cannot stand
    # (the Phil Mafah case — board served 330, markets priced ~nothing).
    assert overmarket_capped(500.0, fc_val=0.0, dp_val=0.0) == 0.0
    assert overmarket_capped(100.0, fc_val=0.0, dp_val=0.0) == 0.0


def test_no_external_coverage_small_value_left_alone():
    # Low-stakes data gap: a small uncovered value is harmless, keep it.
    assert overmarket_capped(50.0, fc_val=0.0, dp_val=0.0) == 50.0


def test_mafah_case_capped_to_trigger_multiple():
    # Board 330 vs FC ~8.6 normalized / DP ~0.2: capped to 2.5 x consensus.
    assert overmarket_capped(329.6, fc_val=8.6, dp_val=0.2) == 21.5


def test_wls_never_adopted_outright_without_market():
    # No market price → WLS cannot create one, whatever its backing.
    assert blend_wls_value(0.0, 900.0, backing=100.0) == 0.0
    assert blend_wls_value(0.0, 900.0, backing=0.0) == 0.0


def test_wls_blend_confidence_weighting():
    # No WLS row → market unchanged; zero backing → market unchanged;
    # backing == k → 50/50 blend.
    assert blend_wls_value(100.0, None, backing=50.0) == 100.0
    assert blend_wls_value(100.0, 900.0, backing=0.0) == 100.0
    assert blend_wls_value(100.0, 300.0, backing=6.0) == 200.0


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
