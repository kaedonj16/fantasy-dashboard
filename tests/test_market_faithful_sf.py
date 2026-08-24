"""Non-QB Superflex value tracks the real SF trade market, not the WLS solve.

The SF WLS regression overshoots skill players who get packaged with QBs (their
outlier overpays drag the least-squares fit up). For non-QBs with enough SF trade
data we instead take the market-faithful value: calibrated 1QB x the player's
observed SF/1QB trade ratio. This locks that contract, incl. the fallback signal
(None) when there's no market ratio and the clamp against a pathological median.
"""
import pytest

pytest.importorskip("numpy")  # trade_value_model imports numpy at load

from data_building.trade_intel.trade_value_model import _market_faithful_sf


def test_applies_market_ratio_to_1qb():
    # Wilson-like: cal_1qb 312, real SF/1QB ratio ~1.07 -> ~334 (not the 410 blend).
    assert _market_faithful_sf(312.0, 1.07) == pytest.approx(312.0 * 1.07)


def test_none_ratio_falls_back():
    # No market signal -> None tells the caller to use the WLS blend instead.
    assert _market_faithful_sf(312.0, None) is None


def test_no_1qb_base_falls_back():
    assert _market_faithful_sf(0.0, 1.1) is None


def test_ratio_clamped_high_and_low():
    # A garbage median can't blow a value up or crush it: clamp to [0.5, 1.2].
    # This path is non-QB only, so hi is a non-QB ceiling (~top of the real
    # 0.8-1.2 range), never a QB premium.
    assert _market_faithful_sf(100.0, 9.0) == 120.0   # hi clamp (non-QB ceiling)
    assert _market_faithful_sf(100.0, 0.01) == 50.0   # lo clamp


def test_supra_range_nonqb_ratio_is_clamped_not_inflated():
    # Regression: a distorted SF/1QB ratio for an elite RB (e.g. 1.35 from thin/
    # whale SF trades) must NOT pass through — that is exactly what floated a lone
    # RB above the whole Superflex board. It is clamped to the non-QB ceiling.
    assert _market_faithful_sf(1000.0, 1.35) == pytest.approx(1000.0 * 1.2)


def test_real_range_nonqb_ratio_passes_through():
    # A normal non-QB SF/1QB ratio inside the real 0.8-1.2 band is applied as-is.
    assert _market_faithful_sf(500.0, 0.95) == pytest.approx(500.0 * 0.95)
    assert _market_faithful_sf(500.0, 1.15) == pytest.approx(500.0 * 1.15)
