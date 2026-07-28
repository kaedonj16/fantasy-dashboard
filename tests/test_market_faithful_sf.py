"""Non-QB Superflex value tracks the real SF trade market, not the WLS solve.

The SF WLS regression overshoots skill players who get packaged with QBs (their
outlier overpays drag the least-squares fit up). For non-QBs with enough SF trade
data we instead take the market-faithful value: calibrated 1QB x the player's
observed SF/1QB trade ratio. This locks that contract, incl. the fallback signal
(None) when there's no market ratio and the clamp against a pathological median.
"""
import pytest

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
    # A garbage median can't blow a value up or crush it: clamp to [0.5, 2.5].
    assert _market_faithful_sf(100.0, 9.0) == 250.0   # hi clamp
    assert _market_faithful_sf(100.0, 0.01) == 50.0   # lo clamp


def test_qb_like_premium_passes_through_within_band():
    # A legit QB-style 1.7x premium is inside the band and applied as-is.
    assert _market_faithful_sf(500.0, 1.7) == 850.0
