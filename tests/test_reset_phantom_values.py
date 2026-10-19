"""Unit tests for the phantom-value cohort rule + corrections (pure parts).

The cohort/reset mechanics were built for the Phil Mafah case (Oct 2026):
board served 329.6 for a player FantasyCalc normalized to ~8.6 and
DynastyProcess to ~0.2, with 0.0 PPG.
"""
import os

import pytest

pytest.importorskip("pandas")  # data_building package import chain

from data_building.reset_phantom_values import (
    corrected_values,
    is_phantom,
    minmax_norm,
)


def test_minmax_norm_endpoints():
    assert minmax_norm(1.0, 1.0, 10912.0) == 0.0
    assert minmax_norm(10912.0, 1.0, 10912.0) == pytest.approx(999.9)
    assert minmax_norm(0.0, 1.0, 10912.0) == 0.0


def test_mafah_is_phantom_and_corrects_to_market_blend():
    # FC raw 95 on a 1..10912 scale; FC-SF raw 87 on 1..10328; DP 3 / 4 raw
    # on 1..10232 / 1..10208 scales (the site's actual files, Sep 2026).
    fc_n = minmax_norm(95, 1, 10912)
    dp_n = minmax_norm(3, 1, 10232)
    fcs_n = minmax_norm(87, 1, 10328)
    dp2_n = minmax_norm(4, 1, 10208)
    assert is_phantom(329.6, max(fc_n, dp_n))
    assert is_phantom(398.0, fcs_n)
    v1, vsf = corrected_values(fc_n, dp_n, fcs_n, dp2_n, "RB")
    assert v1 == pytest.approx(5.8, abs=0.1)
    assert vsf < 10


def test_real_star_is_not_phantom():
    # Consensus 950, served 1000: within the guardrail band, not a phantom.
    assert not is_phantom(1000.0, 950.0)
    # Above-market call both sources support: 600 vs consensus 300 is 2x.
    assert not is_phantom(600.0, 300.0)


def test_small_values_are_never_phantoms():
    assert not is_phantom(50.0, 0.0)
    assert not is_phantom(99.9, 0.0)


def test_no_coverage_at_stake_is_phantom():
    assert is_phantom(100.0, 0.0)
    assert is_phantom(425.0, 0.0)  # the Kareem Hunt shape


def test_corrected_values_zero_when_no_market():
    assert corrected_values(0.0, 0.0, 0.0, 0.0, "RB") == (0.0, 0.0)


def test_corrected_sf_floor_for_nonqb():
    # SF blend near zero but a real 1QB price: non-QB SF floors at 0.85x.
    v1, vsf = corrected_values(400.0, 400.0, 0.0, 0.0, "WR")
    assert v1 == pytest.approx(400.0)
    assert vsf == pytest.approx(0.85 * 400.0)


def test_fc_floor_matches_dp_floor_intent():
    # Source pin: the presence filter's FC membership must clear a floor
    # like DP's, or a bottom-tail FC listing shields a phantom from zeroing.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "app.py"), encoding="utf-8") as f:
        src = f.read()
    assert "_FC_MIN_VALUE = 750.0" in src
    assert "if _fv >= _FC_MIN_VALUE:" in src
    # The rewrite must re-apply the guardrail after the day-over-day clamp
    # and must not adopt WLS outright via the old ternary.
    with open(os.path.join(root, "data_building", "value_model_training.py"),
              encoding="utf-8") as f:
        vmt = f.read()
    assert "Final market-sanity pass" in vmt
    assert "blend_wls_value(" in vmt
    assert "if final_value > 0 else _wls1" not in vmt
    assert "if sf_value > 0 else _wlssf" not in vmt
