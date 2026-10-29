"""Thin-segment WLS robustness (2026-10-29).

At the top of the dynasty board the native WLS segment (one SF flag x one
size bucket) can be too thin to steer values: Bijan Robinson passed Jahmyr
Gibbs at #1 on a tied prior because a +0.58% vs +0.19% calibration
difference survived on ~2.9 decayed backing each, with no native trade
newer than 23 days. These tests pin the three guards in
data_building/trade_intel/trade_value_model.py (segment borrowing,
freshness shrink, top flip guard) and the DynastyProcess staleness
down-weight in data_building/value_model_training.py.
"""
from datetime import date

import pytest

pytest.importorskip("numpy")  # both value modules import numpy at load
pytest.importorskip("pandas")  # value_model_training imports pandas at load
pytest.importorskip("flask")  # utils.utils (via value_model_training) needs flask
pytest.importorskip("bs4")  # utils.utils also imports BeautifulSoup

import numpy as np
import pandas as pd

from data_building import value_model_training as vmt
from data_building.trade_intel import trade_value_model as tvm


# ── Freshness shrink ────────────────────────────────────────────────────────

def test_freshness_factor_curve():
    assert tvm._freshness_factor(0.0) == pytest.approx(1.0)
    assert tvm._freshness_factor(21.0) == pytest.approx(1.0)
    assert tvm._freshness_factor(81.0) == pytest.approx(0.3)
    assert tvm._freshness_factor(200.0) == pytest.approx(0.3)
    # Midpoint of the 21-81 ramp: 1.0 - (30/60) * 0.7 = 0.65.
    assert tvm._freshness_factor(51.0) == pytest.approx(0.65)


def test_freshness_factor_no_native_trade():
    # No dated native trade in a segment that has trades -> floor.
    assert tvm._freshness_factor(None, segment_has_trades=True) == pytest.approx(0.3)
    # Nothing traded anywhere -> leave the deviation alone.
    assert tvm._freshness_factor(None, segment_has_trades=False) == pytest.approx(1.0)


def test_freshness_shrink_moves_value_toward_prior():
    cal = np.array([1200.0, 1000.0])
    prior = np.array([1000.0, 1000.0])
    out = tvm._apply_freshness_shrink(
        cal, prior, [81.0, 5.0], segment_has_trades=True)
    # Stale player keeps 30% of the +200 deviation; fresh player untouched.
    assert out[0] == pytest.approx(1060.0)
    assert out[1] == pytest.approx(1000.0)


# ── Top flip guard ──────────────────────────────────────────────────────────

def test_flip_guard_reverts_sub_1pct_inversion_with_thin_backing():
    prior = np.array([1000.0, 995.0, 500.0])   # gap 1000->995 is 0.5%
    cal = np.array([990.0, 1005.0, 500.0])     # #2 calibrated above #1
    backing = np.array([3.0, 2.9, 0.0])
    ages = [30.0, 23.0, None]
    out = tvm._apply_flip_guard(cal, prior, backing, ages)
    assert out[0] == pytest.approx(1000.0)
    assert out[1] == pytest.approx(995.0)
    assert out[2] == pytest.approx(500.0)


def test_flip_guard_keeps_earned_flip():
    prior = np.array([1000.0, 995.0])
    cal = np.array([990.0, 1005.0])
    backing = np.array([3.0, 12.0])            # >= 10 native backing
    ages = [30.0, 5.0]                         # ...and a trade <= 14d old
    out = tvm._apply_flip_guard(cal, prior, backing, ages)
    assert out[0] == pytest.approx(990.0)
    assert out[1] == pytest.approx(1005.0)


def test_flip_guard_ignores_gaps_of_1pct_or_more():
    prior = np.array([1000.0, 980.0])          # 2% gap: inversion is legal
    cal = np.array([990.0, 1005.0])
    backing = np.array([0.0, 0.0])
    ages = [None, None]
    out = tvm._apply_flip_guard(cal, prior, backing, ages)
    assert out[0] == pytest.approx(990.0)
    assert out[1] == pytest.approx(1005.0)


def test_flip_guard_earned_needs_fresh_trade_too():
    prior = np.array([1000.0, 995.0])
    cal = np.array([990.0, 1005.0])
    backing = np.array([3.0, 12.0])            # backing is there...
    ages = [30.0, 40.0]                        # ...but the newest trade is stale
    out = tvm._apply_flip_guard(cal, prior, backing, ages)
    assert out[0] == pytest.approx(1000.0)
    assert out[1] == pytest.approx(995.0)


# ── DynastyProcess staleness weight ─────────────────────────────────────────

def test_dp_weight_fresh_scrape():
    df = pd.DataFrame({"scrape_date": ["2026-10-27", "2026-10-29"]})
    weight, seen = vmt.dp_effective_weight(df, today=date(2026, 10, 29))
    assert weight == pytest.approx(0.20)
    assert seen == date(2026, 10, 29)


def test_dp_weight_stale_scrape():
    df = pd.DataFrame({"scrape_date": ["2026-10-23"]})
    weight, seen = vmt.dp_effective_weight(df, today=date(2026, 10, 29))
    assert weight == pytest.approx(0.10)
    assert seen == date(2026, 10, 23)


def test_dp_weight_exactly_five_days_is_still_fresh():
    df = pd.DataFrame({"scrape_date": ["2026-10-24"]})
    weight, _seen = vmt.dp_effective_weight(df, today=date(2026, 10, 29))
    assert weight == pytest.approx(0.20)


def test_dp_weight_missing_or_unparseable_scrape_date():
    weight, seen = vmt.dp_effective_weight(
        pd.DataFrame({"player": ["x"]}), today=date(2026, 10, 29))
    assert weight == pytest.approx(0.20)
    assert seen is None

    weight, seen = vmt.dp_effective_weight(
        pd.DataFrame({"scrape_date": ["not-a-date"]}), today=date(2026, 10, 29))
    assert weight == pytest.approx(0.20)
    assert seen is None

    weight, seen = vmt.dp_effective_weight(None, today=date(2026, 10, 29))
    assert weight == pytest.approx(0.20)
    assert seen is None


# ── Segment borrowing: tiers, discounts, gating ─────────────────────────────

def test_borrow_discount_tiers():
    assert tvm._borrow_discount(same_sf=True, same_size=False) == pytest.approx(0.5)
    assert tvm._borrow_discount(same_sf=False, same_size=True) == pytest.approx(0.35)
    # Native (same/same) is not borrowed; doubly-adjacent (neither) neither.
    assert tvm._borrow_discount(same_sf=True, same_size=True) is None
    assert tvm._borrow_discount(same_sf=False, same_size=False) is None


def test_borrowed_trade_gating_requires_a_thin_player():
    thin_idx = {7}
    terms_with_thin = [(3, 1.0), (7, -1.0)]
    terms_without_thin = [(3, 1.0), (4, -1.0), (9, 1.0)]
    assert tvm._terms_touch_thin(terms_with_thin, thin_idx) is True
    assert tvm._terms_touch_thin(terms_without_thin, thin_idx) is False
    assert tvm._terms_touch_thin([], thin_idx) is False


def test_borrowed_teams_filter_complements_native_bucket():
    frag = tvm._borrowed_teams_filter(10)
    assert "l.num_teams < 9" in frag and "l.num_teams > 11" in frag
    assert "BETWEEN 9 AND 11" in tvm._teams_filter(10)
    # Unknown sizes have an unbounded native filter, so no safe complement.
    assert tvm._borrowed_teams_filter(99) is None
