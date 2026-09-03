"""Unit tests for the TE-premium value helpers (pure, no DB/pandas deps).

These encode the logic behind several bugs fixed in the wild: the player modal
not applying the premium, and a ``Decimal * float`` 500 when scaling DB values.
"""
from decimal import Decimal

import pytest

from utils.value_helpers import apply_te_premium, te_premium_from_settings


# ── te_premium_from_settings ────────────────────────────────────────────────
def test_tier_none():
    assert te_premium_from_settings({"bonus_rec_te": 0}) == 0.0
    assert te_premium_from_settings({}) == 0.0


def test_tier_half():
    assert te_premium_from_settings({"bonus_rec_te": 0.5}) == 0.5


def test_tier_full():
    assert te_premium_from_settings({"bonus_rec_te": 1.0}) == 1.0


def test_tier_boundaries():
    # Snap points: >=0.25 -> half, >=0.75 -> full.
    assert te_premium_from_settings({"bonus_rec_te": 0.24}) == 0.0
    assert te_premium_from_settings({"bonus_rec_te": 0.25}) == 0.5
    assert te_premium_from_settings({"bonus_rec_te": 0.74}) == 0.5
    assert te_premium_from_settings({"bonus_rec_te": 0.75}) == 1.0


def test_malformed_settings_are_safe():
    assert te_premium_from_settings(None) == 0.0
    assert te_premium_from_settings({"bonus_rec_te": "not a number"}) == 0.0
    assert te_premium_from_settings({"bonus_rec_te": None}) == 0.0
    assert te_premium_from_settings("nonsense") == 0.0  # .get on a str -> handled


# ── apply_te_premium ────────────────────────────────────────────────────────
def test_full_premium_scales_te_by_20pct():
    assert apply_te_premium(100.0, "TE", 1.0) == pytest.approx(120.0)


def test_half_premium_scales_te_by_10pct():
    assert apply_te_premium(100.0, "TE", 0.5) == pytest.approx(110.0)


def test_no_premium_is_passthrough():
    assert apply_te_premium(100.0, "TE", 0.0) == 100.0


def test_non_te_never_scaled():
    for pos in ("QB", "RB", "WR", "K", "DEF"):
        assert apply_te_premium(100.0, pos, 1.0) == 100.0


def test_position_case_insensitive():
    assert apply_te_premium(100.0, "te", 1.0) == 120.0


def test_decimal_value_does_not_raise():
    # DB numerics arrive as Decimal; Decimal * float used to 500 the modal.
    out = apply_te_premium(Decimal("100"), "TE", 1.0)
    assert out == pytest.approx(120.0)
    assert isinstance(out, float)


def test_none_and_malformed_value_safe():
    assert apply_te_premium(None, "TE", 1.0) == 0.0
    assert apply_te_premium("nope", "TE", 1.0) == 0.0


# ── format-aware value / rank keys ──────────────────────────────────────────
from utils.value_helpers import (
    apply_redraft_display_fields,
    fill_unpriced_redraft_values,
    format_rank_key,
    format_rank_label_key,
    format_value_keys,
    row_format_rank_label,
    row_format_value,
)


def test_format_value_keys_redraft_vs_dynasty():
    assert format_value_keys(is_redraft=True, is_sf=False) == ("redraft_value_1qb", "value")
    assert format_value_keys(is_redraft=True, is_sf=True) == ("redraft_value_sf", "sf_value")
    assert format_value_keys(is_redraft=False, is_sf=False) == ("value", "value")
    assert format_value_keys(is_redraft=False, is_sf=True) == ("sf_value", "value")


def test_format_rank_label_key_redraft_vs_dynasty():
    assert format_rank_label_key(is_redraft=True, is_sf=False) == "redraft_pos_rank_label"
    assert format_rank_label_key(is_redraft=True, is_sf=True) == "redraft_sf_pos_rank_label"
    assert format_rank_label_key(is_redraft=False, is_sf=False) == "pos_rank_label"
    assert format_rank_label_key(is_redraft=False, is_sf=True) == "sf_pos_rank_label"


def test_format_rank_key_redraft_vs_dynasty():
    assert format_rank_key(is_redraft=True, is_sf=False) == "redraft_pos_rank"
    assert format_rank_key(is_redraft=True, is_sf=True) == "redraft_sf_pos_rank"
    assert format_rank_key(is_redraft=False, is_sf=False) == "pos_rank"
    assert format_rank_key(is_redraft=False, is_sf=True) == "sf_pos_rank"


def test_row_format_value_prefers_redraft_over_dynasty():
    row = {"redraft_value_1qb": 40, "value": 400}
    primary, fallback = format_value_keys(is_redraft=True, is_sf=False)
    assert row_format_value(row, primary, fallback) == 40


def test_fill_unpriced_redraft_does_not_use_raw_dynasty():
    """A waiver-wire prospect with dynasty 400 and no redraft price must not
    display 400 in a redraft league — the fill scales it below the priced floor."""
    table = [
        {"id": "star", "position": "WR", "value": 900, "sf_value": 800, "redraft_value_1qb": 200, "redraft_value_sf": 180},
        {"id": "fa", "position": "WR", "value": 400, "sf_value": 350},
    ]
    fill_unpriced_redraft_values(table)
    fa = table[1]
    assert fa["redraft_value_1qb"] < 200
    assert fa["redraft_value_1qb"] == pytest.approx(200 * 0.9)  # only unpriced WR; dyn_max=400
    assert fa["value"] == 400  # dynasty column unchanged
    # Priced player is untouched.
    assert table[0]["redraft_value_1qb"] == 200


def test_fill_unpriced_redraft_is_idempotent():
    table = [
        {"id": "a", "position": "RB", "value": 100, "sf_value": 90, "redraft_value_1qb": 50, "redraft_value_sf": 45},
        {"id": "b", "position": "RB", "value": 80, "sf_value": 70},
    ]
    fill_unpriced_redraft_values(table)
    first = table[1]["redraft_value_1qb"]
    fill_unpriced_redraft_values(table)
    assert table[1]["redraft_value_1qb"] == first


def test_apply_redraft_display_fields_stamps_ranks():
    table = [
        {"id": "a", "position": "WR", "value": 900, "sf_value": 800, "redraft_value_1qb": 200, "redraft_value_sf": 150},
        {"id": "b", "position": "WR", "value": 400, "sf_value": 500, "redraft_value_1qb": 80, "redraft_value_sf": 180},
    ]
    apply_redraft_display_fields(table)
    assert table[0]["redraft_pos_rank_label"] == "WR1"
    assert table[1]["redraft_pos_rank_label"] == "WR2"
    # SF redraft ranks invert because B has the higher SF redraft value.
    assert table[1]["redraft_sf_pos_rank_label"] == "WR1"
    assert table[0]["redraft_sf_pos_rank_label"] == "WR2"
    assert row_format_rank_label(table[0], "redraft_pos_rank_label") == "WR1"
