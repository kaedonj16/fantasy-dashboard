"""Unit tests for utils.tier_thresholds.compute_tier_thresholds.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.tier_thresholds import (
    FALLBACK_THRESHOLDS,
    MAX_DISPLAY_TIERS,
    compute_tier_thresholds,
)


def _table(values, position="RB", key="value"):
    """Build a value table of dicts from a list of numbers."""
    return [{"position": position, key: v} for v in values]


def test_empty_table_returns_fallback():
    assert compute_tier_thresholds([]) == FALLBACK_THRESHOLDS
    assert compute_tier_thresholds(None) == FALLBACK_THRESHOLDS


def test_too_few_players_returns_fallback():
    # Fewer than num_tiers * 3 qualifying players -> fallback.
    assert compute_tier_thresholds(_table([100, 90, 80, 70])) == FALLBACK_THRESHOLDS


def test_thresholds_are_sorted_descending():
    # A clean staircase of well-separated clusters yields real tiers.
    vals = []
    for base in (900, 700, 500, 300, 150, 50):
        vals += [base + j for j in range(8)]  # 6 clusters * 8 = 48 players
    out = compute_tier_thresholds(_table(vals))
    assert out is not FALLBACK_THRESHOLDS
    assert len(out) >= 2
    assert out == sorted(out, reverse=True)


def test_thresholds_lie_between_cluster_values():
    vals = []
    for base in (900, 700, 500, 300, 150, 50):
        vals += [base + j for j in range(8)]
    out = compute_tier_thresholds(_table(vals))
    # Every boundary should fall inside the overall value range.
    assert all(min(vals) < t < max(vals) for t in out)


def test_kickers_defenses_picks_excluded():
    # Only 6 real players + a pile of K/DEF/PICK: still too few -> fallback,
    # proving the excluded positions are not counted toward the population.
    junk = _table([500] * 40, position="K") + _table([400] * 40, position="DEF")
    real = _table([900, 800, 700, 600, 500, 400], position="RB")
    assert compute_tier_thresholds(junk + real) == FALLBACK_THRESHOLDS


def test_non_dict_entries_skipped():
    vals = [b + j for b in (900, 700, 500, 300, 150, 50) for j in range(8)]
    table = _table(vals) + [None, "garbage", 42]
    # Should not raise, and should still produce tiers.
    out = compute_tier_thresholds(table)
    assert len(out) >= 2


def test_values_below_floor_ignored():
    # Values < 5 are dropped; a table of only tiny values -> fallback.
    assert compute_tier_thresholds(_table([4, 3, 2, 1] * 20)) == FALLBACK_THRESHOLDS


def test_sf_uses_sf_value_column():
    vals = [b + j for b in (900, 700, 500, 300, 150, 50) for j in range(8)]
    # Put the real distribution only in sf_value; leave `value` tiny.
    table = [{"position": "RB", "sf_value": v, "value": 1} for v in vals]
    out = compute_tier_thresholds(table, league_type="sf")
    assert out is not FALLBACK_THRESHOLDS
    assert all(min(vals) < t < max(vals) for t in out)


def test_league_size_column_selection():
    vals = [b + j for b in (900, 700, 500, 300, 150, 50) for j in range(8)]
    # 12-team 1QB reads value_12.
    table = [{"position": "RB", "value_12": v, "value": 1} for v in vals]
    out = compute_tier_thresholds(table, league_type="1qb", league_size=12)
    assert out is not FALLBACK_THRESHOLDS


def test_respects_max_display_tiers():
    # Many well-separated clusters, but never more than num_tiers-1 boundaries.
    vals = []
    for base in range(50, 50 + 20 * 60, 60):  # 20 clusters
        vals += [base + j for j in range(6)]
    out = compute_tier_thresholds(_table(vals), num_tiers=MAX_DISPLAY_TIERS)
    assert len(out) <= MAX_DISPLAY_TIERS - 1


def test_flat_wall_gets_span_split():
    # A single long flat ramp with no gaps should still be broken up by the
    # mandatory MAX_SPAN rule rather than collapsing to one tier.
    vals = [1000 - i for i in range(60)]  # smooth 1000..941, span ~60 < 220
    steep = [2000 - 20 * i for i in range(40)]  # span ~800 -> forces splits
    out = compute_tier_thresholds(_table(steep + vals))
    assert len(out) >= 2
