"""Unit tests for utils.tier_stack.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.tier_stack import (
    NUM_TIERS,
    apply_tier_stack_adjustment,
    asset_tier,
    build_tier_caps,
)


# ---- asset_tier -----------------------------------------------------------

def test_asset_tier_top_value_is_tier_1():
    thresholds = [800, 600, 400, 200]
    assert asset_tier(900, thresholds) == 1
    assert asset_tier(800, thresholds) == 1


def test_asset_tier_descends_with_value():
    thresholds = [800, 600, 400, 200]
    assert asset_tier(700, thresholds) == 2
    assert asset_tier(500, thresholds) == 3
    assert asset_tier(300, thresholds) == 4


def test_asset_tier_below_all_thresholds_is_catch_all():
    thresholds = [800, 600, 400, 200]
    assert asset_tier(50, thresholds) == NUM_TIERS


def test_asset_tier_caps_at_num_tiers():
    # More thresholds than NUM_TIERS: the index is clamped.
    many = list(range(200, 0, -10))  # 20 thresholds
    assert asset_tier(5, many) == NUM_TIERS


def test_asset_tier_uses_fallback_when_none():
    # Should not raise with default thresholds.
    assert 1 <= asset_tier(1000) <= NUM_TIERS


# ---- apply_tier_stack_adjustment ------------------------------------------

def _side(values):
    return {
        "breakdown": [{"value": v} for v in values],
        "raw_total": float(sum(values)),
    }


def test_equal_counts_no_adjustment():
    a = _side([300, 200])
    b = _side([250, 250])
    apply_tier_stack_adjustment(a, b)
    # Tiers annotated, but no effective_total / adjustment written.
    assert all("tier" in item for item in a["breakdown"])
    assert "adjustment" not in a and "adjustment" not in b


def test_bigger_side_is_penalized():
    a = _side([300, 100, 50])   # 3 players
    b = _side([400])            # 1 player
    apply_tier_stack_adjustment(a, b)
    # a is bigger -> gets a negative adjustment; b is untouched at raw_total.
    assert a["adjustment"] < 0
    assert a["effective_total"] < a["raw_total"]
    assert b["adjustment"] == 0.0
    assert b["effective_total"] == b["raw_total"]


def test_penalty_grows_with_extra_players():
    two_extra = _side([300, 100, 50])   # 3 vs 1 -> delta 2
    one_extra = _side([300, 100])       # 2 vs 1 -> delta 1
    small1 = _side([400])
    small2 = _side([400])
    apply_tier_stack_adjustment(two_extra, small1)
    apply_tier_stack_adjustment(one_extra, small2)
    assert abs(two_extra["adjustment"]) > abs(one_extra["adjustment"])


def test_value_table_drives_bench_penalty():
    a = _side([300, 100])
    b = _side([400])
    vt = [{"value": v} for v in range(500, 0, -1)]  # deep, sorted-able table
    apply_tier_stack_adjustment(a, b, value_table=vt, league_size=10)
    assert a["adjustment"] < 0


def test_annotations_written_on_each_item():
    a = _side([300, 100])
    b = _side([250, 150])
    apply_tier_stack_adjustment(a, b)
    for side in (a, b):
        for item in side["breakdown"]:
            assert "tier" in item
            assert item["stack_mult"] == 1.0
            assert "effective_value" in item


# ---- build_tier_caps ------------------------------------------------------

def test_tier_caps_endpoints():
    caps = build_tier_caps(NUM_TIERS)
    assert caps[1] == 1.0
    assert caps[NUM_TIERS] == 0.38


def test_tier_caps_monotonically_decreasing():
    caps = build_tier_caps(NUM_TIERS)
    vals = [caps[t] for t in range(1, NUM_TIERS + 1)]
    assert vals == sorted(vals, reverse=True)


def test_tier_caps_single_tier():
    assert build_tier_caps(1) == {1: 1.0}
    assert build_tier_caps(0) == {1: 1.0}
