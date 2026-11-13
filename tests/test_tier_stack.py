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


def _typed_side(players=(), picks=()):
    breakdown = (
        [{"value": v, "type": "player"} for v in players]
        + [{"value": v, "type": "pick"} for v in picks]
    )
    return {
        "breakdown": breakdown,
        "raw_total": float(sum(players) + sum(picks)),
    }


def test_equal_counts_get_tier_retention():
    a = _side([300, 200])
    b = _side([250, 250])
    apply_tier_stack_adjustment(a, b)
    # Tier retention prices every shape now, including equal counts:
    # both sides are written with a (negative) retention adjustment and
    # an effective total equal to the sum of retained values.
    for side in (a, b):
        assert side["adjustment"] < 0
        assert side["effective_total"] == sum(
            item["effective_value"] for item in side["breakdown"])


def test_bigger_side_is_penalized():
    a = _side([300, 100, 50])   # 3 players
    b = _side([400])            # 1 player
    apply_tier_stack_adjustment(a, b)
    # a is bigger -> stack retention PLUS the roster-slot penalty.
    # b is a lone headliner: full value, zero adjustment by design.
    assert a["adjustment"] < 0
    assert a["effective_total"] < a["raw_total"]
    assert b["adjustment"] == 0
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
    caps = build_tier_caps(NUM_TIERS)
    for side in (a, b):
        headliner = max(side["breakdown"], key=lambda i: i["value"])
        for item in side["breakdown"]:
            assert "tier" in item
            if item is headliner:
                # Headliner rule: the side's best player is never
                # discounted, whatever its tier.
                assert item["stack_mult"] == 1.0
            else:
                # Stacked pieces carry the real tier-retention cap,
                # not the old hardcoded 1.0.
                assert item["stack_mult"] == caps[item["tier"]]
            assert item["effective_value"] == round(
                item["value"] * item["stack_mult"], 1)


# ---- effective value: pinning cases ---------------------------------------

def test_stud_for_two_pieces_lands_outside_fair_band():
    # Gibbs ~971 for two ~485 pieces: raw sums are level (971 vs 970),
    # so the old math called this fair. The stud is a lone headliner
    # (full value); on the other side the second 485 is a stacked piece
    # retained at its T4 cap, and the slot penalty stacks on the
    # 2-player side: effective value must land OUTSIDE the 7% band.
    a = _typed_side(players=[971])
    b = _typed_side(players=[485, 485])
    apply_tier_stack_adjustment(a, b)
    assert a["raw_total"] - b["raw_total"] == 1.0  # level on raw value
    baseline = max(a["effective_total"], b["effective_total"])
    assert abs(a["effective_total"] - b["effective_total"]) > 0.07 * baseline


def test_equal_count_consolidation_is_adjusted():
    # 2-for-2, identical raw totals (1000 each): the side holding the
    # stud plus a throw-in is worth more effectively than two mid
    # pieces. Equal counts used to mean zero adjustment by rule.
    a = _typed_side(players=[900, 100])
    b = _typed_side(players=[500, 500])
    apply_tier_stack_adjustment(a, b)
    assert a["effective_total"] > b["effective_total"]


def test_equal_value_one_for_one_stays_fair():
    a = _typed_side(players=[600])
    b = _typed_side(players=[600])
    apply_tier_stack_adjustment(a, b)
    assert a["effective_total"] == b["effective_total"]


def test_picks_are_not_roster_slots():
    # Player + 2 picks vs a single player: player counts are 1 v 1, so
    # no roster-slot penalty may fire even though total assets are 3 v
    # 1, and picks keep their full value (no tier discount).
    a = _typed_side(players=[500], picks=[200, 150])
    b = _typed_side(players=[850])
    apply_tier_stack_adjustment(a, b)
    pick_items = [i for i in a["breakdown"] if i["type"] == "pick"]
    assert all(i["stack_mult"] == 1.0 for i in pick_items)
    assert all(i["effective_value"] == i["value"] for i in pick_items)
    player_item = next(i for i in a["breakdown"] if i["type"] == "player")
    assert a["effective_total"] == player_item["effective_value"] + 200 + 150


def test_sf_bench_lookup_reads_sf_column():
    # Same trade, same table: the bench penalty must come from the
    # sf_value column in a superflex league, not the 1QB column.
    vt = [{"value": 100.0, "sf_value": 20.0} for _ in range(500)]
    a1 = _side([300, 100])
    b1 = _side([400])
    apply_tier_stack_adjustment(a1, b1, value_table=vt, league_size=10,
                                is_sf=False)
    a2 = _side([300, 100])
    b2 = _side([400])
    apply_tier_stack_adjustment(a2, b2, value_table=vt, league_size=10,
                                is_sf=True)
    # 1QB bench reference is 100 (penalty 50); SF reference is 20
    # (penalty 10). Retention is identical, so the adjustments differ
    # by exactly the bench-penalty difference.
    assert a2["adjustment"] - a1["adjustment"] == 40.0


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
