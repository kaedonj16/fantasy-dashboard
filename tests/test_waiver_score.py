"""Unit tests for utils.waiver_score.waiver_pickup_score.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.waiver_score import WAIVER_PRIME_MAX, waiver_pickup_score


def _cand(value=100, age=None, position="RB", rank_change_7d=0, player_id="p1"):
    return {
        "value": value,
        "age": age,
        "position": position,
        "rank_change_7d": rank_change_7d,
        "player_id": player_id,
    }


def test_base_value_with_no_bonuses():
    # No age, no trend, no breakout -> just the raw value.
    c = _cand(value=120, age=None, rank_change_7d=0)
    assert waiver_pickup_score(c, {}) == 120


def test_positive_trend_adds_bonus_capped_at_60():
    strong = _cand(value=100, rank_change_7d=100)   # 100*4 capped at 60
    mild = _cand(value=100, rank_change_7d=5)        # 5*4 = 20
    assert waiver_pickup_score(strong, {}) == 160
    assert waiver_pickup_score(mild, {}) == 120


def test_negative_trend_ignored():
    c = _cand(value=100, rank_change_7d=-30)
    assert waiver_pickup_score(c, {}) == 100


def test_breakout_bonus_capped_at_50():
    c = _cand(value=100, player_id="star")
    big = waiver_pickup_score(c, {"star": 500})   # 500*0.5 capped at 50
    small = waiver_pickup_score(c, {"star": 20})  # 20*0.5 = 10
    assert big == 150
    assert small == 110


def test_prime_age_gives_full_age_bonus():
    # RB prime = 26; at/under prime the age bonus is the full +30.
    c = _cand(value=100, age=24, position="RB")
    assert waiver_pickup_score(c, {}) == 130


def test_past_prime_age_penalized():
    # RB prime 26; age 29 -> 30 - (29-26)*10 = 0 age bonus.
    c = _cand(value=100, age=29, position="RB")
    assert waiver_pickup_score(c, {}) == 100
    # Even further past prime goes negative.
    c2 = _cand(value=100, age=31, position="RB")
    assert waiver_pickup_score(c2, {}) == 100 + (30 - 50)


def test_position_specific_prime():
    # QB prime is 33 vs RB 26; a 30-year-old QB is still in prime.
    qb = _cand(value=100, age=30, position="QB")
    rb = _cand(value=100, age=30, position="RB")
    assert waiver_pickup_score(qb, {}) > waiver_pickup_score(rb, {})


def test_unknown_position_uses_default_prime_28():
    assert WAIVER_PRIME_MAX.get("K") is None
    c = _cand(value=100, age=28, position="K")
    # default prime 28 -> full +30 bonus at age 28.
    assert waiver_pickup_score(c, {}) == 130


def test_all_bonuses_combine():
    c = _cand(value=100, age=24, position="RB", rank_change_7d=10, player_id="x")
    # value 100 + trend min(40,60)=40 + breakout min(30*0.5,50)=15 + age 30
    assert waiver_pickup_score(c, {"x": 30}) == 185
