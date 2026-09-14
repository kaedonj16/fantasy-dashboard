"""Tests for waiver horizons (#2), uncertainty-preserving projection timeline
(#9), and the credible-opportunity discovery floor (#5). Pure logic."""
import pytest

from utils.waiver_score import (
    BREAKOUT_FLOOR,
    WEEK_BYE,
    WEEK_MISSING,
    WEEK_OUT,
    WEEK_PLAYING,
    WEEK_ZERO,
    WEIGHTS,
    classify_projection_week,
    credible_opportunity,
    horizon_weights,
    passes_candidate_floor,
    return_timeline,
    waiver_pickup_score,
    weeks_out_from_projections,
)


# ---- #2 horizons -----------------------------------------------------------

def test_horizon_this_week_mutes_age_and_value():
    base = WEIGHTS
    tw = horizon_weights("this_week")
    assert tw.value_max < base.value_max
    assert tw.age_youth_max < base.age_youth_max
    assert tw.proj_per_ppg > base.proj_per_ppg


def test_horizon_stash_favors_value_and_youth():
    stash = horizon_weights("stash", dynasty=True)
    assert stash.value_max > WEIGHTS.value_max
    assert stash.injury_max < WEIGHTS.injury_max      # short-term vacancy de-emphasized
    assert stash.age_youth_max >= WEIGHTS.age_youth_max


def test_horizon_changes_ranking_meaningfully():
    # A young, high-value, low-projection player vs a productive-now veteran.
    stash_guy = {"value": 1200, "age": 22, "position": "WR", "player_id": "y",
                 "ros_ppg": 4.0}
    now_guy = {"value": 300, "age": 29, "position": "WR", "player_id": "n",
               "ros_ppg": 16.0}
    tw = horizon_weights("this_week")
    st = horizon_weights("stash", dynasty=True)
    # This-week: the productive veteran ranks higher.
    assert waiver_pickup_score(now_guy, {}, w=tw) > waiver_pickup_score(stash_guy, {}, w=tw)
    # Stash: the young high-value asset ranks higher.
    assert waiver_pickup_score(stash_guy, {}, w=st) > waiver_pickup_score(now_guy, {}, w=st)


# ---- #9 uncertainty-preserving timeline ------------------------------------

def test_missing_projection_does_not_extend_absence():
    # Present zeros for weeks 0-1 (confirmed by feed), then MISSING week 2, then
    # a real projection. With missing-as-unknown the run stops at the gap (2),
    # instead of counting the missing week as a third out week.
    series = [0.0, 0.0, None, 12.0]
    assert weeks_out_from_projections(series, treat_missing_as_out=True) == 3   # legacy
    assert weeks_out_from_projections(series, treat_missing_as_out=False) == 2  # #9


def test_explicit_zero_still_counts_as_out():
    assert weeks_out_from_projections([0.0, 0.0, 10.0], treat_missing_as_out=False) == 2


def test_classify_projection_week_distinguishes_states():
    assert classify_projection_week(0.0, bye=True) == WEEK_BYE
    assert classify_projection_week(0.0, confirmed_out=True) == WEEK_OUT
    assert classify_projection_week(None, present=False) == WEEK_MISSING
    assert classify_projection_week(0.0) == WEEK_ZERO
    assert classify_projection_week(14.0) == WEEK_PLAYING
    # An explicit zero alone is NOT WEEK_OUT (not proof of injury).
    assert classify_projection_week(0.0) != WEEK_OUT


def test_return_timeline_reports_coverage_and_basis():
    labels = [WEEK_OUT, WEEK_OUT, WEEK_MISSING, WEEK_PLAYING]
    t = return_timeline(labels)
    assert t["weeks_out"] == 2
    assert t["basis"] == "confirmed"
    assert t["weeks_missing"] == 1
    assert t["unknown_from"] == 2


def test_return_timeline_bye_not_a_missed_game():
    labels = [WEEK_OUT, WEEK_BYE, WEEK_OUT, WEEK_PLAYING]
    t = return_timeline(labels)
    assert t["weeks_out"] == 2          # bye skipped, not counted, not breaking the run


def test_return_timeline_projection_zero_is_estimate():
    labels = [WEEK_ZERO, WEEK_ZERO, WEEK_PLAYING]
    t = return_timeline(labels)
    assert t["weeks_out"] == 2
    assert t["estimated"] is True       # projection-derived => not certain
    # Override with a real return date -> not an estimate.
    t2 = return_timeline(labels, return_week_override=1)
    assert t2["weeks_out"] == 1
    assert t2["basis"] == "return_date"


# ---- #5 credible opportunity / discovery floor -----------------------------

def _low(**kw):
    c = {"player_id": "p", "value": 5, "position": "RB"}
    c.update(kw)
    return c


def test_floor_admits_high_value_directly():
    assert passes_candidate_floor({"value": 500, "player_id": "p"}, {}) is True


def test_floor_blocks_low_value_on_age_or_rank_noise():
    # A low-value player riding only age / rank movement is NOT admitted.
    c = _low(age=22, rank_change_7d=40)
    assert credible_opportunity(c, {}) is False
    assert passes_candidate_floor(c, {}) is False


def test_floor_admits_low_value_with_usage_spike():
    c = _low(usage_stat="snap_pct", usage_delta=10)
    assert credible_opportunity(c, {}) is True
    assert passes_candidate_floor(c, {}) is True


def test_floor_admits_low_value_with_injury_vacancy_next_in_line():
    c = _low(injured_ahead=["IR"], healthy_ahead=0)
    assert passes_candidate_floor(c, {}) is True
    # But not if a healthy body still blocks the role.
    blocked = _low(injured_ahead=["IR"], healthy_ahead=1)
    assert passes_candidate_floor(blocked, {}) is False


def test_floor_admits_low_value_with_engine_breakout():
    c = _low()
    assert passes_candidate_floor(c, {"p": BREAKOUT_FLOOR}) is True
    assert passes_candidate_floor(c, {"p": BREAKOUT_FLOOR - 10}) is False


def test_floor_admits_verified_role_change_or_big_game():
    assert passes_candidate_floor(_low(verified_role_change=True), {}) is True
    assert passes_candidate_floor(_low(big_game_priority=True), {}) is True
