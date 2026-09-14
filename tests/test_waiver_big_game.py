"""Unit tests for utils.waiver_big_game — the shared unexpected-big-game
detector and sustainability classifier. Pure logic, no app/DB import."""
import pytest

from utils.waiver_big_game import (
    BigGameAssessment,
    GameContext,
    absolute_component,
    assess_big_game,
    classify,
    discovery_key,
    merge_assessment,
    performance_surprise,
    relative_component,
    resolve_expectation,
    role_sustainability,
)


# ---- expectation resolution ------------------------------------------------

def test_pregame_snapshot_preferred_over_baseline():
    g = GameContext(player_id="p", position="WR", season=2025, week=5,
                    actual_points=24.0, pregame_projection=8.0,
                    projection_saved_at="2025-10-01T12:00:00Z",
                    baseline_ppg=6.0, baseline_source="season_avg")
    exp, basis, _unc = resolve_expectation(g)
    assert exp == 8.0
    assert basis == "pregame_projection"


def test_projection_without_timestamp_is_not_a_snapshot():
    # A projection with no saved-at timestamp isn't a real pregame snapshot; fall
    # back to the labeled baseline rather than trusting a possibly-revised number.
    g = GameContext(player_id="p", position="WR", season=2025, week=5,
                    actual_points=24.0, pregame_projection=8.0,
                    projection_saved_at=None,
                    baseline_ppg=6.0, baseline_source="trailing4")
    exp, basis, _unc = resolve_expectation(g)
    assert exp == 6.0
    assert basis == "baseline:trailing4"


def test_thin_history_baseline_flagged_uncertain():
    g = GameContext(player_id="rook", position="WR", season=2025, week=3,
                    actual_points=20.0, baseline_ppg=5.0,
                    baseline_source="season_avg", baseline_games=2)
    _exp, _basis, uncertain = resolve_expectation(g)
    assert uncertain is True


def test_rookie_with_no_baseline_uses_position_and_is_uncertain():
    g = GameContext(player_id="rook", position="RB", season=2025, week=1,
                    actual_points=18.0, position_baseline_ppg=9.0, is_rookie=True)
    exp, basis, uncertain = resolve_expectation(g)
    assert exp == 9.0
    assert basis == "position"
    assert uncertain is True


# ---- absolute vs relative surprise (tiny baselines can't explode) ----------

def test_absolute_component_gates_on_real_production():
    assert absolute_component(6.0) == 0.0        # below floor
    assert absolute_component(30.0) == pytest.approx(1.0)
    assert 0.0 < absolute_component(21.0) < 1.0


def test_relative_surprise_shrinks_tiny_expectation():
    # 2 -> 18 (beat by 16 off exp 2) vs 12 -> 28 (beat by 16 off exp 12).
    # The tiny-baseline game gets a bigger ratio but shrinkage keeps it bounded.
    low = relative_component(18.0, 2.0)
    high = relative_component(28.0, 12.0)
    assert low > high
    assert low <= 1.0


def test_tiny_baseline_blowup_still_ranks_below_big_absolute_game():
    # A 2->14 game (huge ratio, modest box score) should not out-surprise a
    # fully-expected-but-massive 18->31 game once absolute production gates it.
    fluky_small = GameContext(player_id="a", position="WR", season=2025, week=4,
                              actual_points=14.0, pregame_projection=2.0,
                              projection_saved_at="t")
    big_box = GameContext(player_id="b", position="WR", season=2025, week=4,
                          actual_points=31.0, pregame_projection=18.0,
                          projection_saved_at="t")
    s_small, _ = performance_surprise(fluky_small)
    s_big, _ = performance_surprise(big_box)
    assert s_big > s_small


# ---- role sustainability (usage) -------------------------------------------

def test_missing_usage_reads_as_unconfirmed_not_zero():
    g = GameContext(player_id="p", position="WR", season=2025, week=6,
                    actual_points=22.0)
    sustain, confirmed, cautions, _factors = role_sustainability(g)
    assert confirmed is False
    assert "role_unconfirmed" in cautions
    assert sustain <= 0.2


def test_rising_usage_lifts_sustainability():
    g = GameContext(player_id="p", position="WR", season=2025, week=6,
                    actual_points=16.0,
                    snap_share=0.82, snap_share_prev=0.45,
                    targets=9, targets_prev=3,
                    target_share=0.26, target_share_prev=0.10)
    sustain, confirmed, cautions, factors = role_sustainability(g)
    assert confirmed is True
    assert sustain >= 0.6
    assert "role_unconfirmed" not in cautions
    assert any("target" in f or "snap" in f for f in factors)


def test_td_dependence_flagged_and_discounts_role():
    # 4 touches, ~two long TDs => TD-dependent, capped sustainability.
    g = GameContext(player_id="p", position="RB", season=2025, week=7,
                    actual_points=20.0, touchdowns=2, touches=4, touches_prev=3,
                    total_yards=90, longest_play_yards=60)
    sustain, _confirmed, cautions, _factors = role_sustainability(g)
    assert "td_dependent" in cautions
    assert "one_big_play" in cautions
    assert sustain < 0.5


# ---- classification (the headline examples from the task) ------------------

def test_two_long_tds_on_four_touches_is_watchlist_with_caution():
    # Example: a player with two long TDs on four touches -> detected, but carries
    # an efficiency/workload caution and should NOT be a priority add.
    g = GameContext(player_id="fluke", position="RB", season=2025, week=8,
                    actual_points=22.0, pregame_projection=5.0, projection_saved_at="t",
                    touchdowns=2, touches=4, touches_prev=3,
                    total_yards=95, longest_play_yards=70)
    a = assess_big_game(g)
    assert a.category in ("watchlist", "speculative")
    assert "td_dependent" in a.cautions
    assert a.performance_surprise > 0  # still detected


def test_sustained_usage_outranks_fluke_even_with_fewer_points():
    # Example: fewer fantasy points but 8 targets and a big snap increase should
    # rank as the more sustainable add.
    sustained = GameContext(player_id="real", position="WR", season=2025, week=8,
                            actual_points=15.0, pregame_projection=7.0, projection_saved_at="t",
                            snap_share=0.80, snap_share_prev=0.40,
                            targets=8, targets_prev=3, target_share=0.24, target_share_prev=0.11)
    fluke = GameContext(player_id="fluke", position="RB", season=2025, week=8,
                        actual_points=22.0, pregame_projection=5.0, projection_saved_at="t",
                        touchdowns=2, touches=4, touches_prev=3,
                        total_yards=95, longest_play_yards=70)
    a_sustained = assess_big_game(sustained)
    a_fluke = assess_big_game(fluke)
    assert a_sustained.role_sustainability > a_fluke.role_sustainability
    # Priority-vs-watchlist ordering, keeping the two dimensions separate.
    assert a_sustained.category == "priority"
    assert a_fluke.category in ("watchlist", "speculative")


def test_classify_boundaries():
    assert classify(0.1, 0.9, 0.9) == "none"          # not surprising enough
    assert classify(0.6, 0.6, 0.6) == "priority"
    assert classify(0.4, 0.1, 0.5) == "watchlist"     # surprising, no role
    assert classify(0.4, 0.4, 0.3) == "speculative"


def test_teammate_out_is_committee_adjusted_not_full_transfer():
    # A confirmed starter absence gives opportunity, but we don't assume the whole
    # workload transfers to one backup.
    full = GameContext(player_id="b", position="RB", season=2025, week=9,
                       actual_points=12.0, teammate_out=True, teammate_out_share=1.0)
    committee = GameContext(player_id="b", position="RB", season=2025, week=9,
                            actual_points=12.0, teammate_out=True, teammate_out_share=0.5)
    s_full, *_ = role_sustainability(full)
    s_comm, *_ = role_sustainability(committee)
    assert s_full > s_comm


# ---- surprise/sustainability/fit kept separate -----------------------------

def test_assessment_keeps_dimensions_separate():
    g = GameContext(player_id="p", position="WR", season=2025, week=5,
                    actual_points=24.0, pregame_projection=8.0, projection_saved_at="t",
                    snap_share=0.8, snap_share_prev=0.5, targets=10, targets_prev=5)
    a = assess_big_game(g)
    d = a.to_dict()
    assert set(d) >= {"performance_surprise", "role_sustainability", "absolute_score"}
    # No roster-fit / availability field leaks into the shared detector.
    assert "faab" not in d and "lineup_gain" not in d and "drop" not in d


def test_scores_are_bounded_not_probabilities():
    g = GameContext(player_id="p", position="WR", season=2025, week=5,
                    actual_points=45.0, pregame_projection=1.0, projection_saved_at="t",
                    snap_share=1.0, snap_share_prev=0.1, targets=20, targets_prev=1)
    a = assess_big_game(g)
    assert 0.0 <= a.performance_surprise <= 1.0
    assert 0.0 <= a.role_sustainability <= 1.0


# ---- idempotent lifecycle (item 8) -----------------------------------------

def test_discovery_key_stable():
    assert discovery_key("p1", 2025, 6) == discovery_key("p1", 2025, 6.0)
    assert discovery_key("p1", 2025, 6) != discovery_key("p1", 2025, 7)


def test_merge_final_supersedes_in_progress():
    live = assess_big_game(GameContext(player_id="p", position="WR", season=2025,
                                       week=6, actual_points=10.0, status="in_progress"))
    final = assess_big_game(GameContext(player_id="p", position="WR", season=2025,
                                        week=6, actual_points=22.0, status="final"))
    merged = merge_assessment(live, final)
    assert merged.status == "final"
    assert merged.performance_surprise == final.performance_surprise
    # A late live update arriving after final does not clobber the final one.
    assert merge_assessment(final, live).status == "final"


def test_merge_no_duplicate_first_seen():
    a = assess_big_game(GameContext(player_id="p", position="WR", season=2025,
                                    week=6, actual_points=22.0))
    assert merge_assessment(None, a) is a
