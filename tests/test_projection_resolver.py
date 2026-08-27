import pytest

from utils.projection_resolver import (
    SEASON_AVERAGE, WEEKLY, projection_cache_key, resolve_projected_ppg,
)


def _weeks(value=16.25):
    return {1: {"rb": {"raw_stats": {"pts_ppr": value, "pts_half_ppr": value - 2,
                                      "pts_std": value - 4, "rec": 4}}},
            2: {"rb": {"raw_stats": {"pts_ppr": value + 2, "pts_half_ppr": value,
                                      "pts_std": value - 2, "rec": 4}}}}


def test_sleeper_is_primary_and_provenance_is_explicit():
    out = resolve_projected_ppg("rb", {"rec": 1}, 2026, weekly_maps=_weeks(), secondary_ppg=99)
    assert out["ppg"] == pytest.approx(17.25)
    assert out["source"] == "sleeper"
    assert out["projection_type"] == SEASON_AVERAGE
    assert out["fallback_used"] is False
    assert out["week"] is None


def test_controlled_chase_brown_fixture_all_consumers_receive_x():
    # The ID is deliberately ordinary test data: no production special case.
    weeks = {w: {"9224": {"raw_stats": {"pts_ppr": 15.7}}} for w in (1, 2, 3)}
    canonical = resolve_projected_ppg("9224", {"rec": 1}, 2026, weekly_maps=weeks)
    consumers = {name: canonical["ppg"] for name in
                 ("draft_room", "modal", "player_page", "cheat_sheet", "pick_score",
                  "recommendation", "draft_grade", "vor")}
    assert set(consumers.values()) == {15.7}


def test_scoring_contexts_agree_and_custom_scoring_is_centralized():
    weeks = {1: {"te": {"raw_stats": {"rec": 5, "rec_yd": 60, "rec_td": 1}}}}
    ppr = resolve_projected_ppg("te", {"rec": 1}, 2026, weekly_maps=weeks, position="TE")
    half = resolve_projected_ppg("te", {"rec": .5}, 2026, weekly_maps=weeks, position="TE")
    tep = resolve_projected_ppg("te", {"rec": 1, "bonus_rec_te": .75}, 2026, weekly_maps=weeks, position="TE")
    assert (ppr["ppg"], half["ppg"], tep["ppg"]) == (17.0, 14.5, 20.75)


def test_weekly_is_distinct_and_fallback_order_is_uniform():
    weeks = _weeks()
    weekly = resolve_projected_ppg("rb", {"rec": 1}, 2026, 1, WEEKLY, weekly_maps=weeks)
    season = resolve_projected_ppg("rb", {"rec": 1}, 2026, weekly_maps=weeks)
    assert weekly["ppg"] != season["ppg"]
    assert weekly["week"] == 1
    missing = resolve_projected_ppg("missing", {"rec": 1}, 2026, weekly_maps=weeks,
                                    secondary_ppg=8, conservative_ppg=4)
    assert (missing["ppg"], missing["source"], missing["fallback_used"]) == (8, "secondary", True)


def test_cache_key_prevents_scoring_and_week_collisions():
    assert projection_cache_key("p", 2026, {"rec": .5}) != projection_cache_key("p", 2026, {"rec": 1})
    assert projection_cache_key("p", 2026, {"rec": 1}, WEEKLY, 1) != projection_cache_key("p", 2026, {"rec": 1}, WEEKLY, 2)
