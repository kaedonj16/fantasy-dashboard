"""Compare page: basic vs detailed advanced-metrics view.

The compare tables (2-player and 3-player) render headline ("basic") metrics
per category up front and hide the rest behind a per-category "Show N more"
expander. These tests pin the backend contract: which metrics are basic, that
every populated category has basic coverage, and that the frontend expander
wiring exists.
"""
import re
from collections import Counter

import pytest

from data_building.advanced_metrics import LEADERBOARD_METRICS

pytest.importorskip("pandas")

# The curated headline set: (key, category). Keep in sync with the "basic"
# flags in data_building/advanced_metrics.py.
EXPECTED_BASIC = {
    # Value
    "vorp", "war",
    # General
    "ppr_pts_per_game", "snap_share", "opportunity_share", "touches_per_game",
    "yards_per_touch", "opportunity_trend", "boom_rate", "bust_rate",
    # Expected Pts
    "expected_ppr_per_game", "ppr_over_expected_per_game", "xtd_per_game",
    "td_over_expected", "breakout_trend_score",
    # Passing
    "yards_per_attempt", "completion_pct", "cpoe", "nfl_passer_rating",
    "epa_per_play", "pass_tds_per_game", "int_rate", "turnover_worthy_rate",
    # Rushing
    "carries_per_game", "rush_yards_per_game", "yards_per_carry",
    "rushing_success_rate", "ngs_rush_yards_over_expected_per_att",
    "rz_opp_share", "rush_tds_per_game",
    # Receiving
    "targets_per_game", "target_share", "rec_yards_per_game",
    "yards_per_target", "catch_rate", "avg_depth_of_target", "wopr",
    "rec_tds_per_game",
}


def _flagged():
    return {k for k, s in LEADERBOARD_METRICS.items()
            if not s.get("hidden") and s.get("basic")}


def test_basic_flag_matches_curated_set():
    flagged = _flagged()
    assert flagged == EXPECTED_BASIC, (
        "basic set drifted: missing=%s extra=%s"
        % (sorted(EXPECTED_BASIC - flagged), sorted(flagged - EXPECTED_BASIC))
    )


def test_every_category_has_basic_coverage():
    by_cat = Counter()
    for k in _flagged():
        by_cat[LEADERBOARD_METRICS[k]["category"]] += 1
    for cat in ("Value", "General", "Expected Pts", "Passing", "Rushing", "Receiving"):
        assert by_cat[cat] >= 2, f"category {cat} has no basic metrics"


def test_basic_is_a_minority_per_category():
    # The point of the split: basic stays scannable, the bulk hides behind
    # the expander.
    for cat in ("General", "Passing", "Rushing", "Receiving"):
        total = sum(1 for s in LEADERBOARD_METRICS.values()
                    if not s.get("hidden") and s.get("category") == cat)
        basic = sum(1 for k in _flagged()
                    if LEADERBOARD_METRICS[k]["category"] == cat)
        assert 0 < basic < total, f"{cat}: basic={basic} total={total}"


def test_config_endpoint_exposes_basic(offline_client):
    resp = offline_client.get("/api/advanced-metrics/config")
    assert resp.status_code == 200
    metrics = resp.get_json()["metrics"]
    assert metrics["vorp"]["basic"] is True
    assert metrics["target_share"]["basic"] is True
    assert metrics["yards_per_carry"]["basic"] is True
    # A detailed-only metric stays collapsed.
    assert metrics["ngs_avg_cushion"]["basic"] is False
    assert metrics["unrealized_air_yards"]["basic"] is False


def test_frontend_expander_wiring():
    src = open("static/app.js").read()
    assert "function cmpToggleCatDetails" in src
    assert "cmp-cat-toggle" in src
    assert "data-cmp-detail" in src
    # Both tables split basic vs detailed.
    assert src.count("_hasBasicFlags") >= 1
    assert "_hasBasic3" in src
    css = open("static/dashboard.css").read()
    assert ".cmp-cat-toggle" in css
    assert "cmp3-detail[hidden]" in css


def test_compare_chips_keep_league_path():
    src = open("static/app.js").read()
    # No hardcoded bare-/compare deep link may remain: chips must preserve
    # the league path via _cmpCompareBase().
    assert '"/compare?' not in src
    assert "function _cmpCompareBase" in src
    # The recent chip for the on-screen comparison is skipped so clicking it
    # can't just reload the identical page.
    assert "already on screen" in src
