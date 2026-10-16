"""Curated easy-metrics additions (Kaedon 2026-10-15, added to the
graph-rail + rate-totals PR): Pass Yds/G, RACR, RZ Opp/G, RZ Touches.
RYOE total lives in test_am_rate_totals.py; Out of Pocket % was already in
the catalog (restored 2026-10-14) and is asserted present here.
"""
import pytest

pytest.importorskip("flask")

from dashboard_services.pages.advanced_metrics_page import (
    build_advanced_metrics_body,
)
from data_building.advanced_metrics import LEADERBOARD_METRICS, PRO_METRICS

PAGE = build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def test_pass_yards_per_game_spec():
    spec = LEADERBOARD_METRICS["pass_yards_per_game"]
    assert spec["label"] == "Pass Yds/G"
    assert spec["category"] == "Passing"
    assert spec["positions"] == ["QB"]
    # Derived from stored columns only (total_pass_yards is itself computed,
    # not a column): Y/A x attempts / games, mirroring rush_yards_per_game.
    assert spec["computed_sql"] == (
        "m.yards_per_attempt * m.total_pass_att / NULLIF(m.games, 0)")
    assert "m.total_pass_yards" not in spec["computed_sql"]
    null = spec["computed_null"]
    for token in ("yards_per_attempt", "total_pass_att", "games"):
        assert token in null
    # Adjacent to its per-game sibling.
    keys = list(LEADERBOARD_METRICS)
    assert keys.index("pass_yards_per_game") == keys.index("pass_tds_per_game") + 1


def test_racr_is_the_stored_pbp_column():
    spec = LEADERBOARD_METRICS["racr"]
    assert spec["label"] == "RACR"
    assert spec["category"] == "Receiving"
    assert spec["positions"] == ["WR", "TE", "RB"]
    assert spec["efficiency"] is True
    assert not spec.get("pct")
    # Plain stored column (builder computes rec yards / total air yards from
    # play-by-play); no computed_sql stand-in.
    assert "computed_sql" not in spec


def test_rz_pair_specs_and_adjacency():
    per_game = LEADERBOARD_METRICS["rz_opps_per_game"]
    total = LEADERBOARD_METRICS["rz_touches"]
    assert per_game["label"] == "RZ Opp/G"
    assert total["label"] == "RZ Touches"
    for spec in (per_game, total):
        assert spec["category"] == "General"
        assert spec["positions"] == ["RB", "WR", "TE", "QB"]
    # Per-game reads the stored red_zone_usage column (per-game RZ carries +
    # targets from Sleeper) under the display key.
    assert per_game["computed_sql"] == "m.red_zone_usage"
    # Season total = per-game rate x games.
    assert total["computed_sql"] == "ROUND(m.red_zone_usage * m.games)"
    assert total["integer"] is True
    assert "red_zone_usage" in total["computed_null"]
    assert "games" in total["computed_null"]
    keys = list(LEADERBOARD_METRICS)
    assert keys.index("rz_touches") == keys.index("rz_opps_per_game") + 1


def test_out_of_pocket_already_in_catalog():
    # Approved in the same curated cut, but it was restored to the catalog on
    # 2026-10-14; pin its shape so it cannot silently drop out again. The
    # builder stores _rate_pct output (0-100), so pct with no pct_frac.
    spec = LEADERBOARD_METRICS["out_of_pocket_rate"]
    assert spec["category"] == "Passing"
    assert spec["positions"] == ["QB"]
    assert spec["pct"] is True
    assert not spec.get("pct_frac")
    assert "computed_sql" not in spec


NEW_KEYS = ("pass_yards_per_game", "racr", "rz_opps_per_game", "rz_touches")


def test_new_keys_are_free():
    for key in NEW_KEYS:
        assert key not in PRO_METRICS


def test_new_keys_render_in_pickers_and_quadrant_labels():
    for key in NEW_KEYS:
        assert f'<option value="{key}">' in PAGE
        assert f"'{key}':" in PAGE  # _qd quadrant/headline labels


def test_rejected_easy_metrics_stay_out():
    # Kaedon's curated cut explicitly rejected these.
    for key in ("ngs_rush_efficiency", "ngs_avg_time_to_los",
                "ngs_avg_expected_yac", "max_air_distance",
                "air_yards_differential"):
        assert key not in LEADERBOARD_METRICS
