"""Metric glossary descriptions are user-facing copy, not code.

Every LEADERBOARD_METRICS desc renders verbatim in the Advanced Metrics
Metric Glossary (and in metric tooltips), so it must never leak raw
snake_case column / field names (yardline_100, is_screen_pass,
pfr_advstats, matchup_ratings, ...) to readers.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from data_building.advanced_metrics import LEADERBOARD_METRICS


def test_no_glossary_desc_contains_underscores():
    offenders = {
        key: spec["desc"]
        for key, spec in LEADERBOARD_METRICS.items()
        if "_" in (spec.get("desc") or "")
    }
    assert offenders == {}


def test_goal_line_desc_is_plain_language():
    desc = LEADERBOARD_METRICS["goal_line_opp_share"]["desc"]
    assert "5-yard line" in desc
    assert "yardline" not in desc
