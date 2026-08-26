"""In-season breakouts must not be replaced with 7-day value movers."""
from pathlib import Path

import pytest


def test_route_source_does_not_fallback_to_value_movers():
    src = (Path(__file__).parents[1] / "routes" / "breakout_api_bp2.py").read_text()
    assert "Fallback to value movers" not in src
    assert "get_top_movers" not in src
    assert '"breakout_score": delta' not in src
    assert "A price move is not a breakout" in src


def test_detection_failure_returns_empty_not_movers():
    mod = pytest.importorskip("routes.breakout_api_bp2")

    def _boom(**kwargs):
        raise RuntimeError("engine down")

    movers = {
        "1": {"name": "Fake Riser", "pos": "WR", "value": 800, "delta": 200},
    }
    assert mod.in_season_breakout_candidates(_boom, movers, movers) == []


def test_engine_rows_map_to_candidates_without_using_delta_as_score():
    mod = pytest.importorskip("routes.breakout_api_bp2")

    def _detect(**kwargs):
        return [{"player_id": "99", "score": 67.3}]

    index = {"99": {"name": "Real Breakout", "pos": "RB", "team": "CHI"}}
    values = {"99": {"age": 23, "value": 400, "sf_value": 420, "pos_rank": 18,
                     "pos_rank_label": "RB18"}}
    out = mod.in_season_breakout_candidates(_detect, index, values)
    assert len(out) == 1
    assert out[0]["player_id"] == "99"
    assert out[0]["name"] == "Real Breakout"
    assert out[0]["breakout_score"] == 67.3
    assert "delta" not in out[0]