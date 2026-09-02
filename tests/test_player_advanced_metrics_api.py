"""Player-modal Adv Metrics API must serialize season rows with non-numeric columns.

Season snapshots from player_advanced_metrics include ``nfl_team`` (VARCHAR)
and rookie boolean flags. The modal Adv Metrics tab requests the latest season
after an auto/career probe; a blanket ``float()`` on every column 500'd that
request and the tab showed "network hiccup".
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


def _season_row(**extra):
    row = {
        "player_id": "4034",
        "position": "WR",
        "season": 2025,
        "as_of_date": date(2025, 12, 28),
        "id": 99,
        "yards_per_target": Decimal("8.50"),
        "catch_rate": Decimal("0.72"),
        "snap_share": Decimal("0.91"),
        "role_score": Decimal("82.0"),
        "grades_offense": Decimal("90.4"),
        "nfl_team": "KC",
        "rookie_eval_is_rookie": False,
        "rookie_eval_true_early_declare": True,
    }
    row.update(extra)
    return row


def test_jsonable_metrics_skips_team_and_bools():
    from routes.players_bp import _jsonable_metrics

    out = _jsonable_metrics(_season_row())
    assert out["yards_per_target"] == 8.5
    assert out["catch_rate"] == 0.72
    assert out["role_score"] == 82.0
    assert "nfl_team" not in out
    assert "rookie_eval_is_rookie" not in out
    assert "rookie_eval_true_early_declare" not in out
    assert "player_id" not in out
    assert "as_of_date" not in out
    assert "id" not in out


def test_jsonable_metrics_drops_nan_and_non_numeric():
    from routes.players_bp import _jsonable_metrics

    out = _jsonable_metrics({
        "yards_per_target": float("nan"),
        "catch_rate": float("inf"),
        "snap_share": "not-a-number",
        "epa_per_play": Decimal("0.15"),
        "nfl_team": "SF",
        "position": "RB",
    })
    assert out["epa_per_play"] == 0.15
    assert "yards_per_target" not in out
    assert "catch_rate" not in out
    assert "snap_share" not in out
    assert "nfl_team" not in out


def test_season_endpoint_survives_nfl_team(offline_client, monkeypatch):
    import data_building.advanced_metrics as am

    monkeypatch.setattr(am, "get_available_seasons_for_player", lambda pid: [2025, 2024])
    monkeypatch.setattr(am, "get_player_metrics_by_season", lambda pid, season: _season_row())
    monkeypatch.setattr(am, "get_available_metric_weeks", lambda pid, season: [1, 2, 3])
    monkeypatch.setattr(am, "get_player_value_metrics", lambda *a, **k: {"metrics": {}})

    resp = offline_client.get("/api/player-advanced-metrics/4034?season=2025")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert data["player_id"] == "4034"
    assert data["position"] == "WR"
    assert data["season"] == 2025
    assert data["available_seasons"] == [2025, 2024]
    assert data["metrics"]["yards_per_target"] == 8.5
    assert data["metrics"]["catch_rate"] == 0.72
    assert "nfl_team" not in data["metrics"]
    # Blended eval score still lands (role + PFF grade): 82*0.65 + 90.4*0.35.
    assert data["metrics"]["player_evaluation_score"] == 84.9


def test_career_probe_still_returns_available_seasons(offline_client, monkeypatch):
    """The modal's auto load omits season (career probe) to learn available years."""
    import data_building.advanced_metrics as am

    monkeypatch.setattr(am, "get_available_seasons_for_player", lambda pid: [2025, 2024])
    monkeypatch.setattr(am, "get_player_career_metrics", lambda pid: {
        "player_id": pid,
        "position": "WR",
        "season": None,
        "as_of_date": date(2025, 12, 28),
        "yards_per_target": 8.1,
    })

    resp = offline_client.get("/api/player-advanced-metrics/4034")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert data["available_seasons"] == [2025, 2024]
    assert data["metrics"]["yards_per_target"] == 8.1


def test_modal_js_treats_404_as_empty_not_network_error():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    load_fn = js[js.find("function loadAdvancedMetrics"): js.find("function pmTrendsSetMode")]
    assert "res.status === 404" in load_fn
    assert "encodeURIComponent(playerId)" in load_fn
    assert "No metrics available for this player" in load_fn
