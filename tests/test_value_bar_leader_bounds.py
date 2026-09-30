"""Leader-relative bar scaling for the player modal Value section.

Expected FP / FP Over Exp / VORP / WAR bars must scale by the current
season leader's value (position-relative), not a fixed end-of-season ceiling.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def test_position_bounds_include_xfp_totals():
    from data_building.advanced_metrics import _compute_position_bounds

    rows = [
        {"player_id": "1", "games": 5, "expected_ppr": 50.0,
         "ppr_over_expected": 5.0, "total_routes": 210},
        {"player_id": "2", "games": 5, "expected_ppr": 80.0,
         "ppr_over_expected": -3.0, "total_routes": 260},
        # Below the games_min qualification: must not move the leader max.
        {"player_id": "3", "games": 1, "expected_ppr": 200.0,
         "ppr_over_expected": 90.0, "total_routes": 40},
    ]
    bounds = _compute_position_bounds(rows, 4)
    assert bounds["expected_ppr"] == [50.0, 80.0]
    assert bounds["ppr_over_expected"] == [-3.0, 5.0]
    assert bounds["total_routes"] == [210.0, 260.0]


def test_value_metrics_return_position_leader_bounds(monkeypatch):
    import data_building.advanced_metrics as am

    recs = [
        {"player_id": "1", "position": "WR", "pts": 100.0, "games": 3,
         "vorp": 20.0, "war": 0.8, "vorp_rank": 2, "war_rank": 2},
        {"player_id": "2", "position": "WR", "pts": 150.0, "games": 3,
         "vorp": 47.6, "war": 1.7, "vorp_rank": 1, "war_rank": 1},
        {"player_id": "3", "position": "RB", "pts": 120.0, "games": 3,
         "vorp": 99.0, "war": 3.5, "vorp_rank": 1, "war_rank": 1},
    ]
    monkeypatch.setattr(am, "_value_table", lambda *a, **k: (2026, recs))

    out = am.get_player_value_metrics("1", 2026, num_teams=12)
    # Bounds are relative to the player's own position leader, not the
    # cross-position max (the RB's 99.0 VORP must not leak into WR bounds).
    assert out["bounds"]["vorp"] == [20.0, 47.6]
    assert out["bounds"]["war"] == [0.8, 1.7]
    # Metrics and ranks unchanged.
    assert out["metrics"] == {"vorp": 20.0, "war": 0.8}
    assert out["ranks"] == {"vorp": 2, "war": 2}

    rb = am.get_player_value_metrics("3", 2026, num_teams=12)
    assert rb["bounds"]["vorp"] == [99.0, 99.0]

    assert am.get_player_value_metrics("99", 2026, num_teams=12) == {}


def test_ranks_endpoint_merges_value_bounds(offline_client, monkeypatch):
    """VORP/WAR position-leader bounds ship alongside the ranks for PRO users."""
    import data_building.advanced_metrics as am
    import routes.players_bp as pb

    monkeypatch.setattr(
        am, "get_player_metric_ranks",
        lambda pid, season=None: {
            "season": 2025,
            "ranks": {"yards_per_target": 12},
            "counts": {"yards_per_target": 100},
            "bounds": {"yards_per_target": [4.0, 12.0]},
        },
    )
    monkeypatch.setattr(
        am, "get_player_value_metrics",
        lambda *a, **k: {"metrics": {"vorp": 47.6, "war": 1.7},
                         "ranks": {"vorp": 3, "war": 3},
                         "bounds": {"vorp": [0.0, 60.0], "war": [0.0, 2.2]}},
    )
    monkeypatch.setattr(pb, "_request_has_pro", lambda: True)

    resp = offline_client.get("/api/player-metric-ranks/4034?season=2025")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert data["bounds"]["vorp"] == [0.0, 60.0]
    assert data["bounds"]["war"] == [0.0, 2.2]
    assert data["bounds"]["yards_per_target"] == [4.0, 12.0]
    assert data["ranks"]["vorp"] == 3


def test_ranks_endpoint_strips_value_bounds_for_non_pro(offline_client, monkeypatch):
    """Non-PRO users must not see VORP/WAR bounds (same as ranks/metrics)."""
    import data_building.advanced_metrics as am
    import routes.players_bp as pb

    monkeypatch.setattr(
        am, "get_player_metric_ranks",
        lambda pid, season=None: {
            "season": 2025, "ranks": {}, "counts": {},
            "bounds": {"yards_per_target": [4.0, 12.0]},
        },
    )
    monkeypatch.setattr(
        am, "get_player_value_metrics",
        lambda *a, **k: {"metrics": {"vorp": 47.6, "war": 1.7},
                         "ranks": {"vorp": 3, "war": 3},
                         "bounds": {"vorp": [0.0, 60.0], "war": [0.0, 2.2]}},
    )
    monkeypatch.setattr(pb, "_request_has_pro", lambda: False)

    resp = offline_client.get("/api/player-metric-ranks/4034?season=2025")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert "vorp" not in data["bounds"]
    assert "war" not in data["bounds"]
    assert data["bounds"]["yards_per_target"] == [4.0, 12.0]


def test_ranks_endpoint_tolerates_legacy_value_shape(offline_client, monkeypatch):
    """A get_player_value_metrics without 'bounds' must not break the endpoint."""
    import data_building.advanced_metrics as am
    import routes.players_bp as pb

    monkeypatch.setattr(
        am, "get_player_metric_ranks",
        lambda pid, season=None: {
            "season": 2025, "ranks": {}, "counts": {}, "bounds": {},
        },
    )
    monkeypatch.setattr(am, "get_player_value_metrics",
                        lambda *a, **k: {"metrics": {}})
    monkeypatch.setattr(pb, "_request_has_pro", lambda: True)

    resp = offline_client.get("/api/player-metric-ranks/4034?season=2025")
    assert resp.status_code == 200, resp.get_data(as_text=True)
