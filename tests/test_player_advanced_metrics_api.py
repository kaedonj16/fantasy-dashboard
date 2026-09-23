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
    # role_score is PRO-gated: non-PRO users don't get it, and the blended
    # eval score (65% role_score) is stripped too so it can't leak.
    assert "role_score" not in data["metrics"]
    assert "player_evaluation_score" not in data["metrics"]


def _pro_row(**extra):
    row = _season_row(
        ppr_over_expected_per_game=2.5,
        wopr=0.55,
        fp_cv=0.35,
        vorp=42.0,
        war=1.8,
        target_quality_score=12.5,
    )
    row.update(extra)
    return row


def test_modal_strips_pro_metrics_for_non_pro(offline_client, monkeypatch):
    """Non-PRO users must not see PRO metrics in the player modal payload."""
    import data_building.advanced_metrics as am
    import routes.players_bp as pb

    monkeypatch.setattr(am, "get_available_seasons_for_player", lambda pid: [2025])
    monkeypatch.setattr(am, "get_player_metrics_by_season", lambda pid, season: _pro_row())
    monkeypatch.setattr(am, "get_available_metric_weeks", lambda pid, season: [1, 2, 3])
    monkeypatch.setattr(am, "get_player_value_metrics", lambda *a, **k: {"metrics": {}})
    monkeypatch.setattr(pb, "_request_has_pro", lambda: False)

    resp = offline_client.get("/api/player-advanced-metrics/4034?season=2025")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    metrics = resp.get_json()["metrics"]
    for key in ("ppr_over_expected_per_game", "wopr", "fp_cv", "vorp", "war",
                "target_quality_score", "role_score", "player_evaluation_score"):
        assert key not in metrics, key
    # Free metrics still ship.
    assert metrics["yards_per_target"] == 8.5
    assert metrics["snap_share"] == 0.91


def test_modal_keeps_pro_metrics_for_pro(offline_client, monkeypatch):
    """PRO users get the full payload including PRO metrics."""
    import data_building.advanced_metrics as am
    import routes.players_bp as pb

    monkeypatch.setattr(am, "get_available_seasons_for_player", lambda pid: [2025])
    monkeypatch.setattr(am, "get_player_metrics_by_season", lambda pid, season: _pro_row())
    monkeypatch.setattr(am, "get_available_metric_weeks", lambda pid, season: [1, 2, 3])
    monkeypatch.setattr(am, "get_player_value_metrics", lambda *a, **k: {"metrics": {}})
    monkeypatch.setattr(pb, "_request_has_pro", lambda: True)

    resp = offline_client.get("/api/player-advanced-metrics/4034?season=2025")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    metrics = resp.get_json()["metrics"]
    assert metrics["ppr_over_expected_per_game"] == 2.5
    assert metrics["wopr"] == 0.55
    assert metrics["vorp"] == 42.0
    # Blended eval score lands for PRO (role + PFF grade): 82*0.65 + 90.4*0.35.
    assert metrics["player_evaluation_score"] == 84.9


def test_strip_pro_metrics_helper():
    from data_building.advanced_metrics import PRO_METRICS, strip_pro_metrics

    payload = {"yards_per_target": 8.5, "ppr_over_expected_per_game": 2.5,
               "vorp": 42.0, "snap_share": 0.91}
    # PRO user: no-op.
    assert strip_pro_metrics(payload, True) is payload
    # Non-PRO: PRO keys removed, free keys kept.
    stripped = strip_pro_metrics(payload, False)
    assert stripped == {"yards_per_target": 8.5, "snap_share": 0.91}
    assert all(k in PRO_METRICS for k in ("ppr_over_expected_per_game", "vorp"))
    # Empty/None safe.
    assert strip_pro_metrics(None, False) is None
    assert strip_pro_metrics({}, False) == {}


def test_pro_metrics_set_covers_full_answers_layer():
    """The 13-metric PRO set: FPOE family, WOPR, trends, consistency,
    composites, value. Free-anchoring metrics stay out."""
    from data_building.advanced_metrics import PRO_METRICS

    assert PRO_METRICS == frozenset({
        "ppr_over_expected_per_game", "ppr_over_expected",
        "half_ppr_over_expected", "standard_over_expected",
        "wopr", "opportunity_trend", "xfp_trend",
        "fp_cv", "xfp_stddev",
        "role_score", "target_quality_score",
        "vorp", "war",
    })
    # Already visible on free surfaces — must stay free.
    for free_key in ("expected_ppr_per_game", "boom_rate", "bust_rate", "schedule_ease"):
        assert free_key not in PRO_METRICS, free_key


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


def test_modal_js_can_combine_available_seasons():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "window.advPickSeason" in js
    assert 'data-year="career"' in js
    assert "Tap more years to combine" in js
    assert "isMultiSeason" in js


def test_multi_season_combines_only_years_the_player_has(offline_client, monkeypatch):
    import data_building.advanced_metrics as am

    seen = {}

    def _career(pid, seasons=None):
        seen["seasons"] = seasons
        return {
            "player_id": pid,
            "position": "WR",
            "season": None,
            "as_of_date": date(2025, 12, 28),
            "yards_per_target": 7.5,
        }

    monkeypatch.setattr(am, "get_available_seasons_for_player", lambda pid: [2025, 2024, 2022])
    monkeypatch.setattr(am, "get_player_career_metrics", _career)

    resp = offline_client.get("/api/player-advanced-metrics/4034?season=2025,2024,2022,2019")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert seen["seasons"] == [2025, 2024, 2022]
    assert data["selected_seasons"] == [2025, 2024, 2022]
    assert data["season"] is None
    assert data["metrics"]["yards_per_target"] == 7.5


def test_multi_season_falls_back_to_single_when_only_one_exists(offline_client, monkeypatch):
    import data_building.advanced_metrics as am

    monkeypatch.setattr(am, "get_available_seasons_for_player", lambda pid: [2024])
    monkeypatch.setattr(am, "get_player_metrics_by_season", lambda pid, season: _season_row(season=season))
    monkeypatch.setattr(am, "get_available_metric_weeks", lambda pid, season: [])
    monkeypatch.setattr(am, "get_player_value_metrics", lambda *a, **k: {"metrics": {}})

    resp = offline_client.get("/api/player-advanced-metrics/4034?season=2024,2019")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert data["season"] == 2024
    assert data["selected_seasons"] == [2024]


def test_trend_endpoint_hides_pro_options_for_non_pro(offline_client, monkeypatch):
    """Trend endpoint: PRO metrics excluded from options for non-PRO users."""
    import data_building.advanced_metrics as am
    import routes.players_bp as pb

    monkeypatch.setattr(am, "get_available_seasons_for_player", lambda pid: [2025])
    monkeypatch.setattr(
        am, "get_player_metrics_by_season",
        lambda pid, season: _pro_row(season=season),
    )
    monkeypatch.setattr(am, "get_player_metric_ranks",
                        lambda pid, season=None: {"ranks": {}, "counts": {}})
    monkeypatch.setattr(pb, "_request_has_pro", lambda: False)

    resp = offline_client.get("/api/player-advanced-metrics-trend/4034")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    option_keys = {o["key"] for o in data["options"]}
    assert "yards_per_target" in option_keys
    for pro_key in ("ppr_over_expected_per_game", "wopr", "fp_cv", "vorp",
                    "role_score", "target_quality_score"):
        assert pro_key not in option_keys, pro_key


def test_trend_endpoint_shows_pro_options_for_pro(offline_client, monkeypatch):
    import data_building.advanced_metrics as am
    import routes.players_bp as pb

    monkeypatch.setattr(am, "get_available_seasons_for_player", lambda pid: [2025])
    monkeypatch.setattr(
        am, "get_player_metrics_by_season",
        lambda pid, season: _pro_row(season=season),
    )
    monkeypatch.setattr(am, "get_player_metric_ranks",
                        lambda pid, season=None: {"ranks": {}, "counts": {}})
    monkeypatch.setattr(pb, "_request_has_pro", lambda: True)

    resp = offline_client.get("/api/player-advanced-metrics-trend/4034")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    option_keys = {o["key"] for o in resp.get_json()["options"]}
    assert "wopr" in option_keys
    assert "vorp" in option_keys


def test_ranks_endpoint_strips_pro_for_non_pro(offline_client, monkeypatch):
    """Ranks endpoint: PRO metric ranks/counts/bounds hidden from non-PRO."""
    import data_building.advanced_metrics as am
    import routes.players_bp as pb

    monkeypatch.setattr(
        am, "get_player_metric_ranks",
        lambda pid, season=None: {
            "season": 2025,
            "ranks": {"yards_per_target": 12, "wopr": 3, "vorp": 8},
            "counts": {"yards_per_target": 100, "wopr": 100, "vorp": 100},
            "bounds": {"yards_per_target": [4.0, 12.0], "wopr": [0.1, 0.7]},
        },
    )
    monkeypatch.setattr(pb, "_request_has_pro", lambda: False)

    resp = offline_client.get("/api/player-metric-ranks/4034?season=2025")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert data["ranks"]["yards_per_target"] == 12
    assert "wopr" not in data["ranks"]
    assert "vorp" not in data["ranks"]
    assert "wopr" not in data["counts"]
    assert "wopr" not in data["bounds"]


def test_ranks_endpoint_keeps_pro_for_pro(offline_client, monkeypatch):
    import data_building.advanced_metrics as am
    import routes.players_bp as pb

    monkeypatch.setattr(
        am, "get_player_metric_ranks",
        lambda pid, season=None: {
            "season": 2025,
            "ranks": {"yards_per_target": 12, "wopr": 3},
            "counts": {"yards_per_target": 100, "wopr": 100},
            "bounds": {},
        },
    )
    monkeypatch.setattr(pb, "_request_has_pro", lambda: True)

    resp = offline_client.get("/api/player-metric-ranks/4034?season=2025")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert resp.get_json()["ranks"]["wopr"] == 3
