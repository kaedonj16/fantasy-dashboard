"""The page and player modal share one authoritative breakout-board set."""
from __future__ import annotations

import pytest


def test_board_membership_uses_resolved_top_n_candidate_pipeline(monkeypatch):
    import dashboard_services.breakout_api as api

    calls = []
    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: 2027)

    def candidates(season, min_score, limit):
        calls.append((season, min_score, limit))
        return {"season": season, "candidates": [{"player_id": 123}]}

    monkeypatch.setattr(api, "get_breakout_candidates", candidates)

    eligible, board = api.breakout_board_membership(" 123 ", 2026)

    assert eligible is True
    assert board["season"] == 2027
    assert calls == [(2027, api.BREAKOUT_BOARD_MIN_SCORE, api.BREAKOUT_BOARD_LIMIT)]


def test_position_filter_is_not_part_of_board_membership_api():
    import inspect
    from dashboard_services.breakout_api import get_breakout_board_candidates

    assert "position" not in inspect.signature(get_breakout_board_candidates).parameters


def test_player_endpoint_exposes_membership_without_premium_detail(monkeypatch):
    flask = pytest.importorskip("flask")
    import dashboard_services.breakout_api as api
    import dashboard_services.subscriptions as subscriptions

    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(api.breakout_bp)
    monkeypatch.setattr(
        api,
        "breakout_board_membership",
        lambda player_id, season: (True, {"season": 2027, "data_available": True}),
    )
    monkeypatch.setattr(subscriptions, "has_premium_for_viewer", lambda *args: False)
    monkeypatch.setattr(
        api,
        "get_breakout_candidate_detail",
        lambda *args: pytest.fail("non-premium membership must not load premium detail"),
    )

    response = app.test_client().get(
        "/api/breakout/player/00123?season=2026&league_id=league-1&platform=espn"
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "board_eligible": True,
        "data_available": True,
        "player_id": "00123",
        "season": 2027,
    }


def test_player_endpoint_merges_membership_into_premium_detail(monkeypatch):
    flask = pytest.importorskip("flask")
    import dashboard_services.breakout_api as api
    import dashboard_services.subscriptions as subscriptions

    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(api.breakout_bp)
    monkeypatch.setattr(
        api,
        "breakout_board_membership",
        lambda player_id, season: (False, {"season": 2027, "data_available": True}),
    )
    monkeypatch.setattr(subscriptions, "has_premium_for_viewer", lambda *args: True)
    detail_calls = []

    def detail(player_id, season):
        detail_calls.append((player_id, season))
        return {"player_name": "Example Player", "board_eligible": True}

    monkeypatch.setattr(api, "get_breakout_candidate_detail", detail)

    response = app.test_client().get(
        "/api/breakout/player/123?season=2026&league_id=league-1&platform=yahoo"
    )

    assert response.status_code == 200
    assert response.get_json()["board_eligible"] is False
    assert response.get_json()["player_name"] == "Example Player"
    assert detail_calls == [("123", 2027)]
