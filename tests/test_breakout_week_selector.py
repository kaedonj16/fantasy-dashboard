"""Week selector for the breakout page.

Covers the /api/breakout/weeks endpoint shape, the week param plumbing
(week=preseason forces the offseason board, an explicit numeric week loads
that exact snapshot, an invalid week reports data_available=False), and the
non-premium preview behavior with a week param.
"""
from __future__ import annotations

import pytest


# --------------------------------------------------------------------------
# Week param parsing
# --------------------------------------------------------------------------

def test_parse_breakout_week_values():
    import dashboard_services.breakout_api as api

    assert api._parse_breakout_week(None) is None
    assert api._parse_breakout_week("") is None
    assert api._parse_breakout_week("latest") is None
    assert api._parse_breakout_week("preseason") == "preseason"
    assert api._parse_breakout_week("Preseason") == "preseason"
    assert api._parse_breakout_week("0") == "preseason"
    assert api._parse_breakout_week("4") == 4
    assert api._parse_breakout_week(7) == 7


def test_parse_breakout_week_rejects_unrecognized_values():
    import dashboard_services.breakout_api as api

    for bad in ("bogus", "week3", "-2", "3.5", "next"):
        assert api._parse_breakout_week(bad) is api._BREAKOUT_WEEK_INVALID


# --------------------------------------------------------------------------
# list_breakout_weeks
# --------------------------------------------------------------------------

def _patch_weeks_deps(monkeypatch, completed, preseason_ready=True):
    import dashboard_services.breakout_api as api
    import data_building.breakout_engine.weekly_store as weekly_store

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: 2026)
    monkeypatch.setattr(
        weekly_store, "list_completed_weeks", lambda season: completed)
    monkeypatch.setattr(api, "opportunity_data_ready", lambda season: preseason_ready)


def test_list_breakout_weeks_shape(monkeypatch):
    import dashboard_services.breakout_api as api

    _patch_weeks_deps(monkeypatch, [
        {"as_of_week": 2, "as_of_date": "2026-09-15"},
        {"as_of_week": 4, "as_of_date": "2026-09-29"},
    ])

    payload = api.list_breakout_weeks(2026)

    assert payload["season"] == 2026
    assert payload["weeks"] == [
        {"value": "preseason", "label": "Preseason"},
        {"value": 2, "label": "Week 2", "as_of_date": "2026-09-15"},
        {"value": 4, "label": "Week 4", "as_of_date": "2026-09-29"},
    ]
    assert payload["latest_week"] == 4
    assert payload["preseason_available"] is True


def test_list_breakout_weeks_without_snapshots_defaults_to_preseason(monkeypatch):
    import dashboard_services.breakout_api as api

    _patch_weeks_deps(monkeypatch, [], preseason_ready=False)

    payload = api.list_breakout_weeks(2026)

    assert payload["weeks"] == [{"value": "preseason", "label": "Preseason"}]
    assert payload["latest_week"] == "preseason"
    assert payload["preseason_available"] is False


def test_weeks_route_serves_week_options(monkeypatch):
    flask = pytest.importorskip("flask")
    import dashboard_services.breakout_api as api

    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(api.breakout_bp)
    monkeypatch.setattr(
        api, "list_breakout_weeks",
        lambda season: {
            "season": 2026,
            "weeks": [{"value": "preseason", "label": "Preseason"},
                      {"value": 4, "label": "Week 4"}],
            "latest_week": 4,
            "preseason_available": True,
        },
    )

    response = app.test_client().get("/api/breakout/weeks?season=2026")

    assert response.status_code == 200
    body = response.get_json()
    assert body["season"] == 2026
    assert body["latest_week"] == 4
    assert body["weeks"][0] == {"value": "preseason", "label": "Preseason"}


# --------------------------------------------------------------------------
# week param through the candidate pipeline
# --------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, *args, **kwargs):
        return None

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return _FakeCursor(self._rows)


def test_week_preseason_forces_offseason_path(monkeypatch):
    """week=preseason skips the weekly branch even when a weekly snapshot is
    available, and runs the offseason opportunity board instead."""
    import dashboard_services.breakout_api as api
    import utils.utils as utils

    def no_weekly_path(*args, **kwargs):
        pytest.fail("weekly path must not run for week=preseason")

    monkeypatch.setattr(api, "_weekly_breakout_available", lambda season: True)
    monkeypatch.setattr(api, "_weekly_history_exists", lambda season: True)
    monkeypatch.setattr(api, "get_weekly_breakout_candidates", no_weekly_path)
    monkeypatch.setattr(api, "opportunity_data_ready", lambda season: True)
    monkeypatch.setattr(api, "get_conn", lambda: _FakeConn([]))
    monkeypatch.setattr(utils, "load_players_index", lambda: {})

    result = api.get_breakout_candidates(2026, week="preseason")

    assert result["data_available"] is True
    assert result.get("mode") != "weekly"
    assert result["data_status"] == "ok"


def test_week_preseason_unavailable_without_offseason_data(monkeypatch):
    import dashboard_services.breakout_api as api

    monkeypatch.setattr(api, "_weekly_breakout_available", lambda season: True)
    monkeypatch.setattr(api, "opportunity_data_ready", lambda season: False)
    monkeypatch.setattr(api, "get_conn", lambda: _FakeConn([]))

    result = api.get_breakout_candidates(2026, week="preseason")

    assert result["data_available"] is False


def test_explicit_numeric_week_loads_that_snapshot(monkeypatch):
    """week=4 goes straight to the Week 4 snapshot; it must not consult the
    latest-week branch at all."""
    import dashboard_services.breakout_api as api

    calls = []

    def fake_weekly(season, min_score=0.0, limit=None, as_of_week=None):
        calls.append((season, min_score, limit, as_of_week))
        return {"season": season, "as_of_week": as_of_week, "candidates": []}

    def no_fallback(*args, **kwargs):
        pytest.fail("default weekly branch must not run for an explicit week")

    monkeypatch.setattr(api, "get_weekly_breakout_candidates", fake_weekly)
    monkeypatch.setattr(api, "_weekly_breakout_available", no_fallback)

    result = api.get_breakout_candidates(2026, week="4")

    assert calls == [(2026, api.BREAKOUT_BOARD_MIN_SCORE, None, 4)]
    assert result["as_of_week"] == 4


def test_missing_snapshot_reports_unavailable_not_latest(monkeypatch):
    """An explicit week with no snapshot reports data_available=False with a
    clear reason, instead of silently serving the latest week's board."""
    import dashboard_services.breakout_api as api
    import data_building.breakout_engine.weekly_store as weekly_store

    monkeypatch.setattr(
        weekly_store, "load_weekly_candidates",
        lambda season, as_of_week=None, **kwargs: {
            "season": season, "as_of_week": as_of_week,
            "candidates": [], "data_available": False,
            "data_status": "unavailable",
        },
    )

    result = api.get_breakout_candidates(2026, week="99")

    assert result["data_available"] is False
    assert "Week 99" in (result.get("reason") or "")


def test_invalid_week_reports_data_unavailable():
    import dashboard_services.breakout_api as api

    result = api.get_breakout_candidates(2026, week="bogus")

    assert result["data_available"] is False
    assert result["data_status"] == "invalid_week"


# --------------------------------------------------------------------------
# route plumbing: week reaches the board pipeline
# --------------------------------------------------------------------------

def _candidates_client(monkeypatch, board_payload, premium=True):
    flask = pytest.importorskip("flask")
    import dashboard_services.breakout_api as api
    import dashboard_services.subscriptions as subscriptions

    calls = []

    def fake_board(season, min_score, limit, week=None):
        calls.append({"season": season, "week": week})
        return dict(board_payload)

    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(api.breakout_bp)
    monkeypatch.setattr(api, "get_breakout_board_candidates", fake_board)
    monkeypatch.setattr(subscriptions, "has_premium_for_viewer",
                        lambda *args: premium)
    return app.test_client(), calls


def test_candidates_route_forwards_week_param(monkeypatch):
    client, calls = _candidates_client(
        monkeypatch, {"candidates": [{"player_id": "1"}], "data_available": True})

    response = client.get("/api/breakout/candidates?season=2026&week=4")

    assert response.status_code == 200
    assert calls == [{"season": 2026, "week": "4"}]


def test_candidates_route_forwards_preseason_week(monkeypatch):
    client, calls = _candidates_client(
        monkeypatch, {"candidates": [{"player_id": "1"}], "data_available": True})

    response = client.get("/api/breakout/candidates?season=2026&week=preseason")

    assert response.status_code == 200
    assert calls == [{"season": 2026, "week": "preseason"}]


def test_nonpremium_preview_keeps_working_with_week(monkeypatch):
    client, calls = _candidates_client(
        monkeypatch,
        {"candidates": [{"player_id": str(i)} for i in range(5)],
         "data_available": True},
        premium=False,
    )

    response = client.get("/api/breakout/candidates?season=2026&week=2")

    assert response.status_code == 200
    body = response.get_json()
    assert len(body["candidates"]) == 3
    assert body["locked_count"] == 2
    assert calls == [{"season": 2026, "week": "2"}]
