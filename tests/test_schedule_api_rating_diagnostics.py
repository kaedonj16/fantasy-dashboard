from __future__ import annotations

import pytest
from flask import Flask

import routes.schedule_api_bp as schedule_routes


pytestmark = pytest.mark.integration


def _client(monkeypatch):
    application = Flask(__name__)
    application.register_blueprint(schedule_routes.schedule_api_bp)
    metadata = {"rating_source": "default-profile-fallback",
                "requested_scoring_profile": "requested-hash",
                "loaded_scoring_profile": "standard-ppr"}
    monkeypatch.setattr(schedule_routes, "_matchup_ratings_metadata", lambda *_a, **_k: metadata)
    monkeypatch.setattr(schedule_routes, "get_league_ctx_from_cache", lambda *_a, **_k: {
        "raw_scoring_settings": {"rec": .5}, "users": [], "rosters": []})
    return application.test_client()


def test_schedule_diagnostics_expose_fallback_source(monkeypatch):
    client = _client(monkeypatch)
    player = {"pid": "1", "pos": "RB", "cells": [], "sos_rank": 1}
    monkeypatch.setattr(schedule_routes, "_compute_schedule_grid", lambda *_a, **_k: [player])
    monkeypatch.setattr(schedule_routes, "_matchup_rank_table", lambda *_a, **_k: (
        {"BUF": 1}, 1, {"BUF": {"completed_through_week": 4}}, True))
    payload = client.get("/api/schedule?season=2026&pids=1&league_id=L").get_json()
    assert payload["diagnostics"]["rating_source"] == "default-profile-fallback"
    assert payload["diagnostics"]["requested_scoring_profile"] == "requested-hash"


def test_schedule_rankings_uses_same_values_and_reports_source(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(schedule_routes, "get_players_index_global", lambda: {
        "1": {"name": "Runner", "pos": "RB", "team": "NE"}})
    monkeypatch.setattr(schedule_routes, "get_model_value_table_cached", lambda: [])
    monkeypatch.setattr(schedule_routes, "_matchup_rank_table", lambda *_a, **_k: (
        {"BUF": 1}, 1,
        {"BUF": {"rank_value": 90.0, "multiplier": None, "fpts": 25}}, True))
    monkeypatch.setattr("utils.utils.load_week_schedule", lambda *_a: [
        {"home": "NE", "away": "BUF"}])

    payload = client.get(
        "/api/schedule-rankings?season=2026&week_start=1&week_end=1&position=RB&league_id=L"
    ).get_json()
    assert payload["diagnostics"]["rating_source"] == "default-profile-fallback"
    assert payload["rankings"][0]["cells"][0]["rank"] == 1
    assert payload["rankings"][0]["sos_rank"] == 1
    assert payload["rankings"][0]["ease_score"] == 50.0
    assert payload["rankings"][0]["adjusted_avg_percent"] is None
