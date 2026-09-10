"""Regression: /api/player-details must not shadow module-level canon_team.

A conditional ``from utils.utils import canon_team`` inside the DEF/DST
fallback made ``canon_team`` a local for the entire handler. Real players
skip that branch, so ``player_team = canon_team(...)`` raised UnboundLocalError.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _player_details_src() -> str:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def api_player_details(player_id: str):")
    end = src.find("def api_player_game_logs", start)
    assert start > 0 and end > start
    return src[start:end]


def test_player_details_does_not_import_canon_team_locally():
    body = _player_details_src()
    assert "player_team = canon_team(" in body
    assert "from utils.utils import canon_team" not in body
    assert "from utils.utils import def_team_logo_urls" in body


def test_api_player_details_known_player_uses_canon_team(monkeypatch):
    """Opening the modal for a found player must not UnboundLocalError."""
    pytest.importorskip("flask")
    try:
        from app import app as flask_app
    except Exception as exc:
        pytest.skip(f"app not importable ({type(exc).__name__})")

    pid = "4046"
    monkeypatch.setattr(
        "utils.utils.load_relevant_index",
        lambda: {pid: {"name": "Patrick Mahomes", "pos": "QB", "team": "WSH"}},
    )
    monkeypatch.setattr("app.get_model_value_table_cached", lambda: [])
    monkeypatch.setattr("app.get_player_value_history", lambda *a, **k: [])
    monkeypatch.setattr("app._load_usage_rows_cached", lambda season: [])
    monkeypatch.setattr("app._oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr("app.get_players_global", lambda: {})

    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        resp = client.get(f"/api/player-details/{pid}")

    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["player_id"] == pid
    assert data["name"] == "Patrick Mahomes"
    assert data["position"] == "QB"
    # Response team is the raw meta team; canon_team is used for oline lookup.
    assert data["team"] == "WSH"
    assert data.get("ok") is not False


def test_api_player_details_def_fallback_still_canonicalizes(monkeypatch):
    pytest.importorskip("flask")
    try:
        from app import app as flask_app
    except Exception as exc:
        pytest.skip(f"app not importable ({type(exc).__name__})")

    monkeypatch.setattr("utils.utils.load_relevant_index", lambda: {})
    monkeypatch.setattr("app.load_players_index", lambda: {})
    monkeypatch.setattr(
        "app.load_teams_index",
        lambda: {"WAS": {"teamId": "32"}},
    )
    monkeypatch.setattr(
        "utils.utils.def_team_logo_urls",
        lambda team: (f"/static/{team}.png", f"https://espn/{team}.png"),
    )
    monkeypatch.setattr("app.get_model_value_table_cached", lambda: [])
    monkeypatch.setattr("app.get_player_value_history", lambda *a, **k: [])
    monkeypatch.setattr("app._load_usage_rows_cached", lambda season: [])
    monkeypatch.setattr("app._oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr("app.get_players_global", lambda: {})

    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        resp = client.get("/api/player-details/WSH")

    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["position"] == "DEF"
    assert data["team"] == "WAS"
    assert data["name"] == "WAS D/ST"
