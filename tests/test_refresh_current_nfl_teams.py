"""Tests for daily NFL team refresh + DB overlay on players_index."""
from __future__ import annotations

import json

import pytest

from data_building.external_data import player_current_team as pct
from data_building.updates import update_players as up


@pytest.fixture(autouse=True)
def _clear_overlay_cache():
    pct.clear_overlay_cache()
    yield
    pct.clear_overlay_cache()


def test_normalize_nfl_team_aliases():
    assert pct.normalize_nfl_team("wsh") == "WAS"
    assert pct.normalize_nfl_team("JAC") == "JAX"
    assert pct.normalize_nfl_team("LA") == "LAR"
    assert pct.normalize_nfl_team("SEA") == "SEA"
    assert pct.normalize_nfl_team("FA") == ""
    assert pct.normalize_nfl_team(None) == ""


def test_apply_team_overlay_updates_team_and_bye(tmp_path, monkeypatch):
    index = {
        "1": {"name": "A", "team": "TB", "byeWeek": 11, "pos": "WR"},
        "2": {"name": "B", "team": "SEA", "byeWeek": 8, "pos": "RB"},
    }
    overlay = {"1": "SF"}
    bye = {"SF": 9, "SEA": 8}

    merged = pct.apply_team_overlay(index, overlay, bye_by_team=bye)

    # Original untouched
    assert index["1"]["team"] == "TB"
    assert merged is not index
    assert merged["1"]["team"] == "SF"
    assert merged["1"]["byeWeek"] == 9
    # Unchanged player shares the same object when no edit needed
    assert merged["2"]["team"] == "SEA"


def test_apply_team_overlay_noop_returns_same_object():
    index = {"1": {"name": "A", "team": "SEA", "pos": "WR"}}
    merged = pct.apply_team_overlay(index, {"1": "SEA"})
    assert merged is index


def test_update_player_teams_from_sleeper_writes_changes(tmp_path, monkeypatch):
    index_path = tmp_path / "players_index.json"
    index_path.write_text(json.dumps({
        "100": {"name": "Trader Joe", "team": "IND", "byeWeek": 11, "pos": "WR"},
        "101": {"name": "Stays Put", "team": "BUF", "byeWeek": 7, "pos": "RB"},
        "102": {"name": "Missing On Sleeper", "team": "KC", "byeWeek": 6, "pos": "TE"},
    }), encoding="utf-8")

    sleeper = {
        "100": {"full_name": "Trader Joe", "team": "PIT"},
        "101": {"full_name": "Stays Put", "team": "BUF"},
        # 102 absent from feed
        "103": {"full_name": "Not In Index", "team": "DAL"},
    }

    monkeypatch.setattr(up, "_bye_by_team", lambda: {"PIT": 5, "BUF": 7})

    result = up.update_player_teams_from_sleeper(
        index_path,
        write=True,
        keep_missing_team=True,
        sleeper_players=sleeper,
    )

    assert result["changed_count"] == 1
    assert result["changed_players"][0]["sleeper_id"] == "100"
    assert result["changed_players"][0]["old_team"] == "IND"
    assert result["changed_players"][0]["new_team"] == "PIT"

    written = json.loads(index_path.read_text(encoding="utf-8"))
    assert written["100"]["team"] == "PIT"
    assert written["100"]["byeWeek"] == 5
    assert written["101"]["team"] == "BUF"
    assert written["102"]["team"] == "KC"  # kept — missing on Sleeper


def test_update_keeps_team_when_sleeper_blank(tmp_path, monkeypatch):
    index_path = tmp_path / "players_index.json"
    index_path.write_text(json.dumps({
        "100": {"name": "Maybe FA", "team": "DAL", "pos": "WR"},
    }), encoding="utf-8")
    monkeypatch.setattr(up, "_bye_by_team", lambda: {})

    result = up.update_player_teams_from_sleeper(
        index_path,
        write=True,
        keep_missing_team=True,
        sleeper_players={"100": {"full_name": "Maybe FA", "team": None}},
    )
    assert result["changed_count"] == 0
    written = json.loads(index_path.read_text(encoding="utf-8"))
    assert written["100"]["team"] == "DAL"


def test_refresh_current_nfl_teams_upserts_db(tmp_path, monkeypatch):
    players = tmp_path / "players_index.json"
    relevant = tmp_path / "players_index_relevant.json"
    payload = {
        "200": {"name": "Move Me", "team": "CHI", "pos": "WR", "byeWeek": 7},
        "201": {"name": "Stay", "team": "GB", "pos": "RB", "byeWeek": 5},
    }
    players.write_text(json.dumps(payload), encoding="utf-8")
    relevant.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(up, "path_players_index", lambda: str(players))
    monkeypatch.setattr(up, "path_relevant_index", lambda: str(relevant))
    monkeypatch.setattr(up, "_bye_by_team", lambda: {"MIN": 6, "GB": 5})
    monkeypatch.setattr(
        up,
        "fetch_sleeper_players",
        lambda: {
            "200": {"full_name": "Move Me", "team": "MIN"},
            "201": {"full_name": "Stay", "team": "GB"},
        },
    )

    upserts = {}
    values = {}

    def fake_upsert(teams):
        upserts.update(teams)
        return len(teams)

    def fake_values(teams):
        values.update(teams)
        return len(teams)

    monkeypatch.setattr(up, "upsert_current_teams", fake_upsert)
    monkeypatch.setattr(up, "update_player_values_teams", fake_values)

    summary = up.refresh_current_nfl_teams(write_files=True, write_db=True)

    assert summary["changed_count"] == 1
    assert upserts["200"] == "MIN"
    assert upserts["201"] == "GB"
    assert values == {"200": "MIN"}
    assert json.loads(players.read_text())["200"]["team"] == "MIN"


def test_load_players_index_wires_team_overlay():
    """Guard that load_players_index overlays DB teams without importing utils
    (utils pulls requests/Flask — not in the slim unit-CI image)."""
    from pathlib import Path

    source = Path("utils/utils.py").read_text(encoding="utf-8")
    assert "def load_players_index()" in source
    assert "def _overlay_players_index(" in source
    assert "apply_team_overlay" in source
    assert "load_current_team_overlay" in source
    assert "player_current_team" in source
