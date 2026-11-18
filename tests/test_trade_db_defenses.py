"""Focused tests for team defenses in the Trade Database (2026-11-18).

Kaedon's screenshot: traded defenses (CLE, PIT) rendered as the raw team
abbreviation with a "?" position and Sleeper's placeholder image. Root
cause: Sleeper keys team defenses by team abbreviation and the players
index carries no DEF entries, so api_trade_database's describe() fell
back to name=pid / position="?", and the card renderer built a Sleeper
player-thumb URL for the abbreviation.

Fix: describe() synthesizes the defense row (full team name, DEF, team)
when a player id canonicalizes to an NFL team, and the renderer paints
the team logo (local crest, ESPN fallback) with the full team name
instead of the shortened initial-surname form.

API tests exec the real api_trade_database via AST with a stubbed
connection, mirroring test_trade_db_side_by_side.py's harness; the
canon_team / get_team_full_name helpers in the exec namespace are the
real ones from utils.nfl.
"""
from __future__ import annotations

import ast
import re
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
PAGE_SRC = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")


# -- Page: defense rendering ---------------------------------------------------

def test_defense_avatar_uses_team_logo_not_sleeper_thumb():
    assert "tdb-hs-def" in PAGE_SRC
    assert "/static/images/team_logos/" in PAGE_SRC
    assert "function tdbDefFallback" in PAGE_SRC
    # ESPN CDN is the second-chance source before the monogram.
    assert "a.espncdn.com/i/teamlogos/nfl/500/" in PAGE_SRC


def test_defense_logo_is_contained_not_cropped():
    m = re.search(r"\.tdb-hs-def img\s*\{\{(.*?)\}\}", PAGE_SRC, re.S)
    assert m, "no .tdb-hs-def img rule"
    assert "object-fit: contain" in m.group(1)


def test_defense_name_is_not_shortened():
    # tdbShortName("Cleveland Browns") would render "C. Browns"; defenses
    # bypass it and show the full team name.
    assert "const displayName = isDef ? (a.name || '') : tdbShortName(a.name);" in PAGE_SRC
    assert "posU === 'DEF' || posU === 'DST'" in PAGE_SRC


def test_defense_team_aliases_canonicalize():
    assert "function tdbCanonTeam" in PAGE_SRC
    for alias, canon in (("WSH", "WAS"), ("JAC", "JAX"), ("LA", "LAR")):
        assert f"t === '{alias}') return '{canon}'" in PAGE_SRC


# -- API: defense asset description --------------------------------------------

class _FakeArgs(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeRequest:
    def __init__(self, args):
        self.args = _FakeArgs(args)


_TRADE_ROW = {
    "id": 1, "transaction_id": "tx1", "season": 2026, "week": 9,
    "created_at": datetime(2026, 11, 5, 12, 0, 0),
    "scoring_type": "half", "is_superflex": False, "num_teams": 12,
    "league_type": 2,
}


def _asset(trade_id, side, player_id):
    return {"trade_id": trade_id, "side": side, "asset_type": "player",
            "player_id": player_id, "pick_season": None, "pick_round": None,
            "pick_order": None, "pick_slot": None, "pick_roster_id": None}


class _FakeCursor:
    def __init__(self, rows=()):
        self._rows = rows

    def fetchone(self):
        return {"n": 5}

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, recorder, asset_rows):
        self._rec = recorder
        self._assets = asset_rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, query, params):
        self._rec.append((query, list(params)))
        if "COUNT(" in query:
            return _FakeCursor()
        if "WHERE trade_id = ANY" in query:
            return _FakeCursor(self._assets)
        return _FakeCursor([_TRADE_ROW])


def _load_endpoint(monkeypatch, asset_rows):
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "api_trade_database"
    )
    seg = ast.get_source_segment(src, node)
    assert seg, "could not extract api_trade_database"

    calls = []
    db_mod = types.ModuleType("dashboard_services.db")
    db_mod.get_conn = lambda: _FakeConn(calls, asset_rows)
    ds_pkg = types.ModuleType("dashboard_services")
    ds_pkg.db = db_mod
    monkeypatch.setitem(sys.modules, "dashboard_services", ds_pkg)
    monkeypatch.setitem(sys.modules, "dashboard_services.db", db_mod)

    utils_mod = types.ModuleType("utils.utils")
    utils_mod.load_players_index = lambda: {
        "9221": {"name": "Jahmyr Gibbs", "pos": "RB", "team": "DET"}
    }
    utils_pkg = types.ModuleType("utils")
    utils_pkg.utils = utils_mod
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.utils", utils_mod)

    lt_mod = types.ModuleType("data_building.trade_intel.league_types")
    lt_mod.league_format_sql_param = lambda fmt: None
    ti_pkg = types.ModuleType("data_building.trade_intel")
    ti_pkg.league_types = lt_mod
    db_build_pkg = types.ModuleType("data_building")
    db_build_pkg.trade_intel = ti_pkg
    monkeypatch.setitem(sys.modules, "data_building", db_build_pkg)
    monkeypatch.setitem(sys.modules, "data_building.trade_intel", ti_pkg)
    monkeypatch.setitem(sys.modules, "data_building.trade_intel.league_types", lt_mod)

    from utils.nfl import canon_team, get_team_full_name

    import datetime as _dt
    import logging as _logging

    ns = {
        "request": _FakeRequest({"season": "2026"}),
        "datetime": _dt.datetime,
        "logger": _logging.getLogger("test"),
        "jsonify": lambda obj, *a: obj,
        "_TRADE_DB_COUNT_CACHE": {},
        "_TRADE_DB_COUNT_TTL": 300,
        "canon_team": canon_team,
        "get_team_full_name": get_team_full_name,
    }
    exec(compile(seg, "api_trade_database", "exec"), ns)  # noqa: S102 - test harness
    return ns["api_trade_database"]


def test_traded_defense_describes_as_defense(monkeypatch):
    fn = _load_endpoint(monkeypatch, [
        _asset(1, "a", "CLE"),
        _asset(1, "b", "9221"),
    ])
    data = fn()
    defense = data["trades"][0]["side_a"][0]
    assert defense == {
        "type": "player", "player_id": "CLE",
        "name": "Cleveland Browns", "position": "DEF", "team": "CLE",
    }
    # The indexed player on the other side is untouched.
    gibbs = data["trades"][0]["side_b"][0]
    assert gibbs["name"] == "Jahmyr Gibbs"
    assert gibbs["position"] == "RB"
    assert gibbs["team"] == "DET"


def test_washington_defense_uses_canonical_abbr(monkeypatch):
    fn = _load_endpoint(monkeypatch, [
        _asset(1, "a", "WSH"),
        _asset(1, "b", "9221"),
    ])
    defense = fn()["trades"][0]["side_a"][0]
    assert defense["name"] == "Washington Commanders"
    assert defense["position"] == "DEF"
    assert defense["team"] == "WAS"


def test_unknown_player_id_is_not_invented_as_a_defense(monkeypatch):
    fn = _load_endpoint(monkeypatch, [
        _asset(1, "a", "99999"),
        _asset(1, "b", "9221"),
    ])
    unknown = fn()["trades"][0]["side_a"][0]
    assert unknown["name"] == "99999"
    assert unknown["position"] == "?"
    assert unknown["team"] == ""
