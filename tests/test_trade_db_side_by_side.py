"""Focused tests for the trade-database side-by-side card rework (2026-10-21).

Kaedon's approved mock: cards show both sides side-by-side at every width
(the old <=480px stacking is gone), a dot-separated meta line instead of
badges, headshot + initial-surname + POS - TEAM per player, round-number
circles for picks, and a navy Analyze bar that opens the trade calculator
preloaded with the trade (?a=&b=&ap=&bp=). A time-range filter
(All season / 7D / 30D / 90D) joins the existing filter rows.

Also pins the honesty fixes that ride along: the old footer computed a
"Fair deal" verdict from side_a_value/side_b_value fields the API never
returns (0 vs 0, always Fair) - the footer and the Fair badge are gone, and
no TEP is rendered because the trade DB has no TEP field.

Page tests read the route source; API tests exec the real
api_trade_database via AST with a stubbed connection, mirroring
test_trade_db_sort.py's harness.
"""
from __future__ import annotations

import ast
import re
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE_SRC = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")


def _rule(selector: str) -> str:
    m = re.search(re.escape(selector) + r"\s*\{\{(.*?)\}\}", PAGE_SRC, re.S)
    assert m, f"no inline CSS rule found for {selector}"
    return m.group(1)


# -- Page: card layout --------------------------------------------------------

def test_card_body_is_two_columns_with_swap_track():
    rule = _rule(".tdb-card-body")
    assert "grid-template-columns: 1fr 28px 1fr" in rule


def test_card_body_never_stacks_on_phone():
    # The old <=480px rule collapsed the body to one column; that stacking
    # is exactly what the rework removes.
    assert ".tdb-card-body {{ grid-template-columns: 1fr; }}" not in PAGE_SRC
    assert "@media(max-width: 480px)" not in PAGE_SRC


def test_swap_badge_is_a_circle():
    rule = _rule(".tdb-swap")
    assert "border-radius: 50%" in rule
    assert "width: 24px" in rule and "height: 24px" in rule


def test_analyze_bar_is_navy_and_bottom_pinned():
    rule = _rule(".tdb-analyze")
    assert "background: #122d4b" in rule
    assert re.search(r"margin\s*:\s*auto", rule)


def test_no_team_labels_footer_or_fair_verdict():
    assert "Team A got" not in PAGE_SRC
    assert "Team B got" not in PAGE_SRC
    assert "tdb-card-foot" not in PAGE_SRC
    assert "Fair deal" not in PAGE_SRC
    assert "chip--fair" not in PAGE_SRC
    # The fake verdict math read fields the API never sends.
    assert "side_a_value" not in PAGE_SRC


def test_renderer_uses_headshots_short_names_and_pick_circles():
    assert "sleepercdn.com/content/nfl/players/thumb/" in PAGE_SRC
    assert "tdbShortName" in PAGE_SRC
    assert "tdb-hs-init" in PAGE_SRC  # initials fallback behind the photo
    assert 'onerror="this.remove()"' in PAGE_SRC
    assert "tdb-pick-ic" in PAGE_SRC
    assert "tdbPickLabel" in PAGE_SRC


def test_analyze_link_preloads_calculator_with_both_sides():
    assert "TDB_LEAGUE_ID" in PAGE_SRC
    assert "calcBase" in PAGE_SRC
    for key in ("'a'", "'b'", "'ap'", "'bp'"):
        assert f"qs.set({key}," in PAGE_SRC
    # Generic round picks use the site's mid-bucket convention.
    assert "'mid'" in PAGE_SRC


def test_meta_line_uses_only_real_fields():
    # Relative date + teams + SF/1QB + scoring + week. No TEP segment is
    # rendered (the trade DB has no TEP field); the word survives only in
    # the renderer comment explaining that, so strip comments first.
    assert "tdbWhen" in PAGE_SRC
    assert "-team" in PAGE_SRC
    code_only = re.sub(r"//[^\n]*", "", PAGE_SRC)
    assert "TEP" not in code_only


# -- Page: time-range filter --------------------------------------------------

def test_time_range_chip_row_exists():
    assert "tdb-tr-filters" in PAGE_SRC
    for val, label in (("all", "All season"), ("7d", "7D"), ("30d", "30D"), ("90d", "90D")):
        assert f'data-tr="{val}"' in PAGE_SRC
        assert f"tdbRangeFilter('{val}')" in PAGE_SRC
        assert label in PAGE_SRC


def test_time_range_is_sent_to_api():
    assert "range: timeRange" in PAGE_SRC
    assert "window.tdbRangeFilter" in PAGE_SRC


# -- API: range filter + asset fields -----------------------------------------

class _FakeArgs(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeRequest:
    def __init__(self, args):
        self.args = _FakeArgs(args)


_TRADE_ROW = {
    "id": 1, "transaction_id": "tx1", "season": 2026, "week": 6,
    "created_at": datetime(2026, 10, 14, 12, 0, 0),
    "scoring_type": "ppr", "is_superflex": True, "num_teams": 12,
}
_ASSET_ROWS = [
    {"trade_id": 1, "side": "a", "asset_type": "player", "player_id": "9221",
     "pick_season": None, "pick_round": None, "pick_order": None,
     "pick_slot": None, "pick_roster_id": None},
    {"trade_id": 1, "side": "b", "asset_type": "pick", "player_id": None,
     "pick_season": 2027, "pick_round": 1, "pick_order": None,
     "pick_slot": None, "pick_roster_id": None},
]


class _FakeCursor:
    def __init__(self, rows=()):
        self._rows = rows

    def fetchone(self):
        return {"n": 5}

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, recorder):
        self._rec = recorder

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, query, params):
        self._rec.append((query, list(params)))
        if "COUNT(DISTINCT" in query:
            return _FakeCursor()
        if "WHERE trade_id = ANY" in query:
            return _FakeCursor(_ASSET_ROWS)
        return _FakeCursor([_TRADE_ROW])


def _load_endpoint(monkeypatch):
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
    db_mod.get_conn = lambda: _FakeConn(calls)
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

    import datetime as _dt
    import logging as _logging

    ns = {
        "request": _FakeRequest({}),
        "datetime": _dt.datetime,
        "logger": _logging.getLogger("test"),
        "jsonify": lambda obj, *a: obj,
        "_TRADE_DB_COUNT_CACHE": {},
        "_TRADE_DB_COUNT_TTL": 300,
    }
    exec(compile(seg, "api_trade_database", "exec"), ns)  # noqa: S102 - test harness
    return ns["api_trade_database"], calls, ns


@pytest.fixture()
def endpoint(monkeypatch):
    fn, calls, ns = _load_endpoint(monkeypatch)
    yield fn, calls, ns


def _run(endpoint, args):
    fn, calls, ns = endpoint
    ns["request"] = _FakeRequest(args)
    ns["_TRADE_DB_COUNT_CACHE"] = {}
    calls.clear()
    return fn(), calls


def test_range_adds_cutoff_to_both_queries_with_param_after_season(endpoint):
    _, calls = _run(endpoint, {"season": "2026", "range": "30d"})
    count_q, count_p = calls[0]
    row_q, row_p = calls[1]
    for q in (count_q, row_q):
        assert "t.created_at >= NOW() - (%s * INTERVAL '1 day')" in q
    # No sf/lf/player filters in this run: params are [season, range, ...].
    assert count_p[:2] == [2026, 30]
    assert row_p[:2] == [2026, 30]


def test_range_7d_and_90d_bind_their_days(endpoint):
    for val, days in (("7d", 7), ("90d", 90)):
        _, calls = _run(endpoint, {"season": "2026", "range": val})
        assert calls[0][1][1] == days


def test_default_and_invalid_range_add_no_cutoff(endpoint):
    for args in ({"season": "2026"}, {"season": "2026", "range": "bogus"}):
        _, calls = _run(endpoint, args)
        for q, _p in calls[:2]:
            assert "INTERVAL" not in q
        assert calls[0][1] == [2026]


def test_response_carries_team_pick_fields_and_created_at(endpoint):
    data, _calls = _run(endpoint, {"season": "2026"})
    trade = data["trades"][0]
    assert trade["created_at"] == "2026-10-14T12:00:00"
    player = trade["side_a"][0]
    assert player["team"] == "DET"
    assert player["position"] == "RB"
    pick = trade["side_b"][0]
    assert pick["pick_season"] == 2027
    assert pick["pick_round"] == 1
    assert pick["pick_slot"] is None
