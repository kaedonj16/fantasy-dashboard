"""Redraft value history surfaces: the player modal chart and Top Movers.

Redraft history has been stored since PR #2845 (late October 2026). These
tests pin the surfacing contract:

- app._cached_value_history passes scoring_type through and caches the
  dynasty and redraft series separately;
- the player-details deferred build serves value_history_redraft next to
  the dynasty value_history: NOT run through the dynasty FC/DP ratio
  rescale (stored redraft values are already the served scale), but with
  the same TE premium the redraft hero values get;
- /api/player-value-history?scoring_type= passes the format through, with
  anything else falling back to dynasty;
- /top-movers?format=redraft and /api/value-movers?scoring_type=redraft
  read redraft movers anchored on a REDRAFT displayed-value map;
- get_top_movers' displayed-value rescale is a no-op for redraft when the
  displayed value equals the stored latest value (the snapshot stores the
  served solve verbatim, from the same player_values columns the model
  table renders), so redraft numbers are never distorted by it;
- the modal JS renders the ACTIVE format's series with explicit labels,
  keeps __pmVH (Trades tab) on dynasty, and re-renders idempotently.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
MODAL_JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
TRADE_PAGE_PY = (ROOT / "dashboard_services" / "pages"
                 / "trade_calculator_page.py").read_text(encoding="utf-8")


# ── _cached_value_history: scoring_type passthrough + separate caches ──────

def test_cached_value_history_scoring_type(monkeypatch):
    import app as appmod

    appmod._VALUE_HISTORY_CACHE.clear()
    monkeypatch.setattr(
        "dashboard_services.player_details_store.get_json",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "dashboard_services.player_details_store.set_json",
        lambda *a, **k: None)

    calls = []

    def _fake(player_id, *, days, league_type, league_size,
              scoring_type="dynasty"):
        calls.append(scoring_type)
        return [{"as_of_date": "2026-10-29", "value": 1.0,
                 "fmt": scoring_type}]

    monkeypatch.setattr(appmod, "get_player_value_history", _fake)

    dyn = appmod._cached_value_history(
        "P1", days=365, league_type="1qb", league_size=10)
    assert dyn[0]["fmt"] == "dynasty"
    # Default format is dynasty and is served from cache on the second call.
    dyn2 = appmod._cached_value_history(
        "P1", days=365, league_type="1qb", league_size=10)
    assert dyn2 == dyn and calls == ["dynasty"]

    rd = appmod._cached_value_history(
        "P1", days=365, league_type="1qb", league_size=10,
        scoring_type="redraft")
    assert rd[0]["fmt"] == "redraft"
    assert calls == ["dynasty", "redraft"], (
        "redraft must not be served the cached dynasty rows")
    # And the redraft entry is cached under its own key too.
    appmod._cached_value_history(
        "P1", days=365, league_type="1qb", league_size=10,
        scoring_type="redraft")
    assert calls == ["dynasty", "redraft"]
    appmod._VALUE_HISTORY_CACHE.clear()


# ── Deferred subset shape ──────────────────────────────────────────────────

def test_deferred_subset_carries_redraft_history():
    import app as appmod

    resp = {
        "player_id": "P1",
        "value_history": [{"value": 1.0}],
        "value_history_redraft": [{"value": 2.0}],
        "stats": {},
    }
    subset = appmod._player_details_deferred_subset(resp)
    assert subset["value_history"] == [{"value": 1.0}]
    assert subset["value_history_redraft"] == [{"value": 2.0}]
    # Missing key degrades to an empty series, never dynasty data.
    subset2 = appmod._player_details_deferred_subset({"player_id": "P2"})
    assert subset2["value_history_redraft"] == []


# ── _displayed_value_map: redraft variant ──────────────────────────────────

_VALUE_TABLE = [
    {"id": "1", "value": 500.0, "sf_value": 600.0,
     "redraft_value_1qb": 700.0, "redraft_value_sf": 650.0},
    {"id": "2", "value": 300.0, "sf_value": 350.0},  # no redraft fields
    {"id": "3", "value": "x", "sf_value": None,
     "redraft_value_1qb": None, "redraft_value_sf": "y"},
]


def test_displayed_value_map_redraft_reads_redraft_fields(monkeypatch):
    import app as appmod

    monkeypatch.setattr(appmod, "get_model_value_table_cached",
                        lambda: list(_VALUE_TABLE))
    assert appmod._displayed_value_map("1qb") == {
        "1": 500.0, "2": 300.0, "3": 0.0}
    assert appmod._displayed_value_map("sf") == {
        "1": 600.0, "2": 350.0, "3": 0.0}
    # Redraft: the redraft columns; a row without them maps to 0.0, exactly
    # like a dynasty row with no value (never the dynasty value).
    assert appmod._displayed_value_map(
        "1qb", scoring_type="redraft") == {"1": 700.0, "2": 0.0, "3": 0.0}
    assert appmod._displayed_value_map(
        "sf", scoring_type="redraft") == {"1": 650.0, "2": 0.0, "3": 0.0}


# ── /api/value-movers: scoring_type passthrough ────────────────────────────

def test_api_value_movers_scoring_type_passthrough(monkeypatch):
    import app as appmod

    monkeypatch.setattr(appmod, "get_model_value_table_cached",
                        lambda: list(_VALUE_TABLE))
    captured = {}

    def _movers(**kwargs):
        captured.update(kwargs)
        return {"used_days": kwargs["days"], "risers": [], "fallers": []}

    monkeypatch.setattr(appmod, "get_top_movers", _movers)
    client = appmod.app.test_client()

    resp = client.get("/api/value-movers?scoring_type=redraft&league_type=sf")
    assert resp.status_code == 200
    assert captured["scoring_type"] == "redraft"
    assert captured["current_values"]["1"] == 650.0  # redraft SF map

    resp = client.get("/api/value-movers?league_type=sf")
    assert resp.status_code == 200
    assert captured["scoring_type"] == "dynasty"
    assert captured["current_values"]["1"] == 600.0  # dynasty SF map

    resp = client.get("/api/value-movers?scoring_type=bogus")
    assert resp.status_code == 200
    assert captured["scoring_type"] == "dynasty"


# ── /api/player-value-history: scoring_type passthrough ────────────────────

def test_player_value_history_api_scoring_type(monkeypatch):
    from flask import Flask

    import routes.players_bp as pb

    calls = []

    def _fake(player_id, **kwargs):
        calls.append(kwargs)
        return [{"as_of_date": "2026-10-29", "value": 1.0}]

    monkeypatch.setattr(pb, "get_player_value_history", _fake)
    app = Flask(__name__)
    app.register_blueprint(pb.players_bp)
    client = app.test_client()

    body = client.get(
        "/api/player-value-history/P1?scoring_type=redraft").get_json()
    assert body["scoring_type"] == "redraft"
    assert calls[-1]["scoring_type"] == "redraft"

    body = client.get("/api/player-value-history/P1").get_json()
    assert body["scoring_type"] == "dynasty"
    assert calls[-1]["scoring_type"] == "dynasty"

    body = client.get(
        "/api/player-value-history/P1?scoring_type=nonsense").get_json()
    assert body["scoring_type"] == "dynasty"
    assert calls[-1]["scoring_type"] == "dynasty"


# ── /top-movers page: format parsing + wiring (source + route) ─────────────

def test_top_movers_page_route_redraft(monkeypatch):
    import routes.seo_pages_bp as sp
    from data_building import player_value_history as db_pvh
    from flask import Flask

    captured = {}

    def _fake_movers(**kwargs):
        captured.update(kwargs)
        return {"risers": [], "fallers": [],
                "latest_date": "2026-10-29", "comparison_date": "2026-10-22"}

    monkeypatch.setattr(db_pvh, "get_top_movers", _fake_movers)
    monkeypatch.setattr(sp, "_displayed_value_map",
                        lambda lt, scoring_type="dynasty": {
                            "called_with": (lt, scoring_type)})
    monkeypatch.setattr(sp, "render_page",
                        lambda title, *a, **k: (title, a, k))
    stub_app = SimpleNamespace(_session_signed_in=lambda: False)
    monkeypatch.setitem(sys.modules, "app", stub_app)

    app = Flask(__name__)
    app.register_blueprint(sp.seo_pages_bp)

    # render_page is stubbed to return a tuple (title, args, kwargs), so
    # call the view through a request context instead of the test client.
    with app.test_request_context("/top-movers?format=redraft&days=30"):
        out = sp.top_movers_page()
    title, _args, kwargs = out
    assert captured["scoring_type"] == "redraft"
    assert captured["days"] == 30
    assert captured["current_values"] == {
        "called_with": ("1qb", "redraft")}
    assert title.startswith("Redraft Top Movers:")
    assert kwargs["description"].startswith("Redraft fantasy football")
    body = _args[2]
    assert "Redraft Fantasy Football Top Movers" in body

    captured.clear()
    with app.test_request_context("/top-movers?days=7"):
        out = sp.top_movers_page()
    title, _args, kwargs = out
    assert captured["scoring_type"] == "dynasty"
    assert captured["current_values"] == {
        "called_with": ("1qb", "dynasty")}
    assert title.startswith("Dynasty Top Movers:")
    assert kwargs["description"].startswith("Dynasty fantasy football")
    assert "Dynasty Fantasy Football Top Movers" in _args[2]

    captured.clear()
    with app.test_request_context("/top-movers?format=banana"):
        sp.top_movers_page()
    assert captured["scoring_type"] == "dynasty"


def test_top_movers_page_source_wiring():
    src = (ROOT / "routes" / "seo_pages_bp.py").read_text(encoding="utf-8")
    seg = src[src.index("def top_movers_page():"):]
    seg = seg[: seg.index("def _compare_name_for_id")]
    assert 'request.args.get("format", "dynasty")' in seg
    assert 'scoring_type=scoring_format' in seg
    assert 'scoring_format=scoring_format' in seg


# ── build_risers_fallers_body: format rendering ────────────────────────────

_MOVERS = {
    "risers": [{"name": "Riser One", "position": "RB", "team": "BUF",
                "new_value": 700.0, "old_value": 640.0, "delta": 60.0,
                "player_id": "42"}],
    "fallers": [],
    "latest_date": "2026-10-29", "comparison_date": "2026-10-22",
}


def test_risers_fallers_body_redraft():
    from dashboard_services.pages.dynasty_pages import (
        build_risers_fallers_body,
    )

    body = build_risers_fallers_body(
        dict(_MOVERS), as_of_date="October 29, 2026", days=30,
        scoring_format="redraft")
    assert "Redraft Fantasy Football Top Movers" in body
    assert "Biggest redraft trade value movers" in body
    # Timeframe links preserve days + format; the format toggle preserves
    # the active days in both directions.
    assert 'href="/top-movers?days=7&format=redraft"' in body
    assert 'href="/top-movers?days=90&format=redraft"' in body
    assert 'href="/top-movers?days=30&format=redraft"' in body
    assert 'href="/top-movers?days=30">Dynasty</a>' in body
    assert "Why Do Redraft Values Change?" in body
    assert "Why Do Dynasty Values Change?" not in body
    # The dynasty Tier-3 editorial (age curves, future seasons) is swapped
    # for the redraft block, not word-swapped into the page.
    assert 'data-tier3="top-movers-editorial"' not in body
    assert 'data-tier3="top-movers-editorial-redraft"' in body
    assert "—" not in body


def test_risers_fallers_body_dynasty_unchanged():
    from dashboard_services.pages.dynasty_pages import (
        build_risers_fallers_body,
    )

    body = build_risers_fallers_body(
        dict(_MOVERS), as_of_date="October 29, 2026", days=30)
    assert "Dynasty Fantasy Football Top Movers" in body
    assert "Biggest dynasty trade value movers" in body
    assert 'href="/top-movers?days=7"' in body
    assert 'href="/top-movers?days=30&format=redraft"' in body
    assert "Why Do Dynasty Values Change?" in body
    assert 'data-tier3="top-movers-editorial"' in body


# ── get_top_movers redraft rescale: verified a no-op at the invariant ──────
#
# The movers rescale anchors shown values to the DISPLAYED value
# (disp / raw_new). For redraft that ratio is 1.0 by construction: the
# history snapshot stores the WLS solve exactly as served, from the same
# player_values columns (redraft_value_1qb / redraft_value_sf) the model
# table's redraft fields render. The double below pins that: with the
# displayed value equal to the stored latest value, old/new/delta pass
# through untouched, and min_current_value filters on the redraft map.

class _MoversCur:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _MoversCursor:
    def __init__(self, handler):
        self._handler = handler
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self._rows = self._handler(sql, params)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _MoversConn:
    def __init__(self, handler):
        self._handler = handler

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        return _MoversCur(self._handler(sql, params))

    def cursor(self):
        return _MoversCursor(self._handler)


_RD_MOVER_ROW = {"player_id": "B", "name": "Redraft Mover", "position": "RB",
                 "team": "BUF", "old_value": 700.0, "new_value": 640.0,
                 "delta": -60.0}


def _rd_movers_handler(sql, params):
    if "AS latest_date" in sql:
        return [{"latest_date": date(2026, 10, 27)}]
    if "AS comparison_date" in sql:
        return [{"comparison_date": date(2026, 10, 20)}]
    if "player_count" in sql:
        return [{"player_count": 500}]
    if "WITH latest_rows AS" in sql:
        return [dict(_RD_MOVER_ROW)]
    return []


def test_top_movers_redraft_rescale_is_noop_at_served_scale(monkeypatch):
    from data_building import player_value_history as H

    monkeypatch.setattr(H, "init_value_history_db", lambda: None)
    monkeypatch.setattr(H, "get_conn",
                        lambda: _MoversConn(_rd_movers_handler))
    H._latest_snapshot_cache.clear()
    H._TOP_MOVERS_CACHE.clear()

    out = H.get_top_movers(days=7, limit=50, scoring_type="redraft",
                           current_values={"B": 640.0})
    row = (out["fallers"] + out["risers"])[0]
    assert (row["old_value"], row["new_value"], row["delta"]) == (
        700.0, 640.0, -60.0)

    H._TOP_MOVERS_CACHE.clear()
    out = H.get_top_movers(days=7, limit=50, scoring_type="redraft",
                           min_current_value=20.0,
                           current_values={"B": 10.0})
    assert not (out["fallers"] or out["risers"]), (
        "a player under the floor on the REDRAFT map must be filtered")


# ── Player-details deferred build: redraft series handling ─────────────────
#
# Fixture mirrors tests/test_player_details_speed_v2.py's proven stub set,
# with a TE in a full TE-premium league and a model row whose corrected
# dynasty value (9999) is far from the raw dynasty history endpoint
# (5000): the dynasty series must come back rescaled to ~9999 (then TE
# x1.2), the redraft series must keep its stored scale (700) with only
# the TE premium applied (840) and its delta scaled with it (100 -> 120).

_LEAGUE = "1312067280816832512"
_PID = "9998"   # TE
_PID2 = "9999"  # QB

_PLAYERS = {
    _PID: {
        "player_id": _PID, "name": "Test Tightend", "full_name": "Test Tightend",
        "first_name": "Test", "last_name": "Tightend",
        "pos": "TE", "position": "TE", "team": "KC",
        "status": "Active", "injury_status": None, "years_exp": 6,
        "age": 30, "number": 87, "depth_chart_order": 1,
    },
    _PID2: {
        "player_id": _PID2, "name": "Bench Quarterback",
        "full_name": "Bench Quarterback",
        "first_name": "Bench", "last_name": "Quarterback",
        "pos": "QB", "position": "QB", "team": "BUF",
        "status": "Active", "injury_status": None, "years_exp": 4,
        "age": 26, "number": 9, "depth_chart_order": 1,
    },
}

_LEAGUE_DICT = {
    "league_id": _LEAGUE, "platform": "sleeper", "settings": {},
    "scoring_settings": {}, "name": "VH Test League", "season": "2026",
}

_SCORING = {"rec": 1.0, "bonus_rec_te": 1.0}  # full TE premium -> x1.2 for TEs

_POLICY = SimpleNamespace(completed_weeks=(1, 2), current_week=3)


def _dyn_rows():
    return [
        {"as_of_date": "2026-10-28", "value": 4000.0, "value_1qb": 4000.0,
         "value_sf": 3800.0, "delta_from_prev": None},
        {"as_of_date": "2026-10-29", "value": 5000.0, "value_1qb": 5000.0,
         "value_sf": 4800.0, "delta_from_prev": 1000.0},
    ]


def _rd_rows():
    return [
        {"as_of_date": "2026-10-28", "value": 600.0, "value_1qb": 600.0,
         "value_sf": 550.0, "delta_from_prev": None},
        {"as_of_date": "2026-10-29", "value": 700.0, "value_1qb": 700.0,
         "value_sf": 650.0, "delta_from_prev": 100.0},
    ]


@pytest.fixture()
def pd_env(monkeypatch):
    import app as appmod

    appmod.app.config.update(TESTING=True)
    vh_calls = []

    for cache in (appmod._PLAYER_DETAILS_CACHE, appmod._VALUE_HISTORY_CACHE,
                  appmod._ELIGIBILITY_CACHE, appmod._COMPLETED_WEEKS_CACHE,
                  appmod._BREAKOUT_FLAGS_CACHE,
                  appmod._WEEKLY_PTS_CACHE, appmod._SEASON_RANK_CACHE):
        cache.clear()
    try:
        from dashboard_services.rookie_api import _cache as _rc
        _rc.clear()
    except Exception:
        pass

    monkeypatch.setattr(appmod, "_player_details_cache_active", lambda: True)
    monkeypatch.setattr(
        "dashboard_services.player_details_store.get_json",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "dashboard_services.player_details_store.set_json",
        lambda *a, **k: None)

    def _players():
        return dict(_PLAYERS)

    monkeypatch.setattr("utils.data_cache.load_relevant_index", _players)
    monkeypatch.setattr("utils.utils.load_relevant_index", _players)
    monkeypatch.setattr(appmod, "load_players_index", lambda: {})
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda: dict(_PLAYERS))
    monkeypatch.setattr(appmod, "get_players_global", lambda: dict(_PLAYERS))
    monkeypatch.setattr(appmod, "get_normalized_scoring_settings",
                        lambda platform: dict(_SCORING))
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: {
                            "rosters": [{
                                "roster_id": 3, "owner_id": "u1",
                                "players": [_PID],
                            }],
                            "users": [{
                                "user_id": "u1", "display_name": "Manager One",
                                "metadata": {"team_name": "Team Alpha"},
                            }],
                            "scoring_settings": dict(_SCORING),
                            "league": dict(_LEAGUE_DICT),
                            "league_settings": {}, "settings": {},
                        })
    monkeypatch.setattr(
        appmod, "get_model_value_table_cached",
        lambda: [
            {"id": _PID, "name": "Test Tightend", "position": "TE",
             "team": "KC", "value": 9999.0, "sf_value": 9999.0,
             "redraft_value_1qb": 700.0, "redraft_value_sf": 650.0},
            {"id": _PID2, "name": "Bench Quarterback", "position": "QB",
             "team": "BUF", "value": 3000.0, "sf_value": 3600.0},
        ])
    monkeypatch.setattr(appmod, "get_league",
                        lambda league_id, **kw: dict(_LEAGUE_DICT))
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_league",
                        lambda league_id, **kw: dict(_LEAGUE_DICT))
    monkeypatch.setattr(appmod, "_load_usage_rows_cached", lambda season: [])
    monkeypatch.setattr(appmod, "_oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_team_bye_map", lambda season: {})
    monkeypatch.setattr(appmod, "_startsit_compare_extras", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "start_score_league_index", lambda *a, **k: 55)
    monkeypatch.setattr(
        "dashboard_services.adp_service.fetch_sleeper_adp", lambda season: {})
    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"week": 7, "season": 2026, "season_type": "regular"})
    monkeypatch.setattr(
        "utils.utils.load_week_projection", lambda season, week: {_PID: {}})
    monkeypatch.setattr(
        "utils.fantasy_scoring.weekly_projection_points", lambda *a, **k: 20.0)
    monkeypatch.setattr(
        "utils.utils.load_week_sched",
        lambda season, week: [{"home": "KC", "away": "BUF",
                               "gameDate": "20261011"}])
    monkeypatch.setattr(
        "utils.game_conditions.build_week_conditions", lambda *a, **k: {})
    monkeypatch.setattr(
        "utils.start_sit_score.bottom_teams_by_implied_total",
        lambda *a, **k: set())
    monkeypatch.setattr(
        "utils.start_sit_context.build_absence_index", lambda *a, **k: {})
    monkeypatch.setattr(
        "utils.start_sit_context.absence_notes",
        lambda *a, **k: {"teammates": [], "opponents": []})
    monkeypatch.setattr(
        "utils.start_sit_context.productive_pids_from_weekly_points",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "utils.start_sit_context.starting_lineman_pids", lambda *a, **k: None)
    monkeypatch.setattr(
        "utils.start_sit_score.compute_start_score",
        lambda *a, **k: (55.0, {"projection": 1.0}, None))
    monkeypatch.setattr(
        "dashboard_services.player_league_trades.get_player_league_trades",
        lambda *a, **k: {"trades": []})
    monkeypatch.setattr(
        "dashboard_services.player_league_trades.get_player_acquisition_events",
        lambda *a, **k: {"events": []})

    def _vh(player_id, *, days, league_type, league_size,
            scoring_type="dynasty"):
        vh_calls.append({"scoring_type": scoring_type, "days": days})
        return _rd_rows() if scoring_type == "redraft" else _dyn_rows()

    monkeypatch.setattr(appmod, "get_player_value_history", _vh)
    monkeypatch.setattr(appmod, "_player_nfl_eligibility",
                        lambda player_id, season: (True, True))

    import data_building.rookie_pipeline.pipeline as pipeline
    monkeypatch.setattr(pipeline, "get_active_rookie_class", lambda: 2026)
    monkeypatch.setattr(pipeline, "get_rookie_rankings_from_db",
                        lambda year: [])
    monkeypatch.setattr(appmod, "_playoff_sos_for",
                        lambda season, team, pos, weeks=None: (5, 32, 0.42))
    monkeypatch.setattr(appmod, "_playoff_sos_weeks_for",
                        lambda *a, **k: [{"week": 15, "opp": "MIA",
                                           "opp_rank": 3, "opp_fpts": 11.0,
                                           "at": False}])
    monkeypatch.setattr(appmod, "_playoff_fpts_max", lambda *a, **k: None)

    import dashboard_services.breakout_api as breakout_api
    monkeypatch.setattr(breakout_api, "_resolve_bo_season", lambda s: s)
    monkeypatch.setattr(breakout_api, "get_breakout_board_candidates",
                        lambda season: {"candidates": [{"player_id": _PID}]})

    import data_building.breakout_engine.weekly_grading as weekly_grading
    monkeypatch.setattr(weekly_grading, "surfaced_player_ids_by_week",
                        lambda season: {})

    import utils.season_qualification as sq
    monkeypatch.setattr(sq, "qualification_policy", lambda season: _POLICY)
    monkeypatch.setattr(
        appmod, "_load_season_weekly_points",
        lambda season, scoring_settings, weeks=None: {
            _PID: [10.0, 20.0], _PID2: [30.0, 10.0]})

    with appmod.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["account_id"] = 7
            sess["user_id"] = 7
        yield SimpleNamespace(appmod=appmod, client=client, vh_calls=vh_calls)
    appmod.app.config.update(TESTING=False)


def _core_url(pid=_PID):
    return (f"/api/player-details/{pid}?platform=sleeper"
            f"&league_id={_LEAGUE}&season=2026&slim=1")


def test_core_payload_has_empty_redraft_history(pd_env):
    resp = pd_env.client.get(_core_url())
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["value_history"] == []
    assert body["value_history_redraft"] == []
    assert pd_env.vh_calls == [], "core must not query value history"


def test_deferred_redraft_history_te_scaled_not_dynasty_rescaled(pd_env):
    resp = pd_env.client.get(_core_url() + "&deferred=1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["deferred"] is True
    assert pd_env.vh_calls == [
        {"scoring_type": "dynasty", "days": 365},
        {"scoring_type": "redraft", "days": 365},
    ]
    # Dynasty: rescaled 5000 -> 9999 toward the corrected value, then the
    # TE premium (x1.2).
    dyn = body["value_history"]
    assert dyn[-1]["value_1qb"] == 11998.8
    # Redraft: stored scale kept (700), TE premium only (x1.2), and the
    # delta scaled by the same multiplier.
    rd = body["value_history_redraft"]
    assert rd[0]["value_1qb"] == 720.0
    assert rd[-1]["value_1qb"] == 840.0
    assert rd[-1]["value_sf"] == 780.0
    assert rd[-1]["value"] == 840.0
    assert rd[-1]["delta_from_prev"] == 120.0


# ── Modal + trade panel JS pins ────────────────────────────────────────────

def test_modal_js_renders_active_format_series():
    # Series selection is driven by the toggle state mirrored on data.
    assert "function _pmActiveValueHistory(data)" in MODAL_JS
    assert "data.value_history_redraft" in MODAL_JS
    assert "data.pmScoringType = pmScoringType" in MODAL_JS
    # Both labels are explicit; the old redraft-shows-dynasty label is gone.
    assert "'Redraft Value History'" in MODAL_JS
    assert "'Dynasty Value History'" in MODAL_JS
    assert "pmScoringType === 'redraft' ? 'Dynasty Value History'" not in MODAL_JS
    # Toggle click re-renders the chart for the new format.
    assert "applyValueHistory" in MODAL_JS
    assert "applyHero();\n              applyValueHistory();" in MODAL_JS
    # Re-render dedupes the range bar and resets the summary.
    assert "querySelectorAll('.pvc-range-bar')" in MODAL_JS
    assert "Redraft history is still building" in MODAL_JS
    assert "tracked since late October" in MODAL_JS


def test_modal_js_deferred_patch_and_pmvh_stays_dynasty():
    assert "data.value_history_redraft = dfd.value_history_redraft" in MODAL_JS
    assert "_vhAny" in MODAL_JS
    assert "dfd.value_history_redraft" in MODAL_JS
    # __pmVH (Trades tab value-change-since-trade) is fed the dynasty
    # series in both the initial stash and the deferred patch.
    assert "hist: Array.isArray(data.value_history)" in MODAL_JS
    assert "hist: dfd.value_history_redraft" not in MODAL_JS
    assert "hist: Array.isArray(dfd.value_history)" in MODAL_JS
    # The block builder labels by the active format.
    assert ("pmScoringType === 'redraft' ? 'Redraft Value History' "
            ": 'Dynasty Value History'") in MODAL_JS


def test_trade_panel_format_toggle_js():
    assert "window.changeMoversFormat = function(format)" in APP_JS
    assert "currentMoversFormat" in APP_JS
    assert "scoring_type=${currentMoversFormat}" in APP_JS
    assert "&scoring_type=" in APP_JS
    # Day-button active toggling must not clear the format buttons (they
    # share the .otc-day-filter styling), so it scopes to [data-days].
    assert ".otc-day-filter[data-days]" in APP_JS
    # Panel HTML carries the toggle, wired to changeMoversFormat.
    assert 'data-movers-format="dynasty"' in TRADE_PAGE_PY
    assert 'data-movers-format="redraft"' in TRADE_PAGE_PY
    assert "changeMoversFormat('redraft')" in TRADE_PAGE_PY


def test_app_py_source_wiring():
    seg = APP_PY[APP_PY.index("def _cached_value_history"):]
    seg = seg[: seg.index("def api_player_indicators")]
    assert 'scoring_type: str = "dynasty"' in seg
    assert '_vh_kwargs = {"scoring_type": "redraft"}' in seg
    seg = APP_PY[APP_PY.index("def api_value_movers"):]
    seg = seg[: seg.index("def api_player_deltas")]
    assert 'request.args.get("scoring_type", "dynasty")' in seg
    assert "scoring_type=scoring_type" in seg


# ── SW bump + changelog (via the canonical pin helpers) ────────────────────

def test_sw_and_changelog_for_this_change():
    from pin_helpers import assert_sw_cache_current, changelog_entry

    assert_sw_cache_current()
    entry = changelog_entry(
        "value history chart follows the Dynasty/Redraft toggle",
        tag="update", link="/top-movers")
    assert "—" not in entry["text"]
