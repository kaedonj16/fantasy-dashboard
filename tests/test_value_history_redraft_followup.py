"""Redraft value history, follow-up surfaces: Beat the Market, the Compare
value-history chart, trade-calculator 7-day deltas, and the weekly EMAIL
digest movers.

Redraft history only accrues from late October 2026 (PR #2845), so every
surface must show the short redraft span honestly: no dynasty numbers
under a redraft label, no loosened thresholds, and an empty redraft
result stays empty. Dynasty behavior is unchanged.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parent.parent

LATEST = date(2026, 10, 20)
DYNASTY_BASELINE = date(2026, 9, 20)
REDRAFT_BASELINE = date(2026, 10, 18)  # redraft tracking is only days old


class _Result:
    def __init__(self, one=None, all_rows=None):
        self._one = one
        self._all = all_rows or []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._all


def _btm_state(**over):
    st = {
        "baseline_sql": [],
        "delta_sql": [],
        "latest_checks": [],
        "redraft_latest": True,
        "delta_rows": [
            {"player_id": "p1", "delta": 10.0},
            {"player_id": "p2", "delta": -6.0},
            {"player_id": "p3", "delta": 0.0},
        ],
    }
    st.update(over)
    return st


class _BTMConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        st = self.state
        if "SELECT 1 AS x" in sql:
            st["latest_checks"].append(sql)
            return _Result(one={"x": 1} if st["redraft_latest"] else None)
        if "MAX(as_of_date)" in sql:
            st["baseline_sql"].append(sql)
            if "redraft" in sql:
                return _Result(one={"d": REDRAFT_BASELINE})
            return _Result(one={"d": DYNASTY_BASELINE})
        st["delta_sql"].append(sql)
        return _Result(all_rows=list(st["delta_rows"]))


def _wire_btm(offline_client, monkeypatch, state, league=None):
    import app as appmod
    import data_building.player_value_history as pvh
    import dashboard_services.db as dbmod

    monkeypatch.setattr(pvh, "init_value_history_db", lambda *a, **k: None)
    monkeypatch.setattr(pvh, "get_latest_snapshot_date", lambda **k: LATEST)
    monkeypatch.setattr(dbmod, "get_conn", lambda *a, **k: _BTMConn(state))
    monkeypatch.setattr(appmod, "get_nfl_state", lambda *a, **k: {"season": "2026"})
    monkeypatch.setattr(appmod, "get_rosters", lambda *a, **k: [
        {"roster_id": 1, "owner_id": "u1", "players": ["p1", "p2"]},
        {"roster_id": 2, "owner_id": "u2", "players": ["p3"]},
    ])
    monkeypatch.setattr(appmod, "get_users", lambda *a, **k: [
        {"user_id": "u1", "display_name": "One", "metadata": {}},
        {"user_id": "u2", "display_name": "Two", "metadata": {}},
    ])
    monkeypatch.setattr(appmod, "load_players_index", lambda *a, **k: {
        "p1": {"pos": "QB", "name": "One QB"},
        "p2": {"pos": "RB", "name": "Two RB"},
        "p3": {"pos": "WR", "name": "Three WR"},
    })
    if league is not None:
        monkeypatch.setattr(appmod, "get_league", lambda *a, **k: league)
    appmod._BEAT_MARKET_CACHE.clear()
    appmod._BEAT_MARKET_FORMAT_CACHE.clear()
    return appmod


_BTM_URL = "/api/beat-the-market?platform=sleeper&league_id=BM1&season=2026&days=30"


# ---------------------------------------------------------------------------
# 1. Beat the Market
# ---------------------------------------------------------------------------

def test_btm_explicit_redraft_reads_redraft_columns(offline_client, monkeypatch):
    state = _btm_state()
    _wire_btm(offline_client, monkeypatch, state)
    r = offline_client.get(_BTM_URL + "&scoring_type=redraft")
    assert r.status_code == 200
    body = r.get_json()
    assert body["scoring_type"] == "redraft"
    # The honest span: redraft baseline is days old, not the 30d dynasty one.
    assert body["baseline_date"] == "2026-10-18"
    assert body["latest_date"] == "2026-10-20"
    assert state["baseline_sql"], "baseline search never ran"
    assert all("redraft_value IS NOT NULL" in s for s in state["baseline_sql"])
    assert state["delta_sql"], "delta query never ran"
    assert "redraft_value" in state["delta_sql"][0]
    assert "COALESCE" not in state["delta_sql"][0]


def test_btm_redraft_without_latest_redraft_data_is_404_not_dynasty(
        offline_client, monkeypatch):
    state = _btm_state(redraft_latest=False)
    _wire_btm(offline_client, monkeypatch, state)
    r = offline_client.get(_BTM_URL + "&scoring_type=redraft")
    assert r.status_code == 404
    assert state["delta_sql"] == []  # never fell back to dynasty deltas


def test_btm_classifies_redraft_league_without_explicit_signal(
        offline_client, monkeypatch):
    state = _btm_state()
    # Sleeper settings.type 0 = redraft.
    _wire_btm(offline_client, monkeypatch, state,
              league={"settings": {"type": 0}, "roster_positions": []})
    r = offline_client.get(_BTM_URL)
    assert r.status_code == 200
    assert r.get_json()["scoring_type"] == "redraft"
    assert "redraft_value" in state["delta_sql"][0]


def test_btm_classifies_dynasty_league_as_dynasty(offline_client, monkeypatch):
    state = _btm_state()
    # Sleeper settings.type 2 = dynasty.
    _wire_btm(offline_client, monkeypatch, state,
              league={"settings": {"type": 2}, "roster_positions": []})
    r = offline_client.get(_BTM_URL)
    assert r.status_code == 200
    body = r.get_json()
    assert body["scoring_type"] == "dynasty"
    assert body["baseline_date"] == "2026-09-20"
    assert "redraft" not in state["delta_sql"][0]


def test_btm_explicit_signal_wins_over_classification(offline_client, monkeypatch):
    state = _btm_state()
    _wire_btm(offline_client, monkeypatch, state,
              league={"settings": {"type": 2}, "roster_positions": []})
    r = offline_client.get(_BTM_URL + "&scoring_type=redraft")
    assert r.status_code == 200
    assert r.get_json()["scoring_type"] == "redraft"


def test_btm_cache_key_separates_formats(offline_client, monkeypatch):
    state = _btm_state()
    _wire_btm(offline_client, monkeypatch, state)
    r_dyn = offline_client.get(_BTM_URL + "&scoring_type=dynasty")
    r_red = offline_client.get(_BTM_URL + "&scoring_type=redraft")
    assert r_dyn.status_code == 200 and r_red.status_code == 200
    assert len(state["delta_sql"]) == 2  # neither served the other's payload
    offline_client.get(_BTM_URL + "&scoring_type=dynasty")
    offline_client.get(_BTM_URL + "&scoring_type=redraft")
    assert len(state["delta_sql"]) == 2  # both warm on repeat


def test_teams_js_sends_mode_toggle_signal():
    src = (ROOT / "static" / "teams.js").read_text()
    assert "_btmFormatSignal" in src
    assert "window.__teamsValueMode" in src
    assert "'&scoring_type=' + _btmFmt" in src
    # Subtitle follows the format the server actually priced.
    assert "data.scoring_type === 'redraft' ? 'redraft' : 'dynasty'" in src


# ---------------------------------------------------------------------------
# 2. Compare value-history chart (frontend selection; the payload field
#    value_history_redraft arrives with the player-details payload)
# ---------------------------------------------------------------------------

def test_compare_chart_selects_series_by_toggle():
    src = (ROOT / "static" / "app.js").read_text()
    assert "function _cmpHistSeries(p)" in src
    assert "_cmpIsRedraft() ? (p.value_history_redraft || []) : (p.value_history || [])" in src
    # Both format labels exist and the chart blocks carry the label class
    # the toggle redraw rewrites.
    assert "'Redraft Value History'" in src
    assert "'Dynasty Value History'" in src
    assert "cmp-vh-label" in src
    # The toggle click handler redraws the charts, not just the tables.
    assert "window._cmpRedrawValueCharts()" in src
    assert "_cmpRedrawValueCharts = function" in src
    # Two-player chart block is actually interpolated into the body now.
    assert "${_chartBlock}" in src
    # Triple chart plots the active format's series too.
    assert "(_cmpHistSeries(p) || []).map(h => h.as_of_date || h.date)" in src
    # Tier-average reference lines never use a dynasty value in redraft mode.
    assert "p.stats.redraft_value_1qb" in src


# ---------------------------------------------------------------------------
# 3. Trade calculator deltas
# ---------------------------------------------------------------------------

def test_trade_response_passes_scoring_type_to_movers(monkeypatch):
    import app as appmod
    import data_building.player_value_history as dpvh

    calls = []

    def fake_movers(**kw):
        calls.append(kw)
        return {"risers": [{"player_id": "1", "delta": 12.5}], "fallers": []}

    monkeypatch.setattr(dpvh, "get_top_movers", fake_movers)
    monkeypatch.setattr(appmod, "_compute_player_indicators", lambda *a, **k: {})
    appmod._LP_BOARD_JSON_CACHE.clear()

    payload = {"players": [{
        "id": "1", "name": "Player 1", "team": "KC", "position": "WR",
        "value": 500.0, "sf_value": 450.0, "search_name": "player 1",
    }]}
    with appmod.app.test_request_context():
        resp = appmod._trade_league_players_response(
            payload, overlay_key=("k-redraft",), is_sf=False,
            league_type="1qb", league_size=10, scoring_type="redraft")
        body = resp.get_data(as_text=True)
        resp2 = appmod._trade_league_players_response(
            payload, overlay_key=("k-dynasty",), is_sf=False,
            league_type="1qb", league_size=10, scoring_type="dynasty")
        body2 = resp2.get_data(as_text=True)
    assert [c.get("scoring_type") for c in calls] == ["redraft", "dynasty"]
    assert "12.5" in body and "12.5" in body2


def test_trade_board_cache_key_separates_formats(monkeypatch):
    import app as appmod
    import dashboard_services.adp_service as adp

    monkeypatch.setattr(appmod, "get_model_value_table_cached", lambda *a, **k: [])
    monkeypatch.setattr(adp, "global_adp_snapshot_signature", lambda *a, **k: 0.0)
    k_red = appmod._lp_overlay_cache_key(False, {"rec": 1.0}, 2026, "redraft")
    k_dyn = appmod._lp_overlay_cache_key(False, {"rec": 1.0}, 2026, "dynasty")
    assert k_red != k_dyn
    # api_league_players derives the movers format from the request format.
    src = Path(appmod.__file__).read_text()
    assert 'scoring_type=("redraft" if _mi_scoring == "redraft" else "dynasty")' in src


def test_player_deltas_scoring_type_passthrough(offline_client, monkeypatch):
    import app as appmod

    calls = []

    def fake_movers(**kw):
        calls.append(kw)
        return {"risers": [{"player_id": "9", "delta": 3.0}], "fallers": []}

    monkeypatch.setattr(appmod, "get_top_movers", fake_movers)
    r = offline_client.get("/api/player-deltas?scoring_type=redraft")
    assert r.status_code == 200
    assert r.get_json() == {"9": 3.0}
    assert calls[-1].get("scoring_type") == "redraft"

    r = offline_client.get("/api/player-deltas?scoring_type=bogus")
    assert r.status_code == 200
    assert calls[-1].get("scoring_type") == "dynasty"


# ---------------------------------------------------------------------------
# 4. Weekly EMAIL digest movers
# ---------------------------------------------------------------------------

def test_digest_movers_for_selects_by_league_format():
    from utils.digest import DigestRunCache

    cache = DigestRunCache()
    cache.movers_1qb = {"risers": [{"player_id": "D1", "delta": 50.0}], "fallers": []}
    cache.movers_sf = {"risers": [{"player_id": "DS", "delta": 55.0}], "fallers": []}
    cache.movers_redraft_1qb = {"risers": [{"player_id": "R1", "delta": 60.0}], "fallers": []}
    cache.movers_redraft_sf = {"risers": [{"player_id": "RS", "delta": 65.0}], "fallers": []}

    assert cache.movers_for(is_superflex=False, fmt={"is_redraft": True}) is cache.movers_redraft_1qb
    assert cache.movers_for(is_superflex=True, fmt={"is_keeper": True}) is cache.movers_redraft_sf
    assert cache.movers_for(is_superflex=False, fmt={"is_dynasty": True}) is cache.movers_1qb
    # No fmt (legacy callers) keeps the historical dynasty behavior.
    assert cache.movers_for(is_superflex=True) is cache.movers_sf


def test_digest_redraft_movers_empty_stays_empty():
    from utils.digest import DigestRunCache

    cache = DigestRunCache()
    cache.movers_1qb = {"risers": [{"player_id": "D1", "delta": 50.0}], "fallers": []}
    # Redraft history too young: no redraft movers loaded. A redraft
    # league's digest gets NO movers, never the dynasty board.
    assert cache.movers_for(is_superflex=False, fmt={"is_redraft": True}) == {}


def test_digest_load_shared_loads_redraft_movers(monkeypatch):
    import dashboard_services.api as api
    import dashboard_services.player_value_history as wpvh
    import utils.data_cache as dc
    from utils.digest import DigestRunCache

    calls = []

    def fake_movers(**kw):
        calls.append(kw)
        if kw.get("scoring_type") == "redraft":
            return {"risers": [{"player_id": "R", "delta": 5.0}], "fallers": []}
        return {"risers": [{"player_id": "D", "delta": 5.0}], "fallers": []}

    monkeypatch.setattr(wpvh, "get_top_movers", fake_movers)
    monkeypatch.setattr(dc, "load_players_index", lambda *a, **k: {})
    monkeypatch.setattr(dc, "load_model_value_table", lambda *a, **k: [])
    monkeypatch.setattr(api, "get_nfl_state", lambda *a, **k: {})
    monkeypatch.setattr(api, "get_nfl_players", lambda *a, **k: {})
    monkeypatch.setattr(DigestRunCache, "_load_schedule_and_proj", lambda self: None)
    monkeypatch.setattr(DigestRunCache, "_load_breakouts", lambda self: None)

    cache = DigestRunCache()
    cache.load_shared()
    redraft_calls = [c for c in calls if c.get("scoring_type") == "redraft"]
    assert {c.get("league_type") for c in redraft_calls} == {"1qb", "sf"}
    assert cache.movers_redraft_1qb["risers"][0]["player_id"] == "R"
    assert cache.movers_1qb["risers"][0]["player_id"] == "D"


def test_weekly_email_callers_pass_league_format():
    src = (ROOT / "utils" / "weekly_email.py").read_text()
    assert src.count("cache.movers_for(is_superflex=bool(fmt.get(\"is_superflex\")), fmt=fmt)") == 2
