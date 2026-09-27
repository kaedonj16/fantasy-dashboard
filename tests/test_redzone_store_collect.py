"""_redzone_collect must prefer the server-side play store over live fetching.

Contracts:
  * A store hit assigns the stored plays directly and never calls
    _rz_fetch_alt_pbp_plays (no upstream PBP fetch on page load / poll).
  * The store hit fetches only the plain boxscore (stat lines), never the
    experimental Tank01 PBP payload.
  * A missing/failed store read falls through to the existing live-fetch
    path unchanged.
"""
import sys
import types

import pytest

# _redzone_collect lives in app.py; the lightweight CI job installs pytest
# only, so follow the suite's dependency-skip convention.
pytest.importorskip("flask")
pytest.importorskip("pandas")

GID = "20260927_KC@BUF"
PID = "9999"

STORED = [
    {
        "play_id": "p1", "seq": 1, "pid": PID, "is_td": False,
        "play_text": "P.Mahomes pass short left to T.Kelce for 8 yds",
    }
]
LIVE_FETCHED = [
    {
        "play_id": "p9", "seq": 9, "pid": PID, "is_td": False,
        "play_text": "live fetch play",
    }
]


def _install_stubs(monkeypatch, store_plays, store_raises=False):
    import app

    players = {PID: {"full_name": "Test Player", "position": "QB", "team": "KC"}}
    team_game = {
        "KC": {
            "gameID": GID, "gameStatusCode": "1", "gameStatus": "live",
            "home": "KC", "away": "BUF",
        }
    }

    api = types.ModuleType("dashboard_services.api")
    api.get_nfl_scores_for_date = lambda *a, **k: {}
    api.build_team_game_lookup = lambda body: dict(team_game)
    api.get_nfl_players = lambda *a, **k: dict(players)
    api.get_normalized_scoring_settings = lambda *a, **k: {}
    api.get_league = lambda *a, **k: {}
    papi = types.ModuleType("dashboard_services.platform_api")
    papi.get_matchups = lambda *a, **k: [{"players": [PID], "starters": []}]
    papi.get_rosters = lambda *a, **k: []
    papi.get_users = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "dashboard_services.api", api)
    monkeypatch.setitem(sys.modules, "dashboard_services.platform_api", papi)

    monkeypatch.setattr(app, "sync_league_globals", lambda *a, **k: None)

    store_mod = types.ModuleType("utils.redzone_store")
    if store_raises:
        def _boom(season, gids):
            raise RuntimeError("db down")

        store_mod.get_plays = _boom
    else:
        store_mod.get_plays = lambda season, gids: dict(store_plays or {})
    monkeypatch.setitem(sys.modules, "utils.redzone_store", store_mod)

    calls = {"box": [], "pbp": []}

    def fake_boxscore(gid, play_by_play=False, ttl=None):
        calls["box"].append({"gid": gid, "play_by_play": play_by_play})
        # Plain boxscore with aggregate stat lines, no PBP keys.
        return {"playerStats": {"x": 1}, "teamStats": {}}

    def fake_pbp(gid, **kwargs):
        calls["pbp"].append(gid)
        return list(LIVE_FETCHED)

    monkeypatch.setattr(app, "_redzone_boxscore", fake_boxscore)
    monkeypatch.setattr(app, "_rz_fetch_alt_pbp_plays", fake_pbp)
    # Keep the tail of the collect quiet/deterministic.
    monkeypatch.setattr(app, "_rz_get_projections", lambda *a, **k: {})
    monkeypatch.setattr(app, "_redzone_trigger_scoring_push",
                        lambda *a, **k: None)
    return calls


def test_store_hit_skips_upstream_pbp(monkeypatch):
    import app

    calls = _install_stubs(monkeypatch, {GID: list(STORED)})
    out = app._redzone_collect("sleeper", "123", 2026, 4)

    assert out["pbp_by_game"][GID] == STORED
    assert calls["pbp"] == [], "store hit must not fetch upstream PBP"
    # Only the plain boxscore (stat lines) is fetched, never the Tank01 PBP.
    assert calls["box"], "stat-line boxscore still fetched"
    assert all(not c["play_by_play"] for c in calls["box"])


def test_store_hit_skips_redundant_plain_boxscore_refetch(monkeypatch):
    import app

    # Plain boxscore without playerStats: the old "missing PBP" fallback
    # would re-fetch the identical plain payload; on a store hit it must not.
    calls = _install_stubs(monkeypatch, {GID: list(STORED)})

    def bare_boxscore(gid, play_by_play=False, ttl=None):
        calls["box"].append({"gid": gid, "play_by_play": play_by_play})
        return {"playerStats": {}, "teamStats": {}}

    monkeypatch.setattr(app, "_redzone_boxscore", bare_boxscore)
    out = app._redzone_collect("sleeper", "123", 2026, 4)

    assert out["pbp_by_game"][GID] == STORED
    assert len(calls["box"]) == 1, "no duplicate plain boxscore fetch on store hit"
    assert calls["pbp"] == []


def test_store_miss_falls_through_to_live_fetch(monkeypatch):
    import app

    calls = _install_stubs(monkeypatch, {})
    out = app._redzone_collect("sleeper", "123", 2026, 4)

    assert out["pbp_by_game"][GID] == LIVE_FETCHED
    assert calls["pbp"] == [GID]


def test_store_failure_falls_through_to_live_fetch(monkeypatch):
    import app

    calls = _install_stubs(monkeypatch, {}, store_raises=True)
    out = app._redzone_collect("sleeper", "123", 2026, 4)

    assert out["pbp_by_game"][GID] == LIVE_FETCHED
    assert calls["pbp"] == [GID]
