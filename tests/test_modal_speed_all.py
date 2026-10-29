"""Modal-speed fixes (audit F1-F9): cache hits must skip the expensive
pre-checks, one league scan serves both trade consumers, repeated modal
requests come from response caches, and every modal fetch is bounded.

Server behavior is pinned through the Flask test client with the heavy
compute seams stubbed; client changes are pinned at the source level,
scoped to the loader function bodies the audit named.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

import app as appmod

ROOT = Path(__file__).resolve().parents[1]
PM = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
APPJS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

_DEFAULT_SCORING = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    "rush_yd": 0.1, "rush_td": 6.0,
    "rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0, "fum_lost": -2.0,
}


def _fn_body(src: str, name: str) -> str:
    """Source of one top-level JS function (up to the next one)."""
    return src.split(f"function {name}", 1)[1].split("\nfunction ", 1)[0]


# ── F1: game-logs cache hit skips eligibility ──────────────────────────────

@pytest.fixture()
def gl_cache():
    appmod._GAME_LOGS_CACHE.clear()
    yield
    appmod._GAME_LOGS_CACHE.clear()


def test_game_logs_cache_hit_skips_eligibility(monkeypatch, gl_cache):
    calls = []

    def fake_elig(pid, season):
        calls.append(pid)
        return (True, False)

    monkeypatch.setattr(appmod, "_player_nfl_eligibility", fake_elig)
    key = appmod._game_logs_cache_key("4046", 2025, _DEFAULT_SCORING)
    appmod._GAME_LOGS_CACHE[key] = (
        time.time(),
        {2025: [{"week": 1, "opponent": "BUF", "fantasy_pts": 12.5}]},
        {2025: "KC"},
        True,  # has_game_logs, stored with the payload on the miss
    )
    resp = appmod.app.test_client().get("/api/player-game-logs/4046?season=2025")
    assert resp.status_code == 200
    data = resp.get_json()
    assert calls == []  # eligibility was NOT consulted on the hit
    assert data["has_game_logs"] is True
    assert data["game_logs_by_year"] == {
        "2025": [{"week": 1, "opponent": "BUF", "fantasy_pts": 12.5}]}
    assert data["season_teams"] == {"2025": "KC"}


def test_game_logs_miss_stores_has_game_logs_then_hit_skips(monkeypatch, gl_cache):
    from utils import utils as _u

    elig_calls = []

    def fake_elig(pid, season):
        elig_calls.append(pid)
        return (True, False)

    monkeypatch.setattr(appmod, "_player_nfl_eligibility", fake_elig)
    monkeypatch.setattr(appmod, "_ensure_sleeper_week_files", lambda season: None)
    monkeypatch.setattr(appmod, "_sleeper_stats_by_week", lambda pid, yr: {})
    monkeypatch.setattr(appmod, "_PLAYER_DETAIL_YEARS_CACHE", {2025})
    monkeypatch.setattr(appmod, "_PLAYER_DETAIL_YEARS_CACHE_TS", time.time())
    monkeypatch.setattr(appmod, "_prefetch_week_projections", lambda season: None)
    monkeypatch.setattr(_u, "load_relevant_index", lambda *a, **k: {})
    monkeypatch.setattr(_u, "load_week_projection", lambda *a, **k: {})

    client = appmod.app.test_client()
    first = client.get("/api/player-game-logs/4046?season=2025")
    assert first.status_code == 200
    assert len(elig_calls) == 1  # computed once, on the miss
    key = appmod._game_logs_cache_key("4046", 2025, _DEFAULT_SCORING)
    assert key in appmod._GAME_LOGS_CACHE
    assert appmod._GAME_LOGS_CACHE[key][3] is True  # has_game_logs in the tuple

    second = client.get("/api/player-game-logs/4046?season=2025")
    assert second.status_code == 200
    assert len(elig_calls) == 1  # the hit did not re-run eligibility
    assert second.get_json() == first.get_json()


def test_game_logs_cache_check_precedes_eligibility_in_source():
    body = Path(appmod.__file__).read_text(encoding="utf-8").split(
        "def api_player_game_logs", 1)[1].split("\ndef ", 1)[0]
    assert body.index("_GAME_LOGS_CACHE.get(_cache_key)") < body.index(
        "_player_nfl_eligibility(player_id, season)")


# ── F7: player-team cache check runs before index loads/resolve ────────────

@pytest.fixture()
def team_cache():
    appmod._TEAM_PAYLOAD_CACHE.clear()
    yield
    appmod._TEAM_PAYLOAD_CACHE.clear()


def test_player_team_cache_hit_skips_index_load(monkeypatch, team_cache):
    def boom(*a, **k):
        raise AssertionError("index load ran on a cache hit")

    monkeypatch.setattr(appmod, "get_players_index_global", boom)
    appmod._TEAM_PAYLOAD_CACHE[("6786", 2025)] = (
        time.time(), {"available": True, "teams": [{"team": "KC"}], "marker": "cached"})
    resp = appmod.app.test_client().get("/api/player-team/6786?season=2025")
    assert resp.status_code == 200
    assert resp.get_json()["marker"] == "cached"


def test_player_team_cache_check_precedes_resolve_in_source():
    body = Path(appmod.__file__).read_text(encoding="utf-8").split(
        "def api_player_team", 1)[1].split("\ndef ", 1)[0]
    assert body.index("_TEAM_PAYLOAD_CACHE.get") < body.index(
        "resolve_team_for_season(str(player_id)")


# ── F3: trades cache ignores limit; one shared transactions scan ───────────

@pytest.fixture()
def trades_cache():
    appmod._PLAYER_TRADES_CACHE.clear()
    yield
    appmod._PLAYER_TRADES_CACHE.clear()


def test_league_trades_cache_ignores_limit_and_truncates(monkeypatch, trades_cache):
    import dashboard_services.player_league_trades as plt

    calls = []

    def stub(**kwargs):
        calls.append(kwargs)
        return {
            "trades": [{"id": i} for i in range(12)],
            "total": 12, "source": "league",
        }

    monkeypatch.setattr(plt, "get_player_league_trades", stub)
    client = appmod.app.test_client()
    base = "/api/player-league-trades/8112?platform=sleeper&league_id=L1&season=2026"
    r10 = client.get(base + "&limit=10")
    assert r10.status_code == 200
    assert len(r10.get_json()["trades"]) == 10  # truncated after the read
    assert r10.get_json()["total"] == 12
    assert len(calls) == 1 and calls[0]["limit"] == 50  # computed once, at max

    r50 = client.get(base + "&limit=50")
    assert len(r50.get_json()["trades"]) == 12
    assert len(calls) == 1  # different limit: same cache entry, no rescan


def test_transactions_fetch_shared_between_trades_and_acquisition(monkeypatch):
    import importlib

    import dashboard_services.player_league_trades as plt
    # Resolve both modules through sys.modules: that is the copy plt's
    # `from dashboard_services.service import ...` reads at call time.
    # `from dashboard_services import service` can return a different
    # module object (the package attribute) if an earlier test re-imported
    # the service module under stubs, which silently detached this patch
    # from the code under test in the full CI suite.
    service = importlib.import_module("dashboard_services.service")
    dapi = importlib.import_module("dashboard_services.api")

    plt._TX_FETCH_CACHE.clear()
    calls = []

    def fake_tx(league_id, weeks, platform="sleeper", season=0):
        calls.append((league_id, season))
        return {}

    monkeypatch.setattr(service, "get_transactions_by_week", fake_tx)
    monkeypatch.setattr(dapi, "build_league_history_map",
                        lambda platform, league_id, season: {2026: "L1"})
    monkeypatch.setattr(dapi, "get_drafts", lambda platform, league_id: [])
    monkeypatch.setattr(plt, "_roster_names", lambda *a, **k: {})

    # The trades pass fetches the season's transactions first...
    plt._transactions_by_week_cached("L1", "sleeper", 2026)
    assert len(calls) == 1
    # ...and the acquisition pass reuses that scan instead of re-fetching
    # weeks 0-18 a second time.
    out = plt.get_player_acquisition_events(
        "111", platform="sleeper", league_id="L1", season=2026)
    assert out == {"events": []}
    assert len(calls) == 1
    plt._TX_FETCH_CACHE.clear()


# ── F6: weekly / advanced metrics response caches ──────────────────────────

@pytest.fixture()
def metrics_cache():
    import routes.players_bp as bp
    bp._METRICS_RESPONSE_CACHE.clear()
    yield bp
    bp._METRICS_RESPONSE_CACHE.clear()


def test_weekly_metrics_second_request_from_cache(monkeypatch, metrics_cache):
    import data_building.weekly_metrics as wm
    import data_building.advanced_metrics as am
    import data_building.external_data.player_team_history as pth
    import routes.players_bp as bp

    calls = []

    def fake_series(pid, season):
        calls.append(pid)
        return [{"week": 1, "fantasy_ppg": 10.0}]

    monkeypatch.setattr(wm, "get_player_weekly_series", fake_series)
    monkeypatch.setattr(am, "get_player_weekly_adv_series", lambda pid, season: [])
    monkeypatch.setattr(pth, "team_for_week", lambda pid, season, week: None)
    monkeypatch.setattr(bp, "get_nfl_state",
                        lambda: {"season": 2025, "season_type": "reg", "week": 8})

    client = appmod.app.test_client()
    url = "/api/player-weekly-metrics/4046?season=2025"
    first = client.get(url)
    second = client.get(url)
    assert first.status_code == second.status_code == 200
    assert len(calls) == 1  # second request served from the response cache
    assert second.get_json() == first.get_json()


def test_advanced_metrics_second_request_from_cache(monkeypatch, metrics_cache):
    import data_building.advanced_metrics as am
    import routes.players_bp as bp

    calls = []

    def fake_by_season(pid, season):
        calls.append((pid, season))
        return {"player_id": pid, "position": "WR", "games_played": 3, "wopr": 0.5}

    monkeypatch.setattr(am, "get_available_seasons_for_player", lambda pid: [2025])
    monkeypatch.setattr(am, "get_player_metrics_by_season", fake_by_season)
    monkeypatch.setattr(am, "get_available_metric_weeks", lambda pid, season: [1, 2, 3])
    monkeypatch.setattr(bp, "get_nfl_state",
                        lambda: {"season": 2025, "season_type": "reg", "week": 8})

    client = appmod.app.test_client()
    url = "/api/player-advanced-metrics/4046?season=2025"
    first = client.get(url)
    second = client.get(url)
    assert first.status_code == second.status_code == 200
    assert len(calls) == 1
    assert second.get_json() == first.get_json()


# ── F9: trade-intel player-trades response cache ───────────────────────────

@pytest.fixture()
def ti_cache():
    appmod._TI_TRADES_CACHE.clear()
    yield
    appmod._TI_TRADES_CACHE.clear()


def test_trade_intel_player_trades_second_request_from_cache(monkeypatch, ti_cache):
    import dashboard_services.db as dbmod
    from utils import utils as _u

    monkeypatch.setattr(appmod, "_deny_unless_trade_intel_premium", lambda: None)
    monkeypatch.setattr(_u, "load_players_index", lambda: {})
    executes = []

    class _Result:
        def fetchone(self):
            return {"n": 0}

        def fetchall(self):
            return []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            executes.append(sql)
            return _Result()

    monkeypatch.setattr(dbmod, "get_conn", lambda: _Conn())

    client = appmod.app.test_client()
    url = ("/api/trade-intel/player-trades/4046?season=2026&platform=sleeper"
           "&league_type=dynasty&page=1&limit=5")
    first = client.get(url)
    assert first.status_code == 200
    assert len(executes) == 2  # count + page queries on the miss
    second = client.get(url)
    assert second.status_code == 200
    assert len(executes) == 2  # second request served from the response cache
    assert second.get_json() == first.get_json()
    assert first.get_json()["total"] == 0


# ── Client pins: player_modal.js ───────────────────────────────────────────

def test_rookie_tab_uses_recent_form_with_retry():
    body = _fn_body(PM, "_pmLoadRookieReturns")
    assert "/api/player-recent-form/" in body
    assert "/api/player-game-logs/" not in body
    assert "_pmFetch(rfUrl, 15000)" in body
    assert ".catch(() => _rfAttempt())" in body  # one quiet retry


def test_in_league_jget_is_bounded():
    assert "const jget = function (u) { return _pmSmallFetch('jget:' + u, u); };" in PM
    assert "_pmTakeEarlyJson(tradesUrl) || jget(tradesUrl)" in PM  # call shape kept


def test_stats_tab_uses_shared_game_logs_promise():
    assert "_pmGameLogsJson(logsUrl)" in PM
    # Failures evict so the Stats tab Retry refetches instead of replaying
    # the rejected promise.
    assert "if (_pmGameLogsCache.get(url) === entry) _pmGameLogsCache.delete(url);" in PM


def test_pm_f5_sites_have_no_raw_fetch():
    assert "_pmFetch(`/api/prospects/comparables/" in PM
    assert "fetch(`/api/prospects/comparables/" not in PM.replace("_pmFetch(", "")
    assert "_pmFetch(url, 15000)" in _fn_body(PM, "_pmLoadUsageBars")
    assert "fetch(url)" not in _fn_body(PM, "_pmLoadUsageBars").replace("_pmFetch(", "")
    assert "_pmFetch(teamUrl, 15000)" in _fn_body(PM, "pmPrefetchTabs")
    assert "_pmFetch(url, 15000)" in _fn_body(PM, "pmSeasonTrendFetch")
    assert PM.count("_pmFetch('/api/player-weekly-metrics/'") == 2
    assert " fetch('/api/player-weekly-metrics/'" not in PM


# ── Client pins: app.js (compare, prospect modal, team modal) ──────────────

def test_compare_game_logs_bounded_with_real_retry():
    body = _fn_body(APPJS, "_cmpLoadGameLogs")
    assert "_advFetch(logsUrl, 20000)" in body
    assert "fetch(logsUrl)" not in APPJS
    # The error state's retry re-invokes the load; it is not null anymore.
    assert ("window.brErrorState(el, 'Could not load game logs.', "
            "() => _cmpLoadGameLogs(pid, position, containerId)") in body
    assert "'Could not load game logs.', null" not in APPJS


def test_appjs_f5_sites_have_no_raw_fetch():
    assert "_advFetch('/api/prospects/comparables/'" in APPJS
    assert " fetch('/api/prospects/comparables/'" not in APPJS
    trades = _fn_body(APPJS, "tmLoadTrades")
    assert "_advFetch(`/api/team-trades/" in trades
    assert "await fetch(`/api/team-trades/" not in trades
    assert "tm-trades-retry" in trades  # error panel now offers a retry
    assert APPJS.count("_advFetch(") >= 1
    assert "_advFetch(`/api/player-weekly-metrics/" in APPJS
    assert "_advFetch('/api/player-weekly-metrics/'" in APPJS
    assert " fetch('/api/player-weekly-metrics/'" not in APPJS
    assert " fetch(`/api/player-weekly-metrics/" not in APPJS
