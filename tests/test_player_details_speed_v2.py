"""Player-modal speed fix v2: core/deferred split, shared caches, and the
single league-wide weekly-points build.

Covered here:
- a slim CORE request fires none of the deferred builders (value-history
  DB, eligibility DB, rookie DB, breakout board/history, playoff SoS,
  per-player completed weeks) and returns neutral defaults for them;
- a slim DEFERRED request runs those builders and returns exactly the
  deferred fields with the full payload's shapes;
- a NON-slim request is unchanged (full build, builders fire once);
- the payload / value-history / eligibility caches serve a second
  "worker" (in-process cache cleared) from the shared Redis layer;
- the weekly-points build runs once per (season, scoring) for two
  players with different completed weeks; the per-player correction is
  deferred-only.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

import dashboard_services.api as api
import dashboard_services.service as service

LEAGUE = "1312067280816832512"
PID = "4046"
PID2 = "9999"

PLAYERS = {
    PID: {
        "player_id": PID, "name": "Josh Allen", "full_name": "Josh Allen",
        "first_name": "Josh", "last_name": "Allen",
        "pos": "QB", "position": "QB", "team": "BUF",
        "status": "Active", "injury_status": None, "years_exp": 8,
        "age": 29, "number": 17, "depth_chart_order": 1,
    },
    PID2: {
        "player_id": PID2, "name": "Bye Week Back", "full_name": "Bye Week Back",
        "first_name": "Bye", "last_name": "Back",
        "pos": "QB", "position": "QB", "team": "KC",
        "status": "Active", "injury_status": None, "years_exp": 4,
        "age": 26, "number": 9, "depth_chart_order": 1,
    },
}

LEAGUE_DICT = {
    "league_id": LEAGUE, "platform": "sleeper", "settings": {},
    "scoring_settings": {}, "name": "Speed Test League", "season": "2026",
}

POLICY = SimpleNamespace(completed_weeks=(1, 2), current_week=3)

TRADES_SENTINEL = {"trades": [{"week": 3, "season": 2026}]}
ACQ_SENTINEL = {"events": [{"kind": "draft", "season": 2026}]}


class FakeRedis:
    """Dict-backed stand-in with the get/setex surface the store uses."""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True


@pytest.fixture()
def env(monkeypatch):
    import app as appmod

    appmod.app.config.update(TESTING=True)
    counters = {
        "vh": 0, "elig": 0, "rookie_db": 0, "playoff": 0,
        "board": 0, "surfaced": 0, "players_lookup": 0,
        "wp_weeks": [], "startsit": 0, "trades": 0, "acq": 0,
    }

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
    _real_wp = appmod._load_season_weekly_points

    def _players():
        counters["players_lookup"] += 1
        return dict(PLAYERS)

    # The handler reads the player meta from the relevant index (with the
    # full players index as fallback) at build time only: a payload cache
    # hit returns before either is consulted.
    monkeypatch.setattr("utils.data_cache.load_relevant_index", _players)
    monkeypatch.setattr("utils.utils.load_relevant_index", _players)
    monkeypatch.setattr(appmod, "load_players_index", lambda: {})
    monkeypatch.setattr(appmod, "get_players_index_global", lambda: dict(PLAYERS))
    monkeypatch.setattr(appmod, "get_players_global", lambda: dict(PLAYERS))
    monkeypatch.setattr(appmod, "get_normalized_scoring_settings",
                        lambda platform: {"rec": 1.0})
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: {
                            "rosters": [{
                                "roster_id": 3, "owner_id": "u1",
                                "players": [PID],
                            }],
                            "users": [{
                                "user_id": "u1", "display_name": "Manager One",
                                "metadata": {"team_name": "Team Alpha"},
                            }],
                            "scoring_settings": {"rec": 1.0},
                            "league": dict(LEAGUE_DICT),
                            "league_settings": {}, "settings": {},
                        })
    monkeypatch.setattr(
        appmod, "get_model_value_table_cached",
        lambda: [
            {"id": PID, "name": "Josh Allen", "position": "QB", "team": "BUF",
             "value_1qb": 5000.0, "value_sf": 6000.0},
            {"id": PID2, "name": "Bye Week Back", "position": "QB",
             "team": "KC", "value_1qb": 3000.0, "value_sf": 3600.0},
        ])
    monkeypatch.setattr(appmod, "get_league",
                        lambda league_id, **kw: dict(LEAGUE_DICT))
    monkeypatch.setattr(api, "get_league", lambda league_id, **kw: dict(LEAGUE_DICT))
    monkeypatch.setattr(appmod, "_load_usage_rows_cached", lambda season: [])
    monkeypatch.setattr(appmod, "_oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_team_bye_map", lambda season: {})
    monkeypatch.setattr(appmod, "_startsit_compare_extras", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "start_score_league_index", lambda *a, **k: 55)
    monkeypatch.setattr(
        "dashboard_services.adp_service.fetch_sleeper_adp", lambda season: {})

    # Live week + a real projection, so the NON-slim start/sit block runs
    # end to end (mirrors tests/test_player_details_slim.py's proven set).
    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"week": 7, "season": 2026, "season_type": "regular"},
    )
    monkeypatch.setattr(
        "utils.utils.load_week_projection", lambda season, week: {PID: {}})
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

    def _compute(*a, **k):
        counters["startsit"] += 1
        return 55.0, {"projection": 1.0}, None

    monkeypatch.setattr("utils.start_sit_score.compute_start_score", _compute)

    def _trades(*a, **k):
        counters["trades"] += 1
        return TRADES_SENTINEL

    def _acq(*a, **k):
        counters["acq"] += 1
        return ACQ_SENTINEL

    monkeypatch.setattr(
        "dashboard_services.player_league_trades.get_player_league_trades",
        _trades)
    monkeypatch.setattr(
        "dashboard_services.player_league_trades.get_player_acquisition_events",
        _acq)

    def _vh(player_id, *, days, league_type, league_size):
        counters["vh"] += 1
        return [{"date": "2026-10-01", "value_sf": 6000.0,
                 "value_1qb": 5000.0}]

    monkeypatch.setattr(appmod, "get_player_value_history", _vh)

    def _elig(player_id, season):
        counters["elig"] += 1
        return (True, True)

    monkeypatch.setattr(appmod, "_player_nfl_eligibility", _elig)

    import data_building.rookie_pipeline.pipeline as pipeline
    monkeypatch.setattr(pipeline, "get_active_rookie_class", lambda: 2026)

    def _rookie_db(year):
        counters["rookie_db"] += 1
        return []

    monkeypatch.setattr(pipeline, "get_rookie_rankings_from_db", _rookie_db)

    def _playoff(season, team, pos, weeks=None):
        counters["playoff"] += 1
        return (5, 32, 0.42)

    monkeypatch.setattr(appmod, "_playoff_sos_for", _playoff)
    monkeypatch.setattr(appmod, "_playoff_sos_weeks_for",
                        lambda *a, **k: [{"week": 15, "opp": "MIA",
                                           "opp_rank": 3, "opp_fpts": 11.0,
                                           "at": False}])
    monkeypatch.setattr(appmod, "_playoff_fpts_max", lambda *a, **k: None)

    import dashboard_services.breakout_api as breakout_api
    monkeypatch.setattr(breakout_api, "_resolve_bo_season", lambda s: s)

    def _board(season):
        counters["board"] += 1
        return {"candidates": [{"player_id": PID}, {"player_id": "7777"}]}

    monkeypatch.setattr(breakout_api, "get_breakout_board_candidates", _board)

    import data_building.breakout_engine.weekly_grading as weekly_grading

    def _surfaced(season):
        counters["surfaced"] += 1
        return {2: {PID2}}

    monkeypatch.setattr(weekly_grading, "surfaced_player_ids_by_week", _surfaced)

    import utils.season_qualification as sq
    monkeypatch.setattr(sq, "qualification_policy", lambda season: POLICY)

    def _wp(season, scoring_settings, weeks=None):
        counters["wp_weeks"].append(weeks)
        return {PID: [10.0, 20.0], PID2: [30.0, 10.0]}

    monkeypatch.setattr(appmod, "_load_season_weekly_points", _wp)

    with appmod.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["account_id"] = 7
            sess["user_id"] = 7
        yield SimpleNamespace(appmod=appmod, client=client, counters=counters,
                              real_wp=_real_wp)
    appmod.app.config.update(TESTING=False)


def _core_url(pid=PID):
    return (f"/api/player-details/{pid}?platform=sleeper"
            f"&league_id={LEAGUE}&season=2026&slim=1")


def _deferred_url(pid=PID):
    return _core_url(pid) + "&deferred=1"


def test_core_slim_fires_no_deferred_builders(env):
    resp = env.client.get(_core_url())
    assert resp.status_code == 200
    body = resp.get_json()
    c = env.counters
    assert c["vh"] == 0, "core must not query value history"
    assert c["elig"] == 0, "core must not run eligibility"
    assert c["rookie_db"] == 0, "core must not load rookie rankings"
    assert c["playoff"] == 0, "core must not build playoff SoS"
    assert c["board"] == 0 and c["surfaced"] == 0, "core must not touch breakout"
    assert c["startsit"] == 0 and c["trades"] == 0 and c["acq"] == 0
    # Neutral defaults keep every existing render path safe.
    assert body["value_history"] == []
    assert body["value_trend"] is None
    assert body["prospect_data"] is None
    assert body["has_prospect_data"] is False
    assert body["playoff_sos"] is None
    assert body["on_breakout_board"] is False
    assert body["has_breakout_call"] is False
    assert body["breakout_call"] is None
    assert body["has_game_logs"] is False
    assert body["has_advanced_metrics"] is False
    # Hero scoring still comes from the shared league-wide build.
    assert body["stats"]["ppg"] == 15.0
    assert body["stats"]["ppg_games"] == 2
    assert body["stats"]["total_pts"] == 30.0
    # Every weekly-points call was league-wide (no per-player week tuple).
    assert c["wp_weeks"] and all(w is None for w in c["wp_weeks"])


def test_deferred_returns_deferred_fields_with_full_shapes(env):
    resp = env.client.get(_deferred_url())
    assert resp.status_code == 200
    body = resp.get_json()
    c = env.counters
    assert body["deferred"] is True
    assert body["player_id"] == PID
    assert c["vh"] == 1
    assert c["elig"] == 1
    assert c["rookie_db"] >= 1
    assert c["playoff"] == 1
    assert c["board"] == 1 and c["surfaced"] == 1
    # Deferred is still slim: no start/sit compute, no league folds.
    assert c["startsit"] == 0 and c["trades"] == 0 and c["acq"] == 0
    # Same shapes as the full payload.
    assert isinstance(body["value_history"], list) and len(body["value_history"]) == 1
    assert "value_trend" in body
    assert body["prospect_data"] is None  # no rookie rows in the fixture DB
    assert body["has_prospect_data"] is False
    assert body["on_breakout_board"] is True  # PID is a board candidate
    # Full-payload semantics: has_breakout_call = on board OR has a call.
    assert body["has_breakout_call"] is True
    assert body["breakout_call"] is None
    assert body["playoff_sos"]["rank"] == 5
    assert body["playoff_sos"]["weeks"][0]["week"] == 15
    assert body["has_game_logs"] is True
    assert body["has_advanced_metrics"] is True
    for key in ("ppg", "ppg_rank", "ppg_games", "total_pts", "total_pts_rank",
                "scoring_data_status"):
        assert key in body["stats"]
    # A player who is not on the board surfaces his earliest call instead.
    resp2 = env.client.get(_deferred_url(PID2))
    body2 = resp2.get_json()
    assert body2["on_breakout_board"] is False
    assert body2["has_breakout_call"] is True
    assert body2["breakout_call"]["label"] == "Week 3"
    assert body2["breakout_call"]["as_of_week"] == 2


def test_breakout_id_sets_built_once_with_aux_cache(env, monkeypatch):
    # With the aux caches live (production), two deferred opens share one
    # board build + one surfaced-ids call.
    monkeypatch.setattr(env.appmod, "_pd_aux_cache_active", lambda: True)
    env.appmod._BREAKOUT_FLAGS_CACHE.clear()
    env.appmod._PLAYER_DETAILS_CACHE.clear()
    env.client.get(_deferred_url(PID))
    env.client.get(_deferred_url(PID2))
    assert env.counters["board"] == 1
    assert env.counters["surfaced"] == 1


def test_non_slim_full_payload_unchanged(env):
    resp = env.client.get(
        f"/api/player-details/{PID}?platform=sleeper"
        f"&league_id={LEAGUE}&season=2026")
    assert resp.status_code == 200
    body = resp.get_json()
    c = env.counters
    assert body.get("deferred") is not True
    assert c["vh"] == 1 and c["elig"] == 1 and c["playoff"] == 1
    assert c["startsit"] == 1 and c["trades"] == 1 and c["acq"] == 1
    assert body["league_trades"] == TRADES_SENTINEL
    assert body["acquisition_events"] == ACQ_SENTINEL
    assert len(body["value_history"]) == 1
    assert body["on_breakout_board"] is True
    assert body["playoff_sos"]["rank"] == 5
    assert body["has_game_logs"] is True
    for key in ("name", "position", "team", "stats", "value_history",
                    "prospect_data", "playoff_sos", "breakout_call"):
        assert key in body


def test_shared_cache_serves_second_worker(env, monkeypatch):
    import dashboard_services.player_details_store as store

    fake = FakeRedis()
    monkeypatch.setattr(store, "_redis_client", lambda: fake)
    monkeypatch.setenv("REDIS_URL", "redis://fake:6379/0")
    assert store.enabled() is True

    appmod = env.appmod
    # First open builds and writes both cache layers.
    resp = env.client.get(_core_url())
    assert resp.status_code == 200
    assert env.counters["players_lookup"] == 1
    assert fake.store, "payload was not written to the shared layer"

    # Second worker: in-process cache empty, shared layer intact.
    appmod._PLAYER_DETAILS_CACHE.clear()
    resp2 = env.client.get(_core_url())
    assert resp2.status_code == 200
    assert resp2.get_json()["name"] == "Josh Allen"
    assert env.counters["players_lookup"] == 1, "rebuild happened despite shared hit"

    # Value history: DB hit once, then shared layer serves.
    rows = appmod._cached_value_history(
        PID, days=365, league_type="dynasty", league_size=12)
    assert rows and env.counters["vh"] == 1
    appmod._VALUE_HISTORY_CACHE.clear()
    rows2 = appmod._cached_value_history(
        PID, days=365, league_type="dynasty", league_size=12)
    assert rows2 == rows
    assert env.counters["vh"] == 1, "value history re-queried despite shared hit"

    # Eligibility: same pattern behind the aux-cache gate.
    monkeypatch.setattr(appmod, "_pd_aux_cache_active", lambda: True)
    appmod._ELIGIBILITY_CACHE.clear()
    assert appmod._player_nfl_eligibility_cached(PID, 2026) == (True, True)
    assert env.counters["elig"] == 1
    appmod._ELIGIBILITY_CACHE.clear()
    assert appmod._player_nfl_eligibility_cached(PID, 2026) == (True, True)
    assert env.counters["elig"] == 1, "eligibility re-queried despite shared hit"


def test_weekly_points_build_once_and_deferred_correction(env, monkeypatch, tmp_path):
    """Real weekly-points loader over real Sleeper week files: one
    league-wide build serves two players whose own completed weeks
    differ; only the deferred payload pays a per-player rebuild, and
    only for the player whose weeks differ."""
    import builtins
    import json as _json

    from utils.projections import score_stats

    appmod = env.appmod
    scoring = {"rec": 1.0}
    week_stats = {
        1: {PID: {"pass_yd": 100, "pass_td": 1, "off_snp": 30},
            PID2: {"pass_yd": 200, "pass_td": 1, "off_snp": 30}},
        2: {PID: {"pass_yd": 300, "pass_td": 2, "off_snp": 30},
            PID2: {"pass_yd": 50, "pass_td": 0, "off_snp": 30}},
    }
    stats_dir = tmp_path / "sleeper_stats"
    stats_dir.mkdir(parents=True)
    for wk, rows in week_stats.items():
        (stats_dir / f"sleeper_stats_s2026_w{wk}.json").write_text(
            _json.dumps(rows))

    opens = []
    real_open = builtins.open

    def _counting_open(file, *a, **k):
        name = str(file)
        if "sleeper_stats_s2026_w" in name:
            opens.append(name.rsplit("_w", 1)[-1].split(".")[0])
        return real_open(file, *a, **k)

    monkeypatch.setattr(appmod, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(appmod, "_ensure_sleeper_week_files",
                        lambda season: None)
    monkeypatch.setattr(appmod, "open", _counting_open, raising=False)
    # Restore the real loader (the fixture replaced it with a stub).
    monkeypatch.setattr(appmod, "_load_season_weekly_points", env.real_wp)
    import utils.season_qualification as sq
    monkeypatch.setattr(
        sq, "player_completed_weeks",
        lambda pid, season: (1, 2) if str(pid) == PID else (1,))
    appmod._WEEKLY_PTS_CACHE.clear()
    appmod._SEASON_RANK_CACHE.clear()
    appmod._PLAYER_DETAILS_CACHE.clear()

    p1_w1 = score_stats(week_stats[1][PID], scoring, "")
    p1_w2 = score_stats(week_stats[2][PID], scoring, "")
    p2_w1 = score_stats(week_stats[1][PID2], scoring, "")

    core1 = env.client.get(_core_url(PID)).get_json()
    core2 = env.client.get(_core_url(PID2)).get_json()
    # Each week file parsed exactly once for both players' core opens.
    assert sorted(opens) == ["1", "2"], opens
    assert core1["stats"]["ppg"] == round((p1_w1 + p1_w2) / 2, 1)
    assert core2["stats"]["ppg_games"] == 2

    # Deferred for the player whose weeks match the league: no re-parse.
    env.client.get(_deferred_url(PID))
    assert sorted(opens) == ["1", "2"], opens

    # Deferred for the player missing week 2: exactly one extra parse
    # (his own week tuple), and his hero PPG corrects to week 1 only.
    def2 = env.client.get(_deferred_url(PID2)).get_json()
    assert sorted(opens) == ["1", "1", "2"], opens
    assert def2["stats"]["ppg"] == round(p2_w1, 1)
    assert def2["stats"]["ppg_games"] == 1


def test_client_fires_deferred_and_early_fetches():
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "static" / "player_modal.js").read_text()
    assert "_pmStartEarlyFetches(playerId, _detailsDescriptor)" in js
    assert "_detailsDescriptor.url + '&deferred=1'" in js
    assert "function _pmPatchDeferred(dfd)" in js
    assert "_pmTakeEarlyResponse(logsUrl) || _pmFetch(logsUrl)" in js
    assert "_pmTakeEarlyJson(tradesUrl) || jget(tradesUrl)" in js
    assert "pmHeroPpgVal" in js and "pmHeroTotalVal" in js
