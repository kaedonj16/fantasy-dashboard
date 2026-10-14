"""Coverage for the player-details first-open performance fixes.

1. The short-TTL server payload cache: a repeat foreground open is served
   without rebuilding, an unresolved-ownership payload is never pinned, and
   speculative requests keep their 204 contract (they never read the cache).
2. The modal warm is kicked from the league-context accessor, so landing on
   a league page starts the fills the first player click used to pay inline.
"""
import time

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

import app as appmod  # noqa: E402

PID = "7564"
LID = "1312067280816832512"
URL = (
    f"/api/player-details/{PID}?league_id={LID}"
    "&platform=sleeper&season=2026&league_type=1qb&league_size=10"
)


class _Counter:
    def __init__(self):
        self.n = 0


def _warm_ctx():
    return {
        "rosters": [
            {"roster_id": 1, "owner_id": "owner-1", "players": [PID]},
        ],
        "users": [
            {"user_id": "owner-1", "metadata": {"team_name": "Team Alpha"}},
        ],
        "scoring_settings": {"rec": 1.0},
    }


@pytest.fixture(autouse=True)
def _clean_payload_cache():
    appmod._PLAYER_DETAILS_CACHE.clear()
    yield
    appmod._PLAYER_DETAILS_CACHE.clear()


def _enable_payload_cache(monkeypatch):
    monkeypatch.setattr(appmod, "_player_details_cache_active", lambda: True)


def _mock_build(monkeypatch, ctx_holder):
    """Stub the handler's heavy dependencies; count full-season builds."""
    builds = _Counter()

    import dashboard_services.api as svc_api
    import dashboard_services.adp_service as adp_service
    import utils.data_cache as data_cache

    monkeypatch.setattr(
        data_cache, "load_relevant_index",
        lambda pid, index: {PID: {
            "full_name": "Ja'Marr Chase", "position": "WR", "team": "CIN",
            "injury_status": None,
        }},
    )
    monkeypatch.setattr(svc_api, "get_league", lambda *a, **k: {}, raising=False)
    monkeypatch.setattr(svc_api, "get_nfl_state", lambda *a, **k: {
        "week": 0, "season": 2026, "season_type": "regular",
    }, raising=False)
    monkeypatch.setattr(adp_service, "fetch_sleeper_adp", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "get_league", lambda *a, **k: {})
    monkeypatch.setattr(
        appmod, "get_normalized_scoring_settings", lambda *a, **k: {"rec": 1.0})
    monkeypatch.setattr(
        appmod, "get_league_ctx_from_cache",
        lambda *a, **k: ctx_holder["ctx"],
    )
    monkeypatch.setattr(appmod, "get_model_value_table_cached", lambda: [])
    monkeypatch.setattr(appmod, "get_player_value_history", lambda *a, **k: [])
    monkeypatch.setattr(appmod, "_player_nfl_eligibility", lambda *a, **k: (False, False))

    def _weekly_points(*a, **k):
        builds.n += 1
        return {}

    monkeypatch.setattr(appmod, "_load_season_weekly_points", _weekly_points)
    return builds


def test_repeat_open_served_from_payload_cache(monkeypatch):
    _enable_payload_cache(monkeypatch)
    builds = _mock_build(monkeypatch, {"ctx": _warm_ctx()})
    client = appmod.app.test_client()

    first = client.get(URL)
    assert first.status_code == 200
    assert builds.n >= 1
    after_first = builds.n

    second = client.get(URL)
    assert second.status_code == 200
    assert second.get_json() == first.get_json()
    assert builds.n == after_first  # no rebuild on the repeat open


def test_unresolved_ownership_payload_is_never_cached(monkeypatch):
    _enable_payload_cache(monkeypatch)
    holder = {"ctx": {}}
    _mock_build(monkeypatch, holder)
    client = appmod.app.test_client()

    cold = client.get(URL)
    assert cold.status_code == 200
    assert cold.get_json()["ownership_unknown"] is True
    assert not appmod._PLAYER_DETAILS_CACHE  # nothing pinned

    holder["ctx"] = _warm_ctx()
    warm = client.get(URL)
    assert warm.status_code == 200
    assert warm.get_json()["ownership_unknown"] is False
    assert warm.get_json()["fantasy_team"] == "Team Alpha"


def test_speculative_request_still_204s_with_a_cached_payload(monkeypatch):
    _enable_payload_cache(monkeypatch)
    _mock_build(monkeypatch, {"ctx": _warm_ctx()})
    appmod.DASHBOARD_CACHE.pop(appmod._cache_key("sleeper", 2026, LID), None)
    client = appmod.app.test_client()

    primed = client.get(URL)
    assert primed.status_code == 200
    assert appmod._PLAYER_DETAILS_CACHE  # foreground open populated the cache

    speculative = client.get(URL, headers={"X-BR-Speculative": "player-details"})
    assert speculative.status_code == 204  # cache never bypasses the 204 gate


def test_modal_league_warm_builds_handler_keys_and_trade_inputs(monkeypatch):
    """The league warm must build with the handler's scoring key (normalized
    RAW league scoring, not the ctx's enriched variant) and must actually
    populate the trade-history inputs. Both failure modes are silent in
    production (best-effort thread), so they are asserted here."""
    # importlib.import_module binds the sys.modules entry, which is what the
    # warm thread's function-level `from ... import` statements read. A plain
    # `import dashboard_services.service as svc` can bind the package
    # attribute instead, which another suite's delete-and-reimport fixture
    # (test_matchup_cards_degrade) leaves pointing at a different module
    # object; patches would then land on a module the warm never reads.
    import importlib

    svc_api = importlib.import_module("dashboard_services.api")
    plt = importlib.import_module("dashboard_services.player_league_trades")
    svc = importlib.import_module("dashboard_services.service")
    from utils.league_scoring import normalize_league_scoring

    monkeypatch.setitem(appmod.app.config, "TESTING", False)
    appmod._MODAL_LEAGUE_WARMED.clear()

    raw_scoring = {"rec": 0.5, "pass_yd": 0.04}
    ctx_scoring = dict(raw_scoring, enriched_only_key=9)
    handler_scoring = normalize_league_scoring(
        "sleeper", raw_scoring, league_id="L-now", season=2026)

    calls = {"weekly": [], "ranks": [], "weekstat": [], "tx": [],
             "names": [], "resolution": [], "slots": []}
    monkeypatch.setattr(svc_api, "get_league",
                        lambda *a, **k: {"scoring_settings": raw_scoring})
    monkeypatch.setattr(svc_api, "build_league_history_map",
                        lambda *a, **k: {2026: "L-now", 2025: "L-prev"})
    monkeypatch.setattr(appmod, "_load_season_weekly_points",
                        lambda season, scoring: calls["weekly"].append((season, scoring)))
    monkeypatch.setattr(appmod, "_season_rank_rows",
                        lambda season, scoring: calls["ranks"].append((season, scoring)))
    monkeypatch.setattr(appmod, "_season_weekstat_points",
                        lambda season, scoring, pos: calls["weekstat"].append((season, scoring, pos)))
    monkeypatch.setattr(svc, "get_transactions_by_week",
                        lambda lid, weeks, platform=None, season=None:
                        calls["tx"].append((lid, season)))
    monkeypatch.setattr(plt, "_roster_names",
                        lambda *a: calls["names"].append(a))
    monkeypatch.setattr(plt, "build_draft_resolution_map",
                        lambda *a: calls["resolution"].append(a))
    monkeypatch.setattr(plt, "_slot_map_for_league",
                        lambda *a: calls["slots"].append(a))

    ctx = {"scoring_settings": ctx_scoring}
    appmod._kick_modal_league_warm("sleeper", "L-now", 2026, ctx)

    deadline = time.time() + 5
    while time.time() < deadline and len(calls["slots"]) < 2:
        time.sleep(0.02)

    try:
        assert calls["weekly"] == [(2026, handler_scoring), (2025, handler_scoring)]
        assert calls["ranks"] == [(2026, handler_scoring)]
        assert {pos for _, _, pos in calls["weekstat"]} == {"QB", "RB", "WR", "TE"}
        assert all(s == handler_scoring for _, s, _ in calls["weekstat"])
        assert sorted(calls["tx"]) == [("L-now", 2026), ("L-prev", 2025)]
        assert len(calls["names"]) == 2
        assert len(calls["resolution"]) == 2
        assert len(calls["slots"]) == 2

        # Deduped: a second kick for the same league/scoring does nothing.
        snapshot = {k: len(v) for k, v in calls.items()}
        appmod._kick_modal_league_warm("sleeper", "L-now", 2026, ctx)
        time.sleep(0.2)
        assert {k: len(v) for k, v in calls.items()} == snapshot
    finally:
        appmod._MODAL_LEAGUE_WARMED.clear()


def test_league_ctx_access_kicks_modal_warm(monkeypatch):
    kicks = _Counter()
    league_warms = _Counter()
    monkeypatch.setattr(
        appmod, "_kick_player_modal_warm",
        lambda: setattr(kicks, "n", kicks.n + 1),
    )
    monkeypatch.setattr(
        appmod, "_kick_modal_league_warm",
        lambda *a, **k: setattr(league_warms, "n", league_warms.n + 1),
    )
    monkeypatch.setattr(appmod, "_maybe_check_roster_freshness", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "get_viewer_session_for_league", lambda *a, **k: {})

    key = appmod._cache_key("sleeper", 2026, LID)
    appmod.DASHBOARD_CACHE[key] = {
        "ctx": _warm_ctx(), "ts": time.time(), "page_html": {},
    }
    try:
        with appmod.app.test_request_context():
            ctx = appmod.get_league_ctx_from_cache("sleeper", LID, 2026)
    finally:
        appmod.DASHBOARD_CACHE.pop(key, None)

    assert ctx  # served from the seeded cache entry
    assert kicks.n == 1
    assert league_warms.n == 1
