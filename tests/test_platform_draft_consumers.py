"""Platform draft-path regressions (the #2151 bug class, three more sites).

1. /api/draft-grades fed every platform's draft id to Sleeper's
   /draft/{id}/picks transport; ESPN/Yahoo synthetic ids 404'd and the route
   500'd. Picks must come from platform_api.get_draft_picks.
2. player_league_trades.get_player_acquisition_events scanned drafts through
   the legacy Sleeper helpers for every platform; the 404 was swallowed and
   the "Drafted, Round X" event silently never appeared off Sleeper.
3. push_notifications.notify_rival_trades read transactions through the
   legacy Sleeper helper for subscribed leagues of ALL platforms, so rival
   trade alerts never fired off Sleeper.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


# ── platform_api.get_draft_picks facade / providers ─────────────────────────

def test_facade_sleeper_delegates_to_legacy_draft_picks(monkeypatch):
    import dashboard_services.api as legacy_api
    from dashboard_services import platform_api

    seen = []

    def fake_legacy(draft_id):
        seen.append(draft_id)
        return [{"player_id": "P1", "pick_no": 1}]

    monkeypatch.setattr(legacy_api, "get_draft_picks", fake_legacy)
    rows = platform_api.get_draft_picks("sleeper", "L1", 2026, draft_id="D9")
    assert rows == [{"player_id": "P1", "pick_no": 1}]
    assert seen == ["D9"]
    # No draft id -> no fetch (Sleeper picks are addressed per draft).
    assert platform_api.get_draft_picks("sleeper", "L1", 2026) == []
    assert seen == ["D9"]


def test_facade_espn_picks_are_canonical_rows(monkeypatch):
    from dashboard_services import platform_api
    from dashboard_services.providers import espn_api

    monkeypatch.setattr(espn_api, "_espn_to_canon_cached",
                        lambda: {"101": "P1", "102": "P2"})
    monkeypatch.setattr(espn_api, "iter_draft_picks", lambda season, lid: [
        {"playerId": 102, "teamId": 2, "overallPickNumber": 2,
         "roundId": 1, "roundPickNumber": 2},
        {"playerId": 101, "teamId": 1, "overallPickNumber": 1,
         "roundId": 1, "roundPickNumber": 1, "keeper": True},
        {"playerId": None, "teamId": 3, "overallPickNumber": 3},  # empty slot
    ])
    rows = platform_api.get_draft_picks("espn", "123", 2026,
                                         draft_id="espn_123_2026")
    assert rows == [
        {"player_id": "P1", "roster_id": "1", "picked_by": "1", "pick_no": 1,
         "round": 1, "draft_slot": 1, "metadata": {"keeper": True}},
        {"player_id": "P2", "roster_id": "2", "picked_by": "2", "pick_no": 2,
         "round": 1, "draft_slot": 2, "metadata": {}},
    ]


def test_facade_yahoo_picks_are_canonical_rows(monkeypatch):
    from dashboard_services import platform_api
    from dashboard_services.providers import yahoo_api

    monkeypatch.setattr(platform_api, "_yahoo_token", lambda lid, season: "tok")
    monkeypatch.setattr(yahoo_api, "_yahoo_id_to_canonical",
                        lambda: {"4499": "P1", "4500": "P2"})
    monkeypatch.setattr(yahoo_api, "get_draft_pick_rows",
                        lambda season, lid, token: [
                            {"pick": 1, "round": 1, "player_id": "4499",
                             "team_id": "3", "cost": None},
                            {"pick": 2, "round": 1, "player_id": "9999",
                             "team_id": "4", "cost": None},  # unmapped: dropped
                            {"pick": 3, "round": 2, "player_id": "4500",
                             "team_id": "3", "cost": 12},
                        ])
    rows = platform_api.get_draft_picks("yahoo", "Y123", 2026,
                                         draft_id="yahoo_Y123_2026")
    assert rows == [
        {"player_id": "P1", "roster_id": "3", "picked_by": "3", "pick_no": 1,
         "round": 1, "draft_slot": 0, "metadata": {}},
        {"player_id": "P2", "roster_id": "3", "picked_by": "3", "pick_no": 3,
         "round": 2, "draft_slot": 0, "metadata": {"amount": 12}},
    ]


def test_facade_mfl_picks_canonicalized(monkeypatch):
    from dashboard_services import platform_api
    from dashboard_services.providers.mfl_api import MFLProvider

    monkeypatch.setattr(MFLProvider, "get_drafts", lambda self, lid, season: [{
        "draft_id": "mfl:2026:55", "status": "complete",
        "picks": [
            {"round": 1, "pick_no": 2, "roster_id": 7, "player_id": "9002",
             "picked_by": "7", "metadata": {}},
            {"round": 1, "pick_no": 1, "roster_id": 6, "player_id": "9001",
             "picked_by": "6", "metadata": {}},
        ],
    }])
    monkeypatch.setattr(MFLProvider, "_canonical_map",
                        lambda self, lid, season: {"9001": "P1", "9002": "P2"})
    rows = platform_api.get_draft_picks("mfl", "55", 2026)
    assert [r["player_id"] for r in rows] == ["P1", "P2"]
    assert [r["pick_no"] for r in rows] == [1, 2]


def test_facade_fleaflicker_picks_canonicalized(monkeypatch):
    from dashboard_services import platform_api
    from dashboard_services.providers.fleaflicker_api import FleaflickerProvider

    monkeypatch.setattr(FleaflickerProvider, "get_drafts",
                        lambda self, lid, season: [{
                            "draft_id": "fleaflicker:2026:77", "status": "complete",
                            "picks": [
                                {"round": 1, "pick_no": 1, "roster_id": 4,
                                 "player_id": "8001", "picked_by": "4",
                                 "metadata": {}},
                            ],
                        }])
    monkeypatch.setattr(FleaflickerProvider, "_canonical_map",
                        lambda self, lid, season: {"8001": "P1"})
    rows = platform_api.get_draft_picks("fleaflicker", "77", 2026)
    assert [r["player_id"] for r in rows] == ["P1"]


# ── /api/draft-grades ────────────────────────────────────────────────────────

_PLAYERS_INDEX = {
    "P1": {"pos": "RB", "name": "Runner One", "team": "BUF"},
    "P2": {"pos": "WR", "name": "Catcher Two", "team": "MIA"},
    "P3": {"pos": "QB", "name": "Passer Three", "team": "KC"},
}

_MODEL_ADP = {
    "P1": {"avg_pick": 1.0, "position": "RB"},
    "P2": {"avg_pick": 2.0, "position": "WR"},
    "P3": {"avg_pick": 3.0, "position": "QB"},
}


def _stub_draft_grades_app(appmod, monkeypatch):
    monkeypatch.setattr(appmod, "get_nfl_state", lambda *a, **k: {
        "season": 2026, "season_type": "regular", "week": 1})
    monkeypatch.setattr(appmod, "get_rosters", lambda *a, **k: [
        {"roster_id": 1, "players": []}, {"roster_id": 2, "players": []}])
    monkeypatch.setattr(appmod, "get_users", lambda *a, **k: [])
    monkeypatch.setattr(appmod, "get_league", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "load_players_index", lambda *a, **k: dict(_PLAYERS_INDEX))
    monkeypatch.setattr(appmod, "_fetch_league_adp_from_db", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_build_model_adp_fallback",
                        lambda *a, **k: dict(_MODEL_ADP))
    monkeypatch.setattr(appmod, "_build_league_players_payload",
                        lambda **k: {"players": [], "tier_thresholds": {}})
    monkeypatch.setattr(
        "data_building.rookie_pipeline.pipeline.is_draft_complete",
        lambda *a, **k: True)
    appmod._DRAFT_GRADES_CACHE.clear()


def test_draft_grades_espn_uses_platform_picks_not_sleeper_transport(monkeypatch):
    import app as appmod
    import dashboard_services.api as legacy_api
    from dashboard_services.providers import espn_api

    _stub_draft_grades_app(appmod, monkeypatch)

    def _no_sleeper_transport(path, *a, **k):
        raise AssertionError(f"Sleeper transport called: {path}")

    monkeypatch.setattr(legacy_api, "fetch_json", _no_sleeper_transport)
    monkeypatch.setattr(espn_api, "get_drafts", lambda season, lid: [{
        "draft_id": "espn_123_2026", "league_id": "123", "season": 2026,
        "status": "complete", "start_time": 1755000000000, "type": "snake",
        "settings": {"rounds": 3},
    }])
    monkeypatch.setattr(espn_api, "_espn_to_canon_cached",
                        lambda: {"101": "P1", "102": "P2", "103": "P3"})
    monkeypatch.setattr(espn_api, "iter_draft_picks", lambda season, lid: [
        {"playerId": 101, "teamId": 1, "overallPickNumber": 1,
         "roundId": 1, "roundPickNumber": 1},
        {"playerId": 102, "teamId": 2, "overallPickNumber": 2,
         "roundId": 1, "roundPickNumber": 2},
        {"playerId": 103, "teamId": 1, "overallPickNumber": 3,
         "roundId": 2, "roundPickNumber": 1},
    ])

    resp = appmod.app.test_client().get(
        "/api/draft-grades?platform=espn&league_id=123&season=2026")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["draft_id"] == "espn_123_2026"
    assert {t["roster_id"] for t in data["teams"]} == {"1", "2"}
    picked = {p["player_id"] for t in data["teams"] for p in t["picks"]}
    assert picked == {"P1", "P2", "P3"}


def test_draft_grades_sleeper_path_unchanged(monkeypatch):
    import app as appmod
    import dashboard_services.api as legacy_api

    _stub_draft_grades_app(appmod, monkeypatch)
    sleeper_picks = [
        {"player_id": "P1", "roster_id": 1, "picked_by": "u1", "pick_no": 1,
         "round": 1, "draft_slot": 1, "metadata": {}},
        {"player_id": "P2", "roster_id": 2, "picked_by": "u2", "pick_no": 2,
         "round": 1, "draft_slot": 2, "metadata": {}},
        {"player_id": "P3", "roster_id": 1, "picked_by": "u1", "pick_no": 3,
         "round": 2, "draft_slot": 1, "metadata": {}},
    ]
    # Sleeper picks keep flowing from the same /draft/{id}/picks source,
    # whether reached via fetch_json (old) or the facade's legacy delegate.
    monkeypatch.setattr(legacy_api, "fetch_json", lambda path, *a, **k: sleeper_picks)
    monkeypatch.setattr(legacy_api, "get_draft_picks", lambda draft_id: sleeper_picks)
    monkeypatch.setattr(appmod, "get_drafts", lambda *a, **k: [{
        "draft_id": "998877", "league_id": "L1", "season": 2026,
        "status": "complete", "start_time": 1755000000000,
        "settings": {"rounds": 3},
    }])

    resp = appmod.app.test_client().get(
        "/api/draft-grades?platform=sleeper&league_id=L1&season=2026")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["draft_id"] == "998877"
    assert {t["roster_id"] for t in data["teams"]} == {"1", "2"}


def test_draft_grades_espn_no_draft_data_is_404_not_500(monkeypatch):
    import app as appmod
    import dashboard_services.api as legacy_api
    from dashboard_services.providers import espn_api

    _stub_draft_grades_app(appmod, monkeypatch)

    def _no_sleeper_transport(path, *a, **k):
        raise AssertionError(f"Sleeper transport called: {path}")

    monkeypatch.setattr(legacy_api, "fetch_json", _no_sleeper_transport)
    monkeypatch.setattr(espn_api, "get_drafts", lambda season, lid: [{
        "draft_id": "espn_123_2026", "league_id": "123", "season": 2026,
        "status": "pre_draft", "start_time": 1755000000000, "type": "snake",
        "settings": {"rounds": 3},
    }])
    monkeypatch.setattr(espn_api, "iter_draft_picks", lambda season, lid: [])

    resp = appmod.app.test_client().get(
        "/api/draft-grades?platform=espn&league_id=123&season=2026")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


# ── get_player_acquisition_events ────────────────────────────────────────────

def _stub_acquisition_common(monkeypatch, plt, season_map):
    import dashboard_services.api as legacy_api
    import dashboard_services.service as service

    monkeypatch.setattr(legacy_api, "build_league_history_map",
                        lambda plat, lid, season: season_map)
    monkeypatch.setattr(service, "get_transactions_by_week", lambda *a, **k: {})
    monkeypatch.setattr(plt, "_roster_names",
                        lambda plat, lid, season: {"3": "Team Three"})


def test_acquisition_events_yahoo_include_drafted_event(monkeypatch):
    import dashboard_services.api as legacy_api
    from dashboard_services import platform_api
    import dashboard_services.player_league_trades as plt

    _stub_acquisition_common(monkeypatch, plt, {2026: "Y123"})

    def _no_legacy(*a, **k):
        raise AssertionError("legacy Sleeper draft helper called")

    monkeypatch.setattr(legacy_api, "get_drafts", _no_legacy)
    monkeypatch.setattr(legacy_api, "get_draft_picks", _no_legacy)
    monkeypatch.setattr(platform_api, "get_drafts", lambda plat, lid, season: [{
        "draft_id": "yahoo_Y123_2026", "status": "complete", "season": 2026}])
    monkeypatch.setattr(platform_api, "get_draft_picks",
                        lambda plat, lid, season, draft_id=None: [
                            {"player_id": "PID9", "roster_id": "3",
                             "picked_by": "3", "round": 2, "draft_slot": 1,
                             "pick_no": 15}])

    out = plt.get_player_acquisition_events(
        "PID9", platform="yahoo", league_id="Y123", season=2026)
    draft_events = [e for e in out["events"] if e["kind"] == "draft"]
    assert len(draft_events) == 1
    assert draft_events[0]["season"] == 2026
    assert draft_events[0]["round"] == 2
    assert draft_events[0]["team"] == "Team Three"


def test_acquisition_events_sleeper_unchanged(monkeypatch):
    import dashboard_services.api as legacy_api
    import dashboard_services.player_league_trades as plt

    _stub_acquisition_common(monkeypatch, plt, {2026: "S123"})
    monkeypatch.setattr(legacy_api, "get_drafts", lambda lid: [
        {"draft_id": "D1", "status": "complete", "season": 2026}])
    monkeypatch.setattr(legacy_api, "get_draft_picks", lambda draft_id: [
        {"player_id": "PID9", "roster_id": 3, "round": 2, "draft_slot": 1,
         "pick_no": 15}])

    out = plt.get_player_acquisition_events(
        "PID9", platform="sleeper", league_id="S123", season=2026)
    draft_events = [e for e in out["events"] if e["kind"] == "draft"]
    assert len(draft_events) == 1
    assert draft_events[0]["round"] == 2
    assert draft_events[0]["team"] == "Team Three"


# ── notify_rival_trades ──────────────────────────────────────────────────────

class _AppStateCursor:
    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return []


class _AppStateConn:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if "FROM app_state" in sql:
            value = self.store.get(params[0])
            return _AppStateCursor({"value": value} if value is not None else None)
        if "INSERT INTO app_state" in sql:
            self.store[params[0]] = params[1]
        return _AppStateCursor()

    def commit(self):
        pass


def test_rival_trades_yahoo_league_uses_platform_transactions(monkeypatch):
    import dashboard_services.api as legacy_api
    import dashboard_services.db as db
    import dashboard_services.platform_api as platform_api
    import utils.push_notifications as pn
    import utils.utils as uu

    monkeypatch.setattr(db, "get_conn", lambda: _AppStateConn({}))
    monkeypatch.setattr(legacy_api, "get_nfl_state", lambda: {
        "season": "2026", "season_type": "reg", "week": 4})

    def _no_legacy(*a, **k):
        raise AssertionError("legacy Sleeper get_transactions called")

    monkeypatch.setattr(legacy_api, "get_transactions", _no_legacy)

    calls = []

    def fake_platform_txns(platform, league_id, week, season):
        calls.append((platform, league_id, week, season))
        return [{"transaction_id": "TX1", "type": "trade",
                 "adds": {"PID9": 1}}]

    monkeypatch.setattr(platform_api, "get_transactions", fake_platform_txns)
    monkeypatch.setattr(uu, "load_model_value_table",
                        lambda: [{"id": "PID9", "value": 5000,
                                  "name": "Star Player"}])
    monkeypatch.setattr(pn, "_get_subscribed_leagues",
                        lambda: [("Y123", "yahoo")])
    monkeypatch.setattr(pn, "_league_display_name",
                        lambda plat, lid, season: "Yahoo League")
    sent = []
    monkeypatch.setattr(pn, "_broadcast_league",
                        lambda league_id, **k: sent.append((league_id, k)))

    pn.notify_rival_trades()

    assert calls == [("yahoo", "Y123", 4, 2026)]
    assert len(sent) == 1
    assert sent[0][0] == "Y123"
    assert sent[0][1]["notif_type"] == "rival_trades"
    assert "Star Player" in sent[0][1]["body"]
