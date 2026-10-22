"""Yahoo team-binding routing for EVERY owner-targeted notification type.

The ScoreZone TD fix (#2779) routed Yahoo touchdown pushes through the
saved user_leagues team binding, but the other owner-targeted senders kept
matching roster owner_id against subscription owner_id -- and Yahoo serves
the FETCHING CALLER's guid as every team's manager guid, so each of those
types handed alerts to whichever subscriber's token fetched the rosters
(Kaedon, for league 1307110) and starved the real owners.

These tests use the real captured payload shape (verified live 2026-10-22):
every roster's owner_id and every user's user_id is the same "CALLER-GUID",
one manager per team, no is_current_login flags. The binding says account 1
owns roster 4 ("Dicker? I hardly know her"); roster 6 is "FromElwaytoNix"
and belongs to nobody with a subscription here. For each type: an event on
roster 6 sends to NOBODY, an event on roster 4 sends ONLY to account 1's
device -- even though account 2's subscription row also carries the caller
guid as its owner_id (guid matching would have sent everything to both).
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import sys
import types
from unittest import mock

import utils.push_notifications as pn

CALLER = "CALLER-GUID"
LEAGUE = "1307110"
SEASON = 2026

_BINDINGS = {(LEAGUE, "4"): [{"account_id": 1, "season": 2026}]}

_SUBS = [
    {"endpoint": "https://push/her", "p256dh": "k", "auth": "a", "prefs": None,
     "platform": "yahoo", "account_key": "1", "owner_id": CALLER,
     "league_id": LEAGUE},
    {"endpoint": "https://push/unbound", "p256dh": "k", "auth": "a", "prefs": None,
     "platform": "yahoo", "account_key": "2", "owner_id": CALLER,
     "league_id": LEAGUE},
]


def _rosters(settings=None):
    out = []
    for i in range(1, 11):
        r = {"roster_id": i, "owner_id": CALLER,
             "players": [f"p{i}"], "starters": [f"p{i}"],
             "metadata": {"team_name": f"Team {i}"}}
        if settings is not None:
            r["settings"] = settings.get(
                i, {"wins": 0, "fpts": 0, "fpts_decimal": 0})
        out.append(r)
    return out


def _users():
    return [
        {"user_id": CALLER, "username": f"mgr{i}", "display_name": f"Mgr {i}",
         "roster_id": i, "metadata": {"team_name": f"Team {i}"}}
        for i in range(1, 11)
    ]


class _R:
    def __init__(self, row=None, rows=None):
        self._row, self._rows = row, rows or []

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Serves every query the notifiers issue against the fake DB: the
    user_leagues binding lookup, both push_subscriptions shapes, the
    playoff_odds table, and app_state get/set/claim. Dict rows, like prod."""

    def __init__(self, bindings=None, subs=None, app_state=None,
                 playoff_rows=None):
        self.bindings = bindings if bindings is not None else _BINDINGS
        self.subs = subs if subs is not None else _SUBS
        self.app_state = app_state if app_state is not None else {}
        self.playoff_rows = playoff_rows or []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def execute(self, q, params=None):
        if "FROM user_leagues" in q:
            return _R(rows=list(self.bindings.get((params[0], params[1]), [])))
        if "FROM playoff_odds" in q:
            return _R(rows=list(self.playoff_rows))
        if "FROM push_subscriptions" in q:
            if "account_key = ANY" in q:
                wanted = set(params[1])
                return _R(rows=[s for s in self.subs
                                if s["league_id"] == params[0]
                                and s["account_key"] in wanted])
            return _R(rows=[s for s in self.subs
                            if s["league_id"] == params[0]])
        if "SELECT value FROM app_state" in q:
            v = self.app_state.get(params[0])
            return _R({"value": v} if v is not None else None)
        if "INSERT INTO app_state" in q:
            if "DO NOTHING" in q:
                if params[0] in self.app_state:
                    return _R(None)
                self.app_state[params[0]] = params[1]
                return _R({"key": params[0]})
            self.app_state[params[0]] = params[1]
            return _R(None)
        return _R(None)


def _install(monkeypatch, conn, sends):
    import dashboard_services.db as db

    monkeypatch.setattr(db, "get_conn", lambda: conn)
    monkeypatch.setattr(pn, "_get_subscribed_leagues",
                        lambda: [(LEAGUE, "yahoo")])
    monkeypatch.setattr(pn, "_league_display_name", lambda *a: "Yahoo League")

    def _capture(rows, title, body, url="/", tag="update", notif_type=None,
                 league_id=None, platform=None):
        rows = list(rows)
        sends.append({"rows": rows, "title": title, "body": body})
        return len(rows)

    monkeypatch.setattr(pn, "_send_with_digest", _capture)


def _patch_state(monkeypatch, **state):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_state", lambda: dict(state))


def _assert_only_her_device(sends, title=None):
    assert len(sends) == 1
    if title is not None:
        assert sends[0]["title"] == title
    assert [r["account_key"] for r in sends[0]["rows"]] == ["1"]
    assert [r["endpoint"] for r in sends[0]["rows"]] == ["https://push/her"]


# ── The shared resolver + dispatcher ────────────────────────────────────────

def test_single_binding_resolver_implementation():
    # One implementation, two names: the TD fix's original helper IS the
    # generalized resolver every sender now shares.
    assert pn._yahoo_binding_account_ids is pn.yahoo_roster_account_ids


def test_dispatcher_yahoo_routes_by_binding_never_guid(monkeypatch):
    sends = []
    _install(monkeypatch, _FakeConn(), sends)
    monkeypatch.setattr(
        pn, "_broadcast_owner",
        mock.Mock(side_effect=AssertionError("guid match must not run for Yahoo")))
    # Roster 6 (no binding): nobody, even though owner_id == every sub's.
    assert pn._broadcast_owner_for_roster(
        "yahoo", LEAGUE, SEASON, 6, CALLER, "T", "B") == 0
    assert sends == []
    # Roster 4 (bound to account 1): only her device.
    assert pn._broadcast_owner_for_roster(
        "yahoo", LEAGUE, SEASON, 4, CALLER, "T", "B") == 1
    _assert_only_her_device(sends, title="T")


def test_dispatcher_non_yahoo_still_guid_matches(monkeypatch):
    seen = []
    monkeypatch.setattr(
        pn, "_broadcast_owner",
        lambda league_id, owner_id, *a, **k: seen.append((league_id, owner_id)) or 3)
    assert pn._broadcast_owner_for_roster(
        "sleeper", "L9", SEASON, 4, "owner-x", "T", "B") == 3
    assert seen == [("L9", "owner-x")]


# ── Value drops ─────────────────────────────────────────────────────────────

def test_yahoo_value_drops_route_by_binding(monkeypatch):
    import dashboard_services.platform_api as papi

    sends = []
    _install(monkeypatch, _FakeConn(), sends)
    _patch_state(monkeypatch, season=SEASON, week=7, season_type="reg")
    # Stubbed in sys.modules, not imported: the real player_value_history
    # pulls numpy, which the lightweight unit shard does not install (the
    # notifier itself imports it lazily at call time, so the stub is what
    # its `from ... import get_top_movers` resolves to).
    pvh = types.ModuleType("dashboard_services.player_value_history")
    pvh.get_top_movers = lambda days=7, limit=50: {
        "fallers": [
            {"player_id": "p4", "delta": -80, "name": "Her Guy"},
            {"player_id": "p6", "delta": -70, "name": "Their Guy"},
        ], "risers": []}
    monkeypatch.setitem(sys.modules, "dashboard_services.player_value_history", pvh)
    monkeypatch.setattr(papi, "get_rosters", lambda *a: _rosters())

    pn.notify_value_drops()
    _assert_only_her_device(sends, title="Dynasty value dropping")
    assert "Her Guy" in sends[0]["body"]


# ── Playoff odds swings ─────────────────────────────────────────────────────

def test_yahoo_playoff_odds_route_by_binding(monkeypatch):
    import dashboard_services.platform_api as papi

    playoff_rows = [
        {"roster_id": 4, "week": 6, "playoff_probability": 40.0},
        {"roster_id": 4, "week": 7, "playoff_probability": 65.0},
        {"roster_id": 6, "week": 6, "playoff_probability": 60.0},
        {"roster_id": 6, "week": 7, "playoff_probability": 35.0},
    ]
    sends = []
    _install(monkeypatch, _FakeConn(playoff_rows=playoff_rows), sends)
    _patch_state(monkeypatch, season=SEASON, week=7, season_type="reg")
    monkeypatch.setattr(papi, "get_rosters", lambda *a: _rosters())

    pn.notify_playoff_odds()
    _assert_only_her_device(sends, title="Playoff picture update")
    assert "moved up to 65%" in sends[0]["body"]


# ── Breakout candidates on roster ───────────────────────────────────────────

def test_yahoo_breakout_roster_routes_by_binding(monkeypatch):
    import dashboard_services.breakout_api as bapi
    import dashboard_services.platform_api as papi

    sends = []
    _install(monkeypatch, _FakeConn(), sends)
    _patch_state(monkeypatch, season=SEASON, week=7, season_type="reg")
    monkeypatch.setattr(bapi, "get_breakout_candidates", lambda **k: {
        "candidates": [
            {"player_id": "p4", "player_name": "Her Guy", "position": "WR",
             "team": "DET", "breakout_opportunity_score": 90},
            {"player_id": "p6", "player_name": "Their Guy", "position": "RB",
             "team": "GB", "breakout_opportunity_score": 85},
        ]})
    monkeypatch.setattr(papi, "get_rosters", lambda *a: _rosters())

    pn.notify_breakout_roster()
    _assert_only_her_device(sends, title="Breakout candidate on your roster")
    assert "Her Guy" in sends[0]["body"]


# ── Matchup preview ─────────────────────────────────────────────────────────

class _TuesdayDate(_dt.date):
    # notify_matchup_preview's gate is weekday() == 1 (Tuesday), whatever
    # its "Monday" docstring says; 2026-10-27 is a Tuesday.
    @classmethod
    def today(cls):
        return cls(2026, 10, 27)


def test_yahoo_matchup_preview_routes_by_binding(monkeypatch):
    import dashboard_services.platform_api as papi

    sends = []
    _install(monkeypatch, _FakeConn(), sends)
    _patch_state(monkeypatch, season=SEASON, week=8, season_type="reg")
    monkeypatch.setattr(_dt, "date", _TuesdayDate)
    monkeypatch.setattr(papi, "get_matchups", lambda *a: [
        {"roster_id": 4, "matchup_id": 1},
        {"roster_id": 6, "matchup_id": 1},
    ])
    monkeypatch.setattr(papi, "get_rosters", lambda *a: _rosters())
    monkeypatch.setattr(papi, "get_users", lambda *a: _users())

    pn.notify_matchup_preview()
    _assert_only_her_device(sends, title="Week 8 matchup preview")
    # Opponent named from the user row carrying roster 6, not the guid map
    # (every user_id is the caller's guid, so the old map named a random
    # manager).
    assert "facing Mgr 6" in sends[0]["body"]


# ── Standings update ────────────────────────────────────────────────────────

class _WednesdayDate(_dt.date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 28)  # a Wednesday


def test_yahoo_standings_update_routes_by_binding(monkeypatch):
    import dashboard_services.platform_api as papi

    app_state = {
        "standings_positions_2026_7": _json.dumps({"4": 1, "6": 2}),
    }
    settings = {i: {"wins": 0, "fpts": i, "fpts_decimal": 0}
                for i in range(1, 11)}
    settings[6] = {"wins": 7, "fpts": 900, "fpts_decimal": 0}   # now 1st (in)
    settings[4] = {"wins": 6, "fpts": 800, "fpts_decimal": 0}   # now 2nd (out)
    sends = []
    _install(monkeypatch, _FakeConn(app_state=app_state), sends)
    _patch_state(monkeypatch, season=SEASON, week=8, season_type="reg")
    monkeypatch.setattr(_dt, "date", _WednesdayDate)
    monkeypatch.setattr(papi, "get_league", lambda *a: {
        "settings": {"playoff_teams": 1}, "name": "Yahoo League"})
    monkeypatch.setattr(papi, "get_rosters",
                        lambda *a: _rosters(settings=settings))

    pn.notify_standings_update()
    # Roster 4 dropped out of the (single) playoff spot -> her device.
    # Roster 6 moved in -> its owner has no binding -> nobody.
    _assert_only_her_device(sends, title="You dropped out of playoff position")


# ── Close game ──────────────────────────────────────────────────────────────

class _MondayNightDateTime(_dt.datetime):
    _now = _dt.datetime(2026, 10, 26, 23, 30, tzinfo=_dt.timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls._now.astimezone(tz) if tz else cls._now


def test_yahoo_close_game_routes_by_binding(monkeypatch):
    import dashboard_services.platform_api as papi

    sends = []
    _install(monkeypatch, _FakeConn(), sends)
    _patch_state(monkeypatch, season=SEASON, week=8, season_type="reg")
    monkeypatch.setattr(pn, "datetime", _MondayNightDateTime)
    monkeypatch.setattr(papi, "get_matchups", lambda *a: [
        {"roster_id": 4, "matchup_id": 1, "points": 55.0},
        {"roster_id": 6, "matchup_id": 1, "points": 50.0},
    ])
    monkeypatch.setattr(papi, "get_rosters", lambda *a: _rosters())
    monkeypatch.setattr(papi, "get_users", lambda *a: _users())

    assert pn.notify_close_game() == 1
    _assert_only_her_device(sends, title="Close matchup tonight")
    assert "You're up 5.0 pts over Mgr 6" in sends[0]["body"]


# ── Injury flips ────────────────────────────────────────────────────────────

_FRIDAY_NOON_UTC = _dt.datetime(2026, 9, 25, 12, 0, tzinfo=_dt.timezone.utc)
_KICKOFF_FUTURE = int(_dt.datetime(2026, 9, 26, 0, 15,
                                   tzinfo=_dt.timezone.utc).timestamp())


class _FridayDateTime(_dt.datetime):
    _now = _FRIDAY_NOON_UTC

    @classmethod
    def now(cls, tz=None):
        return cls._now.astimezone(tz) if tz else cls._now


def test_yahoo_injury_flip_routes_by_binding(monkeypatch):
    import dashboard_services.injury_return as ir
    import dashboard_services.platform_api as papi
    from utils import data_cache

    app_state = {
        "injury_last_seen_2026_4": _json.dumps(
            {f"{LEAGUE}:p4": "Questionable", f"{LEAGUE}:p6": "Questionable"}),
    }
    sends = []
    _install(monkeypatch, _FakeConn(app_state=app_state), sends)
    _patch_state(monkeypatch, season=SEASON, week=4, season_type="reg")
    monkeypatch.setattr(pn, "datetime", _FridayDateTime)
    monkeypatch.setattr(ir, "refresh_espn_return_dates", lambda force=False: {
        "p4": {"status": "Out", "team": "DET", "name": "Her Guy"},
        "p6": {"status": "Out", "team": "DET", "name": "Their Guy"},
    })
    monkeypatch.setattr(data_cache, "load_week_schedule", lambda s, w: [
        {"home": "DET", "away": "KC", "gameTime_epoch": _KICKOFF_FUTURE}])
    monkeypatch.setattr(papi, "get_rosters", lambda *a: _rosters())

    assert pn.notify_injury_flip() == 1
    _assert_only_her_device(sends, title="Starter injury alert")
    assert "Her Guy" in sends[0]["body"]


# ── Lineup lock ─────────────────────────────────────────────────────────────

def test_yahoo_lineup_lock_routes_by_binding(monkeypatch):
    import dashboard_services.api as api
    import dashboard_services.platform_api as papi

    # Keep the projection build out of this test: no bench-swap scan, the
    # hard-issue path is what routes.
    app_stub = types.ModuleType("app")
    app_stub.build_projections_by_week = lambda *a: {}
    monkeypatch.setitem(sys.modules, "app", app_stub)

    sends = []
    _install(monkeypatch, _FakeConn(), sends)
    monkeypatch.setattr(api, "get_nfl_players", lambda: {
        "p4": {"full_name": "Her Guy", "team": "DET", "injury_status": "Out"},
        "p6": {"full_name": "Their Guy", "team": "KC", "injury_status": "Out"},
    })
    monkeypatch.setattr(papi, "get_league",
                        lambda *a: {"roster_positions": ["QB", "RB", "WR"]})
    monkeypatch.setattr(papi, "get_rosters", lambda *a: _rosters())

    games = [{"home": "DET", "away": "KC", "gameTime_epoch": 1999999999}]
    pn._lineup_lock_send(
        games, SEASON, 4,
        dedupe_key="lineup_lock_day", dedupe_value="test",
        tag="lineup-lock-test", kickoff_line="Kickoff soon.",
        soon_line="Lock soon.", log_label="test", week_games=games)

    _assert_only_her_device(sends, title="Your lineup needs attention")
    assert "Her Guy" in sends[0]["body"]
