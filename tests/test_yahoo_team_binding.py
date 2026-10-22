"""Yahoo team-binding fix: ownership comes from the saved binding, never
from Yahoo payload guids.

Real payload shape behind every fake here (verified live in production
2026-10-22 for league 1307110): Yahoo serves the FETCHING CALLER's guid as
the manager guid on every team -- one manager entry per team, distinct
manager_id/nickname/felo per team, and no ``is_current_login`` /
``is_owned_by_current_login`` flags anywhere. So in these fakes every
roster's owner_id and every user's user_id is the same "CALLER-GUID",
while each row's roster_id is that team's own id (4 = the subscriber's
team "Dicker? I hardly know her", 6 = "FromElwaytoNix").
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest import mock

import pytest

import utils.push_notifications as pn
from utils.league import resolve_viewer_for_league

CALLER = "CALLER-GUID"
LEAGUE = "1307110"


def _yahoo_rosters():
    """get_rosters shape: owner_id is the caller's guid on ALL rosters."""
    return [
        {"roster_id": i, "owner_id": CALLER,
         "players": [f"p{i}"], "starters": [f"p{i}"],
         "metadata": {"team_name": f"Team {i}"}}
        for i in range(1, 11)
    ]


def _yahoo_users():
    """get_users shape: user_id is the caller's guid on ALL rows; each row
    still carries its own team's roster_id and team name."""
    return [
        {"user_id": CALLER, "username": f"mgr{i}", "display_name": f"Mgr {i}",
         "roster_id": i, "metadata": {"team_name": f"Team {i}"}}
        for i in range(1, 11)
    ]


# ── A. ScoreZone TD push targeting ───────────────────────────────────────────

class _R:
    def __init__(self, row=None, rows=None):
        self._row, self._rows = row, rows or []

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Dispatches the exact queries the TD path issues: app_state claims,
    the user_leagues binding lookup, and the account_key subscription
    lookup. Rows are dicts, like production get_conn."""

    def __init__(self, bindings, subs):
        self.bindings = bindings      # {(league_id, team_id): [row, ...]}
        self.subs = subs              # [push_subscriptions row, ...]
        self.claimed = set()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def execute(self, q, params=None):
        if "ON CONFLICT (key) DO NOTHING" in q:
            key = params[0]
            if key in self.claimed:
                return _R(None)
            self.claimed.add(key)
            return _R({"key": key})
        if "FROM user_leagues" in q:
            return _R(rows=list(self.bindings.get((params[0], params[1]), [])))
        if "FROM push_subscriptions" in q and "account_key = ANY" in q:
            wanted = set(params[1])
            return _R(rows=[s for s in self.subs
                            if s["league_id"] == params[0]
                            and s["account_key"] in wanted])
        return _R(None)


def _run_yahoo_notify(pbp, rosters, bindings, subs, sends):
    conn = _FakeConn(bindings, subs)
    fake_db = types.ModuleType("dashboard_services.db")
    fake_db.get_conn = lambda: conn
    pkg = sys.modules.get("dashboard_services") or types.ModuleType("dashboard_services")
    with mock.patch.dict(sys.modules, {"dashboard_services": pkg,
                                       "dashboard_services.db": fake_db}), \
         mock.patch.object(pn, "_league_display_name", lambda *a: "Yahoo League"), \
         mock.patch.object(
             pn, "_send_with_digest",
             side_effect=lambda rows, *a, **k: sends.extend(rows) or len(rows)), \
         mock.patch.object(
             pn, "_broadcast_owner",
             side_effect=AssertionError("guid broadcast must not run for Yahoo")):
        return pn.notify_scorezone_scores(
            LEAGUE, "yahoo", pbp, _PLAYER_INFO, rosters, _SCORING,
            season=2026, week=7,
        )


_PLAYER_INFO = {
    "p4": {"name": "Her Starter", "pos": "WR"},
    "p6": {"name": "Their Starter", "pos": "RB"},
}
_SCORING = {"rush_td": 6.0, "rec_td": 6.0, "pass_td": 4.0}


def _td(pid, play_id):
    return {"pid": pid, "play_id": play_id, "is_td": True, "play_state": "VALID",
            "stat_line": {"rec_td": 1}, "play_text": "TOUCHDOWN"}


_BINDINGS = {(LEAGUE, "4"): [{"account_id": 1, "season": 2026}]}
_SUBS = [{"endpoint": "https://push/1", "p256dh": "k", "auth": "a", "prefs": None,
          "platform": "yahoo", "account_key": "1", "league_id": LEAGUE}]


def test_yahoo_td_on_unbound_roster_goes_to_nobody():
    sends = []
    n = _run_yahoo_notify({"g1": [_td("p6", "play6")]}, _yahoo_rosters(),
                          _BINDINGS, _SUBS, sends)
    assert n == 0
    assert sends == []


def test_yahoo_td_on_bound_roster_goes_to_binding_owner():
    sends = []
    n = _run_yahoo_notify({"g1": [_td("p4", "play4")]}, _yahoo_rosters(),
                          _BINDINGS, _SUBS, sends)
    assert n == 1
    assert [s["account_key"] for s in sends] == ["1"]


def test_yahoo_binding_account_ids_season_preference():
    conn = _FakeConn({
        (LEAGUE, "4"): [{"account_id": 9, "season": 2025},
                        {"account_id": 1, "season": 2026}],
        (LEAGUE, "6"): [{"account_id": 9, "season": 2025}],
    }, [])
    fake_db = types.ModuleType("dashboard_services.db")
    fake_db.get_conn = lambda: conn
    pkg = sys.modules.get("dashboard_services") or types.ModuleType("dashboard_services")
    with mock.patch.dict(sys.modules, {"dashboard_services": pkg,
                                       "dashboard_services.db": fake_db}):
        # Current-season binding wins over last season's for the same team.
        assert pn._yahoo_binding_account_ids(LEAGUE, 2026, "4") == ["1"]
        # No season known: newest binding wins.
        assert pn._yahoo_binding_account_ids(LEAGUE, None, "4") == ["1"]
        # Only a prior-season binding exists: it is still the newest record.
        assert pn._yahoo_binding_account_ids(LEAGUE, 2026, "6") == ["9"]
        # No binding at all: nobody.
        assert pn._yahoo_binding_account_ids(LEAGUE, 2026, "8") == []


# ── B. Login-team capture (yahoo_api helper + OAuth callback) ────────────────

def _login_teams_payload(*team_keys):
    teams = {str(i): {"team": [{"team_key": tk, "team_id": tk.rsplit(".t.", 1)[-1]}]}
             for i, tk in enumerate(team_keys)}
    return {"fantasy_content": {"users": {"0": {"user": [
        {"guid": CALLER},
        {"games": {"0": {"game": [{"game_key": "461"}, {"teams": teams}]}}},
    ]}}}}


def test_get_login_team_id_reads_login_users_own_teams(monkeypatch):
    from dashboard_services.providers import yahoo_api

    payload = _login_teams_payload("461.l.999999.t.2", "461.l.1307110.t.4")
    monkeypatch.setattr(yahoo_api, "_nfl_game_keys", lambda tok: [(2026, "461")])
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda tok, path: payload)
    assert yahoo_api.get_login_team_id("tok", LEAGUE, 2026) == "4"
    # A league the login user has no team in: unknown, never a guess.
    assert yahoo_api.get_login_team_id("tok", "555555", 2026) is None


def test_get_login_team_id_none_when_lookup_fails(monkeypatch):
    from dashboard_services.providers import yahoo_api

    def boom(tok, path):
        raise RuntimeError("403")

    monkeypatch.setattr(yahoo_api, "_nfl_game_keys", lambda tok: [(2026, "461")])
    monkeypatch.setattr(yahoo_api, "_yahoo_get", boom)
    assert yahoo_api.get_login_team_id("tok", LEAGUE, 2026) is None


flask = pytest.importorskip("flask")


def _run_yahoo_callback(monkeypatch, login_team_id):
    from routes.auth_bp import auth_bp

    attached = []
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.exchange_code_for_tokens",
        lambda code: {"xoauth_yahoo_guid": CALLER, "access_token": "token",
                      "refresh_token": "refresh", "expires_in": 3600},
    )
    monkeypatch.setattr("dashboard_services.providers.yahoo_api.save_tokens", lambda *a: None)
    monkeypatch.setattr("dashboard_services.providers.yahoo_api.save_league_owner", lambda *a: None)
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.resolve_league_key",
        lambda *a: {"status": "found", "season": 2026, "name": "Yahoo League"},
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_login_team_id",
        lambda *a: login_team_id,
    )
    monkeypatch.setattr("dashboard_services.api.get_nfl_state", lambda: {"season": 2026})
    monkeypatch.setattr(
        "dashboard_services.accounts.link_platform_identity",
        lambda *a, **k: attached.append(("identity", a, k)),
    )
    monkeypatch.setattr(
        "dashboard_services.accounts.add_user_league",
        lambda *a, **k: attached.append(("league", a, k)),
    )
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(auth_bp)
    with app.test_client() as client:
        with client.session_transaction() as session:
            session["yahoo_oauth_state"] = "state"
            session["yahoo_oauth_ctx"] = {"league_id": LEAGUE, "team_name": "My Team"}
            session["account_id"] = 77
        response = client.get("/auth/yahoo/callback?code=code&state=state")
    assert response.status_code == 302
    league_calls = [c for c in attached if c[0] == "league"]
    assert len(league_calls) == 1
    return league_calls[0]


def test_callback_stores_login_teams_team_id(monkeypatch):
    call = _run_yahoo_callback(monkeypatch, "4")
    assert call[1][:3] == (77, "yahoo", LEAGUE)
    assert call[2]["team_id"] == "4"


def test_callback_stores_null_team_when_login_teams_indeterminate(monkeypatch):
    # get_users would offer roster 1 first (every user_id is the caller's
    # guid); the binding must store NO team rather than that first-team guess.
    call = _run_yahoo_callback(monkeypatch, None)
    assert call[2]["team_id"] is None


# ── C. Username sign-in resolution (utils/league.py) ─────────────────────────

def test_yahoo_username_resolution_uses_users_own_roster_id():
    users = _yahoo_users()
    users[5]["metadata"]["team_name"] = "FromElwaytoNix"
    viewer = resolve_viewer_for_league(users, _yahoo_rosters(), "FromElwaytoNix")
    assert viewer is not None
    # First-match over owner_ids would return roster 1 (all owner_ids are
    # the caller's guid); the matched user's own roster_id is 6.
    assert viewer["viewer_roster_id"] == "6"


def test_sleeper_resolution_without_roster_id_still_owner_matches():
    users = [
        {"user_id": "u1", "username": "one", "display_name": "One"},
        {"user_id": "u2", "username": "two", "display_name": "Two"},
    ]
    rosters = [
        {"roster_id": 1, "owner_id": "u1", "metadata": {"team_name": "T1"}},
        {"roster_id": 2, "owner_id": "u2", "metadata": {"team_name": "T2"}},
    ]
    viewer = resolve_viewer_for_league(users, rosters, "two")
    assert viewer["viewer_roster_id"] == "2"


# ── D. Account viewer resolution (accounts.py) ───────────────────────────────

from dashboard_services import accounts


class _AcctResult:
    def __init__(self, value):
        self.value = value

    def fetchone(self):
        return self.value

    def fetchall(self):
        return self.value


class _AcctConn:
    def __init__(self, membership, identities):
        self.membership = membership
        self.identities = identities
        self.updated = []

    def execute(self, sql, params):
        if "ORDER BY season DESC" in sql:
            return _AcctResult(None)
        if "SELECT team_id" in sql and "FROM user_leagues" in sql:
            return _AcctResult(self.membership)
        if "SELECT platform_user_id,handle" in sql:
            return _AcctResult(self.identities)
        if "UPDATE user_leagues SET team_id" in sql:
            self.updated.append(params)
            return _AcctResult(None)
        raise AssertionError(sql)

    def commit(self):
        pass


def _install_accounts_db(monkeypatch, membership, identities):
    conn = _AcctConn(membership, identities)

    @contextmanager
    def get_conn():
        yield conn

    monkeypatch.setattr(accounts, "init_accounts_tables", lambda: None)
    import dashboard_services.db
    monkeypatch.setattr(dashboard_services.db, "get_conn", get_conn)
    return conn


def test_yahoo_pointer_present_resolution_unchanged(monkeypatch):
    conn = _install_accounts_db(
        monkeypatch, {"team_id": "4", "season": 2026},
        [{"platform_user_id": "HER-GUID", "handle": "her"}],
    )
    monkeypatch.setattr(
        accounts, "_yahoo_login_team_id",
        lambda *a: (_ for _ in ()).throw(AssertionError("login lookup must not run")),
    )
    viewer = accounts.resolve_account_viewer_for_league(
        1, "yahoo", LEAGUE, 2026, _yahoo_users(), _yahoo_rosters(),
    )
    assert viewer["viewer_roster_id"] == "4"
    assert conn.updated == []


def test_yahoo_no_pointer_resolves_via_login_teams_not_owner_match(monkeypatch):
    conn = _install_accounts_db(
        monkeypatch, {"team_id": None, "season": 2026},
        [{"platform_user_id": "HER-GUID", "handle": "her"}],
    )
    # Rosters fetched with HER token: every owner_id is her guid, so the old
    # owner match returned roster 1. Login-teams says her team is 4.
    rosters = _yahoo_rosters()
    for r in rosters:
        r["owner_id"] = "HER-GUID"
    monkeypatch.setattr(accounts, "_yahoo_login_team_id", lambda *a: "4")
    viewer = accounts.resolve_account_viewer_for_league(
        1, "yahoo", LEAGUE, 2026, _yahoo_users(), rosters,
    )
    assert viewer["viewer_roster_id"] == "4"
    assert conn.updated == [("4", 1, "yahoo", LEAGUE, 2026)]


def test_yahoo_no_pointer_and_login_teams_unknown_resolves_nothing(monkeypatch):
    _install_accounts_db(
        monkeypatch, {"team_id": None, "season": 2026},
        [{"platform_user_id": "HER-GUID", "handle": "her"}],
    )
    rosters = _yahoo_rosters()
    for r in rosters:
        r["owner_id"] = "HER-GUID"
    monkeypatch.setattr(accounts, "_yahoo_login_team_id", lambda *a: None)
    viewer = accounts.resolve_account_viewer_for_league(
        1, "yahoo", LEAGUE, 2026, _yahoo_users(), rosters,
    )
    assert viewer is None
