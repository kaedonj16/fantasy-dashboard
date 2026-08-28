from contextlib import contextmanager

from dashboard_services import accounts


class FakeConnection:
    def __init__(self, membership, identities, fallback=None):
        self.membership = membership
        self.identities = identities
        self.fallback = fallback
        self.updated = []

    def execute(self, sql, params):
        if "ORDER BY season DESC" in sql:
            return FakeResult(self.fallback)
        if "SELECT team_id" in sql and "FROM user_leagues" in sql:
            return FakeResult(self.membership)
        if "SELECT platform_user_id,handle" in sql:
            return FakeResult(self.identities)
        if "UPDATE user_leagues SET team_id" in sql:
            self.updated.append(params)
            return FakeResult(None)
        raise AssertionError(sql)

    def commit(self):
        pass


class FakeResult:
    def __init__(self, value):
        self.value = value

    def fetchone(self):
        return self.value

    def fetchall(self):
        return self.value


def install_db(monkeypatch, membership, identities, fallback=None, espn_creds=None):
    connection = FakeConnection(membership, identities, fallback=fallback)

    @contextmanager
    def get_conn():
        yield connection

    monkeypatch.setattr(accounts, "init_accounts_tables", lambda: None)
    monkeypatch.setattr(accounts, "get_espn_league_credentials", lambda *a, **k: espn_creds)
    import dashboard_services.db
    monkeypatch.setattr(dashboard_services.db, "get_conn", get_conn)
    return connection


def test_stable_sleeper_owner_resolves_and_persists_league_roster(monkeypatch):
    connection = install_db(monkeypatch, {"team_id": None}, [
        {"platform_user_id": "stable-user-2", "handle": "second"},
    ])
    viewer = accounts.resolve_account_viewer_for_league(
        7, "sleeper", "league-a", 2026,
        [{"user_id": "stable-user-2", "username": "second", "display_name": "Second"}],
        [{"roster_id": 19, "owner_id": "stable-user-2", "metadata": {"team_name": "Team Two"}}],
    )
    assert viewer == {"viewer_username": "second", "viewer_user_id": "stable-user-2",
                      "viewer_roster_id": "19", "viewer_team_name": "Team Two"}
    assert connection.updated == [("19", 7, "sleeper", "league-a", 2026)]


def test_stored_team_is_scoped_to_selected_league(monkeypatch):
    install_db(monkeypatch, {"team_id": "42"}, [
        {"platform_user_id": "owner-b", "handle": "owner"},
    ])
    viewer = accounts.resolve_account_viewer_for_league(
        7, "sleeper", "league-b", 2026,
        [{"user_id": "owner-b", "username": "owner"}],
        [{"roster_id": 5, "owner_id": "somebody-else"},
         {"roster_id": 42, "owner_id": "owner-b"}],
    )
    assert viewer["viewer_roster_id"] == "42"


def test_identity_from_another_platform_cannot_claim_roster(monkeypatch):
    install_db(monkeypatch, {"team_id": None}, [])
    viewer = accounts.resolve_account_viewer_for_league(
        7, "espn", "league-c", 2026,
        [{"user_id": "sleeper-owner", "username": "same-name"}],
        [{"roster_id": 3, "owner_id": "sleeper-owner"}],
    )
    assert viewer is None


def test_espn_viewer_resolves_when_saved_season_lags_current(monkeypatch):
    connection = install_db(
        monkeypatch, None,
        [{"platform_user_id": "{SW-1}", "handle": None}],
        fallback={"team_id": None, "season": 2025},
    )
    viewer = accounts.resolve_account_viewer_for_league(
        7, "espn", "league-e", 2026,
        [{"user_id": "{SW-1}", "display_name": "Kaedon"}],
        [{"roster_id": 4, "owner_id": "{SW-1}", "metadata": {"team_name": "Biters"}}],
    )
    assert viewer["viewer_roster_id"] == "4"
    assert viewer["viewer_team_name"] == "Biters"
    assert connection.updated == [("4", 7, "espn", "league-e", 2025)]


def test_espn_swid_without_braces_still_matches_roster(monkeypatch):
    install_db(monkeypatch, {"team_id": None, "season": 2026}, [
        {"platform_user_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "handle": None},
    ])
    viewer = accounts.resolve_account_viewer_for_league(
        7, "espn", "league-f", 2026,
        [{"user_id": "{aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee}"}],
        [{"roster_id": 2, "owner_id": "{aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee}"}],
    )
    assert viewer["viewer_roster_id"] == "2"


def test_espn_stored_cookies_resolve_team_without_saved_team_id(monkeypatch):
    connection = install_db(
        monkeypatch, {"team_id": None, "season": 2026}, [],
        espn_creds={"swid": "{SW-COOKIE}"},
    )
    viewer = accounts.resolve_account_viewer_for_league(
        7, "espn", "league-g", 2026, [],
        [{"roster_id": 9, "owner_id": "{SW-COOKIE}"}],
    )
    assert viewer["viewer_roster_id"] == "9"
    assert connection.updated == [("9", 7, "espn", "league-g", 2026)]


def test_espn_connect_stores_swid_as_platform_identity():
    source = open("dashboard_services/accounts.py", encoding="utf-8").read()
    fn = source.split("def add_espn_league_connection")[1].split("\ndef ")[0]
    assert 'link_platform_identity(account_id, "espn"' in fn
