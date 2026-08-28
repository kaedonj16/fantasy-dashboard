"""Fleaflicker viewer resolution for personalized dashboard features."""
from contextlib import contextmanager

from dashboard_services import accounts


class FakeConnection:
    def __init__(self, membership, identities):
        self.membership = membership
        self.identities = identities
        self.updated = []

    def execute(self, sql, params):
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


def install_db(monkeypatch, membership, identities):
    connection = FakeConnection(membership, identities)

    @contextmanager
    def get_conn():
        yield connection

    monkeypatch.setattr(accounts, "init_accounts_tables", lambda: None)
    import dashboard_services.db
    monkeypatch.setattr(dashboard_services.db, "get_conn", get_conn)
    return connection


def test_fleaflicker_resolves_team_from_stored_flea_user_id(monkeypatch):
    install_db(monkeypatch, {"team_id": None, "season": 2026}, [])
    monkeypatch.setattr(
        accounts,
        "get_provider_league_credentials",
        lambda *args: {"flea_user_id": "532417", "token": "abc"},
    )
    users = [{
        "user_id": "1020439",
        "roster_id": 1020439,
        "display_name": "East Bay Biters",
        "metadata": {"team_name": "East Bay Biters", "flea_owner_id": "532417"},
    }]
    rosters = [{
        "roster_id": 1020439,
        "owner_id": "1020439",
        "metadata": {"team_name": "East Bay Biters"},
    }]
    viewer = accounts.resolve_account_viewer_for_league(
        7, "fleaflicker", "92916", 2026, users, rosters,
    )
    assert viewer is not None
    assert viewer["viewer_roster_id"] == "1020439"
    assert viewer["viewer_team_name"] == "East Bay Biters"


def test_fleaflicker_resolves_team_from_platform_identity_owner_id(monkeypatch):
    install_db(monkeypatch, {"team_id": None, "season": 2026}, [
        {"platform_user_id": "532417", "handle": "manager"},
    ])
    monkeypatch.setattr(
        accounts,
        "get_provider_league_credentials",
        lambda *args: {},
    )
    users = [{
        "user_id": "1020439",
        "roster_id": 1020439,
        "display_name": "East Bay Biters",
        "metadata": {"team_name": "East Bay Biters", "flea_owner_id": "532417"},
    }]
    rosters = [{
        "roster_id": 1020439,
        "owner_id": "1020439",
        "metadata": {"team_name": "East Bay Biters"},
    }]
    viewer = accounts.resolve_account_viewer_for_league(
        7, "fleaflicker", "92916", 2026, users, rosters,
    )
    assert viewer is not None
    assert viewer["viewer_roster_id"] == "1020439"


def test_fleaflicker_resolves_when_saved_season_lags_current(monkeypatch):
    connection = install_db(monkeypatch, {"team_id": "1020439", "season": 2025}, [])

    def execute(sql, params):
        if "AND season=%s" in sql and params[-1] == 2026:
            return FakeResult(None)
        if "ORDER BY season DESC" in sql:
            return FakeResult({"team_id": "1020439", "season": 2025})
        if "SELECT team_id" in sql and "FROM user_leagues" in sql:
            return FakeResult(connection.membership)
        if "SELECT platform_user_id,handle" in sql:
            return FakeResult(connection.identities)
        if "UPDATE user_leagues SET team_id" in sql:
            connection.updated.append(params)
            return FakeResult(None)
        raise AssertionError(sql)

    connection.execute = execute
    users = [{
        "user_id": "1020439",
        "roster_id": 1020439,
        "display_name": "East Bay Biters",
        "metadata": {"team_name": "East Bay Biters", "flea_owner_id": "532417"},
    }]
    rosters = [{
        "roster_id": 1020439,
        "owner_id": "1020439",
        "metadata": {"team_name": "East Bay Biters"},
    }]
    viewer = accounts.resolve_account_viewer_for_league(
        7, "fleaflicker", "92916", 2026, users, rosters,
    )
    assert viewer is not None
    assert viewer["viewer_roster_id"] == "1020439"


def test_add_provider_league_connection_links_flea_owner_identity():
    source = open("dashboard_services/accounts.py", encoding="utf-8").read()
    fn = source.split("def add_provider_league_connection")[1].split("\ndef ")[0]
    assert 'link_platform_identity(account_id, "fleaflicker"' in fn
    assert "flea_user_id" in fn


def test_link_add_persists_fleaflicker_viewer():
    source = open("routes/link_bp.py", encoding="utf-8").read()
    fn = source.split("def link_add")[1].split("\n@")[0]
    assert 'platform == "fleaflicker"' in fn
    assert "_persist_fleaflicker_viewer" in fn
