"""Fleaflicker viewer resolution for personalized dashboard features."""
from contextlib import contextmanager

from dashboard_services import accounts


class FakeConnection:
    def __init__(self, membership, identities):
        self.membership = membership
        self.identities = identities
        self.updated = []

    def execute(self, sql, params):
        if "SELECT team_id FROM user_leagues" in sql:
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
    install_db(monkeypatch, {"team_id": None}, [])
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
    install_db(monkeypatch, {"team_id": None}, [
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
