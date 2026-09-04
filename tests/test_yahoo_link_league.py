"""Yahoo link-modal preview and add-league persistence."""
from unittest import mock

import pytest

flask = pytest.importorskip("flask")

from routes.link_bp import link_bp


@pytest.fixture
def link_app():
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(link_bp)
    return app


def test_yahoo_preview_uses_db_token_when_session_access_token_missing(link_app, monkeypatch):
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.yahoo_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_valid_access_token",
        lambda guid: "db-token" if guid == "yahoo-guid" else None,
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.resolve_league_key",
        lambda *args: {"status": "found", "season": 2026, "name": "Yahoo League"},
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_league",
        lambda *args: {"name": "Yahoo League"},
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_users",
        lambda *args: [{"user_id": "yahoo-guid", "roster_id": 3, "display_name": "Mine"}],
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_login_guid",
        lambda *args: "yahoo-guid",
    )
    monkeypatch.setattr("dashboard_services.api.get_nfl_state", lambda: {"season": 2026})

    with link_app.test_client() as client:
        with client.session_transaction() as session:
            session["yahoo_guid"] = "yahoo-guid"
            # Stale session bearer must be ignored; DB token wins and is not
            # written back into the session cookie.
            session["yahoo_access_token"] = "expired-session-token"
        res = client.get("/api/link/yahoo/preview?league_id=1307110")
        with client.session_transaction() as session:
            assert "yahoo_access_token" not in session
            assert session.get("yahoo_guid") == "yahoo-guid"

    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert body["my_team_id"] == "3"


def test_yahoo_preview_reauths_on_token_expired(link_app, monkeypatch):
    monkeypatch.setattr("dashboard_services.providers.yahoo_api.yahoo_enabled", lambda: True)
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_valid_access_token",
        lambda guid, force_refresh=False: ("stale-but-db-says-ok" if not force_refresh else None),
    )

    def boom(*args, **kwargs):
        raise Exception(
            '401 Client Error: Please provide valid credentials. '
            'OAuth oauth_problem="token_expired", realm="yahooapis.com"'
        )

    monkeypatch.setattr("dashboard_services.providers.yahoo_api.resolve_league_key", boom)
    monkeypatch.setattr("dashboard_services.api.get_nfl_state", lambda: {"season": 2026})

    with link_app.test_client() as client:
        with client.session_transaction() as session:
            session["yahoo_guid"] = "yahoo-guid"
            session["yahoo_access_token"] = "expired"
            session["account_id"] = 7
        res = client.get("/api/link/yahoo/preview?league_id=1307110")
        with client.session_transaction() as session:
            assert "yahoo_access_token" not in session

    assert res.status_code == 401
    body = res.get_json()
    assert body["needs_oauth"] is True
    assert "expired" in (body.get("error") or "").lower()
    assert "reauth=1" in body["auth_url"]


def test_yahoo_preview_retries_after_forced_token_refresh(link_app, monkeypatch):
    monkeypatch.setattr("dashboard_services.providers.yahoo_api.yahoo_enabled", lambda: True)

    def fake_valid(guid, force_refresh=False):
        return "fresh-token" if force_refresh else "expired-token"

    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_valid_access_token", fake_valid,
    )
    calls = {"n": 0}

    def fake_resolve(token, league_id):
        calls["n"] += 1
        if token == "expired-token":
            raise Exception('401 oauth_problem="token_expired"')
        return {"status": "found", "season": 2026, "name": "Yahoo League"}

    monkeypatch.setattr("dashboard_services.providers.yahoo_api.resolve_league_key", fake_resolve)
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_league",
        lambda *args: {"name": "Yahoo League"},
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_users",
        lambda *args: [{"user_id": "yahoo-guid", "roster_id": 3}],
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_login_guid",
        lambda *args: "yahoo-guid",
    )
    monkeypatch.setattr("dashboard_services.api.get_nfl_state", lambda: {"season": 2026})

    with link_app.test_client() as client:
        with client.session_transaction() as session:
            session["yahoo_guid"] = "yahoo-guid"
        res = client.get("/api/link/yahoo/preview?league_id=1307110")
        with client.session_transaction() as session:
            assert "yahoo_access_token" not in session
            assert session.get("yahoo_guid") == "yahoo-guid"

    assert res.status_code == 200
    assert res.get_json()["ok"] is True
    assert calls["n"] == 2


def test_yahoo_preview_oauth_url_includes_league_for_signed_in_account(link_app, monkeypatch):
    monkeypatch.setattr("dashboard_services.providers.yahoo_api.yahoo_enabled", lambda: True)
    monkeypatch.setattr("dashboard_services.providers.yahoo_api.get_valid_access_token", lambda guid: None)

    with link_app.test_client() as client:
        with client.session_transaction() as session:
            session["account_id"] = 42
        res = client.get("/api/link/yahoo/preview?league_id=1307110")

    assert res.status_code == 401
    body = res.get_json()
    assert body["needs_oauth"] is True
    assert "league_id=1307110" in body["auth_url"]
    assert "next=%2Fportfolio" in body["auth_url"] or "next=/portfolio" in body["auth_url"]


def test_yahoo_link_add_persists_identity_and_league_owner(link_app, monkeypatch):
    saved = {"identity": [], "owner": [], "league": []}
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.resolve_session_yahoo_token",
        lambda session: ("yahoo-guid", "token"),
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.resolve_league_key",
        lambda *args: {"status": "found", "season": 2026},
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.save_league_owner",
        lambda *args: saved["owner"].append(args),
    )
    monkeypatch.setattr(
        "dashboard_services.accounts.link_platform_identity",
        lambda *args, **kwargs: saved["identity"].append((args, kwargs)),
    )
    monkeypatch.setattr(
        "dashboard_services.accounts.add_user_league",
        lambda *args, **kwargs: saved["league"].append((args, kwargs)),
    )

    with link_app.test_client() as client:
        with client.session_transaction() as session:
            session["account_id"] = 77
            session["yahoo_guid"] = "yahoo-guid"
        res = client.post("/api/link/add", json={
            "platform": "yahoo",
            "league_id": "1307110",
            "season": 2026,
            "team_id": "8",
            "name": "My Yahoo League",
        })

    assert res.status_code == 200
    assert saved["identity"]
    assert saved["owner"] == [("1307110", 2026, "yahoo-guid")]
    assert saved["league"][0][0][:3] == (77, "yahoo", "1307110")
    assert saved["league"][0][1]["season"] == 2026
