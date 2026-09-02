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
        res = client.get("/api/link/yahoo/preview?league_id=1307110")
        with client.session_transaction() as session:
            assert session.get("yahoo_access_token") == "db-token"

    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert body["my_team_id"] == "3"


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
