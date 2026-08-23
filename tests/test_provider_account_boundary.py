import pytest

flask = pytest.importorskip("flask")

from routes.auth_bp import auth_bp
from routes.link_bp import link_bp
from routes.yahoo_auth_bp import yahoo_auth_bp


def test_sleeper_identify_sets_provider_session_but_never_google_account(monkeypatch):
    monkeypatch.setattr(
        "dashboard_services.api.get_sleeper_user_by_username",
        lambda username: {"user_id": "sleeper-user", "username": username},
    )
    monkeypatch.setattr("dashboard_services.api.get_sleeper_user_leagues", lambda *args: [])
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(auth_bp)

    with app.test_client() as client:
        response = client.post("/api/identify", json={"username": "same-as-saved-account"})
        with client.session_transaction() as session:
            assert session["viewer_user_id"] == "sleeper-user"
            assert "account_id" not in session

    assert response.status_code == 200


def _mock_yahoo(monkeypatch, attached):
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.exchange_code_for_tokens",
        lambda code: {
            "xoauth_yahoo_guid": "yahoo-guid", "access_token": "token",
            "refresh_token": "refresh", "expires_in": 3600,
        },
    )
    monkeypatch.setattr("dashboard_services.providers.yahoo_api.save_tokens", lambda *args: None)
    monkeypatch.setattr("dashboard_services.providers.yahoo_api.save_league_owner", lambda *args: None)
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.resolve_league_key",
        lambda *args: {"status": "found", "season": 2026, "name": "Yahoo League"},
    )
    monkeypatch.setattr(
        "dashboard_services.providers.yahoo_api.get_users",
        lambda *args: [{"user_id": "yahoo-guid", "roster_id": "team-8"}],
    )
    monkeypatch.setattr("dashboard_services.api.get_nfl_state", lambda: {"season": 2026})
    monkeypatch.setattr(
        "dashboard_services.accounts.link_platform_identity",
        lambda *args, **kwargs: attached.append(("identity", args, kwargs)),
    )
    monkeypatch.setattr(
        "dashboard_services.accounts.add_user_league",
        lambda *args, **kwargs: attached.append(("league", args, kwargs)),
    )


@pytest.mark.parametrize("google_account", [None, 77])
def test_yahoo_oauth_only_attaches_league_when_google_is_already_active(
    monkeypatch, google_account,
):
    attached = []
    _mock_yahoo(monkeypatch, attached)
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(yahoo_auth_bp)

    with app.test_client() as client:
        with client.session_transaction() as session:
            session["yahoo_oauth_state"] = "state"
            session["yahoo_oauth_ctx"] = {"league_id": "league-3", "team_name": "My Team"}
            if google_account:
                session["account_id"] = google_account
        response = client.get("/auth/yahoo/callback?code=code&state=state")
        with client.session_transaction() as session:
            assert session["yahoo_guid"] == "yahoo-guid"
            assert session.get("account_id") == google_account

    assert response.status_code == 302
    if google_account:
        league_call = next(call for call in attached if call[0] == "league")
        assert league_call[1][:4] == (77, "yahoo", "league-3",)
        assert league_call[2]["season"] == 2026
        assert league_call[2]["team_id"] == "team-8"
    else:
        assert attached == []


def test_signed_in_espn_connection_is_persisted_idempotently(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        "dashboard_services.providers.espn_api.connect_league",
        lambda *args, **kwargs: {"name": "ESPN League"},
    )
    monkeypatch.setattr(
        "dashboard_services.accounts.add_espn_league_connection",
        lambda *args, **kwargs: persisted.append((args, kwargs)),
    )
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(link_bp)

    with app.test_client() as client:
        with client.session_transaction() as session:
            session["account_id"] = 77
        first = client.post("/api/link/espn/public", json={"league_id": "2", "season": 2026})
        second = client.post("/api/link/espn/public", json={"league_id": "2", "season": 2026})

    assert first.status_code == second.status_code == 200
    assert [call[0][:4] for call in persisted] == [
        (77, "2", 2026, "ESPN League"),
        (77, "2", 2026, "ESPN League"),
    ]


def test_provider_only_quick_view_does_not_attach_google_account(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        "dashboard_services.accounts.add_user_league",
        lambda *args, **kwargs: persisted.append((args, kwargs)),
    )
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(auth_bp)

    with app.test_client() as client:
        response = client.post("/api/quick-set-viewer", json={
            "username": "ESPN Team", "roster_id": "4", "platform": "espn",
            "league_id": "2", "season": 2026,
        })
        with client.session_transaction() as session:
            assert "account_id" not in session

    assert response.status_code == 200
    assert persisted == []


@pytest.mark.parametrize("platform", ["sleeper", "espn", "yahoo"])
def test_pre_google_link_stages_provider_context_without_authenticating_account(platform):
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(link_bp)

    with app.test_client() as client:
        response = client.post("/api/link/pending", json={
            "platform": platform, "league_id": "league-1", "season": 2026,
            "team_id": "team-7", "name": "Connected League",
        })
        with client.session_transaction() as session:
            assert "account_id" not in session
            assert session["pending_link"] == {
                "platform": platform, "league_id": "league-1", "season": 2026,
                "team_id": "team-7", "name": "Connected League", "username": None,
            }

    assert response.status_code == 200
    assert response.json["auth_url"].startswith("/auth/google")


def test_explicit_unlink_requires_google_and_removes_only_requested_membership(monkeypatch):
    removed = []
    monkeypatch.setattr(
        "dashboard_services.accounts.remove_user_league",
        lambda *args, **kwargs: removed.append((args, kwargs)),
    )
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(link_bp)

    with app.test_client() as client:
        denied = client.post("/api/link/remove", json={
            "platform": "yahoo", "league_id": "league-3", "season": 2026,
        })
        with client.session_transaction() as session:
            session["account_id"] = 77
        removed_response = client.post("/api/link/remove", json={
            "platform": "yahoo", "league_id": "league-3", "season": 2026,
        })

    assert denied.status_code == 401
    assert removed_response.status_code == 200
    assert removed == [((77, "yahoo", "league-3"), {"season": 2026})]


def test_google_callback_persists_verified_sleeper_team_immediately():
    source = open("routes/google_auth_bp.py", encoding="utf-8").read()
    viewer_block = source[source.index("if viewer:"):source.index("if vuid:")]
    assert "add_user_league(" in viewer_block
    assert 'team_id=viewer.get("viewer_roster_id")' in viewer_block
