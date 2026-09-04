import pytest

flask = pytest.importorskip("flask")

from routes.auth_bp import auth_bp
from routes.link_bp import link_bp
from routes.yahoo_auth_bp import yahoo_auth_bp
from routes.google_auth_bp import google_auth_bp


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


def test_sleeper_identify_conflict_does_not_keep_foreign_viewer(monkeypatch):
    monkeypatch.setattr(
        "dashboard_services.api.get_sleeper_user_by_username",
        lambda username: {"user_id": "other-sleeper", "username": username},
    )
    monkeypatch.setattr("dashboard_services.api.get_sleeper_user_leagues", lambda *args: [])
    monkeypatch.setattr(
        "dashboard_services.accounts.link_platform_identity",
        lambda *args, **kwargs: "conflict",
    )
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(auth_bp)

    with app.test_client() as client:
        with client.session_transaction() as session:
            session["account_id"] = 42
        response = client.post("/api/identify", json={"username": "someone-else"})
        with client.session_transaction() as session:
            assert session.get("account_id") == 42
            assert "viewer_user_id" not in session
            assert "viewer_username" not in session

    assert response.status_code == 409
    assert b"already linked" in response.data


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
            assert "yahoo_access_token" not in session
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


class _JsonResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def _stub_google_callback(monkeypatch, attached):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost/auth/google/callback")
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _JsonResp({"id_token": "tok"}))
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda *args, **kwargs: {
            "sub": "g-sub", "email": "user@example.com",
            "nonce": "nonce", "given_name": "Pat",
        },
    )
    monkeypatch.setattr(
        "dashboard_services.accounts.upsert_google_account",
        lambda *args, **kwargs: 77,
    )
    monkeypatch.setattr(
        "dashboard_services.accounts.link_platform_identity",
        lambda *args, **kwargs: "ok",
    )
    monkeypatch.setattr(
        "dashboard_services.accounts.add_user_league",
        lambda *args, **kwargs: attached.append(("league", args, kwargs)),
    )
    monkeypatch.setattr(
        "dashboard_services.accounts.get_post_login_destination",
        lambda *args, **kwargs: "/",
    )


def test_google_callback_sends_pending_yahoo_to_yahoo_oauth(monkeypatch):
    attached = []
    _stub_google_callback(monkeypatch, attached)
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(google_auth_bp)

    with app.test_client() as client:
        with client.session_transaction() as session:
            session["google_oauth_state"] = "state"
            session["google_oauth_nonce"] = "nonce"
            session["google_pkce_verifier"] = "verifier"
            session["google_oauth_next"] = "/"
            session["pending_link"] = {
                "platform": "yahoo", "league_id": "123456",
                "season": 2026, "username": "Dynasty Monsters",
            }
        response = client.get("/auth/google/callback?code=code&state=state")
        with client.session_transaction() as session:
            assert session["account_id"] == 77
            assert "yahoo_access_token" not in session

    assert response.status_code == 302
    from urllib.parse import parse_qs, urlparse
    loc = urlparse(response.headers["Location"])
    assert loc.path == "/auth/yahoo"
    qs = parse_qs(loc.query)
    assert qs["league_id"] == ["123456"]
    assert qs["team_name"] == ["Dynasty Monsters"]
    assert attached == []


def test_google_callback_attaches_pending_yahoo_when_yahoo_is_already_authorized(monkeypatch):
    attached = []
    _stub_google_callback(monkeypatch, attached)
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(google_auth_bp)

    with app.test_client() as client:
        with client.session_transaction() as session:
            session["google_oauth_state"] = "state"
            session["google_oauth_nonce"] = "nonce"
            session["google_pkce_verifier"] = "verifier"
            session["google_oauth_next"] = "/"
            session["yahoo_guid"] = "yahoo-guid"
            session["pending_link"] = {
                "platform": "yahoo", "league_id": "123456",
                "season": 2026, "name": "Yahoo League",
            }
        response = client.get("/auth/google/callback?code=code&state=state")

    assert response.status_code == 302
    assert response.headers["Location"] == "/yahoo/2026/123456/dashboard"
    assert attached[0][1][:3] == (77, "yahoo", "123456")
