from types import SimpleNamespace

import pytest

flask = pytest.importorskip("flask")

from routes.link_bp import link_bp


@pytest.fixture
def client(monkeypatch):
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(link_bp)
    import dashboard_services.providers.espn_api as espn
    import dashboard_services.accounts as accounts
    monkeypatch.setattr(espn, "connect_league", lambda *a, **k: {"name": "Test League"})
    monkeypatch.setattr(accounts, "add_espn_league_connection", lambda *a, **k: None)
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["account_id"] = 7
        yield test_client


def test_public_rejects_credentials(client):
    response = client.post("/api/link/espn/public", json={"league_id": "123", "swid": "secret"})
    assert response.status_code == 400
    assert response.json == {"ok": False, "error": "Unexpected fields for this connection method."}


@pytest.mark.parametrize("payload", [
    {"league_id": "123", "swid": "x"},
    {"league_id": "123", "espn_s2": "x"},
])
def test_private_requires_both_credentials(client, payload):
    response = client.post("/api/link/espn/private", json=payload)
    assert response.status_code == 400
    assert response.json["error"] == "SWID and ESPN_S2 are required."


def test_public_connection_succeeds(client):
    response = client.post("/api/link/espn/public", json={"league_id": "123", "season": 2026})
    assert response.status_code == 200
    assert response.json["connection_method"] == "public"
    assert response.json["redirect_url"] == "/espn/2026/123/dashboard"


def test_private_connection_succeeds_without_returning_secrets(client):
    response = client.post("/api/link/espn/private", json={
        "league_id": "123", "season": 2026, "swid": "{owner}", "espn_s2": "secret",
    })
    assert response.status_code == 200
    assert "swid" not in response.json and "espn_s2" not in response.json


def test_private_malformed_espn_response_is_reported_as_expired_session(client, monkeypatch, caplog):
    import dashboard_services.providers.espn_api as espn

    def malformed(*args, **kwargs):
        raise espn.ESPNMalformedResponse("ESPN returned an invalid response.")

    monkeypatch.setattr(espn, "connect_league", malformed)
    response = client.post("/api/link/espn/private", json={
        "league_id": "123", "season": 2026, "swid": "{owner}", "espn_s2": "secret",
    })

    assert response.status_code == 403
    assert response.is_json
    assert "Copy fresh SWID and espn_s2" in response.json["error"]
    assert "Reference:" in response.json["error"]
    assert "secret" not in response.get_data(as_text=True)
    assert "error_type=ESPNMalformedResponse" in caplog.text
    assert "message_fingerprint=" in caplog.text
    assert "secret" not in caplog.text


def test_public_malformed_espn_response_does_not_use_proxy_502(client, monkeypatch):
    import dashboard_services.providers.espn_api as espn

    def malformed(*args, **kwargs):
        raise espn.ESPNMalformedResponse("ESPN returned an invalid response.")

    monkeypatch.setattr(espn, "connect_league", malformed)
    response = client.post("/api/link/espn/public", json={
        "league_id": "123", "season": 2026,
    })

    assert response.status_code == 422
    assert response.is_json
    assert "incomplete league data" in response.json["error"]


def test_connection_requires_account(monkeypatch):
    app = flask.Flask(__name__); app.secret_key = "test"; app.register_blueprint(link_bp)
    with app.test_client() as test_client:
        response = test_client.post("/api/link/espn/public", json={"league_id": "123"})
    assert response.status_code == 401


def test_saved_private_connection_opens_without_submitted_credentials(client, monkeypatch):
    import dashboard_services.accounts as accounts
    monkeypatch.setattr(accounts, "get_espn_league_credentials", lambda *a: {
        "swid": "{owner}", "espn_s2": "secret",
    })
    response = client.post("/api/link/espn/private/saved", json={
        "league_id": "123", "season": 2026,
    })
    assert response.status_code == 200
    assert response.json["redirect_url"] == "/espn/2026/123/dashboard"
    assert "swid" not in response.json and "espn_s2" not in response.json


def test_saved_private_connection_prompts_only_when_missing(client, monkeypatch):
    import dashboard_services.accounts as accounts
    monkeypatch.setattr(accounts, "get_espn_league_credentials", lambda *a: None)
    response = client.post("/api/link/espn/private/saved", json={
        "league_id": "123", "season": 2026,
    })
    assert response.status_code == 409
    assert response.json["needs_credentials"] is True


def test_reconnect_checks_ownership_before_contacting_espn(client, monkeypatch):
    import dashboard_services.accounts as accounts
    import dashboard_services.providers.espn_api as espn
    contacted = []
    monkeypatch.setattr(accounts, "owns_user_league", lambda *a: False)
    monkeypatch.setattr(espn, "connect_league", lambda *a, **k: contacted.append(True))
    response = client.post("/api/link/espn/reconnect", json={
        "league_id": "999", "season": 2026, "swid": "{owner}", "espn_s2": "secret",
    })
    assert response.status_code == 404
    assert contacted == []


def test_signed_out_private_connection_is_staged_before_google(monkeypatch):
    app = flask.Flask(__name__); app.secret_key = "test"; app.register_blueprint(link_bp)
    import dashboard_services.providers.espn_api as espn
    import dashboard_services.accounts as accounts
    monkeypatch.setattr(espn, "connect_league", lambda *a, **k: {
        "name": "Private", "teams": [{"team_id": "7", "name": "My Team"}],
    })
    monkeypatch.setattr(accounts, "stage_private_espn_connection", lambda *a: "opaque-token")
    with app.test_client() as test_client:
        response = test_client.post("/api/link/espn/private/pending", json={
            "league_id": "123", "season": 2026, "swid": "{owner}", "espn_s2": "secret",
        })
        with test_client.session_transaction() as sess:
            assert sess["pending_provider_connection_token"] == "opaque-token"
            assert "swid" not in sess["onboarding_progress"]
            assert "espn_s2" not in sess["onboarding_progress"]
    assert response.status_code == 200
    assert response.json["auth_url"].startswith("/auth/google?intent=onboarding")
    assert response.json["teams"] == [{"team_id": "7", "name": "My Team"}]


def test_private_pending_reports_missing_encryption_configuration(monkeypatch):
    app = flask.Flask(__name__); app.secret_key = "test"; app.register_blueprint(link_bp)
    import dashboard_services.providers.espn_api as espn
    import dashboard_services.accounts as accounts
    monkeypatch.setattr(espn, "connect_league", lambda *a, **k: {"name": "Private"})
    monkeypatch.setattr(
        accounts,
        "stage_private_espn_connection",
        lambda *a: (_ for _ in ()).throw(accounts.ProviderCredentialConfigurationError()),
    )
    with app.test_client() as test_client:
        response = test_client.post("/api/link/espn/private/pending", json={
            "league_id": "123", "season": 2026,
            "swid": "{owner}", "espn_s2": "secret",
        })

    assert response.status_code == 503
    assert "server encryption key is not configured" in response.json["error"]
    assert "Reference:" in response.json["error"]


def test_staged_private_connection_can_continue_without_account(monkeypatch):
    app = flask.Flask(__name__); app.secret_key = "test"; app.register_blueprint(link_bp)
    import dashboard_services.accounts as accounts
    monkeypatch.setattr(accounts, "peek_private_espn_connection", lambda *a: {
        "league_id": "123", "season": 2026, "team_id": "7",
    })
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["pending_provider_connection_token"] = "opaque-token"
        response = test_client.post("/api/link/espn/private/guest", json={
            "league_id": "123", "season": 2026,
        })
    assert response.status_code == 200
    assert response.json["redirect_url"] == "/espn/2026/123/dashboard"


def test_private_team_selection_validates_and_saves_roster(monkeypatch):
    app = flask.Flask(__name__); app.secret_key = "test"; app.register_blueprint(link_bp)
    import dashboard_services.accounts as accounts
    import dashboard_services.providers.espn_api as espn
    monkeypatch.setattr(accounts, "peek_private_espn_connection", lambda *a: {
        "league_id": "123", "season": 2026, "swid": "{owner}", "espn_s2": "secret",
    })
    selected = []
    monkeypatch.setattr(accounts, "select_pending_private_espn_team", lambda *a: selected.append(a) or True)
    monkeypatch.setattr(espn.ESPNFantasyClient, "get_league", lambda *a: {
        "teams": [{"team_id": "7", "name": "My Team"}],
    })
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["pending_provider_connection_token"] = "opaque-token"
        response = test_client.post("/api/link/espn/private/select-team", json={
            "league_id": "123", "season": 2026, "team_id": "7",
        })
        with test_client.session_transaction() as sess:
            assert sess["viewer_roster_id"] == "7"
            assert sess["viewer_team_name"] == "My Team"
    assert response.status_code == 200
    assert selected[0][-1] == "7"
