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


def test_connection_requires_account(monkeypatch):
    app = flask.Flask(__name__); app.secret_key = "test"; app.register_blueprint(link_bp)
    with app.test_client() as test_client:
        response = test_client.post("/api/link/espn/public", json={"league_id": "123"})
    assert response.status_code == 401
