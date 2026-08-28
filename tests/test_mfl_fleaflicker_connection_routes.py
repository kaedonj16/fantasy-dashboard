from types import SimpleNamespace

import pytest

flask = pytest.importorskip("flask")

from routes.link_bp import link_bp


@pytest.fixture
def client(monkeypatch):
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(link_bp)
    import dashboard_services.accounts as accounts
    monkeypatch.setattr(accounts, "add_provider_league_connection", lambda *a, **k: None)
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["account_id"] = 7
        yield test_client


def _patch_mfl(monkeypatch, connect=None):
    from dashboard_services.providers import mfl_api
    provider = SimpleNamespace(
        connect_league=connect or (lambda *a, **k: {"name": "MFL League", "league_id": "123", "season": 2026}),
        get_league=lambda *a, **k: {"name": "MFL League", "league_id": "123", "season": 2026},
        get_users=lambda *a, **k: [{"roster_id": 1, "display_name": "Ada", "metadata": {"team_name": "Owls"}}],
    )
    monkeypatch.setattr("dashboard_services.providers.registry.get_provider", lambda key: provider)
    return provider


def _patch_flea(monkeypatch, connect=None):
    provider = SimpleNamespace(
        connect_league=connect or (lambda *a, **k: {"name": "Flea League", "league_id": "14153", "season": 2026}),
        get_league=lambda *a, **k: {"name": "Flea League", "league_id": "14153", "season": 2026},
        get_users=lambda *a, **k: [{"roster_id": 1, "display_name": "Ada", "metadata": {"team_name": "Owls"}}],
    )
    monkeypatch.setattr("dashboard_services.providers.registry.get_provider", lambda key: provider)
    return provider


def test_mfl_public_connection_succeeds(client, monkeypatch):
    _patch_mfl(monkeypatch)
    response = client.post("/api/link/mfl/public", json={"league_id": "123", "season": 2026})
    assert response.status_code == 200
    assert response.json["connection_method"] == "public"
    assert response.json["redirect_url"] == "/mfl/2026/123/dashboard"


def test_mfl_private_requires_credentials(client):
    response = client.post("/api/link/mfl/private", json={"league_id": "123", "season": 2026})
    assert response.status_code == 400
    assert "APIKEY" in response.json["error"] or "cookie" in response.json["error"].lower()


def test_mfl_private_stores_apikey_without_password(client, monkeypatch):
    captured = {}

    def add_conn(account_id, provider, league_id, season, name, method, *, credentials=None):
        captured["credentials"] = credentials
        captured["method"] = method

    import dashboard_services.accounts as accounts
    monkeypatch.setattr(accounts, "add_provider_league_connection", add_conn)
    _patch_mfl(monkeypatch)
    response = client.post("/api/link/mfl/private", json={
        "league_id": "123", "season": 2026, "apikey": "league-key",
    })
    assert response.status_code == 200
    assert captured["method"] == "private"
    assert captured["credentials"] == {"apikey": "league-key"}
    assert "password" not in (captured["credentials"] or {})
    assert "apikey" not in response.json


def test_fleaflicker_public_connection_succeeds(client, monkeypatch):
    _patch_flea(monkeypatch)
    response = client.post("/api/link/fleaflicker/public", json={"league_id": "14153", "season": 2026})
    assert response.status_code == 200
    assert response.json["platform"] == "fleaflicker"
    assert response.json["redirect_url"] == "/fleaflicker/2026/14153/dashboard"


def test_fleaflicker_private_login_stores_token_only(client, monkeypatch):
    captured = {}

    def add_conn(account_id, provider, league_id, season, name, method, *, credentials=None, team_id=None):
        captured["credentials"] = credentials
        captured["team_id"] = team_id

    import dashboard_services.accounts as accounts
    monkeypatch.setattr(accounts, "add_provider_league_connection", add_conn)
    monkeypatch.setattr(
        "dashboard_services.providers.fleaflicker_api.login",
        lambda email, password: {"token": "session-token", "user_id": "532417"},
    )
    provider = _patch_flea(monkeypatch)
    provider.get_users = lambda *a, **k: [
        {"roster_id": 1020439, "user_id": "1020439",
         "metadata": {"team_name": "East Bay Biters", "flea_owner_id": "532417"}},
    ]
    response = client.post("/api/link/fleaflicker/private", json={
        "league_id": "14153", "season": 2026, "email": "a@b.com", "password": "secret",
    })
    assert response.status_code == 200
    assert captured["credentials"] == {"token": "session-token", "flea_user_id": "532417"}
    assert captured["team_id"] == "1020439"
    assert "password" not in response.get_data(as_text=True)
    assert "secret" not in response.get_data(as_text=True)


def test_fleaflicker_preview_public(client, monkeypatch):
    _patch_flea(monkeypatch)
    response = client.get("/api/link/fleaflicker/preview?league_id=14153&season=2026")
    assert response.status_code == 200
    assert response.json["ok"] is True
    assert response.json["teams"][0]["name"] == "Owls"


def test_fleaflicker_draft_detect(monkeypatch):
    from routes.draft_api_bp import draft_api_bp
    app = flask.Flask(__name__)
    app.register_blueprint(draft_api_bp)
    draft_ms = 1_735_689_600_000
    monkeypatch.setattr(
        "routes.draft_api_bp.get_drafts",
        lambda platform, league_id, season: [{
            "draft_id": "fleaflicker:2026:14153",
            "league_id": "14153",
            "season": "2026",
            "status": "pre_draft",
            "type": "snake",
            "start_time": draft_ms,
        }],
    )
    with app.test_client() as client:
        response = client.get("/api/draft/detect?platform=fleaflicker&league_id=14153&season=2026")
    assert response.status_code == 200
    body = response.json
    assert body["drafts"][0]["start_time"] == draft_ms
    assert body["drafts"][0]["status"] == "pre_draft"
