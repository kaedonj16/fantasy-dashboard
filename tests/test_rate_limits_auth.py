"""Rate limits on auth endpoints (GROUP C / audit follow-up).

/api/identify fans out to the Sleeper API per call and the OAuth callbacks
run the token-exchange path -- all three were uncapped. Each now carries a
generous "30 per minute" per-IP limit that legitimate use (a handful of
calls per session / one hit per login) can never trip.

These are functional tests against the real blueprints. The only stub is
dashboard_services.api (Sleeper lookups), scoped per-test via monkeypatch
(per AGENTS.md: never leave stub modules in sys.modules at import time).
"""
from __future__ import annotations

import sys
import types

import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("flask_limiter")

IDENTIFY_LIMIT = 30
CALLBACK_LIMIT = 30


@pytest.fixture()
def app(monkeypatch):
    api_stub = types.ModuleType("dashboard_services.api")
    api_stub.get_sleeper_user_by_username = (
        lambda username: {"username": username, "user_id": "uid-1"}
    )
    api_stub.get_sleeper_user_leagues = lambda user_id, season: []
    monkeypatch.setitem(sys.modules, "dashboard_services.api", api_stub)

    import routes.auth_bp as auth_mod
    import routes.google_auth_bp as google_mod
    import routes.yahoo_auth_bp as yahoo_mod
    from extensions import limiter

    test_app = flask.Flask(__name__)
    test_app.config["TESTING"] = True
    test_app.secret_key = "test-secret"
    limiter.init_app(test_app)
    test_app.register_blueprint(auth_mod.auth_bp)
    test_app.register_blueprint(google_mod.google_auth_bp)
    test_app.register_blueprint(yahoo_mod.yahoo_auth_bp)
    return test_app


def test_identify_is_rate_limited(app):
    client = app.test_client()
    statuses = set()
    for _ in range(IDENTIFY_LIMIT):
        resp = client.post("/api/identify", json={"username": "someone"})
        statuses.add(resp.status_code)
    assert statuses == {200}, f"legitimate identify calls must succeed, got {statuses}"
    resp = client.post("/api/identify", json={"username": "someone"})
    assert resp.status_code == 429


def test_google_oauth_callback_is_rate_limited(app):
    client = app.test_client()
    for _ in range(CALLBACK_LIMIT):
        resp = client.get("/auth/google/callback")
        assert resp.status_code in (302, 303), resp.status_code
    resp = client.get("/auth/google/callback")
    assert resp.status_code == 429


def test_yahoo_oauth_callback_is_rate_limited(app):
    client = app.test_client()
    for _ in range(CALLBACK_LIMIT):
        resp = client.get("/auth/yahoo/callback")
        assert resp.status_code in (302, 303), resp.status_code
    resp = client.get("/auth/yahoo/callback")
    assert resp.status_code == 429


def test_memory_limiter_backend_degradation_is_loud():
    # The audit flagged the silent memory:// fallback: without REDIS_URL each
    # gunicorn worker enforces its own budget. app.py must log at warning
    # level (not info) when the shared redis backend is absent.
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    block = src[src.index("limiter.init_app(app)"):]
    block = block[: block.index("# Response compression")]
    assert "set REDIS_URL" in block
    assert "logger.warning" in block
    assert 'logger.info("[limiter] Flask-Limiter enabled (%s backend)"' not in block
