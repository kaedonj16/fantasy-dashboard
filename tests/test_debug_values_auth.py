"""Auth gate on /api/debug-values (CRON_SECRET, fail-closed).

Functional tests against the real blueprint. routes.admin_api_bp imports the
app.py monolith and dashboard_services.db at module/request time, neither of
which is importable in this env, so both are stubbed -- scoped to each test via
a monkeypatch fixture (per AGENTS.md: never leave stub modules in sys.modules
at import time).
"""
from __future__ import annotations

import sys
import types

import pytest

flask = pytest.importorskip("flask")

pytest.importorskip("flask_limiter")


class _FakeResult:
    def fetchall(self):
        return []


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args, **kwargs):
        return _FakeResult()


def _fake_get_conn():
    return _FakeConn()


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with the real admin_api_bp and stubbed heavy imports."""
    app_stub = types.ModuleType("app")
    app_stub.DASHBOARD_CACHE = {}
    app_stub.CACHE_TTL = 300
    monkeypatch.setitem(sys.modules, "app", app_stub)

    # The view does `from dashboard_services.db import get_conn`; the
    # dashboard_services package __init__ is empty, so stubbing the submodule
    # entry is enough.
    db_stub = types.ModuleType("dashboard_services.db")
    db_stub.get_conn = _fake_get_conn
    monkeypatch.setitem(sys.modules, "dashboard_services.db", db_stub)

    # Import only after the stubs are in place. No other test module imports
    # routes.admin_api_bp, so this is the first (cached) import.
    import routes.admin_api_bp as admin_mod
    from extensions import limiter

    test_app = flask.Flask(__name__)
    test_app.config["TESTING"] = True
    limiter.init_app(test_app)
    test_app.register_blueprint(admin_mod.admin_api_bp)
    return test_app.test_client()


def test_debug_values_unauthenticated_is_rejected(monkeypatch, client):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/debug-values")
    assert resp.status_code == 403
    assert resp.get_json() == {"error": "unauthorized"}


def test_debug_values_wrong_secret_is_rejected(monkeypatch, client):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/debug-values", query_string={"secret": "wrong"})
    assert resp.status_code == 403


def test_debug_values_fails_closed_when_secret_unset(monkeypatch, client):
    # Even a "correct-looking" provided secret must fail when the server-side
    # secret is unset -- the old `if secret and ...` pattern passed everything.
    monkeypatch.delenv("CRON_SECRET", raising=False)
    resp = client.get("/api/debug-values", query_string={"secret": "anything"})
    assert resp.status_code == 403


def test_debug_values_authed_via_query_param_succeeds(monkeypatch, client):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/debug-values", query_string={"secret": "s3cret"})
    assert resp.status_code == 200
    body = resp.get_json()
    # Response shape unchanged for legitimate callers.
    assert body["top_players"] == []
    assert "pipeline_state" in body
    assert "model_values_json_mtime" in body


def test_debug_values_authed_via_header_succeeds(monkeypatch, client):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/debug-values", headers={"X-Cron-Secret": "s3cret"})
    assert resp.status_code == 200


def test_debug_values_authed_via_json_body_succeeds(monkeypatch, client):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/debug-values", json={"secret": "s3cret"})
    assert resp.status_code == 200
