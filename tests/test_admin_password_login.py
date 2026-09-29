"""Tests for admin password login (ADMIN_PASSWORD + /admin/login).

Builds a minimal Flask app with just the analytics blueprint, so no app.py
import (which would require pandas) is needed.
"""
import pytest
from flask import Flask, session

from dashboard_services import admin_auth
from dashboard_services import analytics as analytics_svc
from extensions import limiter
from routes.analytics_bp import analytics_bp


def _stub_analytics_data(monkeypatch):
    """Stub the dashboard queries so /admin/analytics renders without a DB."""
    monkeypatch.setattr(analytics_svc, "dau_last_30_days", lambda: [])
    monkeypatch.setattr(analytics_svc, "wau_last_12_weeks", lambda: [])
    monkeypatch.setattr(analytics_svc, "signups_per_day", lambda: [])
    monkeypatch.setattr(analytics_svc, "feature_usage_by_week", lambda weeks=8: [])
    monkeypatch.setattr(analytics_svc, "week_over_week_return", lambda: [])
    monkeypatch.setattr(
        analytics_svc,
        "funnel_last_30_days",
        lambda: {"visitors": 0, "signups": 0, "linked": 0, "pro": 0},
    )
    monkeypatch.setattr(analytics_svc, "events_table_ready", lambda: False)


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.delenv("ADMIN_KEY", raising=False)
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")
    storage = getattr(limiter, "_storage", None)
    if storage is not None and hasattr(storage, "reset"):
        storage.reset()
    _stub_analytics_data(monkeypatch)
    flask_app = Flask(__name__)
    flask_app.secret_key = "test-secret"
    flask_app.register_blueprint(analytics_bp)
    limiter.init_app(flask_app)
    return flask_app


def _real_limiter():
    return type(limiter).__name__ == "Limiter"


# ── Login form ────────────────────────────────────────────────────────────────

def test_login_get_renders_form(app):
    resp = app.test_client().get("/admin/login")
    assert resp.status_code == 200
    assert "Admin password" in resp.get_data(as_text=True)


def test_login_get_redirects_when_already_admin(app):
    client = app.test_client()
    client.post("/admin/login", data={"password": "secret123"})
    resp = client.get("/admin/login")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/admin/analytics")


def test_login_get_shows_note_when_no_password_configured(app, monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD")
    resp = app.test_client().get("/admin/login")
    assert resp.status_code == 200
    assert "No admin password is configured" in resp.get_data(as_text=True)


# ── Password verification ─────────────────────────────────────────────────────

def test_login_correct_password_grants_analytics(app):
    client = app.test_client()
    resp = client.post("/admin/login", data={"password": "secret123"})
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/admin/analytics")
    # Session flag alone grants the dashboard even with ADMIN_KEY unset.
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    assert "Product Analytics" in resp.get_data(as_text=True)


def test_login_wrong_password_denied(app):
    client = app.test_client()
    resp = client.post("/admin/login", data={"password": "wrong"})
    assert resp.status_code == 200
    assert "Incorrect password" in resp.get_data(as_text=True)
    assert client.get("/admin/analytics").status_code == 404


def test_login_missing_password_denied(app):
    client = app.test_client()
    resp = client.post("/admin/login", data={})
    assert resp.status_code == 200
    assert "Incorrect password" in resp.get_data(as_text=True)
    assert client.get("/admin/analytics").status_code == 404


def test_login_post_fails_closed_when_no_password_configured(app, monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD")
    client = app.test_client()
    resp = client.post("/admin/login", data={"password": "anything"})
    assert resp.status_code == 200
    assert "Incorrect password" in resp.get_data(as_text=True)
    assert client.get("/admin/analytics").status_code == 404


def test_verify_admin_password_unit(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")
    assert admin_auth.verify_admin_password("secret123") is True
    assert admin_auth.verify_admin_password("wrong") is False
    assert admin_auth.verify_admin_password("") is False
    monkeypatch.delenv("ADMIN_PASSWORD")
    assert admin_auth.verify_admin_password("secret123") is False


def test_is_admin_honors_session_flag_without_admin_key(app):
    with app.test_request_context("/"):
        session[admin_auth.ADMIN_SESSION_KEY] = True
        assert admin_auth.is_admin() is True


# ── ADMIN_KEY path unchanged ──────────────────────────────────────────────────

def test_admin_key_query_param_still_grants_access(app, monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", "key123")
    client = app.test_client()
    resp = client.get("/admin/analytics?admin_key=key123")
    assert resp.status_code == 200
    # The key check marks the session, so later requests pass without the key.
    assert client.get("/admin/analytics").status_code == 200


def test_admin_key_header_still_grants_access(app, monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", "key123")
    client = app.test_client()
    resp = client.get("/admin/analytics", headers={"X-Admin-Key": "key123"})
    assert resp.status_code == 200


def test_admin_key_wrong_still_denied(app, monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", "key123")
    client = app.test_client()
    assert client.get("/admin/analytics?admin_key=nope").status_code == 404


# ── Rate limiting ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _real_limiter(), reason="flask-limiter not installed")
def test_login_post_rate_limited(app):
    client = app.test_client()
    statuses = [
        client.post("/admin/login", data={"password": "wrong"}).status_code
        for _ in range(11)
    ]
    assert statuses[-1] == 429
    # The GET form is not rate limited.
    assert client.get("/admin/login").status_code == 200
