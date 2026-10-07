"""The /compare page's "Connect your league" CTA must only render for logged-out
visitors.

Regression: the CTA was unconditional, so Google-linked managers (account_id in
the session, no Sleeper username) saw the logged-out landing variant even with
their league connected.
"""
from __future__ import annotations

import pytest

flask = pytest.importorskip("flask")

from flask import Flask, session

import routes.seo_pages_bp as seo_bp


CTA = "Connect your league to compare your players"


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setattr(
        seo_bp, "render_page", lambda title, league_id, active, body, *a, **k: body
    )
    flask_app = Flask(__name__)
    flask_app.secret_key = "test-secret"
    flask_app.register_blueprint(seo_bp.seo_pages_bp)
    return flask_app


def _get(client, path="/compare", sess=None):
    if sess:
        with client.session_transaction() as s:
            s.update(sess)
    return client.get(path).get_data(as_text=True)


def test_cta_shown_when_logged_out(app):
    assert CTA in _get(app.test_client())


def test_cta_hidden_for_google_linked_session(app):
    # Kaedon's case: Google sign-in sets account_id, never a Sleeper identity.
    body = _get(app.test_client(), sess={"account_id": "123"})
    assert CTA not in body


def test_cta_hidden_for_sleeper_session(app):
    body = _get(app.test_client(), sess={"viewer_username": "someone"})
    assert CTA not in body


def test_cta_hidden_on_league_scoped_url(app):
    body = _get(
        app.test_client(),
        path="/sleeper/2026/1312067280816832512/compare",
        sess={"account_id": "123"},
    )
    assert CTA not in body
    assert "compare-page" in body
