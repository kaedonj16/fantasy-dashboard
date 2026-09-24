"""Shareable public links for Season / Weekly Wrapped decks.

Covers: POST /api/wrapped/share mints a token + URL, GET /wrapped/<token>
renders the stored deck with no auth, 404 for bad/expired tokens, the 200KB
payload cap, and the copy-link button in both overlay namespaces.

The Postgres layer is faked with an in-memory dict (monkeypatched), so these
run without a database.
"""
import json

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

import dashboard_services.wrapped_shares as WS
from dashboard_services.pages import history_page as H


@pytest.fixture
def fake_store(monkeypatch):
    store = {}

    def fake_init():
        return None

    def fake_create(*, kind, ns, overlay_html, share_data, label):
        import secrets
        token = secrets.token_urlsafe(16)
        store[token] = {
            "token": token, "kind": kind, "ns": ns,
            "overlay_html": overlay_html, "share_data": share_data,
            "label": label,
        }
        return token

    def fake_get(token):
        return store.get((token or "").strip())

    monkeypatch.setattr(WS, "init_wrapped_shares_table", fake_init)
    monkeypatch.setattr(WS, "create_wrapped_share", fake_create)
    monkeypatch.setattr(WS, "get_wrapped_share", fake_get)
    return store


def _deck_html(ns="wrapped"):
    slides = [
        {"kind": "intro", "eyebrow": "2026 SEASON", "big": "Blackedraw",
         "num": False, "dp": 0, "suffix": "", "label": "Wrapped",
         "sub": "A look back", "bgword": "'26"},
        {"kind": "topscore", "eyebrow": "MOST POINTS", "big": "1720.4",
         "num": True, "dp": 1, "suffix": " PTS", "label": "Butter Boys",
         "sub": "Top scoring machine", "bgword": "1720"},
        {"kind": "luck", "eyebrow": "LUCK INDEX", "big": "3.0",
         "num": True, "dp": 1, "suffix": " W", "label": "Jiggy",
         "sub": "Luckiest", "bgword": "LUCK"},
    ]
    return H._wrapped_overlay_markup(
        slides, {"league": "Blackedraw", "season": "2026",
                 "highlights": [{"k": "TOP SCORER", "n": "Butter Boys", "v": "1720.4"}]},
        season=2026, ns=ns)


def test_share_create_returns_url_and_stores(offline_client, fake_store):
    import app
    client = app.app.test_client()
    html = _deck_html()
    resp = client.post("/api/wrapped/share", json={
        "kind": "season", "ns": "wrapped", "overlay_html": html,
        "share_data": {"league": "Blackedraw", "season": "2026"},
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert data["url"].startswith("http")
    assert "/wrapped/" in data["url"]
    token = data["url"].rsplit("/wrapped/", 1)[1]
    assert token in fake_store
    assert fake_store[token]["kind"] == "season"


def test_share_create_weekly_kind(offline_client, fake_store):
    import app
    client = app.app.test_client()
    html = _deck_html(ns="weekly-wrapped")
    resp = client.post("/api/wrapped/share", json={
        "kind": "weekly", "ns": "weekly-wrapped", "overlay_html": html,
        "share_data": {"league": "Blackedraw", "week": 2},
    })
    assert resp.status_code == 200
    token = resp.get_json()["url"].rsplit("/wrapped/", 1)[1]
    assert fake_store[token]["kind"] == "weekly"
    assert fake_store[token]["ns"] == "weekly-wrapped"


def test_share_create_rejects_bad_input(offline_client, fake_store):
    import app
    client = app.app.test_client()
    # bad kind
    r = client.post("/api/wrapped/share", json={
        "kind": "nope", "ns": "wrapped", "overlay_html": _deck_html(),
        "share_data": {}})
    assert r.status_code == 400
    # missing slides
    r = client.post("/api/wrapped/share", json={
        "kind": "season", "ns": "wrapped", "overlay_html": "<div>hi</div>",
        "share_data": {}})
    assert r.status_code == 400
    # oversize payload
    big = "<div class='wrapped-slide'>" + ("x" * (210 * 1024)) + "</div>"
    r = client.post("/api/wrapped/share", json={
        "kind": "season", "ns": "wrapped", "overlay_html": big,
        "share_data": {}})
    assert r.status_code == 413


def test_public_view_renders_deck(offline_client, fake_store):
    import app
    client = app.app.test_client()
    html = _deck_html()
    token = client.post("/api/wrapped/share", json={
        "kind": "season", "ns": "wrapped", "overlay_html": html,
        "share_data": {"league": "Blackedraw", "season": "2026",
                       "highlights": [{"k": "TOP SCORER", "n": "Butter Boys", "v": "1720.4"}]},
    }).get_json()["url"].rsplit("/wrapped/", 1)[1]

    resp = client.get(f"/wrapped/{token}")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # Deck content present, noindex set, OG tags for unfurling.
    assert "wrapped-slide" in body
    assert "Butter Boys" in body
    assert 'name="robots" content="noindex, nofollow"' in body
    assert 'property="og:title"' in body
    assert "Blackedraw" in body
    # Auto-open public bootstrap, no lazy launcher.
    assert "__wrappedSharePublic" in body
    assert "data-wrapped-url" not in body


def test_public_view_404_for_bad_token(offline_client, fake_store):
    import app
    client = app.app.test_client()
    resp = client.get("/wrapped/does-not-exist-123")
    assert resp.status_code == 404
    assert "noindex" in resp.get_data(as_text=True)


def test_copy_link_button_in_both_namespaces(offline_client):
    for ns in ("wrapped", "weekly-wrapped"):
        html = _deck_html(ns=ns)
        assert f'id="{ns}Link"' in html, ns
        js = H._wrapped_bootstrap_js(ns)
        assert f"'{ns}Link'" in js, ns
        assert "/api/wrapped/share" in js
        # Public variant auto-opens and copies location.href.
        pub = H._wrapped_public_bootstrap_js(ns)
        assert "openWrapped();" in pub
        assert "__wrappedSharePublic" in js  # link handler checks the flag


def test_share_rate_limit_declared(offline_client, fake_store):
    """POST /api/wrapped/share is rate-limited to 20/hour (behavioral check).

    Uses a dedicated REMOTE_ADDR so this test's budget is independent of other
    tests hitting the same endpoint in-process. When flask_limiter isn't
    installed the NoopLimiter applies no throttling.
    """
    try:
        import flask_limiter  # noqa
        have_limiter = True
    except ImportError:
        have_limiter = False

    import app
    client = app.app.test_client()
    payload = {
        "kind": "season", "ns": "wrapped", "overlay_html": "<div class='wrapped-slide'>x</div>",
        "share_data": {"league": "Blackedraw", "season": "2026"},
    }
    env = {"REMOTE_ADDR": "10.9.9.9"}
    codes = [client.post("/api/wrapped/share", json=payload,
                         environ_overrides=env).status_code
             for _ in range(21)]
    if have_limiter:
        assert codes[:20] == [200] * 20, codes
        assert codes[20] == 429, codes
    else:
        assert codes == [200] * 21, codes


def test_share_page_bootstrap_counts_up_without_app_bundle():
    """The public /wrapped/<token> page doesn't load the app JS bundle, so
    window.brCountUp is undefined there. The bootstrap must carry its own
    count-up fallback or the big numbers stay 0 (regression: pts showed 0
    on share links)."""
    from dashboard_services.pages.history_page import (
        _wrapped_public_bootstrap_js,
        render_wrapped_share_page,
    )
    js = _wrapped_public_bootstrap_js("weekly-wrapped")
    assert "function wrappedCountUp" in js
    assert "window.brCountUp || wrappedCountUp" in js
    html = render_wrapped_share_page(
        overlay_html="<div class='wrapped-slide'></div>",
        share_data={"week": 3, "league": "Test League"},
        label="Test League — Week 3 Wrapped",
        ns="weekly-wrapped",
        css_url="/static/dashboard.css",
        logo_url="/static/BR_Logo_dark.png",
    )
    assert "function wrappedCountUp" in html
