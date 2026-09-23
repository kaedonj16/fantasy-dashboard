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
    """The create endpoint must carry a rate limit (enforced in prod where
    flask_limiter is installed; here we verify the declaration)."""
    import app
    from routes import wrapped_share_bp as WSB

    try:
        import flask_limiter  # noqa
        have_limiter = True
    except ImportError:
        have_limiter = False

    view = app.app.view_functions["wrapped_share.api_wrapped_share_create"]
    # flask-limiter tags the view with its limits when installed.
    limits = getattr(view, "_rate_limiting", None) or getattr(view, "rate_limits", None)
    if have_limiter:
        assert limits, "expected rate limits on POST /api/wrapped/share"
    else:
        # NoopLimiter in this env: at least confirm the decorator was applied
        # (source still carries the limit string).
        import inspect
        src = inspect.getsource(WSB)
        assert '@limiter.limit("20 per hour")' in src
