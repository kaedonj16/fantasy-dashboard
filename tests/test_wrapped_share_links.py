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


def _legacy_deck_html(ns="weekly-wrapped"):
    """Overlay markup shaped like decks minted before the nav chrome existed:
    overlay + progress + stage only, logo imgs without src attributes."""
    slides = "".join(
        f"<section class='wrapped-slide' data-kind='s{i}'>"
        f"<div class='wrapped-num'><span class='wrapped-big' "
        f"data-w-count='{10 + i}' data-w-dp='0'>0</span></div></section>"
        for i in range(3))
    bars = "".join("<span class='wrapped-bar'><i></i></span>" for _ in range(4))
    return (
        f'<div class="wrapped-overlay" id="{ns}Overlay" hidden aria-hidden="true">'
        f'<div class="wrapped-progress">{bars}</div>'
        f'<div class="wrapped-stage" id="{ns}Stage">'
        f"<section class='wrapped-slide' data-kind='intro'>"
        f"<img alt='BR Fantasy' class='wrapped-intro-logo'>"
        f"<div class='wrapped-league'>blackedraw</div></section>"
        f"{slides}"
        f"<div class='wrapped-foot'><img alt=''>"
        f"<span class='wrapped-foot-season'>WEEK 3</span></div>"
        f"</div></div>"
    )


def test_public_bootstrap_has_no_unguarded_element_binds():
    """Regression: a stored deck minted before the overlay carried its nav
    chrome (no Close/Next/Prev/Pause/Share/Link buttons, no ShareData script)
    rendered a blank page, because bindOverlay called
    getElementById('<ns>Close').addEventListener unguarded and the TypeError
    killed the bootstrap before openWrapped() ran. Every nav binding must be
    null-guarded in both namespaces."""
    import re
    from dashboard_services.pages.history_page import _wrapped_public_bootstrap_js
    for ns in ("wrapped", "weekly-wrapped"):
        js = _wrapped_public_bootstrap_js(ns)
        bad = re.findall(r"getElementById\('[^']+'\)\.addEventListener", js)
        assert not bad, (ns, bad)


def test_public_view_repairs_legacy_deck(offline_client, fake_store):
    """End-to-end with a legacy stored deck: the page must open the deck
    (guarded bootstrap) and backfill the missing logo img srcs."""
    import app
    client = app.app.test_client()
    html = _legacy_deck_html()
    token = client.post("/api/wrapped/share", json={
        "kind": "weekly", "ns": "weekly-wrapped", "overlay_html": html,
        "share_data": {"league": "blackedraw", "week": 3},
    }).get_json()["url"].rsplit("/wrapped/", 1)[1]

    resp = client.get(f"/wrapped/{token}")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # Guarded nav bindings for the weekly namespace.
    assert "if (closeBtn) closeBtn.addEventListener" in body
    assert "if (nextBtn) nextBtn.addEventListener" in body
    assert "if (prevBtn) prevBtn.addEventListener" in body
    # Logo backfill wiring for the src-less imgs in legacy decks.
    assert "window.__wrappedShareLogo" in body
    assert "img.wrapped-intro-logo, .wrapped-foot img" in body


def test_public_view_restores_pause_and_tap_zones(offline_client, fake_store):
    """Regression: the share sanitizer strips every <button>/<svg>/<script>
    from the stored overlay, so public decks arrived with no Pause pill and
    no Prev/Next tap zones: auto-play worked, tapping did nothing. The
    public page must re-add that trusted chrome at render time, for both
    namespaces, along with the ShareData JSON the Share card paints from."""
    import app
    client = app.app.test_client()
    for ns, kind, share_data in (
        ("weekly-wrapped", "weekly", {"league": "Blackedraw", "week": 2}),
        ("wrapped", "season", {"league": "Blackedraw", "season": "2026"}),
    ):
        token = client.post("/api/wrapped/share", json={
            "kind": kind, "ns": ns, "overlay_html": _deck_html(ns=ns),
            "share_data": share_data,
        }).get_json()["url"].rsplit("/wrapped/", 1)[1]

        resp = client.get(f"/wrapped/{token}")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        # Pause pill, complete with the pause/play icons paintPauseBtn toggles.
        assert f'id="{ns}Pause"' in body, ns
        assert 'class="wrapped-pause"' in body, ns
        assert "wp-ic-pause" in body and "wp-ic-play" in body, ns
        # Tap zones are what tapping a slide actually hits.
        assert f'id="{ns}Next"' in body and "wrapped-tap-next" in body, ns
        assert f'id="{ns}Prev"' in body and "wrapped-tap-prev" in body, ns
        # The rest of the chrome the sanitizer stripped.
        assert f'id="{ns}Share"' in body, ns
        assert f'id="{ns}Link"' in body, ns
        assert f'id="{ns}Close"' in body, ns
        assert "wrapped-hint" in body, ns
        # Exactly one of each (restore must not duplicate existing chrome).
        assert body.count(f'id="{ns}Pause"') == 1, ns
        assert body.count(f'id="{ns}Next"') == 1, ns
        # ShareData restored from the stored share payload, not the deck.
        assert f'id="{ns}ShareData"' in body, ns
        assert "Blackedraw" in body, ns


def test_public_view_restores_chrome_for_legacy_deck(offline_client, fake_store):
    """Decks already stored without any chrome (minted before the Pause
    control existed, then stripped again by the render sanitizer) get the
    Pause pill and tap zones back without being re-minted."""
    import app
    client = app.app.test_client()
    token = client.post("/api/wrapped/share", json={
        "kind": "weekly", "ns": "weekly-wrapped",
        "overlay_html": _legacy_deck_html(),
        "share_data": {"league": "blackedraw", "week": 3},
    }).get_json()["url"].rsplit("/wrapped/", 1)[1]

    body = client.get(f"/wrapped/{token}").get_data(as_text=True)
    assert 'id="weekly-wrappedPause"' in body
    assert 'id="weekly-wrappedNext"' in body
    assert 'id="weekly-wrappedPrev"' in body
    assert 'id="weekly-wrappedShareData"' in body


def test_restore_chrome_does_not_duplicate_present_chrome():
    html = _deck_html(ns="weekly-wrapped")
    out = H._restore_wrapped_share_chrome(
        html, "weekly-wrapped", {"league": "Blackedraw", "week": 2})
    assert out == html


def test_public_bootstrap_has_stage_tap_fallback():
    """If a deck ever reaches bindOverlay without tap zones again, tapping
    the stage itself must still advance instead of sitting stuck."""
    for ns in ("wrapped", "weekly-wrapped"):
        pub = H._wrapped_public_bootstrap_js(ns)
        assert "if (!nextBtn)" in pub, ns
        assert "stage.addEventListener('click'" in pub, ns


def test_public_view_cta_does_not_promise_standalone_wrapped():
    """The bottom CTA links to the home page (the dashboard product), so it
    must not promise a standalone 'make your own Wrapped' tool."""
    html = H.render_wrapped_share_page(
        overlay_html=_deck_html(ns="weekly-wrapped"),
        share_data={"league": "Blackedraw", "week": 2},
        label="Blackedraw: Week 2 Wrapped",
        ns="weekly-wrapped",
        css_url="/static/dashboard.css",
        logo_url="/static/BR_Logo_dark.png",
    )
    assert "Make your own Wrapped" not in html
    assert "See your league on BR Fantasy" in html
