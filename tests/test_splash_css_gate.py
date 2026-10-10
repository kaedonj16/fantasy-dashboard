"""Hard-refresh CSS flash: the branded splash must stay up until the main
stylesheet has applied.

Signed-in pages load the ~1.2 MB dashboard.css async via the media="print"
trick, with only thin critical CSS inlined. On a hard refresh the sheet trails
the DOM by seconds; the splash used to hide at DOMContentLoaded, painting the
page unstyled (nav as a bare link list) until the sheet arrived. These tests
lock the fixed shape: the async link carries id="brMainCss" and releases the
splash via window.__brCssReady, and the splash script gates its hide on the
sheet being applied.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")


def _signin(client):
    with client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "u1"


def _main_css_link(html: str) -> str | None:
    m = re.search(r'<link[^>]*id="brMainCss"[^>]*>', html)
    return m.group(0) if m else None


def test_signed_in_css_link_releases_splash(offline_client):
    """Async dashboard.css link must be identifiable and hook splash release."""
    _signin(offline_client)
    r = offline_client.get("/compare")
    assert r.status_code == 200
    html = r.get_data(as_text=True)

    link = _main_css_link(html)
    assert link, "signed-in page should emit the async main CSS link"
    assert 'media="print"' in link
    assert "dashboard" in link and ".css" in link
    # onload applies the sheet AND releases the splash; onerror releases too so
    # a failed sheet never traps the user behind the splash.
    assert "__brCssReady" in link
    assert re.search(r'onload="[^"]*this\.media=\'all\'[^"]*__brCssReady', link), link
    assert re.search(r'onerror="[^"]*__brCssReady', link), link


def test_splash_hides_only_after_css_applies(offline_client):
    """Splash script must gate its hide on DOM + stylesheet readiness."""
    _signin(offline_client)
    r = offline_client.get("/compare")
    assert r.status_code == 200
    html = r.get_data(as_text=True)

    assert "window.__brCssReady" in html
    # The gate: look the sheet up by id and confirm it landed in styleSheets.
    assert 'getElementById(\'brMainCss\')' in html
    assert "document.styleSheets" in html
    # Hide requires both DOM-ready and css-ready; the old unguarded
    # DOMContentLoaded -> hide() path must be gone.
    assert "addEventListener('DOMContentLoaded',function(){" in html
    assert "_domReady" in html
    assert "maybeHide()" in html
    assert "addEventListener('DOMContentLoaded',hide)" not in html
    # Safety net survives so a hung sheet can never trap the splash forever.
    assert re.search(r"setTimeout\(hide,\d+\)", html)


def test_guest_lite_page_has_no_async_main_css(offline_client):
    """Lite pages keep their render-blocking sheet; no splash gating needed."""
    import app as app_mod

    if not getattr(app_mod, "_FEATURES_JS_FILE", None):
        pytest.skip("app-features.js bundle not built in this environment")

    r = offline_client.get("/compare")
    assert r.status_code == 200
    html = r.get_data(as_text=True)

    assert _main_css_link(html) is None
    assert re.search(r"/static/seo_lite(?:\.min)?\.css", html)
    # Splash markup still present; the gate no-ops (cssReady true, no link).
    assert 'id="appSplash"' in html
    assert "window.__brCssReady" in html
