"""Logged-out SEO pages serve slim public.js (R14.2), not the full app bundle.

Signed-in visitors keep full app.js even when the route opts into lite_js.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")

# Public SEO shells that should opt into lite_js for guests.
_SEO_LITE_PATHS = [
    "/compare",
    "/players",
    "/top-movers",
    "/rankings/dynasty",
    "/rankings/dynasty-qb",
    "/breakouts",
    "/prospects",
    "/dynasty-trade-value-chart",
]


def _script_srcs(html: str) -> list[str]:
    import re
    return re.findall(r'<script[^>]+src="([^"]+)"', html)


def _page_js(html: str) -> str | None:
    for src in _script_srcs(html):
        if "/static/public" in src or "/static/app." in src or "/static/app?" in src:
            # Match public.js / public.min.js / app.js / app.min.js (with ?v=).
            if "public" in src.split("/")[-1]:
                return "public"
            if src.split("/")[-1].startswith("app.") or "/static/app.js" in src or "/static/app.min.js" in src:
                return "app"
    # Fallback: look for app-features marker which only lite pages set.
    return None


@pytest.mark.parametrize("path", _SEO_LITE_PATHS)
def test_guest_seo_page_serves_public_js(offline_client, path):
    """Guests on SEO surfaces get public.js + a features-bundle URL."""
    import app as app_mod
    if not getattr(app_mod, "_FEATURES_JS_FILE", None):
        pytest.skip("app-features.js bundle not built in this environment")

    r = offline_client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    html = r.get_data(as_text=True)
    srcs = _script_srcs(html)
    assert any("/static/public" in s for s in srcs), (
        f"{path} should serve public.js for guests; got {srcs}"
    )
    assert not any(
        s.split("?")[0].endswith("/static/app.js")
        or s.split("?")[0].endswith("/static/app.min.js")
        for s in srcs
    ), f"{path} should not load full app.js for guests; got {srcs}"
    assert "__FEATURES_JS" in html and "app-features" in html
    # Lite pages omit the blocking player_modal.js tag (features bundle owns it).
    assert not any("player_modal.js" in s for s in srcs)


def test_signed_in_seo_page_keeps_full_app_js(offline_client):
    """lite_js is ignored when a session is signed in — full app.js stays."""
    import app as app_mod
    if not getattr(app_mod, "_FEATURES_JS_FILE", None):
        pytest.skip("app-features.js bundle not built in this environment")

    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "u1"

    r = offline_client.get("/compare")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    srcs = _script_srcs(html)
    assert any(
        s.split("?")[0].endswith("/static/app.js")
        or s.split("?")[0].endswith("/static/app.min.js")
        for s in srcs
    ), f"signed-in /compare should serve app.js; got {srcs}"
    assert not any("/static/public" in s for s in srcs)


def test_app_js_eager_features_for_interactive_seo_shells():
    """Compare / prospects / breakouts should eager-load the features half."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    assert 'data-page="compare"' in src
    assert 'data-page="prospects"' in src
    assert 'data-page="breakouts"' in src
    assert "_eagerLite" in src
