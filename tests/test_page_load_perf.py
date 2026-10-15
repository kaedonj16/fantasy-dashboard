"""First-paint asset weight: minify leftover JS/CSS and lazy-load the player modal.

Signed-in shells used to parse unminified player_modal.js + paywall.js on every
page, and deferred scripts sat after the (often huge) body so the preload
scanner found them late. These tests lock the cheaper shape.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"


def _script_srcs(html: str) -> list[str]:
    import re
    return re.findall(r'<script[^>]+src="([^"]+)"', html)


def _head_html(html: str) -> str:
    start = html.find("<head")
    end = html.find("</head>")
    assert start != -1 and end != -1
    return html[start:end]


def test_extra_assets_are_minified_smaller_than_source():
    """Boot minify must actually shrink the first-paint leftovers."""
    import app as app_mod
    from utils.static_minify import STATIC_DIR, served_name

    assert STATIC_DIR == STATIC
    pairs = [
        ("player_modal.js", app_mod._PLAYER_MODAL_JS_FILE),
        ("paywall.js", app_mod._PAYWALL_JS_FILE),
        ("rankings.js", app_mod._RANKINGS_JS_FILE),
        ("seo_lite.css", app_mod._SEO_LITE_CSS_FILE),
        ("landing_lite.css", app_mod._LANDING_LITE_CSS_FILE),
    ]
    for src_name, served in pairs:
        src = STATIC / src_name
        out = STATIC / served
        assert src.exists(), src_name
        assert out.exists(), served
        assert out.stat().st_size < src.stat().st_size, (
            f"{served} should be smaller than {src_name}"
        )
        if src_name.endswith(".js"):
            assert served.endswith(".min.js"), f"{src_name} should serve a .min.js"
        else:
            assert served.endswith(".min.css"), f"{src_name} should serve a .min.css"
    # Draft room is the largest remaining page bundle; minify at boot too.
    assert served_name("draft_room.js").endswith(".min.js")
    dr_src = STATIC / "draft_room.js"
    dr_min = STATIC / served_name("draft_room.js")
    assert dr_min.stat().st_size < dr_src.stat().st_size


def test_signed_in_page_lazy_loads_player_modal(offline_client):
    """Signed-in HTML must not parse player_modal.js up front; it sets a lazy URL."""
    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "u1"

    r = offline_client.get("/compare")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    srcs = _script_srcs(html)
    assert not any("player_modal" in s.split("?")[0] for s in srcs), srcs
    assert "__PLAYER_MODAL_JS" in html
    assert "player_modal" in html
    # Features bundle stays null so we don't double-bind signed-in app.js.
    import re
    m = re.search(r"window\.__FEATURES_JS\s*=\s*([^;]+);", html)
    assert m, html[:500]
    assert m.group(1).strip() == "null"
    m = re.search(r"window\.__PLAYER_MODAL_JS\s*=\s*([^;]+);", html)
    assert m
    assert "player_modal" in m.group(1)
    assert m.group(1).strip() != "null"


def test_guest_page_does_not_set_player_modal_url(offline_client):
    """Lite guests get the modal inside app-features.js, not a second URL."""
    import app as app_mod
    if not getattr(app_mod, "_FEATURES_JS_FILE", None):
        pytest.skip("app-features.js bundle not built in this environment")

    r = offline_client.get("/compare")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    import re
    m = re.search(r"window\.__PLAYER_MODAL_JS\s*=\s*([^;]+);", html)
    assert m
    assert m.group(1).strip() == "null"
    assert not any("player_modal" in s for s in _script_srcs(html))


def test_deferred_shell_scripts_are_in_head(offline_client):
    """app.js + paywall.js must be discovered in <head>, not after the body HTML."""
    import re
    r = offline_client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    head = _head_html(html)
    assert re.search(r'<script[^>]+src="/static/(?:public|app)[^"]*" defer', head)
    assert re.search(r'<script[^>]+src="/static/paywall[^"]*" defer', head)
    # Body must not repeat those shell tags.
    body = html[html.find("</head>"):]
    assert not re.search(r'<script[^>]+src="/static/(?:public|app)[^"]*" defer', body)
    assert not re.search(r'<script[^>]+src="/static/paywall[^"]*" defer', body)


def test_signed_in_first_paint_bundle_weight_budget(offline_client):
    """Weight budget for the deferred first-paint bundles in the page head.

    Measured 2026-10-14 on a signed-in /compare render in the full-stack
    environment, where the boot minifiers run: the deferred head scripts
    (app bundle + paywall.js) plus the head stylesheets (dashboard bundle
    + paywall.css) totalled 1,839,121 bytes served. Where the minifiers
    are unavailable the same four source files total 2,699,989 bytes, so
    the cap covers that unminified fallback: 2,699,989 plus 10%, rounded
    to 2,970,000 bytes. A change that pushes first-paint weight past the
    cap fails here on purpose: shrink the bundle or raise the cap
    deliberately with a new measurement.
    """
    import re

    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "u1"

    r = offline_client.get("/compare")
    assert r.status_code == 200
    head = _head_html(r.get_data(as_text=True))
    urls = set(re.findall(r'<script[^>]+src="(/static/[^"]+)"[^>]*\bdefer\b', head))
    urls |= set(
        re.findall(r'<link[^>]+rel="stylesheet"[^>]+href="(/static/[^"]+)"', head)
    )
    assert urls, "signed-in head should carry deferred JS and CSS bundles"
    total = 0
    for url in urls:
        name = url.split("?")[0].rsplit("/", 1)[-1]
        served = STATIC / name
        assert served.exists(), name
        total += served.stat().st_size
    cap = 2_970_000
    assert total <= cap, (
        f"deferred first-paint bundles total {total} bytes, over the "
        f"{cap} byte budget (see measurement note in this test)"
    )


def test_ensure_features_loads_player_modal_url():
    src = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "window.__FEATURES_JS || window.__PLAYER_MODAL_JS" in src
    assert "window.__PLAYER_MODAL_JS" in src
