"""Refresh data must actually update the mobile freshness timestamp.

The More-sheet Refresh row used to expire the league cache then
``location.reload()``. On a PWA the service worker paints the 3.5s cached shell
when the rebuild is slow, so the page looks like it reloaded but ``data-cache-ts``
(and the "2h" label) stay put. These contracts lock the fix:

  * SW skips the cached-shell timeout for reload / bypass-cache navigations.
  * The client waits for fresh HTML (in-place swap) or flags a user refresh
    so a late ``nav-fresh`` still replaces the stale paint.
  * Soft-nav copies ``data-cache-ts`` so the sheet time tracks the new document.
  * ``data-cache-ts`` is the league-context build time, not HTML render time.
  * A Refresh POST busts sibling gunicorn workers, not only the one that
    handled the request.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
ADMIN = (ROOT / "routes" / "admin_api_bp.py").read_text(encoding="utf-8")


def _freshness_iife() -> str:
    start = APP_JS.index("// ── Discord + data-freshness")
    end = APP_JS.index("window.addEventListener('beforeunload'", start)
    return APP_JS[start:end]


def test_sw_skips_stale_shell_on_explicit_refresh():
    assert "bypass-cache" in SW
    assert "forceNetworkNav" in SW
    assert "request.cache === 'reload'" in SW
    assert "skipStaleShell" in SW
    # Explicit refresh waits longer for the network, but still races a timeout
    # so a hung fetch cannot blank the PWA forever.
    assert "NAV_REFRESH_TIMEOUT_MS" in SW
    assert "skipStaleShell ? NAV_REFRESH_TIMEOUT_MS : NAV_TIMEOUT_MS" in SW
    # Message handler acks so the page can reload after the SW is armed.
    assert "event.ports[0].postMessage" in SW


def test_client_refresh_swaps_in_place_or_bypasses_sw():
    src = _freshness_iife()
    assert "function doRefresh()" in src
    assert "doRefresh._busy" in src
    assert "brRefreshOverlay" in src
    assert "Refreshing data" in src
    assert "canSwapInPlace" in src
    assert "brSwapPageRoot" in src
    assert "brUserRefresh" in src
    assert "bypass-cache" in src
    assert "cache: 'reload'" in src
    assert "/api/refresh-league" in src
    assert "credentials: 'same-origin'" in src
    # Must not blindly reload without first expiring / fetching fresh HTML.
    assert "function hardReload()" in src


def test_soft_nav_copies_cache_timestamp():
    assert "window.brSwapPageRoot" in APP_JS
    assert "newRoot.dataset.cacheTs" in APP_JS
    assert "curRoot.dataset.cacheTs = newRoot.dataset.cacheTs" in APP_JS
    assert "window.brUpdateFreshness" in APP_JS


def test_nav_fresh_honors_user_refresh_on_warm_launch():
    start = APP_JS.index("// ── Stale-page auto-refresh")
    stale = APP_JS[start: APP_JS.index("})();", start)]
    assert "brUserRefresh" in stale
    assert "__brWarmLaunch && !userRefresh" in stale


def test_splash_stays_up_during_user_refresh():
    assert "brUserRefresh" in APP_PY
    assert "_warm && !_userRefresh" in APP_PY
    assert "__brWarmLaunch = _warm && !_userRefresh" in APP_PY


def test_refresh_page_stamps_cache_ts():
    handler = APP_JS[APP_JS.index("// Refresh Button Handler"):]
    handler = handler[: handler.index("// ── ESPN email OTP")]
    assert 'root.dataset.cacheTs = String(Date.now())' in handler


def test_render_uses_league_cache_timestamp():
    assert "cache_ts=_league_cache_ts_ms(platform, season, league_id)" in APP_PY
    assert "def _league_cache_ts_ms" in APP_PY
    assert "def _league_ctx_cache_valid" in APP_PY
    assert "def _touch_league_bust" in APP_PY


def test_refresh_league_touches_cross_worker_bust():
    fn = ADMIN[ADMIN.index("def api_refresh_league"):]
    fn = fn[: fn.index("def api_flush_value_cache")]
    assert "_touch_league_bust" in fn
    assert 'DASHBOARD_CACHE[key]["ts"] = 0' in fn


def test_league_ctx_cache_valid_respects_bust_and_ttl(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    import app

    platform, season, league_id = "sleeper", 2026, "busttest"
    monkeypatch.setattr(app, "_league_bust_path",
                        lambda *a: str(tmp_path / "bust"))
    now = time.time()
    fresh = {"ts": now, "ctx": {}}
    assert app._league_ctx_cache_valid(fresh, platform, season, league_id) is True
    assert app._league_ctx_cache_valid({"ts": 0}, platform, season, league_id) is False
    assert app._league_ctx_cache_valid(None, platform, season, league_id) is False
    stale = {"ts": now - app.CACHE_TTL - 10}
    assert app._league_ctx_cache_valid(stale, platform, season, league_id) is False

    app._touch_league_bust(platform, season, league_id)
    # Bust file is newer than the cached ts (written after ``now``).
    older = {"ts": now - 5}
    assert app._league_ctx_cache_valid(older, platform, season, league_id) is False


def test_league_cache_ts_ms_uses_entry_ts(monkeypatch):
    pytest.importorskip("flask")
    import app

    key = app._cache_key("sleeper", 2026, "tsleague")
    built = 1_700_000_000.0
    monkeypatch.setitem(app.DASHBOARD_CACHE, key, {"ts": built, "ctx": {}})
    assert app._league_cache_ts_ms("sleeper", 2026, "tsleague") == int(built * 1000)
    # Missing / zero ts falls back to "now", not 0 (which would hide the chip).
    assert app._league_cache_ts_ms("sleeper", 2026, "missing-league") > 0


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_do_refresh_busy_guard_and_overlay_node():
    """Smoke-check the Refresh IIFE still parses and exposes the overlay id."""
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    src = _freshness_iife()
    assert "brRefreshOverlay" in src
    assert "doRefresh._busy" in src
    # Syntax-check just the IIFE by wrapping it; it references window/document.
    harness = (
        "var window = global; var document = { addEventListener: function(){}, "
        "getElementById: function(){ return null; }, "
        "createElement: function(){ return { style: {}, setAttribute: function(){}, "
        "innerHTML: '', querySelector: function(){ return null; } }; }, "
        "body: { appendChild: function(){} } };\n"
        "var navigator = { serviceWorker: null };\n"
        "var location = { href: 'http://x/sleeper/2026/abc/dashboard', "
        "pathname: '/sleeper/2026/abc/dashboard', reload: function(){} };\n"
        "window.matchMedia = function(){ return { matches: true }; };\n"
        "var setInterval = function(){ return 0; };\n"
        + src + "\n"
        "if (typeof window.brUpdateFreshness !== 'function') process.exit(2);\n"
        "process.exit(0);\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "freshness_check.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", "--check", fp], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        res = subprocess.run(["node", fp], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr or res.stdout
