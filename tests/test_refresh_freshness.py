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
    assert stale.index("if (!userRefresh && Date.now() - loadedAt > 20000)") > stale.index("brUserRefresh")


def test_sw_late_network_notifies_after_explicit_refresh_fallback():
    block = SW[SW.index("async function handleNavigate") : SW.index("// ── Push notifications")]
    assert "notifyNavFresh(request, networkFetch)" in block
    assert "if (cached)" in block
    assert block.index("notifyNavFresh(request, networkFetch)") > block.index("if (cached)")


def test_splash_stays_up_during_user_refresh():
    assert "brUserRefresh" in APP_PY
    assert "_warm && !_userRefresh" in APP_PY
    assert "__brWarmLaunch = _warm && !_userRefresh" in APP_PY


def test_legacy_refresh_button_uses_global_transaction_without_fake_timestamp():
    handler = APP_JS[APP_JS.index("// Refresh Button Handler"):]
    handler = handler[: handler.index("// ── ESPN email OTP")]
    assert "window.brRefreshLeague()" in handler
    assert "dataset.cacheTs = String(Date.now())" not in handler
    assert 'fetch("/api/refresh-page"' not in handler


def test_refresh_requires_new_authoritative_document_timestamp():
    src = _freshness_iife()
    assert "function extractFreshDocument(html, beforeTs)" in src
    assert "nextTs <= beforeTs" in src
    assert "stale refreshed document" in src
    assert "attempts = 3" in src
    assert "cache: 'reload'" in src
    assert "cacheTs() !== acceptedTs" in src


def test_refresh_failure_keeps_old_timestamp_and_clears_busy_in_finally():
    src = _freshness_iife()
    assert "setRefreshFailure()" in src
    assert "} finally {" in src
    assert "doRefresh._busy = false" in src
    failure = src[src.index("function setRefreshFailure") : src.index("function setRefreshingLabel")]
    assert "dataset.cacheTs" not in failure


def test_hard_reload_records_expected_authoritative_timestamp():
    src = _freshness_iife()
    assert "brRefreshExpectedTs" in src
    assert "hardReload.expectedTs = acceptedTs" in src
    assert "ts >= expected" in src


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
    assert "clear_league_provider_cache_for_league" in fn


def test_context_rebuild_clears_platform_specific_provider_cache():
    fn = APP_PY[APP_PY.index("def get_league_ctx_from_cache") : APP_PY.index("# /api/trade-count")]
    assert 'platform == "sleeper"' in fn
    assert "clear_league_provider_cache_for_league(league_id)" in fn
    assert 'platform == "espn"' in fn
    assert "clear_espn_league_caches(league_id, season)" in fn


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


def test_invalidated_context_rebuild_advances_authoritative_timestamp(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    import app

    platform, season, league_id = "sleeper", 2026, "rebuildtest"
    key = app._cache_key(platform, season, league_id)
    old_ts = time.time() - 10
    monkeypatch.setattr(app, "_league_bust_path", lambda *a: str(tmp_path / "bust"))
    monkeypatch.setitem(app.DASHBOARD_CACHE, key, {"ts": old_ts, "ctx": {"users": [], "rosters": []}})
    monkeypatch.setattr(app, "build_league_context", lambda *a: {"users": [], "rosters": []})
    monkeypatch.setattr(app, "get_viewer_session_for_league", lambda *a: {})

    app._touch_league_bust(platform, season, league_id)
    app.get_league_ctx_from_cache(platform, league_id, season)
    assert app.DASHBOARD_CACHE[key]["ts"] > old_ts
    assert app._league_cache_ts_ms(platform, season, league_id) == int(app.DASHBOARD_CACHE[key]["ts"] * 1000)


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

@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_refresh_transaction_success_stale_failure_and_double_click_node():
    """Exercise the shared transaction rather than only matching source strings."""
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    src = _freshness_iife()
    harness = r"""
var window = globalThis;
var location = { href: 'http://x/sleeper/2026/abc/dashboard', pathname: '/sleeper/2026/abc/dashboard', reload: function(){ reloads++; } };
window.location = location;
var navigator = { serviceWorker: null };
var reloads = 0, postCount = 0, getCount = 0, mode = 'success';
var root = { dataset: { cacheTs: '1000' }, querySelectorAll: function(){ return []; } };
var sheetTime = { textContent: '', classList: { toggle: function(){} } };
var button = { disabled: false, _brWired: false, setAttribute: function(){}, removeAttribute: function(){}, addEventListener: function(){} };
var elements = { 'page-root': root, 'brSheetRefreshTime': sheetTime, 'brSheetRefresh': button };
var document = {
  visibilityState: 'visible',
  getElementById: function(id){ return elements[id] || null; },
  querySelector: function(){ return null; },
  addEventListener: function(){},
  createElement: function(){ return { style: {}, setAttribute: function(){}, innerHTML: '' }; },
  body: { appendChild: function(el){ elements[el.id] = el; } }
};
window.matchMedia = function(){ return { matches: true }; };
var setInterval = function(){ return 0; };
var realSetTimeout = globalThis.setTimeout;
var DOMParser = function(){};
DOMParser.prototype.parseFromString = function(html){
  var m = /data-cache-ts=["'](\d+)/.exec(html);
  return { getElementById: function(id){ return id === 'page-root' && m ? { dataset: { cacheTs: m[1] } } : null; } };
};
window.brSwapPageRoot = function(html){
  var m = /data-cache-ts=["'](\d+)/.exec(html);
  if (!m) return false;
  root.dataset.cacheTs = m[1];
  return true;
};
function response(ok, body){ return { ok: ok, status: ok ? 200 : 500, text: function(){ return Promise.resolve(body || ''); } }; }
window.brFetchWithTimeout = function(url, opts){
  if (url === '/api/refresh-league') { postCount++; return Promise.resolve(response(true)); }
  getCount++;
  if (mode === 'failure') return Promise.reject(new Error('network'));
  var ts = mode === 'stale' ? '1000' : '2000';
  return Promise.resolve(response(true, '<main id="page-root" data-cache-ts="' + ts + '"></main>'));
};
""" + src + r"""
function wait(ms){ return new Promise(function(resolve){ realSetTimeout(resolve, ms); }); }
(async function(){
  var a = window.brRefreshLeague();
  var b = window.brRefreshLeague();
  await Promise.all([a, b]);
  if (postCount !== 1 || getCount !== 1 || root.dataset.cacheTs !== '2000' || window.brRefreshLeague._busy) process.exit(2);

  mode = 'stale'; root.dataset.cacheTs = '1000'; postCount = 0; getCount = 0;
  await window.brRefreshLeague();
  if (postCount !== 1 || getCount !== 3 || root.dataset.cacheTs !== '1000' || window.brRefreshLeague._busy || sheetTime.textContent.indexOf('Failed') !== 0) process.exit(3);

  mode = 'failure'; postCount = 0; getCount = 0;
  await window.brRefreshLeague();
  if (postCount !== 1 || getCount !== 3 || root.dataset.cacheTs !== '1000' || window.brRefreshLeague._busy || button.disabled) process.exit(4);
  process.exit(0);
})().catch(function(e){ console.error(e); process.exit(5); });
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "refresh_transaction.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=8)
    assert res.returncode == 0, res.stderr or res.stdout
