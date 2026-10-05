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
ADMIN = (ROOT / "routes" / "internal_bp.py").read_text(encoding="utf-8")


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
    # so a hung fetch cannot blank the PWA forever. ScoreZone gets its own
    # longer head start (its cached shell holds stale plays).
    assert "NAV_REFRESH_TIMEOUT_MS" in SW
    assert "skipStaleShell ? NAV_REFRESH_TIMEOUT_MS" in SW
    assert "rzNav ? NAV_RZ_TIMEOUT_MS : NAV_TIMEOUT_MS" in SW
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


def _nav_fresh_block() -> str:
    start = APP_JS.index("// ── Service-worker nav-fresh")
    end = APP_JS.index("// ──", start + 10)
    return APP_JS[start:end]


def test_nav_fresh_only_reloads_for_explicit_user_refresh():
    """The stale-page auto-reload is gone; nav-fresh only swaps in the fresh
    copy after an EXPLICIT user refresh (iOS paints the cached shell first)."""
    block = _nav_fresh_block()
    assert "d.type !== 'nav-fresh'" in block
    assert "brUserRefresh" in block
    assert "if (!userRefresh) return;" in block
    assert "location.reload();" in block
    # None of the removed auto-reload machinery may come back.
    for dead in ("STALE_MS", "reloadOnce", "maybeResumeReload", "hiddenSince",
                 "brAutoRefreshTs", "liveSurface", "visibilitychange",
                 "pageshow", "__brWarmLaunch"):
        assert dead not in block, dead


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_nav_fresh_explicit_refresh_swap_behavior_node():
    """Behavioral: nav-fresh only reloads when the explicit-refresh flag is
    set. No flag, no reload; a nav-fresh for another URL is ignored."""
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    src = _nav_fresh_block()
    harness = r"""
var window = globalThis;
var reloads = 0;
var store = {};
var sessionStorage = {
  getItem: function(k) { return Object.prototype.hasOwnProperty.call(store, k) ? store[k] : null; },
  setItem: function(k, v) { store[k] = String(v); },
  removeItem: function(k) { delete store[k]; }
};
var document = { visibilityState: 'visible' };
var swListeners = {};
var navigator = { serviceWorker: { addEventListener: function(n, fn) { swListeners[n] = fn; } } };
var location = { href: 'http://x/advanced-metrics', reload: function() { reloads++; } };
""" + src + r"""
function navFresh(url) {
  swListeners['message']({ data: { type: 'nav-fresh', url: url } });
}
if (typeof swListeners['message'] !== 'function') process.exit(2);
// No explicit-refresh flag: nav-fresh must not reload.
navFresh('http://x/advanced-metrics');
if (reloads !== 0) process.exit(3);
// nav-fresh for a different URL: ignored.
store['brUserRefresh'] = '1';
navFresh('http://x/other-page');
if (reloads !== 0) process.exit(4);
// Explicit refresh + matching URL: exactly one reload, flag consumed.
navFresh('http://x/advanced-metrics');
if (reloads !== 1) process.exit(5);
if (store['brUserRefresh'] !== undefined) process.exit(6);
// A late duplicate nav-fresh after the flag was consumed: no second reload.
navFresh('http://x/advanced-metrics');
if (reloads !== 1) process.exit(7);
process.exit(0);
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "nav_fresh_swap_check.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=8)
        assert res.returncode == 0, (res.stderr or res.stdout)


def test_sw_late_network_notifies_after_explicit_refresh_fallback():
    block = SW[SW.index("async function handleNavigate") : SW.index("// ── Push notifications")]
    assert "notifyNavFresh(request, networkFetch)" in block
    # Every page paints its cached shell on timeout (ScoreZone included: its
    # shell live-polls the API on boot), with the late-network nudge after it.
    assert "if (cached) {" in block
    assert "if (cached && !rzNav)" not in block
    assert block.index("notifyNavFresh(request, networkFetch)") > block.index("if (cached) {")


def test_splash_stays_up_during_user_refresh():
    assert "brUserRefresh" in APP_PY
    assert "_warm && !_userRefresh" in APP_PY
    # __brWarmLaunch is gone: app.js no longer auto-reloads on warm
    # navigations, so the splash script no longer needs to broadcast it.
    assert "__brWarmLaunch" not in APP_PY


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
    assert "attempts = 3" not in src
    assert "function fetchFreshDocument(beforeTs, signal)" in src
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
    # Refresh marks the entry for a forced rebuild via force_refresh instead of
    # zeroing ts, so the stale-fallback keeps the old ts if the rebuild fails.
    # (internal_bp uses the lazy _dashboard_cache() accessor to avoid the
    # app circular import, so the cache var may be _dc rather than
    # DASHBOARD_CACHE.)
    assert '["force_refresh"] = True' in fn
    assert '["ts"] = 0' not in fn
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
    # Missing freshness is not a successful refresh and must stay unknown.
    assert app._league_cache_ts_ms("sleeper", 2026, "missing-league") == 0


def test_freshness_labels_handle_unknown_seconds_future_and_days():
    src = _freshness_iife()
    assert "value < 100000000000" in src
    assert "value > Date.now() + 5 * 60000" in src
    # Two label sites (updateSheetTime + updateChip) plus the auto-revalidate
    # freshness gate, which reuses the same seconds/future-tolerant normalization.
    assert src.count("normalizeTimestamp(cacheTs())") == 3
    assert "t.textContent = ts ? 'Updated ' + fmtAge(ts) : ''" in src
    assert "el.textContent = t ? fmtAge(t, true) : 'Unknown'" in src
    assert "'d ago'" in src


def test_root_swap_rejects_an_older_same_league_snapshot():
    assert "sameSnapshot" in APP_JS
    assert "incomingTs < currentTs" in APP_JS
    assert "stale league snapshot" in APP_JS
    assert "['platform', 'season', 'leagueId']" in APP_JS


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
  if (postCount !== 1 || getCount !== 1 || root.dataset.cacheTs !== '1000' || window.brRefreshLeague._busy || sheetTime.textContent.indexOf('Failed') !== 0) process.exit(3);

  mode = 'failure'; postCount = 0; getCount = 0;
  await window.brRefreshLeague();
  if (postCount !== 1 || getCount !== 1 || root.dataset.cacheTs !== '1000' || window.brRefreshLeague._busy || button.disabled) process.exit(4);
  process.exit(0);
})().catch(function(e){ console.error(e); process.exit(5); });
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "refresh_transaction.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=8)
    assert res.returncode == 0, res.stderr or res.stdout


def test_auto_revalidate_never_swaps_advanced_metrics_page():
    """The silent stale-while-revalidate pass must not swap Advanced Metrics.

    The page's whole UI (selected metric, compare columns, filters, open
    pickers) lives in in-memory page state, and its script is inline, so
    canSwapInPlace() cannot tell it is swap-unsafe. A root swap landing
    mid-interaction re-runs the bootstrap and resets the page, which reads
    as a random repaint. The pass may still warm the league cache.
    """
    src = _freshness_iife()
    assert "function autoSwapBlocked()" in src
    assert "document.getElementById('amCmdBar')" in src
    # The silent pass gates its swap on the blocklist...
    assert (
        "if (!autoSwapBlocked() && canSwapInPlace() && "
        "window.brSwapPageRoot(fresh.html))"
    ) in src
    # ...while the explicit, user-initiated Refresh keeps its own swap path.
    assert "if (canSwapInPlace() && window.brSwapPageRoot(fresh.html))" in src


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_auto_revalidate_warms_but_does_not_swap_metrics_node():
    """Behavioral: on an Advanced Metrics page the silent pass still expires
    and refetches (warming the cache) but never calls brSwapPageRoot."""
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    src = _freshness_iife()
    harness = r"""
var window = globalThis;
var location = { href: 'http://x/sleeper/2026/abc/metrics', pathname: '/sleeper/2026/abc/metrics', reload: function(){} };
window.location = location;
var navigator = { serviceWorker: null };
var postCount = 0, getCount = 0, swapCount = 0;
var staleTs = String(Date.now() - 120000);
var root = { dataset: { cacheTs: staleTs, season: String(new Date().getFullYear()) }, querySelectorAll: function(){ return []; } };
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
var DOMParser = function(){};
DOMParser.prototype.parseFromString = function(html){
  var m = /data-cache-ts=[\"'](\d+)/.exec(html);
  return { getElementById: function(id){ return id === 'page-root' && m ? { dataset: { cacheTs: m[1] } } : null; } };
};
window.brSwapPageRoot = function(html){
  swapCount++;
  var m = /data-cache-ts=[\"'](\d+)/.exec(html);
  if (!m) return false;
  root.dataset.cacheTs = m[1];
  return true;
};
function response(ok, body){ return { ok: ok, status: ok ? 200 : 500, text: function(){ return Promise.resolve(body || ''); } }; }
window.brFetchWithTimeout = function(url, opts){
  if (url === '/api/refresh-league') { postCount++; return Promise.resolve(response(true)); }
  getCount++;
  return Promise.resolve(response(true, '<main id="page-root" data-cache-ts="' + Date.now() + '"></main>'));
};
""" + src + r"""
(async function(){
  // Control page (no Advanced Metrics marker): a stale snapshot swaps in place.
  await window.brMaybeAutoRevalidate();
  if (postCount !== 1 || getCount !== 1 || swapCount !== 1) process.exit(2);
  if (root.dataset.cacheTs === staleTs) process.exit(3);
  // Advanced Metrics page: stale again, marker present. The pass must still
  // expire + refetch (cache warmed for the next load) but never swap.
  root.dataset.cacheTs = staleTs;
  elements['amCmdBar'] = {};
  postCount = 0; getCount = 0;
  await window.brMaybeAutoRevalidate();
  if (postCount !== 1 || getCount !== 1) process.exit(4);
  if (swapCount !== 1) process.exit(5);
  if (root.dataset.cacheTs !== staleTs) process.exit(6);
  process.exit(0);
})().catch(function(e){ console.error(e); process.exit(7); });
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "auto_revalidate_metrics.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=8)
    assert res.returncode == 0, res.stderr or res.stdout
