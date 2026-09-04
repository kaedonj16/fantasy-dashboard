"""PWA navigations must never hang on a blank white screen.

The service worker used to race the network against a timeout ONLY when a
cached copy of the page already existed. Cold PWA launches (empty cache after
a SW update, first install, or a start_url that had never been cached) awaited
``fetch()`` forever — and on a slow/sleeping origin or a fetch that never
settles (common on mobile / iOS standalone) that is a permanent blank white
screen.

These contracts lock the fix:
  * Every navigation races the network against a timeout, cached or not.
  * Timed-out navigations with a warm cache paint the cache (and notify
    nav-fresh); uncached navigations get a longer grace before offline.html
    so slow-but-online loads are not painted as "You're offline".
  * A settled HTTP error (404/500) is preferred over the offline shell.
  * Redirected responses are rebuilt without Content-Encoding headers.
  * The offline shell reloads when the SW later posts ``nav-fresh``, and
    does not claim "You're offline" when ``navigator.onLine`` is true.
  * Cache name is bumped so poisoned entries from older SWs are purged.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")
OFFLINE = (ROOT / "static" / "offline.html").read_text(encoding="utf-8")


def _handle_navigate() -> str:
    start = SW.index("async function handleNavigate")
    end = SW.index("// ── Push notifications", start)
    return SW[start:end]


def test_cache_name_bumped_for_blank_screen_fix():
    assert "br-fantasy-v27" in SW


def test_nav_timeout_always_races_even_without_cache():
    body = _handle_navigate()
    # The forever-await path is gone: we always Promise.race against a timeout.
    assert "Promise.race([networkFetch, timeout])" in body
    assert "NAV_TIMEOUT_MS" in body
    assert "NAV_REFRESH_TIMEOUT_MS" in body
    assert "NAV_UNCACHED_GRACE_MS" in body
    # Must not gate the race on an existing cached copy.
    assert "if (cached && !skipStaleShell)" not in body
    assert "const net = await networkFetch;" not in body


def test_navigation_fallback_chain():
    assert "async function navigationFallback" in SW
    assert "cache.match(OFFLINE_URL)" in SW
    assert "cache.match('/')" in SW
    body = _handle_navigate()
    assert "navigationFallback(cache, null)" in body
    assert "if (cached) {" in body
    assert "if (networkError) return networkError;" in body
    assert "notifyNavFresh(request, networkFetch)" in body
    # Uncached timeout must wait for grace / in-flight fetch before offline.
    assert "NAV_UNCACHED_GRACE_MS" in body
    assert "Promise.race([networkFetch, grace])" in body


def test_unredirect_strips_encoding_headers():
    start = SW.index("async function unredirect")
    fn = SW[start: SW.index("function notifyNavFresh", start)]
    assert "headers.delete('content-encoding')" in fn
    assert "headers.delete('content-length')" in fn
    assert "headers.delete('transfer-encoding')" in fn
    assert "response.redirected" in fn


def test_network_only_wins_when_ok():
    body = _handle_navigate()
    assert "clean.ok" in body
    # Non-OK must not be returned as a race winner over a cached shell.
    assert "return null;" in body


def test_offline_shell_reloads_on_nav_fresh():
    assert "nav-fresh" in OFFLINE
    assert "serviceWorker" in OFFLINE
    assert "location.reload()" in OFFLINE


def test_offline_shell_does_not_claim_offline_when_online():
    """False-positive fix: slow online loads must not keep saying You're offline."""
    assert "Still connecting" in OFFLINE
    assert "navigator.onLine" in OFFLINE
    assert "You're online, but BR Fantasy is taking longer than usual" in OFFLINE


@pytest.mark.skipif(
    __import__("os").environ.get("SKIP_NODE") == "1", reason="node skipped"
)
def test_hung_fetch_resolves_via_timeout_node():
    """Simulate the SW race: a never-settling fetch must not block forever."""
    import os
    import shutil
    import subprocess
    import tempfile

    if shutil.which("node") is None:
        pytest.skip("Node.js not available")

    harness = r"""
const NAV_TIMEOUT_MS = 50;
const NAV_UNCACHED_GRACE_MS = 80;
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function handleNavigate({ networkFetch, cached, offline }) {
  const timeout = new Promise(resolve => setTimeout(() => resolve(null), NAV_TIMEOUT_MS));
  const winner = await Promise.race([networkFetch, timeout]);
  if (winner) return { source: 'network', body: winner };
  if (cached) return { source: 'cache', body: cached };
  const grace = new Promise(resolve => setTimeout(() => resolve(null), NAV_UNCACHED_GRACE_MS));
  const late = await Promise.race([networkFetch, grace]);
  if (late) return { source: 'network-late', body: late };
  if (offline) return { source: 'offline', body: offline };
  return { source: 'error' };
}

(async () => {
  // Hung network, no cache → offline shell after grace (NOT a permanent hang).
  const hung = new Promise(() => {});
  const started = Date.now();
  const r1 = await handleNavigate({
    networkFetch: hung,
    cached: null,
    offline: 'OFFLINE',
  });
  const elapsed = Date.now() - started;
  if (r1.source !== 'offline' || r1.body !== 'OFFLINE') process.exit(2);
  if (elapsed > 2000) process.exit(3);  // must not wait forever
  if (elapsed < 100) process.exit(7);   // must wait past the short + grace timeouts

  // Hung network, warm cache → cached shell (no grace wait required).
  const r2Started = Date.now();
  const r2 = await handleNavigate({
    networkFetch: new Promise(() => {}),
    cached: 'CACHED',
    offline: 'OFFLINE',
  });
  if (r2.source !== 'cache' || r2.body !== 'CACHED') process.exit(4);
  if (Date.now() - r2Started > 500) process.exit(8);

  // Fast OK network still wins.
  const r3 = await handleNavigate({
    networkFetch: Promise.resolve('FRESH'),
    cached: 'CACHED',
    offline: 'OFFLINE',
  });
  if (r3.source !== 'network' || r3.body !== 'FRESH') process.exit(5);

  // Slow network (after short timeout) loses to cache.
  const r4 = await handleNavigate({
    networkFetch: sleep(200).then(() => 'LATE'),
    cached: 'CACHED',
    offline: 'OFFLINE',
  });
  if (r4.source !== 'cache') process.exit(6);

  // Slow network, no cache → wait through grace and accept the late response
  // instead of painting offline while online.
  const r5 = await handleNavigate({
    networkFetch: sleep(100).then(() => 'LATE_OK'),
    cached: null,
    offline: 'OFFLINE',
  });
  if (r5.source !== 'network-late' || r5.body !== 'LATE_OK') process.exit(9);

  process.exit(0);
})().catch(err => { console.error(err); process.exit(1); });
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "nav_race.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=5)
    assert res.returncode == 0, res.stderr or res.stdout
