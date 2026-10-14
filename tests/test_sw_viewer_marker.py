"""Viewer-state marker: a cached guest shell must not stick after sign-in.

The service worker caches page HTML keyed by URL only and paints the cached
copy when the network loses the 3.5s race (routine on mobile). One dashboard
paint captured while the viewer was unresolved (guest paint, expired
session) therefore kept serving the "Link your team to see this week's
actions" card to a signed-in, already-linked manager, and the nav-fresh
handler deliberately never swapped the fresh copy in outside an explicit
user refresh.

The surgical fix: the server stamps every page head with a
``br-viewer-state`` identity marker (signed-in / account / viewer ids); the
SW compares the painted copy's marker with the fresh copy's and flags
``viewerChanged`` on the nav-fresh message only when both markers exist and
differ; app.js then performs the swap-in reload (rate-limited per URL).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")


def test_server_stamps_viewer_state_meta_in_shell():
    assert '<meta name="br-viewer-state" content="{viewer_state_meta}">' in APP_PY
    # The marker is computed in render_page from the same session state the
    # window._viewerRid/_isSignedIn script uses, and passed to the shell.
    assert "_viewer_state_meta" in APP_PY
    assert "viewer_state_meta=_viewer_state_meta" in APP_PY
    assert '_session_signed_in()' in APP_PY
    assert 'session.get("account_id")' in APP_PY


def test_sw_extracts_and_compares_viewer_markers():
    assert "function viewerStateOf" in SW
    assert '<meta name="br-viewer-state" content="' in SW
    # Both copies must carry a marker and they must differ; a missing marker
    # (older cached shell, non-shell page) never forces a swap.
    assert "before && after && before !== after" in SW
    assert "viewerChanged: viewerChanged" in SW
    # The painted copy is cloned synchronously, before the navigation
    # consumes its body.
    assert "cachedResponse.clone()" in SW
    # handleNavigate hands the painted copy to the notifier on both the
    # timeout-paint path and the offline-fallback path.
    block = SW[SW.index("async function handleNavigate") : SW.index("// ── Push notifications")]
    assert block.count("notifyNavFresh(request, networkFetch, cached)") == 2


def test_client_reloads_only_on_flagged_viewer_change():
    start = APP_JS.index("// ── Service-worker nav-fresh")
    end = APP_JS.index("// ──", start + 10)
    block = APP_JS[start:end]
    assert "d.viewerChanged === true" in block
    # Rate limit: one viewer swap per URL per 30s window, in sessionStorage.
    assert "brViewerSwapAt" in block
    assert "30000" in block


def _sw_marker_block() -> str:
    start = SW.index("function viewerStateOf")
    end = SW.index("async function navigationFallback")
    return SW[start:end]


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_sw_viewer_changed_flag_behavior_node():
    """Behavioral: notifyNavFresh flags viewerChanged only when the painted
    copy's marker and the fresh copy's marker both exist and differ."""
    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    src = _sw_marker_block()
    harness = r"""
var posted = [];
var clients = {
  matchAll: function() {
    return Promise.resolve([{ url: 'http://x/dash', postMessage: function(m) { posted.push(m); } }]);
  }
};
function fakeResponse(html) {
  return {
    clone: function() { return fakeResponse(html); },
    text: function() { return Promise.resolve(html); }
  };
}
var request = { url: 'http://x/dash' };
""" + src + r"""
function page(marker) {
  return '<html><head><meta name="br-viewer-state" content="' + marker + '"></head><body>x</body></html>';
}
async function run(cachedHtml, freshHtml) {
  posted = [];
  var cached = cachedHtml === null ? null : fakeResponse(cachedHtml);
  var networkFetch = Promise.resolve(fakeResponse(freshHtml));
  notifyNavFresh(request, networkFetch, cached);
  await new Promise(function(r) { setTimeout(r, 20); });
  if (posted.length !== 1) process.exit(2);
  return posted[0].viewerChanged;
}
(async function() {
  // Guest paint, signed-in fresh copy: changed.
  if (await run(page('v1:0:0::'), page('v1:1:1:12345:3')) !== true) process.exit(3);
  // Same identity on both copies: not changed.
  if (await run(page('v1:1:1:12345:3'), page('v1:1:1:12345:3')) !== false) process.exit(4);
  // Signed-in paint, guest fresh copy (signed out elsewhere): changed.
  if (await run(page('v1:1:1:12345:3'), page('v1:0:0::')) !== true) process.exit(5);
  // Marker missing from the painted copy (pre-marker cache): never forced.
  if (await run('<html><head></head><body>x</body></html>', page('v1:1:1:12345:3')) !== false) process.exit(6);
  // Marker missing from the fresh copy: never forced.
  if (await run(page('v1:0:0::'), '<html><head></head><body>x</body></html>') !== false) process.exit(7);
  // No painted copy at all (offline fallback path): not changed.
  if (await run(null, page('v1:1:1:12345:3')) !== false) process.exit(8);
  process.exit(0);
})().catch(function(e) { console.error(e); process.exit(9); });
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "sw_viewer_marker_check.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=15)
        assert res.returncode == 0, (res.stderr or res.stdout)
