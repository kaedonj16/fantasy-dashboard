"""Loading overlays and soft-nav must never strand the user without an exit."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
RANKINGS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_br_fetch_with_timeout_is_core_utility():
    assert "window.brFetchWithTimeout = function" in APP_JS
    assert "ctl.abort()" in APP_JS


def test_br_loading_escape_exposes_arm_and_disarm():
    start = APP_JS.index("Escape hatch for full-screen loading overlays")
    block = APP_JS[start: APP_JS.index("})();", start) + 5]
    assert "window.brLoadingEscape = {" in block
    assert "arm: function" in block
    assert "disarm: disarmOne" in block
    assert "fullscreen-loading-actions" in block
    assert "Escape" in block


def test_soft_nav_has_fetch_and_progress_failsafes():
    start = APP_JS.index("(function initSoftNav()")
    block = APP_JS[start: APP_JS.index("})();", start) + 5]
    assert "progressFailsafe = setTimeout" in block
    assert "25000" in block
    assert "brFetchWithTimeout" in block


def test_dashboard_and_refresh_overlays_arm_escape():
    assert "brLoadingEscape.arm(overlay" in APP_JS
    assert "brLoadingEscape.arm(el" in APP_JS
    assert "onCancel: function" in APP_JS


def test_do_refresh_uses_timed_fetch():
    start = APP_JS.index("// ── Discord + data-freshness")
    block = APP_JS[start: APP_JS.index("window.addEventListener('beforeunload'", start)]
    assert "brFetchWithTimeout('/api/refresh-league'" in block
    assert "brFetchWithTimeout(location.href" in block


def test_league_switcher_never_blocks_forever():
    start = APP_JS.index("function navigateToLeague(leagueId")
    block = APP_JS[start: start + 1800]
    assert "setTimeout(go, 12000)" in block
    assert "brLoadingEscape.arm(overlay" in APP_JS


def test_rankings_hydrate_uses_timed_fetch():
    fn = RANKINGS.split("function prLoadData()")[1].split("prLoadData();")[0]
    assert "brFetchWithTimeout" in fn


def test_loading_escape_css_present():
    assert ".fullscreen-loading-actions {" in CSS
    assert ".fullscreen-loading-cancel" in CSS
    assert ".fullscreen-loading-retry" in CSS


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_br_fetch_with_timeout_aborts_node():
    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    harness = """
var global = globalThis;
global.window = global;
var AbortController = global.AbortController;
var setTimeout = global.setTimeout;
var clearTimeout = global.clearTimeout;
var fetchCalls = 0;
global.fetch = function(url, opts) {
  fetchCalls++;
  return new Promise(function(resolve, reject) {
    if (opts && opts.signal) {
      opts.signal.addEventListener('abort', function() {
        var err = new Error('aborted');
        err.name = 'AbortError';
        reject(err);
      });
    }
  });
};
window.brFetchWithTimeout = function (url, opts, ms) {
  ms = ms || 25000;
  opts = opts || {};
  var ctl = new AbortController();
  var timer = setTimeout(function () { ctl.abort(); }, ms);
  var merged = Object.assign({}, opts, { signal: ctl.signal });
  return fetch(url, merged).finally(function () { clearTimeout(timer); });
};
window.brFetchWithTimeout('/slow', {}, 30).catch(function(e) {
  if (e.name !== 'AbortError') process.exit(2);
  process.exit(0);
});
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "fetch_timeout.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=5)
    assert res.returncode == 0, res.stderr or res.stdout
