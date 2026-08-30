"""League switch must land on that league's current page and data.

Grades and Teams panels are now peer-relative / lazy-loaded. Switching rooms
used to keep a last-segment allowlist (dropping /draft and /draft/history) and
navigate into a prewarmed DASHBOARD_CACHE snapshot of the destination. These
tests lock the shipped JS/server contract:

  * ``navigateToLeague`` POSTs ``/api/refresh-league`` for the destination,
    then navigates. Nested paths after the league id are preserved.
  * ``api_refresh_league`` drops ``_DRAFT_GRADES_CACHE`` entries for that room.
  * Teams lazy loaders re-read ``window.__teamsCfg`` when the league identity
    changes so they do not keep serving the previous room's flags/data.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_APP_JS = _ROOT / "static" / "app.js"
_TEAMS_JS = _ROOT / "static" / "teams.js"
_ADMIN = (_ROOT / "routes" / "admin_api_bp.py").read_text(encoding="utf-8")

_NAV_RE = re.compile(
    r"function navigateToLeague\(leagueId, platform, season\) \{.*?"
    r"\}\n\n    // Top-bar league chip",
    re.DOTALL,
)


def _extract_navigate() -> str:
    src = _APP_JS.read_text(encoding="utf-8")
    m = _NAV_RE.search(src)
    assert m, "could not locate navigateToLeague() in static/app.js"
    return m.group(0).rsplit("\n\n    // Top-bar league chip", 1)[0]


def test_navigate_to_league_refreshes_destination_and_keeps_page():
    src = _extract_navigate()
    assert "/api/refresh-league" in src
    assert "pathParts.slice(3)" in src
    assert "pageParts.join('/')" in src
    assert "window.location.href = dest" in src
    # Do not fall back to a last-segment allowlist that drops /draft.
    assert "leaguePages" not in src
    assert "lastSegment" not in src


def test_refresh_league_drops_draft_grades_cache():
    fn = _ADMIN[_ADMIN.index("def api_refresh_league") :]
    fn = fn[: fn.index("def api_flush_value_cache")]
    assert "_DRAFT_GRADES_CACHE" in fn
    assert "league_id" in fn


def test_teams_loaders_resync_league_cfg():
    src = _TEAMS_JS.read_text(encoding="utf-8")
    assert "function _syncLeagueCfg()" in src
    for name in (
        "loadBtm",
        "loadSos",
        "loadDraft",
        "loadRosterIntel",
        "loadPowerRankings",
    ):
        block = src[src.index("function %s()" % name) :]
        head = block[: block.index("\n      function ") if "\n      function " in block[:800] else 200]
        # Each loader must re-read cfg before the lazy-load short-circuit.
        assert "_syncLeagueCfg()" in head, name
        assert head.index("_syncLeagueCfg()") < head.index("if (_loaded.")


_CASES = [
    {
        "from": "/sleeper/2026/aaa/draft",
        "to": ["bbb", "espn", "2026"],
        "want_dest": "/espn/2026/bbb/draft",
        "want_league": "bbb",
        "want_platform": "espn",
    },
    {
        "from": "/sleeper/2026/aaa/draft/history",
        "to": ["ccc", "sleeper", "2026"],
        "want_dest": "/sleeper/2026/ccc/draft/history",
        "want_league": "ccc",
        "want_platform": "sleeper",
    },
    {
        "from": "/sleeper/2026/aaa/teams",
        "to": ["ddd", None, None],
        "want_dest": "/sleeper/2026/ddd/teams",
        "want_league": "ddd",
        "want_platform": "sleeper",
    },
    {
        "from": "/sleeper/2026/aaa",
        "to": ["eee", "yahoo", "2025"],
        "want_dest": "/yahoo/2025/eee/dashboard",
        "want_league": "eee",
        "want_platform": "yahoo",
    },
]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_navigate_to_league_node_harness():
    harness = r"""
var currentLeagueId = 'aaa';
var currentPlatform = 'sleeper';
var currentSeason = '2026';
var fetches = [];
var locationHref = null;
var window = { location: { pathname: '', href: '' } };
function showFullscreenLoading() {}
function fetch(url, opts) {
  fetches.push({ url: url, opts: opts });
  return { then: function (ok, err) { ok(); return { then: function () {} }; } };
}
"""
    harness += _extract_navigate() + "\n"
    harness += "var cases = " + json.dumps(_CASES) + ";\n"
    harness += r"""
var bad = [];
cases.forEach(function (c) {
  fetches = [];
  locationHref = null;
  window.location.pathname = c.from;
  window.location.href = '';
  navigateToLeague(c.to[0], c.to[1], c.to[2]);
  if (fetches.length !== 1 || fetches[0].url !== '/api/refresh-league') {
    bad.push(c.from + ' missing refresh-league fetch');
    return;
  }
  var body = JSON.parse(fetches[0].opts.body);
  if (String(body.league_id) !== String(c.want_league)) {
    bad.push(c.from + ' refresh league_id=' + body.league_id + ' want ' + c.want_league);
  }
  if (String(body.platform) !== String(c.want_platform)) {
    bad.push(c.from + ' refresh platform=' + body.platform + ' want ' + c.want_platform);
  }
  if (window.location.href !== c.want_dest) {
    bad.push(c.from + ' dest=' + window.location.href + ' want ' + c.want_dest);
  }
});
// Same-league is a no-op (no fetch, no navigation).
fetches = [];
window.location.pathname = '/sleeper/2026/aaa/draft';
window.location.href = '';
navigateToLeague('aaa', 'sleeper', '2026');
if (fetches.length || window.location.href) {
  bad.push('same-league switch should be a no-op');
}
if (bad.length) { console.error(bad.join('\n')); process.exit(1); }
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "league_switch_check.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True)
    assert res.returncode == 0, "navigateToLeague mismatches:\n" + (res.stderr or res.stdout)
