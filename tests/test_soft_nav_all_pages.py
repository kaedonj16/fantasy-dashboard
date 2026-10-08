"""Soft-nav on every shell page: script allowlist, reexec bridge, rendered coverage.

Locks down the pieces that let previously native-only pages swap in place:

  * ``scriptReRunnable()`` must match the *minified* srcs the server actually
    serves (``/static/rankings.min.js?v=...``) -- a substring match against the
    unminified list entries never matched, so allowlisted pages silently
    bailed to a full navigation. Executed from the real app.js source via Node.

  * ``reexecScripts()`` must bridge DOMContentLoaded / window load: both
    already fired for the document, so a swapped-in page script registering
    for them would never initialise. The bridge captures those registrations
    while the new scripts run and invokes them immediately after. Executed
    from the real source against a minimal DOM stub.

  * Every page in the SOFT_NAV_PAGES allowlist, rendered through the offline
    client with the X-Soft-Nav header, must only pull page-root external
    scripts that are on the SOFT_OK_SCRIPTS allowlist (normalized the same
    way the client normalizes them).

  * The two live-polling page modules (scorezone, draft room) must stop their
    timers when their page DOM is swapped out, and the graphs radar must use
    the readyState init pattern instead of a bare DOMContentLoaded listener.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_JS = os.path.join(_REPO_ROOT, "static", "app.js")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _ok_scripts() -> list[str]:
    src = _read(_APP_JS)
    m = re.search(r"var SOFT_OK_SCRIPTS = \[(.*?)\];", src, re.S)
    assert m, "SOFT_OK_SCRIPTS not found in static/app.js"
    return re.findall(r"'([^']+)'", m.group(1))


def _normalize_base(src: str) -> str:
    base = src.split("?")[0].split("#")[0].rstrip("/").split("/")[-1]
    return re.sub(r"\.min(\.js)$", r"\1", base, flags=re.I)


def _run_node(harness: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "check.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr or res.stdout


# ── scriptReRunnable: minified src matching ──────────────────────────────────

_SCRIPT_CASES = [
    # The srcs the server actually serves (minified, versioned query string).
    ("/static/teams.min.js?v=abc123", True),
    ("/static/rankings.min.js?v=abc123", True),
    ("/static/scorezone.min.js?v=1", True),
    ("/static/keeper.min.js", True),
    ("/static/cheat_sheet.min.js?v=2", True),
    ("/static/custom_selects.min.js", True),
    ("/static/draft_room.min.js?v=3", True),
    ("/static/draft_board_core.min.js", True),
    ("/static/draft_grade_team.min.js", True),
    ("/static/pick_score.min.js", True),
    # Unminified forms (dev / fresh checkout) match too.
    ("/static/teams.js", True),
    ("/static/rankings.js?v=9", True),
    ("/static/keeper.js", True),
    # Shell bundles and anything unknown stay full-navigation-only.
    ("/static/app.min.js?v=1", False),
    ("/static/app-features.min.js?v=1", False),
    ("/static/public.min.js?v=1", False),
    ("/static/paywall.js", False),
    ("/static/player_modal.min.js?v=1", False),
    ("/static/plotly.min.js", False),
    ("/static/some-random-lib.min.js", False),
    ("https://cdn.example.com/teams.min.js", True),  # basename match, any host
]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_script_rerunnable_matches_minified_srcs():
    src = _read(_APP_JS)
    m = re.search(
        r"var SOFT_OK_SCRIPTS = \[.*?\];.*?"
        r"function scriptReRunnable\(src\) \{.*?\n  \}",
        src,
        re.S,
    )
    assert m, "could not locate SOFT_OK_SCRIPTS/scriptReRunnable in app.js"
    harness = (
        m.group(0)
        + "\nvar cases = " + json.dumps(_SCRIPT_CASES) + ";\n"
        + "var bad = [];\n"
        + "cases.forEach(function (c) {\n"
        + "  var got = scriptReRunnable(c[0]);\n"
        + "  if (got !== c[1]) bad.push(c[0] + ' -> ' + got + ' (want ' + c[1] + ')');\n"
        + "});\n"
        + "if (bad.length) { console.error(bad.join('\\n')); process.exit(1); }\n"
    )
    _run_node(harness)


# ── reexecScripts: DOMContentLoaded / load bridge ────────────────────────────

@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_reexec_bridges_domcontentloaded_and_load():
    src = _read(_APP_JS)
    m = re.search(r"function reexecScripts\(container\) \{.*?\n  \}", src, re.S)
    assert m, "could not locate reexecScripts in app.js"
    page_script = (
        "document.addEventListener('DOMContentLoaded', function () { globalThis.__dcl = 1; });\n"
        "window.addEventListener('DOMContentLoaded', function () { globalThis.__wdcl = 1; });\n"
        "window.addEventListener('load', function () { globalThis.__load = 1; });\n"
        "document.addEventListener('click', function () { globalThis.__click = 1; });\n"
    )
    harness = (
        "var realDoc = [], realWin = [];\n"
        "globalThis.document = {\n"
        "  addEventListener: function (t) { realDoc.push(t); },\n"
        "  createElement: function () { return { attrs: {}, setAttribute: function (k, v) { this.attrs[k] = v; } }; },\n"
        "};\n"
        "globalThis.window = { addEventListener: function (t) { realWin.push(t); } };\n"
        + m.group(0)
        + "\nvar old = { attributes: [], src: '', textContent: " + json.dumps(page_script) + ",\n"
        + "  parentNode: { replaceChild: function (s) { eval(s.textContent); } } };\n"
        "reexecScripts({ querySelectorAll: function () { return [old]; } });\n"
        "var bad = [];\n"
        "if (globalThis.__dcl !== 1) bad.push('document DOMContentLoaded not bridged');\n"
        "if (globalThis.__wdcl !== 1) bad.push('window DOMContentLoaded not bridged');\n"
        "if (globalThis.__load !== 1) bad.push('window load not bridged');\n"
        "if (realDoc.indexOf('click') === -1) bad.push('click did not pass through to real addEventListener');\n"
        "if (realDoc.indexOf('DOMContentLoaded') !== -1) bad.push('DOMContentLoaded leaked to real addEventListener');\n"
        "if (realWin.indexOf('load') !== -1) bad.push('load leaked to real addEventListener');\n"
        "if (document.addEventListener.toString().indexOf('realDoc') === -1 && realDoc.length !== 1) bad.push('addEventListener not restored');\n"
        "if (bad.length) { console.error(bad.join('\\n')); process.exit(1); }\n"
    )
    _run_node(harness)


# ── Rendered pages: page-root external scripts are all allowlisted ──────────

_LEAGUE = "/sleeper/2026/tourdemo"
_PAGES = [
    f"{_LEAGUE}/dashboard", f"{_LEAGUE}/standings", f"{_LEAGUE}/teams",
    f"{_LEAGUE}/activity", f"{_LEAGUE}/weekly", f"{_LEAGUE}/recap",
    f"{_LEAGUE}/awards", f"{_LEAGUE}/history", f"{_LEAGUE}/league_health",
    f"{_LEAGUE}/players", f"{_LEAGUE}/breakouts", f"{_LEAGUE}/waivers",
    f"{_LEAGUE}/schedule", f"{_LEAGUE}/graphs", f"{_LEAGUE}/metrics",
    f"{_LEAGUE}/nfl-teams", f"{_LEAGUE}/scorezone", f"{_LEAGUE}/keeper",
    f"{_LEAGUE}/compare", f"{_LEAGUE}/trade", f"{_LEAGUE}/trade-database",
    f"{_LEAGUE}/prospects", f"{_LEAGUE}/draft", f"{_LEAGUE}/draft/history",
    f"{_LEAGUE}/draft/cheat-sheet",
    "/players", "/prospects", "/compare", "/trade", "/trade-database",
    "/keeper", "/draft", "/dynasty-trade-value-chart", "/rankings/dynasty",
    "/top-movers", "/oline-rankings",
    "/about", "/glossary", "/guides", "/faq", "/pricing", "/privacy",
    "/terms", "/support", "/contact",
]

_SCRIPT_RE = re.compile(r"""<script[^>]+src=["']([^"']+)["']""", re.I)


def _page_root(html: str) -> str:
    start = html.find('id="page-root"')
    assert start != -1, "no #page-root in response"
    open_end = html.find(">", start)
    end = html.find("</main>", open_end)
    assert end != -1, "no closing </main> for #page-root"
    return html[open_end:end]


@pytest.mark.parametrize("path", _PAGES)
def test_soft_nav_page_scripts_allowlisted(offline_client, path):
    r = offline_client.get(path, headers={"X-Soft-Nav": "1"})
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    root = _page_root(r.get_data(as_text=True))
    ok = set(_ok_scripts())
    bad = [
        s for s in _SCRIPT_RE.findall(root)
        if _normalize_base(s) not in ok
    ]
    assert not bad, f"{path} pulls non-re-runnable page scripts: {bad}"


# ── Live-polling modules stop when swapped out; graphs radar init pattern ────

def test_scorezone_stops_polling_when_swapped_out():
    src = _read(os.path.join(_REPO_ROOT, "static", "scorezone.js"))
    tick = re.search(r"function _tick\(\) \{(.*?)\n  \}", src, re.S)
    assert tick and "root.isConnected" in tick.group(1), (
        "_tick must bail (and clear its interval) once #rz-root is detached"
    )
    refresh = re.search(r"async function _refresh\(opts\) \{(.*?)\n    if \(_streaming\)", src, re.S)
    assert refresh and "root.isConnected" in refresh.group(1), (
        "_refresh must no-op once #rz-root is detached"
    )


def test_draft_room_stops_timers_when_swapped_out():
    src = _read(os.path.join(_REPO_ROOT, "static", "draft_room.js"))
    assert "_drRootEl" in src and "_drGone()" in src
    assert "_stopTimersOnly" in src
    poll = re.search(r"function pollOnce\(\)\s*\{(.*?)\n    if \(_pollInFlight\)", src, re.S)
    assert poll and "_drGone()" in poll.group(1), "pollOnce must stop when the draft page is gone"
    ticker = re.search(r"function startPolling\(\)\s*\{(.*?)pollOnce\(\);", src, re.S)
    assert ticker and "_drGone()" in ticker.group(1), "the 1s poll ticker must stop when gone"


def test_graphs_tabs_init_inline():
    # The tab switcher must be inline (IIFE at the end of the body) so it
    # works after a soft-nav swap without a DOMContentLoaded listener.
    src = _read(os.path.join(
        _REPO_ROOT, "dashboard_services", "pages", "graphs_page.py"))
    assert ".gs-tabs .gs-tab[data-tab]" in src
    assert "document.addEventListener('DOMContentLoaded', () =>" not in src


def test_graphs_radar_uses_readystate_init():
    # The radar chart's init must use the readyState guard (not a bare
    # DOMContentLoaded listener) so it renders after a soft-nav swap.
    src = _read(os.path.join(
        _REPO_ROOT, "dashboard_services", "pages", "graphs_page.py"))
    assert "_initRadar" in src, "radar chart is back on the Performance tab"
    assert "document.readyState" in src
