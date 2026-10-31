"""Pull-to-refresh (PWA, Tier 1) hook contracts.

Tier 1 extends the v1 contract (tests/test_pull_to_refresh.py) to five
more pages. Each page registers window.brPageRefresh() returning a
promise that settles when its OWN existing in-place load settles:

  Portfolio  the owner refresh() flow the sheet Refresh button already
             uses through brRefreshCurrentPage (POST batches + per-card
             tracker phase), registered per owner in bind()
  Waivers    wvLoad(), the page's main load; settle polled off the
             candidates skeleton it replaces on render/error
  Standings  the #standingsWeek change handler (same /api/standings-week
             fetch as a week scrub); settle is its .st-week-loading class
  Players /  prLoadData() in static/rankings.js. These are ONE surface:
  Rankings   /players is the Player Rankings board rankings.js renders,
             and the loadPlayers() in app.js is the nav-search dropdown's
             one-shot cached list, not this page. prLoadData now returns
             its load chain so the hook can wait on it.

No hook fetches anything itself and there is no reload fallback.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
RANKINGS_JS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
WAIVERS_PY = (
    ROOT / "dashboard_services" / "pages" / "waivers_page.py"
).read_text(encoding="utf-8")
CHANGELOG_PY = (
    ROOT / "dashboard_services" / "changelog.py"
).read_text(encoding="utf-8")


def _hook_block(src: str, start_marker: str, end_marker: str) -> str:
    a = src.index(start_marker)
    b = src.index(end_marker, a)
    return src[a:b]


# ── Portfolio ────────────────────────────────────────────────────────────────
def test_portfolio_hook_registered_per_owner_in_bind():
    assert "window.brPageRefresh = owner.ptrRefresh;" in APP_JS
    # Registered at the same point as the refresh-button entry point.
    bind_at = APP_JS.index("window.brRefreshCurrentPage = refresh;")
    hook_at = APP_JS.index("window.brPageRefresh = owner.ptrRefresh;")
    assert hook_at > bind_at


def test_portfolio_hook_calls_the_existing_refresh_flow():
    block = _hook_block(
        APP_JS, "owner.ptrRefresh = function () {",
        "window.brPageRefresh = owner.ptrRefresh;",
    )
    # The same refresh() whose promise resolves after the per-card
    # tracker phase, not a bare POST-and-return.
    assert "return refresh();" in block
    assert "owner.alive()" in block
    assert "delete window.brPageRefresh" in block
    # The promise-completion machinery it relies on is still the real one.
    assert "schedule(owner, card, 0, true, function (result)" in APP_JS


def test_portfolio_destroy_drops_the_ptr_registration():
    assert (
        "if (window.brPageRefresh === owner.ptrRefresh) "
        "delete window.brPageRefresh;"
    ) in APP_JS


# ── Waivers ──────────────────────────────────────────────────────────────────
def test_waivers_hook_reinvokes_wvload_without_fetching():
    block = _hook_block(
        WAIVERS_PY, "window.brPageRefresh = function()",
        "function wvLoadBigGames()",
    )
    assert "wvLoad();" in block
    # No parallel loader: the hook itself never fetches.
    assert "fetch(" not in block
    # Settle: the candidates skeleton + the request seq, capped just past
    # the loader's own 25 s failsafe.
    assert "wvCandidateRequestSeq" in block
    assert "querySelector('.skeleton')" in block
    assert "26000" in block


def test_waivers_hook_detach_guard_and_scroll_restore():
    block = _hook_block(
        WAIVERS_PY, "window.brPageRefresh = function()",
        "function wvLoadBigGames()",
    )
    assert "list.isConnected === false" in block
    assert "delete window.brPageRefresh" in block
    assert "window.scrollTo(0, y)" in block


# ── Standings ────────────────────────────────────────────────────────────────
def test_standings_hook_lives_in_the_week_selector_script():
    hook_at = APP_PY.index("window.brPageRefresh = function()")
    # The enclosing script is the week selector's: its change handler with
    # the /api/standings-week fetch precedes the hook in the same script.
    before = APP_PY[:hook_at]
    handler_at = before.rindex("'/api/standings-week?platform='")
    assert handler_at > APP_PY.index('id="standingsWeek"')


def test_standings_hook_reinvokes_the_change_handler():
    block = _hook_block(
        APP_PY, "window.brPageRefresh = function()",
        "}})();",
    )
    assert "sel.dispatchEvent(new Event('change'))" in block
    assert "fetch(" not in block
    # Settle signal is the handler's own loading class; week untouched.
    assert ".st-week-loading" in block
    assert "sel.isConnected === false" in block
    assert "delete window.brPageRefresh" in block
    assert "window.scrollTo(0, y)" in block
    assert "30000" in block


# ── Players / Rankings (one rankings.js board) ───────────────────────────────
def test_rankings_hook_registered_and_returns_the_real_load():
    block = _hook_block(
        RANKINGS_JS, "window.brPageRefresh = function () {", "\n};",
    )
    assert "getElementById('prList')" in block
    assert "list.isConnected === false" in block
    assert "delete window.brPageRefresh" in block
    assert "return prLoadData();" in block


def test_prloaddata_returns_its_load_chain():
    start = RANKINGS_JS.index("function prLoadData()")
    block = RANKINGS_JS[start:RANKINGS_JS.index("]).then(", start)]
    assert "return Promise.all([" in block


def test_players_and_rankings_are_the_one_rankings_js_board():
    # The /players route assembles the Player Rankings shell and loads
    # rankings.js; there is no second in-app rankings list loader.
    players_py = (
        ROOT / "dashboard_services" / "pages" / "players_page.py"
    ).read_text(encoding="utf-8")
    assert "Player Rankings (/players) page shell" in players_py
    assert "prSetScoringType" in players_py  # shell controls call rankings.js


# ── Changelog ────────────────────────────────────────────────────────────────
def test_changelog_prepend_names_the_tier1_pages():
    entry = "Pull to refresh now works on more pages in the app"
    assert entry in CHANGELOG_PY
    block = CHANGELOG_PY[CHANGELOG_PY.index(entry):CHANGELOG_PY.index(entry) + 400]
    for page in ("My Leagues", "Waivers", "Standings", "Player Rankings"):
        assert page in block
    # Found by content, never by position: any later PR that prepends its
    # own entry pushes this one down the list, and that must not fail here
    # (the repo's changelog pins are content lookups, not index pins).


# ── Behavioral harness (Node) ─────────────────────────────────────────────────
def test_ptr_tier1_hook_harness():
    """Drives the real extracted hook blocks with fake DOM: each calls its
    page entry point exactly once, settles with it, and self-unregisters
    without loading when its page root is detached/dead."""
    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not available for the JS behavioral harness")
    result = subprocess.run(
        [node, str(ROOT / "tests" / "pull_to_refresh_tier1_harness.mjs")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL PULL-TO-REFRESH TIER-1 HARNESS CHECKS PASSED" in result.stdout
