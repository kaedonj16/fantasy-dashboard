"""Refresh modal + targeted refresh contracts.

The Refresh Data popup: a centered modal card (site styles) shown at the
START of a user-initiated refresh, with a brief success state and a
failure state that stays up with Try again / Dismiss instead of
vanishing silently.

Targeted refresh: pages that already register a verified in-place load
as window.brPageRefresh (Standings, Waivers, Weekly Hub, Rankings,
ScoreZone) are refreshed through that load after the league expire,
instead of fetching the whole page document / hard-reloading. Pages
without a hook keep the global document path, and the silent
auto-revalidate pass never shows the modal.

Second pass (everywhere applicable): every page was audited the same
way. Pages with a real loader register it as brPageRefresh (Schedule,
Advanced Metrics, NFL Teams, Breakouts, Prospects, Top Movers, Trade
Intel, Trade Database, Draft Database, Cheat Sheet, Draft History,
History, Autopilot, Trade Calculator/Hub, Compare, Home). Weekly Hub's
hook also refills its Scout/Lineup tabs. Server-rendered pages without
a loader (Dashboard, Teams, Activity, League Health, Scout, Recap,
Awards, Graphs, Draft Recap, offseason dashboard) take the targeted
section tier: brSwapChangedSections swaps only the sections whose
markup changed from the fresh document. Pages whose content is live
user/draft state (Draft Room, Keeper Assistant) stay excluded with
named reasons in the code.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _freshness_iife() -> str:
    start = APP_JS.index("// ── Discord + data-freshness")
    end = APP_JS.index("window.addEventListener('beforeunload'", start)
    return APP_JS[start:end]


def _do_refresh() -> str:
    start = APP_JS.index("async function doRefresh()")
    end = APP_JS.index("window.brRefreshLeague = doRefresh;", start)
    return APP_JS[start:end]


# ── Modal markup / states ────────────────────────────────────────────────────
def test_refresh_modal_card_markup_and_states():
    src = _freshness_iife()
    assert "br-refresh-modal-overlay" in src
    assert "br-refresh-modal-card" in src
    assert "Refreshing data..." in src
    assert "You're up to date" in src
    assert "Couldn't refresh" in src
    assert "br-refresh-modal-retry" in src
    assert "br-refresh-modal-dismiss" in src
    # No em dashes in the modal's UI copy.
    modal = src[src.index("function setRefreshModalState"):src.index("function ackBypassCache")]
    assert "—" not in modal


def test_refresh_modal_settles_visibly_in_do_refresh():
    body = _do_refresh()
    assert "outcome = 'success'" in body
    assert "outcome = 'fail'" in body
    assert "showRefreshSuccess(false)" in body
    assert "showRefreshFailure(failRateLimit)" in body


def test_refresh_modal_shared_surface_for_pull_to_refresh():
    src = _freshness_iife()
    assert "window.brRefreshModal = {" in src
    # The PTR controller shows the modal at the start of a pull and
    # settles it with the hook's promise.
    ptr = APP_JS[APP_JS.index("/* br-ptr:start */"):APP_JS.index("/* br-ptr:end */")]
    assert "window.brRefreshModal.show()" in ptr
    assert "window.brRefreshModal.succeed()" in ptr
    assert "window.brRefreshModal.fail(" in ptr


def test_refresh_modal_css_uses_site_tokens():
    assert ".br-refresh-modal-overlay {" in CSS
    assert "rgba(10, 25, 42, 0.56)" in CSS
    assert ".br-refresh-modal-card {" in CSS
    assert "border-radius: 16px" in CSS
    card = CSS[CSS.index(".br-refresh-modal-spinner {"):CSS.index(".br-refresh-modal-actions {")]
    assert "var(--spinner)" in card


def test_auto_revalidate_never_shows_the_modal():
    src = _freshness_iife()
    auto = src[src.index("async function autoRevalidate()"):src.index("window.brMaybeAutoRevalidate")]
    assert "showRefreshOverlay" not in auto
    assert "brRefreshModal" not in auto


# ── Targeted refresh ─────────────────────────────────────────────────────────
def test_do_refresh_prefers_the_pages_own_targeted_load():
    body = _do_refresh()
    # Portfolio's handler keeps first precedence.
    assert body.index("brRefreshCurrentPage") < body.index("window.brPageRefresh")
    # The targeted path expires the league, then runs only the page hook,
    # and only falls back to the full document fetch when it fails.
    assert body.index("window.brPageRefresh") < body.index("fetchFreshDocument")
    assert "await pageHook();" in body
    assert "leagueExpired" in body


# ── Everywhere applicable: section tier ─────────────────────────────────────
def test_section_swap_exists_with_exclusions_and_skip_list():
    assert "window.brSwapChangedSections = function (html)" in APP_JS
    assert "window.brSectionSwapBlockedRoot = sectionSwapBlockedRoot" in APP_JS
    blocked = APP_JS[APP_JS.index("function sectionSwapBlockedRoot"):APP_JS.index("window.brSectionSwapBlockedRoot")]
    # Loader-owned live surfaces and the Draft Room never section-swap:
    # their server shell would wipe painted data back to skeletons.
    for marker in ("#drSideTabs", "#rz-root", "#breakoutsContainer", "#amCmdBar", "#tdbList"):
        assert marker in blocked
    # Recap bodies still carrying async fill slots are excluded too.
    assert "data-recap-activity-slot" in blocked
    skip = APP_JS[APP_JS.index("var SECTION_SKIP_IDS"):APP_JS.index("function sectionSwapBlockedRoot")]
    # Load-once JS mounts inside swappable pages are preserved, not wiped.
    for mount in ("sinceLastVisitCard", "os-jump-actions", "btmPanel", "sosPanel",
                  "nflNewsCard", "historyAwardsContent", "historyChartContent"):
        assert mount in skip


def test_do_refresh_prefers_section_tier_over_full_root_swap():
    body = _do_refresh()
    assert body.index("brSwapChangedSections(fresh.html)") < body.index("canSwapInPlace() && window.brSwapPageRoot(fresh.html)")
    assert "sectionRes" in body


def test_section_refresh_surface_and_home_hook():
    src = _freshness_iife()
    assert "window.brSectionRefreshEligible = sectionRefreshEligible" in src
    assert "window.brSectionRefresh = async function ()" in src
    # Home registers its own loaders (league list + ticker) as the hook.
    home = src[src.index("if (location.pathname === '/') {"):]
    assert "window.brPageRefresh = function ()" in home
    assert "refreshHomeLeagues({ force: true })" in home
    assert "populateHomeTicker" in home


def test_ptr_falls_back_to_section_refresh():
    ptr = APP_JS[APP_JS.index("/* br-ptr:start */"):APP_JS.index("/* br-ptr:end */")]
    assert "function ptrHook()" in ptr
    assert "window.brSectionRefreshEligible()" in ptr
    assert "ptrHook()()" in ptr


# ── Everywhere applicable: per-page loader registrations ────────────────────
def test_schedule_rankings_registers_its_render_loader():
    src = _read("dashboard_services/pages/schedule_page.py")
    hook = src[src.index("window.brPageRefresh = function()"):]
    assert "mainCache = {};" in hook
    assert "render();" in hook


def test_advanced_metrics_registers_forced_fetch_data():
    src = _read("dashboard_services/pages/advanced_metrics_page.py")
    hook = src[src.index("window.brPageRefresh = function()"):]
    # The board-signature guard must be cleared or the "refresh" is a no-op.
    assert "state._boardSig = null;" in hook
    assert "fetchData();" in hook
    assert "_loadMovers();" in hook


def test_nfl_teams_registers_load_rankings():
    src = _read("dashboard_services/pages/nfl_teams_page.py")
    hook = src[src.index("window.brPageRefresh = function()"):]
    assert "DATA = null;" in hook
    assert "loadRankings();" in hook


def test_breakouts_registers_current_week_loader():
    hook = APP_PY[APP_PY.index("window.brPageRefresh = function()"):]
    assert "loadBreakouts(sel ? sel.value : currentWeek)" in hook


def test_prospects_registers_class_reload():
    src = _read("dashboard_services/pages/rookies_page.py")
    hook = src[src.index("window.brPageRefresh = function()"):]
    assert "rkSetClass(rkViewSel)" in hook


def test_top_movers_registers_its_swap_without_navigation():
    src = _read("dashboard_services/pages/dynasty_pages.py")
    assert "return swap(window.location.href, false, { noNav: true });" in src
    assert "function swap(url, push, opts)" in src


def test_trade_intel_registers_current_page_loader():
    src = _read("dashboard_services/pages/trade_intel_page.py")
    hook = src[src.index("window.brPageRefresh = function()"):]
    assert "loadTIPage(currentPage)" in hook


def test_trade_database_registers_current_page_loader():
    src = _read("routes/trade_bp.py")
    hook = src[src.index("window.brPageRefresh = function()"):]
    assert "loadTDBPage(currentPage)" in hook


def test_draft_database_registers_current_view_loader():
    src = _read("routes/draft_db_bp.py")
    hook = src[src.index("window.brPageRefresh = function()"):]
    assert "loadAdp()" in hook
    assert "openBoard(boardDraftId)" in hook
    assert "loadPage(currentPage)" in hook


def test_cheat_sheet_registers_load_players():
    src = _read("static/cheat_sheet.js")
    hook = src[src.index("window.brPageRefresh = function ()"):]
    assert "var req = loadPlayers();" in hook
    assert "if (loadError) throw new Error(loadError);" in hook


def test_draft_history_registers_load_list():
    src = _read("dashboard_services/pages/draft_room_page.py")
    hook = src[src.index("window.brPageRefresh = function()"):]
    assert "loadList()" in hook


def test_history_registers_sections_loader():
    hook = APP_JS[APP_JS.index("window.brPageRefresh = () => loadHistorySections()") - 200:]
    assert "histSectionsFailed" in APP_JS
    assert "loadHistorySections" in hook


def test_autopilot_registers_soft_refresh():
    src = _read("dashboard_services/pages/autopilot_page.py")
    hook = src[src.index("window.brPageRefresh = function()"):]
    assert "softRefresh(function(ok)" in hook


def test_trade_pages_register_pool_revalidation():
    hook = APP_JS[APP_JS.index("window.brPageRefresh = function () {\n    var startedAt = Date.now();"):]
    assert "brGetLeaguePlayersData(_lpFeedKey" in hook
    assert "recomputeTrade()" in hook


def test_compare_registers_chosen_refetch():
    idx = APP_JS.index("window.brPageRefresh = function () {\n    if (!chosen[1] || !chosen[2] || !resultEl)")
    hook = APP_JS[idx:idx + 400]
    assert "_maybeCompare();" in hook


def test_weekly_hub_hook_refills_scout_and_lineup_tabs():
    src = _read("dashboard_services/pages/weekly_hub_page.py")
    assert "window.__hubRefillLazyTabs" in src
    assert "fillTab(t, true)" in src
    assert "window.__hubRefillBakedTabs" in src
    assert "__hubRefillLazyTabs()" in src


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_section_tier_swaps_sections_not_whole_root_node():
    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    src = _freshness_iife()
    harness = r"""
var window = globalThis;
var location = { href: 'http://x/sleeper/2026/abc/dashboard', pathname: '/sleeper/2026/abc/dashboard', reload: function(){ this.reloaded = true; } };
window.location = location;
var navigator = { serviceWorker: null };
var postCount = 0, docFetches = 0;
var root = { dataset: { cacheTs: '1000' }, querySelectorAll: function(){ return []; } };
var sheetTime = { textContent: '', classList: { toggle: function(){} } };
var button = { disabled: false, _brWired: false, setAttribute: function(){}, removeAttribute: function(){}, addEventListener: function(){} };
var elements = { 'page-root': root, 'brSheetRefreshTime': sheetTime, 'brSheetRefresh': button };
var DOMParser = function(){};
DOMParser.prototype.parseFromString = function(){
  return { getElementById: function(){ return { dataset: { cacheTs: '2000' } }; } };
};
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
window.brFetchWithTimeout = function(url, opts){
  if (url === '/api/refresh-league') { postCount++; return Promise.resolve({ ok: true, status: 200 }); }
  docFetches++;
  return Promise.resolve({ ok: true, status: 200, text: function(){ return Promise.resolve('<html>fresh</html>'); } });
};
var sectionCalls = 0, rootSwapCalls = 0;
window.brSwapChangedSections = function(html){
  sectionCalls++;
  root.dataset.cacheTs = '2000';
  return { swapped: 3, skipped: 1 };
};
window.brSwapPageRoot = function(){ rootSwapCalls++; return true; };
""" + src + r"""
function wait(ms){ return new Promise(function(resolve){ realSetTimeout(resolve, ms); }); }
(async function(){
  await window.brRefreshLeague();
  // League expired once, ONE document fetched, sections swapped, and
  // the whole-root swap (and any hard reload) never ran.
  if (postCount !== 1 || docFetches !== 1) process.exit(2);
  if (sectionCalls !== 1 || rootSwapCalls !== 0) process.exit(3);
  if (location.reloaded) process.exit(4);
  if (window.brRefreshLeague._busy) process.exit(5);
  var overlay = elements['brRefreshOverlay'];
  if (!overlay || overlay.style.display !== 'flex') process.exit(6);
  await wait(1100);
  if (overlay.style.display !== 'none') process.exit(7);
  process.exit(0);
})().catch(function(e){ console.error(e); process.exit(8); });
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "refresh_section_tier.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=8)
    assert res.returncode == 0, res.stderr or res.stdout


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_targeted_refresh_skips_document_fetch_and_modal_succeeds_node():
    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    src = _freshness_iife()
    harness = r"""
var window = globalThis;
var location = { href: 'http://x/sleeper/2026/abc/standings', pathname: '/sleeper/2026/abc/standings', reload: function(){} };
window.location = location;
var navigator = { serviceWorker: null };
var postCount = 0, getCount = 0;
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
window.brFetchWithTimeout = function(url, opts){
  if (url === '/api/refresh-league') { postCount++; return Promise.resolve({ ok: true, status: 200 }); }
  getCount++;
  return Promise.resolve({ ok: true, status: 200, text: function(){ return Promise.resolve(''); } });
};
var hookCalls = 0;
window.brPageRefresh = function(){ hookCalls++; return Promise.resolve(); };
""" + src + r"""
function wait(ms){ return new Promise(function(resolve){ realSetTimeout(resolve, ms); }); }
(async function(){
  await window.brRefreshLeague();
  // League expired once, the page hook ran, and NO document was fetched.
  if (postCount !== 1 || hookCalls !== 1 || getCount !== 0) process.exit(2);
  if (window.brRefreshLeague._busy) process.exit(3);
  var overlay = elements['brRefreshOverlay'];
  if (!overlay || overlay.style.display !== 'flex') process.exit(4);
  // The success card auto-dismisses.
  await wait(1100);
  if (overlay.style.display !== 'none') process.exit(5);
  process.exit(0);
})().catch(function(e){ console.error(e); process.exit(6); });
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "refresh_targeted.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=8)
    assert res.returncode == 0, res.stderr or res.stdout


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_failed_refresh_keeps_the_modal_up_node():
    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    src = _freshness_iife()
    harness = r"""
var window = globalThis;
var location = { href: 'http://x/sleeper/2026/abc/standings', pathname: '/sleeper/2026/abc/standings', reload: function(){} };
window.location = location;
var navigator = { serviceWorker: null };
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
window.brFetchWithTimeout = function(url, opts){
  if (url === '/api/refresh-league') return Promise.resolve({ ok: true, status: 200 });
  return Promise.reject(new Error('network'));
};
// The targeted hook fails too, so the run falls back and fails globally.
window.brPageRefresh = function(){ return Promise.reject(new Error('hook failed')); };
""" + src + r"""
(async function(){
  await window.brRefreshLeague();
  if (window.brRefreshLeague._busy) process.exit(2);
  if (button.disabled) process.exit(3);
  var overlay = elements['brRefreshOverlay'];
  // Failure state: the modal stays up (Try again / Dismiss) instead of
  // vanishing silently.
  if (!overlay || overlay.style.display !== 'flex') process.exit(4);
  if (sheetTime.textContent.indexOf('Failed') !== 0) process.exit(5);
  process.exit(0);
})().catch(function(e){ console.error(e); process.exit(6); });
"""
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "refresh_modal_fail.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=8)
    assert res.returncode == 0, res.stderr or res.stdout
