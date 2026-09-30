"""Focused browser and source contracts for the Lineup league controls."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _browser():
    playwright = pytest.importorskip("playwright.sync_api")
    pw = playwright.sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception:
        pw.stop()
        pytest.skip("Chromium is not installed")
    return pw, browser


def _leaderboard_markup():
    stats = "".join(
        f'<span class="opt-team-stat"><span class="opt-team-label">{label}</span>'
        f'<span class="opt-team-value{extra}">{value}</span></span>'
        for label, value, extra in (("Actual", "123.4", ""), ("Optimal", "125.0", ""),
                                   ("Missed", "1.6", " is-missed"))
    )
    return f'''<div id="optimalLineupContent"><nav class="opt-nav">
      <div class="opt-tab-group"><a class="opt-tab active">My Team</a><a class="opt-tab">League</a></div>
      <div class="opt-tab-group"><a class="opt-tab active">Weekly</a><a class="opt-tab">Season</a></div>
      <select class="opt-week-select" aria-label="Completed week"><option>Week 1</option><option>Week 2</option></select>
    </nav><div class="opt-leaderboard"><div class="opt-leaderboard-head">Efficiency leaderboard</div>
      <div class="opt-leaderboard-columns"><span>Rank</span><span>Team</span><span>Efficiency</span><span>Actual</span><span>Optimal</span><span>Missed</span></div>
      <details class="card opt-team is-viewer"><summary><span class="opt-rank opt-rank-1">#1</span>
        <span class="opt-team-identity"><strong class="opt-team-name">A deliberately very long fantasy football team name that wraps</strong>
        <span class="opt-team-sub">123.4 of 125.0 pts</span></span>
        <span class="opt-team-hero"><span class="opt-team-label">Efficiency</span><span class="opt-team-effnum">98.7%</span>
        <span class="opt-team-bar" aria-hidden="true"><span style="width:98.7%"></span></span></span>
        <span class="opt-team-stats">{stats}</span>
        <span class="opt-team-weeks"><span class="opt-team-label">Weeks counted</span> <span class="opt-team-value">1 of 1 weeks</span></span>
      </summary></details></div></div>'''


@pytest.mark.parametrize("panel_width", [360, 480, 600, 900])
def test_league_leaderboard_uses_panel_width_without_clipping(panel_width):
    pw, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        css = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
        page.set_content(f'<style>{css}</style><main style="width:{panel_width}px">{_leaderboard_markup()}</main>')
        page.add_script_tag(path=str(ROOT / "static/custom_selects.js"))
        result = page.eval_on_selector("main", """e => ({client:e.clientWidth, scroll:e.scrollWidth,
          labels:[...e.querySelectorAll('.opt-team-label')].map(x => x.textContent),
          columns:getComputedStyle(e.querySelector('.opt-leaderboard-columns')).display})""")
        assert result["scroll"] <= result["client"] + 1
        assert {"Efficiency", "Actual", "Optimal", "Missed", "Weeks counted"} <= set(result["labels"])
        assert result["columns"] == ("grid" if panel_width >= 650 else "none")
    finally:
        browser.close()
        pw.stop()


def test_week_picker_reinitializes_idempotently_and_opens_above_content():
    pw, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 480, "height": 400})
        css = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
        page.set_content(f'<style>{css}</style>{_leaderboard_markup()}')
        page.add_script_tag(path=str(ROOT / "static/custom_selects.js"))
        page.evaluate("initCustomSelects(document.querySelector('#optimalLineupContent'))")
        assert page.locator(".opt-nav > .csd-wrap").count() == 1
        page.click(".csd-trigger")
        state = page.eval_on_selector(".csd-list", "e => ({position:getComputedStyle(e).position, visible:getComputedStyle(e).display, z:getComputedStyle(e).zIndex})")
        assert state["position"] == "fixed"
        assert state["visible"] == "block"
        assert int(state["z"]) > 0
    finally:
        browser.close()
        pw.stop()


def test_optimal_fragment_runs_shared_dropdown_enhancer():
    source = (ROOT / "static/app.js").read_text(encoding="utf-8")
    replacement = source.index("host.innerHTML = data.html")
    enhancer = source.index("window.initCustomSelects(host)", replacement)
    history = source.index("history.pushState", replacement)
    assert replacement < enhancer < history


def test_card_tabs_persist_weekly_left_tab_in_url():
    """The weekly-hub Lineup tab must survive reloads: clicking a tab in the
    weekly hub's left card writes ?tab=<data-tab> via history.replaceState (no
    navigation), so the existing ?tab= activation restores it on reload."""
    source = (ROOT / "static/app.js").read_text(encoding="utf-8")
    start = source.index("function initCardTabs")
    end = source.index("function initPlayoffOdds", start)
    block = source[start:end]
    # Scoped to the weekly hub's left tabs card only.
    assert block.count("weeklyLeftTabs") == 1
    assert 'card.id === "weeklyLeftTabs"' in block
    assert "history.replaceState" in block
    assert 'searchParams.set("tab"' in block
    # replaceState (not pushState/location assignment): must never navigate or
    # reload. Reading location.href to build the new URL is fine.
    assert "pushState" not in block
    assert "location.assign" not in block and "location.replace" not in block
    assert "location.href =" not in block


_NODE_TAB_HARNESS = r"""'use strict';
// Minimal fake DOM sufficient to execute the weekly-hub tab + reflow script.
const changeHandlers = [];
let MATCHES = false;

function matchSel(e, sel) {
  let rest = sel, id = null, classes = [], tabVal;
  let m = rest.match(/^#([\w-]+)/);
  if (m) { id = m[1]; rest = rest.slice(m[0].length); }
  m = rest.match(/^\.([\w-]+(?:\.[\w-]+)*)/);
  if (m) { classes = m[1].split('.'); rest = rest.slice(m[0].length); }
  m = rest.match(/^\[data-tab="([^"]*)"\]/);
  if (m) { tabVal = m[1]; rest = rest.slice(m[0].length); }
  if (rest !== '') return false;
  if (id && e.attrs.id !== id) return false;
  for (const c of classes) if (!e._cls.has(c)) return false;
  if (tabVal !== undefined && e.attrs['data-tab'] !== tabVal) return false;
  return true;
}
function queryAll(root, sel) {
  const out = [];
  (function walk(n) { if (matchSel(n, sel)) out.push(n); for (const c of n.children) walk(c); })(root);
  return out;
}
function makeEl(tag, attrs) {
  const e = {
    tag, attrs: attrs || {}, children: [], parent: null, style: {},
    _cls: new Set(), _closestMap: {},
    getAttribute(n) { return Object.prototype.hasOwnProperty.call(this.attrs, n) ? this.attrs[n] : null; },
    setAttribute(n, v) { this.attrs[n] = String(v); },
    appendChild(c) { if (c.parent) c.parent.removeChild(c); c.parent = this; this.children.push(c); return c; },
    removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parent = null; return c; },
    insertBefore(c, ref) { if (c.parent) c.parent.removeChild(c); c.parent = this; const i = ref ? this.children.indexOf(ref) : -1; if (i >= 0) this.children.splice(i, 0, c); else this.children.push(c); return c; },
    closest(sel) { return this._closestMap[sel] || null; },
    querySelector(sel) { const r = queryAll(this, sel); return r.length ? r[0] : null; },
    querySelectorAll(sel) { return queryAll(this, sel); },
  };
  e.classList = {
    add: (c) => e._cls.add(c),
    remove: (c) => e._cls.delete(c),
    contains: (c) => e._cls.has(c),
    toggle: (c, force) => { const on = force === undefined ? !e._cls.has(c) : !!force; if (on) e._cls.add(c); else e._cls.delete(c); return on; },
  };
  return e;
}

const mqObj = {
  get matches() { return MATCHES; },
  addEventListener(t, fn) { changeHandlers.push(fn); },
  addListener(fn) { changeHandlers.push(fn); },
};

function buildDom() {
  const pageLayout = makeEl('div'); pageLayout._cls.add('page-layout');
  const main = makeEl('main'); main._cls.add('page-main');
  main._closestMap = { '.page-layout': pageLayout, '.page-main': main };
  const tabs = makeEl('div', { id: 'weeklyLeftTabs' }); tabs._cls.add('card-tabs');
  tabs._closestMap = { '.page-layout': pageLayout, '.page-main': main };
  const bar = makeEl('div'); bar._cls.add('tab-bar');
  const panels = makeEl('div'); panels._cls.add('tab-panels');
  for (const n of ['matchups', 'scorers', 'scout', 'optimal']) {
    const b = makeEl('button', { 'data-tab': n }); b._cls.add('tab-btn');
    const p = makeEl('div', { 'data-tab': n }); p._cls.add('tab-panel');
    bar.appendChild(b); panels.appendChild(p);
  }
  bar.children[0]._cls.add('active'); panels.children[0]._cls.add('active');
  tabs.appendChild(bar); tabs.appendChild(panels);
  main.appendChild(tabs); pageLayout.appendChild(main);
  const shell = makeEl('div'); shell._cls.add('matchups-shell');
  const mc = makeEl('div', { id: 'weeklyMatchupsContainer' });
  mc._closestMap = { '.matchups-shell': shell };
  const band = makeEl('div'); band._cls.add('week-leaders-band');
  return { tabs, mc, band };
}

const SCRIPT = __SCRIPT_JSON__;

function activeTab(dom) {
  const b = dom.tabs.querySelector('.tab-btn.active');
  return b ? b.getAttribute('data-tab') : null;
}

function runScenario(search, matches, afterChange) {
  MATCHES = matches;
  changeHandlers.length = 0;
  const dom = buildDom();
  const byId = { weeklyLeftTabs: dom.tabs, weeklyMatchupsContainer: dom.mc };
  const document = {
    getElementById: (id) => byId[id] || null,
    createElement: (t) => makeEl(t),
    querySelector: (s) => (s === '.week-leaders-band' ? dom.band : null),
  };
  const window = {
    location: { search, href: 'https://www.brfantasyfootball.com/x/2026/1/weekly' + search },
    matchMedia: () => mqObj,
  };
  eval(SCRIPT);
  if (afterChange !== undefined && afterChange !== null) {
    MATCHES = afterChange;
    changeHandlers.forEach((fn) => fn());
  }
  return activeTab(dom);
}

const results = {
  desktop_optimal: runScenario('?tab=optimal', true),
  mobile_optimal: runScenario('?tab=optimal', false),
  desktop_default: runScenario('', true),
  mobile_default: runScenario('', false),
  desktop_bogus: runScenario('?tab=bogus', true),
  mobile_to_desktop_change: runScenario('?tab=optimal', false, true),
  desktop_to_mobile_change: runScenario('?tab=scout', true, false),
};
console.log(JSON.stringify(results));
"""


def _weekly_hub_tab_script():
    """Extract the inline tab-activation + desktop-reflow script from
    weekly_hub_page.py, undoing the Python f-string brace doubling."""
    src = (ROOT / "dashboard_services/pages/weekly_hub_page.py").read_text(encoding="utf-8")
    start = src.index("// Activate a weekly left-tab by name")
    end = src.index("</script>", start)
    script = src[start:end].replace("{{", "{").replace("}}", "}")
    assert "{show_matchups_js}" in script, "expected the single Python interpolation in the tab script"
    return script.replace("{show_matchups_js}", "true")


def test_weekly_hub_tab_survives_reflow_on_reload():
    """Behavioral regression test for the Lineup-tab reset bug: ?tab=optimal
    must survive the weekly-hub desktop-reflow apply() on a full-page reload
    (mobile and desktop widths), and crossing the 1100px breakpoint must
    preserve the currently active tab instead of resetting to the default.
    Runs the real inline script from weekly_hub_page.py in node against a
    minimal fake DOM; skipped where node is unavailable."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = _weekly_hub_tab_script()
    program = _NODE_TAB_HARNESS.replace("__SCRIPT_JSON__", json.dumps(script))
    proc = subprocess.run([node], input=program, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr[-3000:]}"
    assert json.loads(proc.stdout) == {
        "desktop_optimal": "optimal",
        "mobile_optimal": "optimal",
        "desktop_default": "scorers",
        "mobile_default": "matchups",
        "desktop_bogus": "scorers",
        "mobile_to_desktop_change": "optimal",
        "desktop_to_mobile_change": "scout",
    }


def _opt_player(name, pos, score):
    return (f'<span class="opt-player"><span class="opt-pos opt-pos-{pos.lower()}">{pos}</span>'
            f'<button type="button" class="opt-player-name player-clickable" data-player-id="1" '
            f'data-player-name="{name}">{name}</button>'
            f'<strong class="opt-score">{score:.1f}</strong></span>')


def _comparison_markup():
    """Mirror of optimal_page._comparison output (same classes/nesting)."""
    def row(slot, actual, optimal, gain, changed=False):
        return (f'<div class="opt-lineup-row{" is-changed" if changed else ""}">'
                f'<div class="opt-slot">{slot}</div>'
                f'<div class="opt-side"><span class="opt-side-label">Actual</span>{actual}</div>'
                f'<div class="opt-arrow" aria-hidden="true">→</div>'
                f'<div class="opt-side"><span class="opt-side-label">Optimal after results</span>{optimal}</div>'
                f'<div class="opt-gain">{gain}</div></div>')
    return ('<div class="opt-comparison">'
            + row("QB", _opt_player("Brock Purdy", "QB", 31.3), _opt_player("Brock Purdy", "QB", 31.3), "—")
            + row("RB", _opt_player("Jeremiyah Love", "RB", 21.9), _opt_player("Jeremiyah Love", "RB", 21.9), "—")
            + row("RB", _opt_player("Christian McCaffrey", "RB", 12.4),
                  _opt_player("Breece Hall", "RB", 28.4), "+16.0", changed=True)
            + row("WR", _opt_player("Ja'Marr Chase", "WR", 9.8),
                  _opt_player("Justin Jefferson", "WR", 24.1), "+14.3", changed=True)
            + '</div>')


def _browser_opt():
    playwright = pytest.importorskip("playwright.sync_api")
    pw = playwright.sync_playwright().start()
    chrome = Path("/opt/meta-chromium/chrome")
    try:
        if chrome.exists():
            browser = pw.chromium.launch(headless=True, executable_path=str(chrome))
        else:
            browser = pw.chromium.launch(headless=True)
    except playwright.Error:
        pw.stop()
        pytest.skip("Chromium is not installed")
    return pw, browser


@pytest.mark.parametrize("panel_width", [360, 480, 600, 900])
def test_optimal_comparison_names_never_collapse(panel_width):
    """Regression: in a narrow panel the two sides of a lineup row used to be
    squeezed side-by-side until each player name was ~1 character wide and
    wrapped letter-per-line (rows hundreds of px tall). The comparison must
    stack by PANEL width (container query), keep names readable, keep the
    arrow and gain visible, and never overflow horizontally. At >=650px the
    single-row desktop layout is unchanged."""
    pw, browser = _browser_opt()
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        css = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
        page.set_content(f'<style>{css}</style><main style="width:{panel_width}px">{_comparison_markup()}</main>')
        result = page.eval_on_selector("main", """e => {
          const rows = [...e.querySelectorAll('.opt-lineup-row')].map(r => {
            const sides = r.querySelectorAll('.opt-side');
            return {stacked: sides[1].offsetTop > sides[0].offsetTop + 4,
                    arrow: getComputedStyle(r.querySelector('.opt-arrow')).display};
          });
          const names = [...e.querySelectorAll('.opt-player-name')].map(n => ({w: n.offsetWidth, h: n.offsetHeight}));
          return {client: e.clientWidth, scroll: e.scrollWidth, rows, names};
        }""")
        assert result["scroll"] <= result["client"] + 1
        for name in result["names"]:
            assert name["w"] >= 48, f"name crushed to {name['w']}px wide at panel {panel_width}"
            assert name["h"] <= 44, f"name wrapped to {name['h']}px tall at panel {panel_width}"
        for r in result["rows"]:
            assert r["arrow"] != "none"
            assert r["stacked"] == (panel_width < 650)
    finally:
        browser.close()
        pw.stop()


def _css_rule(css, selector):
    start = css.index(selector)
    end = css.index("}", start)
    return css[start:end]


def test_optimal_comparison_stacks_by_container_width():
    """Source contract: the narrow layout is keyed to the comparison panel's
    own width (container query, like the leaderboard cards), not only to the
    viewport, and the name can no longer be sized below its longest word."""
    css = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
    assert "container-type:inline-size" in _css_rule(css, ".opt-comparison {")
    start = css.index("@container (max-width:649px)")
    block = css[start:css.index("\n}", start)]
    assert ".opt-lineup-row" in block and "54px minmax(0,1fr) 48px" in block
    assert ".opt-slot + .opt-side" in block and ".opt-arrow + .opt-side" in block
    assert "display:none" not in block  # arrow stays visible while stacked
    name_rule = _css_rule(css, ".opt-player-name {")
    assert "overflow-wrap:anywhere" not in name_rule
    assert "overflow-wrap:break-word" in name_rule


def test_optimal_viewport_media_block_no_longer_restacks_rows():
    """The old @media (max-width:720px) row overrides (which hid the arrow and
    overlapped the actual score with the gain) are superseded by the
    container query; the summary/nav rules in that block stay."""
    css = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
    anchor = css.index(".opt-nav { gap:7px")
    start = css.rindex("@media (max-width:720px)", 0, anchor)
    block = css[start:css.index("\n}", anchor)]
    assert ".opt-summary" in block
    assert ".opt-lineup-row" not in block
    assert ".opt-arrow" not in block
