"""Up-front filters for Advanced Metrics (approved mock v5, built for real).

Season, week-range quick options, and Min move onto the main bar as a
boxy outlined strip (the real control nodes are relocated by JS, so the
server markup, IDs, and listeners stay single-sourced). Every active
non-default filter gets an editable chip under the strip, and an empty
board caused by filters names them and offers one-tap relax actions
instead of the generic no-data copy.
"""
import os
import re
import shutil
import subprocess

import pytest

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _body(html):
    return html.split("<style>")[0]


def _style(html):
    return html[html.index("<style>"):html.index("</style>")]


def _page_js():
    html = _html()
    return html[html.index("<script>"):html.index("</script>")]


def _extract_fn(js, name):
    m = re.search(r"function %s\([^)]*\) \{.*?\n  \}" % re.escape(name), js, re.S)
    assert m, f"{name} not found in page script"
    return m.group(0)


def _run_node(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def _sheet_span(body):
    start = body.index('id="amFilterSheet"')
    open_idx = body.rindex("<div", 0, start)
    depth = 0
    i = open_idx
    while True:
        nxt_open = body.find("<div", i + 1)
        nxt_close = body.find("</div>", i + 1)
        if nxt_open != -1 and nxt_open < nxt_close:
            depth += 1
            i = nxt_open
        else:
            if depth == 0:
                return open_idx, nxt_close + len("</div>")
            depth -= 1
            i = nxt_close


# ── Markup: strip + chips hosts on the main bar, controls unmoved ──────

def test_strip_and_chips_hosts_live_on_the_main_bar():
    body = _body(_html())
    sheet_start, sheet_end = _sheet_span(body)
    for el_id in ("amFilterStrip", "amStripSeason", "amStripWeeks", "amStripMin",
                  "amActiveWrap", "amActiveChips", "amChipEditor"):
        count = len(re.findall(r'\bid="%s"' % el_id, body))
        assert count == 1, f"#{el_id} must appear exactly once, found {count}"
        idx = body.index('id="%s"' % el_id)
        assert not (sheet_start < idx < sheet_end), \
            f"#{el_id} is a main-bar host and must stay outside the filter sheet"


def test_relocated_controls_stay_in_the_sheet_server_side():
    """The strip is filled by relocating the real nodes at runtime; the
    server markup must keep them in the sheet so there is exactly one
    #amSeasonMulti / #amQuickRanges / #amGamesCtrl in the document."""
    body = _body(_html())
    sheet_start, sheet_end = _sheet_span(body)
    for el_id in ("amSeasonMulti", "amQuickRanges", "amGamesCtrl"):
        idx = body.index('id="%s"' % el_id)
        assert sheet_start < idx < sheet_end, f"#{el_id} must start inside the sheet"
    js = _page_js()
    assert "function amRelocateFilterStrip" in js or "amRelocateFilterStrip" in js
    assert "seasonHost.appendChild(seasonCtrl)" in js
    assert "weekHost.appendChild(qr)" in js
    assert "minHost.appendChild(gamesCtrl)" in js


def test_desktop_grid_opens_strip_and_chips_rows():
    css = _style(_html())
    assert css.count('"strip strip"') == 2  # base + has-compare templates
    assert css.count('"chips chips"') == 2
    assert ".am-cmd-controls .am-filter-strip { grid-area:strip; }" in css
    assert ".am-cmd-controls .am-active-wrap { grid-area:chips; }" in css


# ── Chips: model, render hook, editor wiring ───────────────────────────

def test_chips_functions_and_badge_hook_exist():
    js = _page_js()
    for name in ("amActiveFilterChips", "amRenderActiveChips", "amOpenChipEditor",
                 "amApplyChipEditor", "amResetFilterChip", "amClearAllFilters",
                 "amRenderFilterEmpty"):
        assert f"function {name}(" in js, f"{name} missing from page script"
    badge = _extract_fn(js, "amRefreshFilterBadge")
    assert "amRenderActiveChips();" in badge, \
        "the badge choke point must also refresh the chips row"
    # The editor covers custom metric filters in place, not just built-ins.
    editor = _extract_fn(js, "amOpenChipEditor")
    assert "amEdKey" in editor and "amEdOp" in editor and "amEdVal" in editor


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_active_filter_chips_model():
    js = _page_js()
    fns = "\n".join(_extract_fn(js, n) for n in (
        "_amMinUnitWord", "_amMinChipLabel", "_amWeekChipLabel",
        "_amCustomChipLabel", "amActiveFilterChips"))
    script = (
        "var cfg = { seasons: [2026, 2025] };\n"
        "var state = { metric: 'targets_per_snap', weekRange: '', minVol: '', team: '',\n"
        "  ageMin: '', ageMax: '', rosterOnly: false, comboFilters: [], _seasons: [2026], _range: null };\n"
        "var volLabel = { textContent: 'Min Snaps' };\n"
        "function amSelectedSeasons() { return state._seasons; }\n"
        "function amSeasonLabel(y) { return y.join(' + '); }\n"
        "function resolveWeekRange() { return state._range || { ws: null, we: null }; }\n"
        "function _mLabel(k) { return k === 'target_share' ? 'Target Share' : k; }\n"
        + fns + "\n"
        "function check(name, cond) { if (!cond) { console.error('FAIL ' + name); process.exit(1); } }\n"
        "function labels() { return amActiveFilterChips().map(function(c) { return c.label; }); }\n"
        "check('defaults produce no chips', labels().length === 0);\n"
        "state.minVol = '100';\n"
        "check('min chip carries value and unit', JSON.stringify(labels()) === JSON.stringify(['Min 100 snaps']));\n"
        "state.weekRange = 'custom'; state._range = { ws: 7, we: 9 };\n"
        "check('custom week chip resolves the range', labels().indexOf('Weeks 7-9') !== -1);\n"
        "state.team = 'SF';\n"
        "check('team chip', labels().indexOf('Team SF') !== -1);\n"
        "state.comboFilters = [{ key: 'target_share', op: 'gte', val: '20' }];\n"
        "check('custom filter chip', labels().indexOf('Target Share ≥ 20') !== -1);\n"
        "state._seasons = [2025];\n"
        "check('non-default season chips', amActiveFilterChips().some(function(c) { return c.kind === 'season'; }));\n"
        "state._seasons = [2026];\n"
        "check('default season does not chip', !amActiveFilterChips().some(function(c) { return c.kind === 'season'; }));\n"
    )
    _run_node(script)


def test_editor_chip_state_is_initialized_before_the_init_render():
    """Regression (2026-11-06 live report): amRenderActiveChips() runs
    during page init and, with no active chips (the default load), calls
    amCloseChipEditor(), which writes _amEditorChip. When that variable
    was declared with `let` down in the chip-editor section, the init
    call hit the temporal dead zone: 'Cannot access _amEditorChip before
    initialization', killing the rest of the page script."""
    js = _page_js()
    decl = js.index("let _amEditorChip = null;")
    init_call = js.index("\n  amRenderActiveChips();\n")
    assert decl < init_call, \
        "_amEditorChip must be initialized before the init amRenderActiveChips() call"
    assert js.count("let _amEditorChip") == 1, "exactly one declaration"


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_render_active_chips_with_no_chips_closes_editor_cleanly():
    js = _page_js()
    decl = re.search(r"let _amEditorChip = null;", js).group(0)
    fns = "\n".join(_extract_fn(js, n) for n in (
        "amActiveFilterChips", "amRenderActiveChips", "amCloseChipEditor"))
    script = (
        "var cfg = { seasons: [2026] };\n"
        "var state = { weekRange: '', minVol: '', team: '', ageMin: '', ageMax: '',\n"
        "  rosterOnly: false, comboFilters: [] };\n"
        + decl + "\n"
        "var _wrap = { style: {} };\n"
        "var _row = { innerHTML: 'stale' };\n"
        "var _ed = { hidden: false, innerHTML: 'stale' };\n"
        "var document = { getElementById: function(id) {\n"
        "  if (id === 'amActiveWrap') return _wrap;\n"
        "  if (id === 'amActiveChips') return _row;\n"
        "  if (id === 'amChipEditor') return _ed;\n"
        "  return null; } };\n"
        "function amSelectedSeasons() { return [2026]; }\n"
        "function amSeasonLabel(y) { return y.join(' + '); }\n"
        + fns + "\n"
        "amRenderActiveChips();\n"
        "if (_wrap.style.display !== 'none') { console.error('FAIL wrap not hidden'); process.exit(1); }\n"
        "if (_row.innerHTML !== '') { console.error('FAIL chips row not cleared'); process.exit(1); }\n"
        "if (!_ed.hidden || _ed.innerHTML !== '') { console.error('FAIL editor not closed'); process.exit(1); }\n"
        "if (_amEditorChip !== null) { console.error('FAIL editor chip not reset'); process.exit(1); }\n"
    )
    _run_node(script)


# ── Honest empty state ─────────────────────────────────────────────────

def test_render_branches_on_active_filters_for_empty_state():
    js = _page_js()
    assert "amActiveFilterChips().length" in js
    assert "amRenderFilterEmpty(empty)" in js
    assert "No players pass these filters" in js
    # The genuine no-data copy stays for the no-filters case.
    assert "This metric doesn’t have enough sample to rank for the current filters." in js


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_filter_empty_state_names_constraints_and_relax_actions():
    js = _page_js()
    fns = "\n".join(_extract_fn(js, n) for n in (
        "_amMinUnitWord", "_amMinChipLabel", "_amWeekChipLabel",
        "_amCustomChipLabel", "amActiveFilterChips", "_amEsc",
        "_amLowerMinOption", "amRenderFilterEmpty"))
    script = (
        "var cfg = { seasons: [2026, 2025] };\n"
        "var state = { metric: 'targets_per_snap', weekRange: 'custom', minVol: '200', team: '',\n"
        "  ageMin: '', ageMax: '', rosterOnly: false, comboFilters: [] };\n"
        "var volLabel = { textContent: 'Min Snaps' };\n"
        "var minGamesSel = { options: [{ value: '' }, { value: '100' }, { value: '200' }, { value: '400' }] };\n"
        "function amSelectedSeasons() { return [2026]; }\n"
        "function amSeasonLabel(y) { return y.join(' + '); }\n"
        "function resolveWeekRange() { return { ws: 7, we: 9 }; }\n"
        "function _mLabel(k) { return k; }\n"
        "function _posList() { return ['WR']; }\n"
        + fns + "\n"
        "function check(name, cond) { if (!cond) { console.error('FAIL ' + name); process.exit(1); } }\n"
        "var el = { innerHTML: '' };\n"
        "amRenderFilterEmpty(el);\n"
        "check('honest title', el.innerHTML.indexOf('No players pass these filters') !== -1);\n"
        "check('names the min constraint', el.innerHTML.indexOf('Min 200 snaps') !== -1);\n"
        "check('names the week constraint', el.innerHTML.indexOf('Weeks 7-9') !== -1);\n"
        "check('names the position', el.innerHTML.indexOf('WR') !== -1);\n"
        "check('min relax steps one option down', el.innerHTML.indexOf('data-relax=\"min\" data-lower=\"100\"') !== -1);\n"
        "check('min relax label', el.innerHTML.indexOf('Lower Min to 100 snaps') !== -1);\n"
        "check('weeks relax offered', el.innerHTML.indexOf('data-relax=\"weeks\"') !== -1);\n"
        "state.minVol = '100';\n"
        "amRenderFilterEmpty(el);\n"
        "check('smallest min relax clears instead', el.innerHTML.indexOf('Clear Min filter') !== -1);\n"
    )
    _run_node(script)


# ── Styling contract + changelog ───────────────────────────────────────

def test_strip_and_chip_css_is_boxy_outlined_and_nowrap():
    css = open(os.path.join(os.path.dirname(__file__), "..", "static", "dashboard.css"),
               encoding="utf-8").read()
    marker = css.index("MERGED: icons.css")
    start = css.index("Advanced Metrics: up-front filter strip")
    assert start < marker, "new AM filter CSS must land before the icons merge marker"
    block = css[start:marker]
    assert ".am-filter-strip {" in block
    assert ".am-active-chip {" in block
    assert "flex-wrap: nowrap" in block
    assert "border-radius: 6px" in block
    assert "999px" not in block, "pill radius is banned site-wide; the strip is boxy"


def test_changelog_entry_present_by_content():
    from dashboard_services.changelog import CHANGELOG
    hits = [e for e in CHANGELOG
            if "Advanced Metrics" in e.get("text", "") and "up front" in e.get("text", "")]
    assert hits, "a changelog entry for the up-front filters is missing"
    assert hits[0].get("link") == "/advanced-metrics"
