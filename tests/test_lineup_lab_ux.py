"""Lineup Lab UX polish contracts.

Covers the 2026-09-29 UX pass on the Lab (Start/Sit tab):
- mobile full-width CSS (no wasted side gutters, reduced bench-sheet indent)
- no mid-interaction repaint (open sheets + scroll survive wvRenderLab)
- change highlighting after Optimize / manual swap / Chase upside (flash, summary, scroll)
- Chase upside applies ceiling-maximizing swaps to the lineup (like Optimize),
  with a no-gain note when the lineup is already top-ceiling
"""
import os
import re
import subprocess
import tempfile

import pytest

from dashboard_services.pages.waivers_page import build_waivers_body


@pytest.fixture(scope="module")
def page():
    return build_waivers_body("sleeper", 2026, "12345", {})


@pytest.fixture(scope="module")
def lab_js(page):
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    s = [x for x in scripts if "function wvLoad(" in x]
    assert s, "waivers inline script not found"
    script = s[0]
    start = script.index("// ── Lineup Lab (Start/Sit tab) ──")
    end = script.index("if (!window.__brctx)", start)
    return script[start:end]


def _media_blocks(page, query):
    """Return the bodies of @media blocks whose prelude contains query."""
    blocks = []
    for m in re.finditer(r"@media\s*([^{]+)\{", page):
        if query not in m.group(1):
            continue
        depth, i = 1, m.end()
        while i < len(page) and depth:
            if page[i] == "{":
                depth += 1
            elif page[i] == "}":
                depth -= 1
            i += 1
        blocks.append(page[m.end():i - 1])
    return blocks


# ---------------------------------------------------------------------------
# CSS contracts
# ---------------------------------------------------------------------------

def test_mobile_lab_goes_full_width(page):
    blocks = _media_blocks(page, "max-width: 768px")
    assert blocks, "no (max-width: 768px) media block found"
    assert any(".wv-page { padding: 12px; }" in b for b in blocks), \
        "page padding must drop to 12px on mobile"
    assert any(".wv-lab-sheet { padding: 0 0 12px 12px; }" in b for b in blocks), \
        "bench sheet must lose its 54px desktop indent on mobile"


def test_change_flash_css(page):
    assert "@keyframes wv-lab-changed-flash" in page
    assert ".wv-lab-slot.wv-lab-changed" in page
    assert ".wv-lab-changes" in page


def test_change_flash_disabled_for_reduced_motion(page):
    blocks = _media_blocks(page, "prefers-reduced-motion")
    assert blocks, "no prefers-reduced-motion block found"
    assert any(".wv-lab-changed" in b and "animation: none" in b for b in blocks), \
        "change flash must be disabled under prefers-reduced-motion"


# ---------------------------------------------------------------------------
# JS contracts (node harness over the real Lab script section)
# ---------------------------------------------------------------------------

_NODE_PRELUDE = r"""
var __scrollToCalls = [];
var __scrolledIntoView = null;
var __fakeBody = { innerHTML: '' };
function __makeEl() {
  var cls = {};
  return {
    classList: {
      toggle: function(c) { cls[c] = !cls[c]; return !!cls[c]; },
      contains: function(c) { return !!cls[c]; }
    },
    scrollIntoView: function(opts) { __scrolledIntoView = opts || true; }
  };
}
var __els = {};
var window = {
  pageYOffset: 321,
  scrollTo: function(x, y) { __scrollToCalls.push([x, y]); },
  __brctx: {}
};
var document = {
  getElementById: function(id) { return id === 'wvLabBody' ? __fakeBody : null; },
  querySelector: function(sel) {
    if (!__els[sel]) __els[sel] = __makeEl();
    return __els[sel];
  },
  documentElement: { scrollTop: 0 }
};
"""

_LAB_SETUP = r"""
wvLabData = { opponent: { name: 'Test Opp' }, corr: {} };
wvLabResult = { winPct: 0.62, p10: 100, p90: 140, median: 120 };
wvLabOppStats = null;
function __mkEntry(pid, name, pos, proj, floor, ceiling, bench) {
  return { player_id: pid, slot: pos, name: name, pos: pos, proj: proj,
           floor: floor, ceiling: ceiling, matchup: 'vs DEN', tags: [],
           profile: {}, eligible: [pos], usage_stat: null, usage_avg: null,
           bench: bench || [] };
}
wvLabLineup = [
  __mkEntry('p1', 'Starter A', 'QB', 18, 10, 28, [
    __mkEntry('p2', 'Bench B', 'QB', 17, 5, 31, []),
    __mkEntry('p3', 'Bench C', 'QB', 12, 4, 20, [])
  ]),
  __mkEntry('p4', 'Starter D', 'RB', 15, 8, 24, [])
];
"""


def _run_node(lab_js, body):
    harness = _NODE_PRELUDE + lab_js + "\n" + body
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        out = subprocess.run(["node", path], capture_output=True, text=True,
                             timeout=120)
    finally:
        os.unlink(path)
    assert out.returncode == 0, "node harness failed: %s" % out.stderr[-2000:]
    vals = {}
    for m in re.finditer(r"^([A-Z_]+)=(.*)$", out.stdout, re.M):
        vals[m.group(1)] = m.group(2).strip()
    return vals


def test_toggle_open_state_survives_rerender(lab_js):
    vals = _run_node(lab_js, _LAB_SETUP + r"""
wvLabSwapDelta = function(si, b) { return 0.02; };  // render-only: skip sims
wvLabToggleSlot(0);
wvRenderLab();
var html = __fakeBody.innerHTML;
console.log('OPEN=' + (html.indexOf('wv-lab-slot open') !== -1));
console.log('SCROLL=' + JSON.stringify(__scrollToCalls[__scrollToCalls.length - 1]));
wvLabToggleSlot(0);  // collapse again
wvRenderLab();
console.log('CLOSED=' + (__fakeBody.innerHTML.indexOf('wv-lab-slot open') === -1));
""")
    assert vals["OPEN"] == "true", "open slot must keep its open class across wvRenderLab"
    assert vals["SCROLL"] == "[0,321]", "scroll position must be restored after render"
    assert vals["CLOSED"] == "true", "collapsed slot must not render open"


def test_toggle_is_pure_class_toggle(lab_js):
    # Toggling a sheet must not rebuild the body at all.
    vals = _run_node(lab_js, _LAB_SETUP + r"""
__fakeBody.innerHTML = 'SENTINEL';
wvLabToggleSlot(1);
console.log('UNTOUCHED=' + (__fakeBody.innerHTML === 'SENTINEL'));
console.log('TRACKED=' + (wvLabOpenSlots[1] === true));
""")
    assert vals["UNTOUCHED"] == "true", "wvLabToggleSlot must not re-render"
    assert vals["TRACKED"] == "true", "wvLabToggleSlot must record the open slot"


def test_chase_upside_applies_max_ceiling_swaps(lab_js):
    vals = _run_node(lab_js, _LAB_SETUP + r"""
wvLabSwapDelta = function(si, b) { return 0.02; };  // render-only: skip sims
wvLabEvaluate = function(lineup) { return { winPct: 0.70, median: 125, p10: 105, p90: 145 }; };
// Give slot 1 a bench option with a higher ceiling than Starter D (24).
wvLabLineup[1].bench = [__mkEntry('p5', 'Bench E', 'RB', 14, 6, 30, [])];
wvLabChaseUpside();
var html = __fakeBody.innerHTML;
console.log('CHANGES=' + JSON.stringify(wvLabLastChanges));
console.log('NAMES=' + wvLabLineup[0].name + '/' + wvLabLineup[1].name);
console.log('ACTION=' + wvLabLastAction);
console.log('SUMMARY=' + (html.indexOf('Chased upside: 2 swaps:') !== -1
  && html.indexOf('Bench B in for Starter A') !== -1
  && html.indexOf('Bench E in for Starter D') !== -1));
console.log('FLASH=' + (html.split('wv-lab-changed').length - 1));
console.log('OPEN=' + (html.split('wv-lab-slot open').length - 1));
console.log('SCROLLINTOVIEW=' + JSON.stringify(__scrolledIntoView));
console.log('BTN=' + (html.indexOf('disabled title="Already your highest-ceiling lineup"') !== -1));
console.log('SHEET=' + (html.indexOf('bench \u00b7 win% change') !== -1
  && html.indexOf('sorted by ceiling') === -1));
console.log('NODASH=' + (html.indexOf('\u2014') === -1));
""")
    assert vals["CHANGES"] == '[{"si":0,"out":"Starter A","inn":"Bench B"},' \
        '{"si":1,"out":"Starter D","inn":"Bench E"}]', \
        "chase upside must swap in the top-ceiling bench option per slot"
    assert vals["NAMES"] == "Bench B/Bench E", "swaps must actually apply"
    assert vals["ACTION"] == "upside"
    assert vals["SUMMARY"] == "true", "chase upside must render a labeled swap summary"
    assert vals["FLASH"] == "2", "each changed row must flash"
    assert vals["OPEN"] == "2", "changed slots must stay expanded"
    assert vals["SCROLLINTOVIEW"] == '{"block":"nearest","behavior":"smooth"}', \
        "chase upside must smoothly scroll the first changed row into view"
    assert vals["BTN"] == "true", "after a full chase the Chase button must park disabled"
    assert vals["SHEET"] == "true", "bench sheets must stay in win%-delta form"
    assert vals["NODASH"] == "true", "no em dashes in Lab UI copy"


def test_chase_upside_no_gain_shows_note(lab_js):
    vals = _run_node(lab_js, _LAB_SETUP + r"""
wvLabSwapDelta = function(si, b) { return 0.02; };
wvLabEvaluate = function(lineup) { return { winPct: 0.62, median: 120, p10: 100, p90: 140 }; };
// Starter A (ceil 28) already beats every bench ceiling (31 -> 20).
wvLabLineup[0].bench = [__mkEntry('p3', 'Bench C', 'QB', 12, 4, 20, [])];
wvLabChaseUpside();
var html = __fakeBody.innerHTML;
console.log('CHANGES=' + JSON.stringify(wvLabLastChanges));
console.log('NAMES=' + wvLabLineup[0].name);
console.log('NOTE=' + (html.indexOf('Already your highest-ceiling lineup.') !== -1));
console.log('BTN=' + (html.indexOf('disabled title="Already your highest-ceiling lineup"') !== -1));
console.log('FLASH=' + (html.indexOf('wv-lab-changed') !== -1));
console.log('NODASH=' + (html.indexOf('\u2014') === -1));
// The note must clear on the next action instead of going stale.
wvLabLineup[0].bench = [__mkEntry('p2', 'Bench B', 'QB', 17, 5, 31, [])];
wvLabSwap(0, 0);
html = __fakeBody.innerHTML;
console.log('CLEARED=' + (html.indexOf('Already your highest-ceiling lineup.') === -1
  && html.indexOf('1 swap:') !== -1));
""")
    assert vals["CHANGES"] == "[]", "no swaps when the starter already has the top ceiling"
    assert vals["NAMES"] == "Starter A", "lineup must be untouched"
    assert vals["NOTE"] == "true", "no-gain chase upside must say so"
    assert vals["BTN"] == "true", "no-gain chase upside must park the Chase button disabled"
    assert vals["FLASH"] == "false", "nothing changed, so nothing flashes"
    assert vals["NODASH"] == "true", "no em dashes in Lab UI copy"
    assert vals["CLEARED"] == "true", "the note must clear on the next action"


def test_chase_upside_threshold(lab_js):
    vals = _run_node(lab_js, _LAB_SETUP + r"""
wvLabSwapDelta = function(si, b) { return 0.02; };
wvLabEvaluate = function(lineup) { return { winPct: 0.62, median: 120, p10: 100, p90: 140 }; };
// Starter A ceiling is 28. A 0.04 gain must not swap; a 0.06 gain must.
wvLabLineup[0].bench = [__mkEntry('p9', 'Edge', 'QB', 17, 5, 28.04, [])];
wvLabChaseUpside();
console.log('TIE=' + wvLabLineup[0].name);
wvLabLineup[0].bench = [__mkEntry('p9', 'Edge', 'QB', 17, 5, 28.06, [])];
wvLabChaseUpside();
console.log('GAIN=' + wvLabLineup[0].name);
""")
    assert vals["TIE"] == "Starter A", "a 0.04 ceiling gain must not swap"
    assert vals["GAIN"] == "Edge", "a 0.06 ceiling gain must swap"


def test_manual_swap_highlights_and_stays_open(lab_js):
    vals = _run_node(lab_js, _LAB_SETUP + r"""
wvLabSwapDelta = function(si, b) { return 0.05; };
wvLabEvaluate = function(lineup) { return { winPct: 0.70, median: 125, p10: 105, p90: 145 }; };
wvLabSwap(0, 0);
var html = __fakeBody.innerHTML;
console.log('CHANGES=' + JSON.stringify(wvLabLastChanges));
console.log('FLASH=' + (html.indexOf('wv-lab-changed') !== -1));
console.log('OPEN=' + (html.indexOf('wv-lab-slot open') !== -1));
console.log('SUMMARY=' + (html.indexOf('1 swap:') !== -1 && html.indexOf('Bench B in for Starter A') !== -1));
console.log('NODASH=' + (html.indexOf('\u2014') === -1));
""")
    assert vals["CHANGES"] == '[{"si":0,"out":"Starter A","inn":"Bench B"}]'
    assert vals["FLASH"] == "true", "swapped row must flash"
    assert vals["OPEN"] == "true", "swapped slot must stay expanded"
    assert vals["SUMMARY"] == "true", "swap summary line must render"
    assert vals["NODASH"] == "true", "no em dashes in Lab UI copy"


def test_optimize_reports_changes_with_real_engine(lab_js):
    vals = _run_node(lab_js, r"""
wvLabData = { opponent: { name: 'Opp' }, corr: {} };
wvLabOppStats = null;
function __prof(mean) { return { mean: mean, std: 3, skew_alpha: 2, dud_risk: 0 }; }
function __e(pid, name, pos, mean, bench) {
  return { player_id: pid, slot: pos, name: name, pos: pos, proj: mean,
           floor: mean - 4, ceiling: mean + 10, matchup: '', tags: [],
           profile: __prof(mean), bench: bench || [] };
}
wvLabLineup = [ __e('p1', 'Starter A', 'QB', 10, [ __e('p2', 'Bench B', 'QB', 20, []) ]) ];
wvLabBase = wvLabBuildBase(['p1', 'p2'], WV_LAB_SIMS, 42);
var __r = wvLabRng(99);
wvLabOppDraws = new Float64Array(WV_LAB_SIMS);
for (var s = 0; s < WV_LAB_SIMS; s++) wvLabOppDraws[s] = 12 + (__r() * 2 - 1) * 3;
wvLabResult = wvLabEvaluate(wvLabLineup);
wvLabOptimize();
var html = __fakeBody.innerHTML;
console.log('CHANGES=' + JSON.stringify(wvLabLastChanges));
console.log('SWAPPED=' + (wvLabLineup[0].name === 'Bench B'));
console.log('SUMMARY=' + (html.indexOf('1 swap:') !== -1 && html.indexOf('Bench B in for Starter A') !== -1));
console.log('FLASH=' + (html.indexOf('wv-lab-changed') !== -1));
console.log('OPTBTN=' + (html.indexOf('disabled title="Already optimized"') !== -1
  && html.indexOf('Lineup optimized') !== -1));
console.log('SCROLLINTOVIEW=' + JSON.stringify(__scrolledIntoView));
console.log('SCROLLKEPT=' + JSON.stringify(__scrollToCalls[__scrollToCalls.length - 1]));
""")
    assert vals["CHANGES"] == '[{"si":0,"out":"Starter A","inn":"Bench B"}]', \
        "optimize must report the applied swap"
    assert vals["SWAPPED"] == "true", "optimize must actually apply the swap"
    assert vals["SUMMARY"] == "true", "optimize must render a swap summary"
    assert vals["FLASH"] == "true", "changed row must flash"
    assert vals["OPTBTN"] == "true", \
        "after optimizing, the Optimize button must park disabled as optimized"
    assert vals["SCROLLINTOVIEW"] == '{"block":"nearest","behavior":"smooth"}', \
        "optimize must smoothly scroll the first changed row into view"
    assert vals["SCROLLKEPT"] == "[0,321]", "render must still preserve scroll first"


def test_pos_badges_have_color_css(page):
    for pos in ("qb", "rb", "wr", "te", "flex", "k", "def"):
        assert ".wv-lab-pos.%s" % pos in page, \
            "position badge color missing for %s" % pos.upper()


def test_slot_row_renders_colored_pos_badge(lab_js):
    vals = _run_node(lab_js, _LAB_SETUP + r"""
wvLabSwapDelta = function(si, b) { return 0; };
wvLabLineup = [
  __mkEntry('p1', 'Starter A', 'QB', 18, 10, 28, []),
  __mkEntry('p2', 'Starter B', 'RB', 15, 8, 24, []),
  __mkEntry('p3', 'Starter C', 'FLEX', 14, 7, 22, []),
  __mkEntry('p4', 'Starter D', 'WRRB_FLEX', 13, 6, 20, [])
];
var html = wvLabRenderSlots();
console.log('QB=' + (html.indexOf('wv-lab-pos qb') !== -1));
console.log('RB=' + (html.indexOf('wv-lab-pos rb') !== -1));
console.log('FLEX=' + (html.indexOf('wv-lab-pos flex') !== -1));
console.log('UNKNOWN=' + (html.indexOf('wv-lab-pos wrrb_flex') === -1 && html.indexOf('wv-lab-pos">') !== -1));
""")
    assert vals["QB"] == "true", "QB row must carry the qb color class"
    assert vals["RB"] == "true", "RB row must carry the rb color class"
    assert vals["FLEX"] == "true", "FLEX row must carry the flex color class"
    assert vals["UNKNOWN"] == "true", "unknown slot must not get a color class"


def test_range_line_is_horizontal(page):
    assert ".wv-lab-rangeline" in page, "range must be one horizontal line"
    assert ".wv-lab-rangeblock" not in page, "the stacked range block is gone"
    assert ".wv-lab-range-ends" not in page, "the separate MIN/MAX ends row is gone"


def test_proj_is_start_sit_style_hero(page):
    m = re.search(r"(?m)^\.wv-lab-proj \.n \{([^}]*)\}", page)
    assert m, "missing .wv-lab-proj .n CSS rule"
    assert "font-size: 23px" in m.group(1), (
        "the Lab projection must be the Start/Sit hero number (23px), "
        "pinned to the right of the row"
    )
    m2 = re.search(r"(?m)^\.wv-lab-line2 \{([^}]*)\}", page)
    assert m2 and "display: flex" in m2.group(1), (
        "usage/matchup/tags and the range must share one flex line under the name"
    )


def test_phone_detail_line_wraps_range_full_width(page):
    m = re.search(r"@media \(max-width: 560px\) \{(.*?)\n\}", page, re.S)
    assert m, "missing the phone breakpoint for the Lab detail line"
    block = m.group(1)
    assert ".wv-lab-line2 { flex-wrap: wrap" in block, (
        "on phones the detail line must wrap instead of crushing the range bar"
    )
    assert ".wv-lab-line2 .wv-lab-rangeline { flex: 1 1 100%" in block, (
        "the wrapped range line must go full width so the bar stays a real gauge"
    )
    assert ".wv-lab-line2 .wv-lab-meta { flex: 1 1 100%" in block, (
        "the meta line must take its own full-width line above the range"
    )


def test_tags_are_outline_pills(page):
    assert "border: 1px solid currentColor" in page, "tags must be outline pills"
    assert ".wv-lab-tag.td" in page


def test_slot_row_renders_horizontal_range_with_labels(lab_js):
    vals = _run_node(lab_js, _LAB_SETUP + r"""
wvLabSwapDelta = function(si, b) { return 0; };
var qb = __mkEntry('p1', 'Starter A', 'QB', 18, 4.4, 31.2, []);
qb.usage_stat = 'snap_pct'; qb.usage_avg = 98.2;
var rb = __mkEntry('p2', 'Starter B', 'RB', 15, 8, 24, []);
rb.usage_stat = 'touches'; rb.usage_avg = 17.34;
wvLabLineup = [qb, rb];
var html = wvLabRenderSlots();
var i1 = html.indexOf('wv-lab-line1'), i2 = html.indexOf('wv-lab-line2');
var line1 = (i1 !== -1 && i2 !== -1) ? html.slice(i1, i2) : '';
console.log('LINE=' + (html.indexOf('wv-lab-rangeline') !== -1 && i1 !== -1 && i2 !== -1 && i1 < i2));
console.log('NAMEOWN=' + (line1.indexOf('Starter A') !== -1 && line1.indexOf('wv-lab-meta') === -1 && line1.indexOf('MIN') === -1));
var line2 = i2 !== -1 ? html.slice(i2) : '';
console.log('DETAIL=' + (line2.indexOf('wv-lab-meta') !== -1 && line2.indexOf('wv-lab-rangeline') !== -1 && line2.indexOf('wv-lab-meta') < line2.indexOf('wv-lab-rangeline')));
console.log('STACKED=' + (html.indexOf('wv-lab-rangeblock') !== -1 || html.indexOf('wv-lab-sub') !== -1));
console.log('MIN=' + (html.indexOf('<em>MIN</em>') !== -1 && html.indexOf('>4.4<') !== -1));
console.log('MAX=' + (html.indexOf('<em>MAX</em>') !== -1 && html.indexOf('>31.2<') !== -1));
console.log('META=' + (html.indexOf('wv-lab-meta') !== -1));
console.log('SNAP=' + (html.indexOf('98% snaps') !== -1));
console.log('TOUCHES=' + (html.indexOf('17.3 touches/g') !== -1));
console.log('NOSHARE=' + (html.indexOf('% share') === -1));
console.log('NODASH=' + (html.indexOf('\u2014') === -1));
""")
    assert vals["LINE"] == "true", "row must render the horizontal range line"
    assert vals["NAMEOWN"] == "true", "the name must sit on its own row, no meta or range in line 1"
    assert vals["DETAIL"] == "true", "usage/matchup/tags and the range must share the line under the name"
    assert vals["STACKED"] == "false", "row must not use the old stacked blocks"
    assert vals["MIN"] == "true", "range line must label its MIN end with the floor value"
    assert vals["MAX"] == "true", "range line must label its MAX end with the ceiling value"
    assert vals["META"] == "true", "usage/matchup/tags must sit in the meta line"
    assert vals["SNAP"] == "true", "a QB must show snap %, not a points share"
    assert vals["TOUCHES"] == "true", "an RB must show touches per game"
    assert vals["NOSHARE"] == "true", "the meaningless points-share stat is gone"
    assert vals["NODASH"] == "true", "no em dashes in Lab UI copy"


def test_range_track_is_block_level(page):
    m = re.search(r"(?m)^\.wv-lab-range \{([^}]*)\}", page)
    assert m, "missing standalone .wv-lab-range CSS rule"
    assert "display" in m.group(1) and "block" in m.group(1), (
        "the range track must be display:block: as an inline span its height is "
        "ignored and the bar collapses to zero height"
    )


def test_chase_upside_promotes_shared_bench_player_once(lab_js):
    # Regression: one FLEX-eligible bench player copied under three slots
    # must be promoted exactly once, not once per slot.
    vals = _run_node(lab_js, r"""
wvLabData = { opponent: { name: 'Opp' }, corr: {} };
wvLabOppStats = null;
wvLabResult = { winPct: 0.62, p10: 100, p90: 140, median: 120 };
wvLabSwapDelta = function(si, b) { return 0.02; };  // render-only: skip sims
wvLabEvaluate = function(lineup) { return { winPct: 0.70, median: 125, p10: 105, p90: 145 }; };
function __e2(pid, name, pos, slot, proj, floor, ceiling, eligible, bench) {
  return { player_id: pid, slot: slot, name: name, pos: pos, proj: proj,
           floor: floor, ceiling: ceiling, matchup: '', tags: [],
           profile: {}, eligible: eligible, usage_stat: null, usage_avg: null,
           bench: bench || [] };
}
function __flexStar() { return __e2('w1', 'Flex Star', 'WR', 'BN', 12, 5, 40, [], []); }
wvLabLineup = [
  __e2('s1', 'Wide One', 'WR', 'WR', 14, 8, 20, ['WR'], [__flexStar()]),
  __e2('s2', 'Wide Two', 'WR', 'WR', 14, 8, 21, ['WR'], [__flexStar()]),
  __e2('s3', 'Flex Guy', 'RB', 'FLEX', 13, 7, 19, ['RB', 'TE', 'WR'], [__flexStar()])
];
wvLabChaseUpside();
var html = __fakeBody.innerHTML;
var starters = wvLabLineup.map(function(e) { return e.name; }).join('/');
var starStarts = wvLabLineup.filter(function(e) { return e.player_id === 'w1'; }).length;
var starBenched = 0;
wvLabLineup.forEach(function(e) { (e.bench || []).forEach(function(b) {
  if (b.player_id === 'w1') starBenched++;
}); });
console.log('CHANGES=' + JSON.stringify(wvLabLastChanges));
console.log('STARTERS=' + starters);
console.log('STARSTARTS=' + starStarts);
console.log('STARBENCHED=' + starBenched);
console.log('SUMMARY=' + (html.indexOf('Chased upside: 1 swap:') !== -1
  && html.indexOf('Flex Star in for Flex Guy') !== -1));
console.log('NAMECOUNT=' + (html.split('Flex Star').length - 1));
console.log('DEMOTED=' + (html.indexOf('Flex Guy') !== -1));
console.log('NODASH=' + (html.indexOf('\u2014') === -1));
""")
    assert vals["CHANGES"] == '[{"si":2,"out":"Flex Guy","inn":"Flex Star"}]', \
        "the shared bench player must be promoted exactly once"
    assert vals["STARTERS"] == "Wide One/Wide Two/Flex Star"
    assert vals["STARSTARTS"] == "1", "one player, one start"
    assert vals["STARBENCHED"] == "0", "a starter must not remain on any bench"
    assert vals["SUMMARY"] == "true", "summary must list the single real swap"
    assert vals["NAMECOUNT"] == "2", "Flex Star appears once as a starter, once in the summary"
    assert vals["DEMOTED"] == "true", "the demoted starter takes the FLEX bench seat"
    assert vals["NODASH"] == "true", "no em dashes in Lab UI copy"


def test_optimize_no_gain_disables_button(lab_js):
    vals = _run_node(lab_js, _LAB_SETUP + r"""
wvLabSwapDelta = function(si, b) { return -0.03; };  // every swap loses win%
wvLabEvaluate = function(lineup) { return { winPct: 0.62, median: 120, p10: 100, p90: 140 }; };
wvLabOptimize();
var html = __fakeBody.innerHTML;
console.log('CHANGES=' + JSON.stringify(wvLabLastChanges));
console.log('NOTE=' + (html.indexOf('Already optimized. No swaps improve your win probability.') !== -1));
console.log('BTN=' + (html.indexOf('disabled title="Already optimized"') !== -1
  && html.indexOf('Lineup optimized') !== -1));
console.log('FLASH=' + (html.indexOf('wv-lab-changed') !== -1));
// A manual swap must re-enable the button: the state is sticky, not permanent.
wvLabSwapDelta = function(si, b) { return 0.05; };
wvLabSwap(0, 0);
html = __fakeBody.innerHTML;
console.log('REENABLED=' + (html.indexOf('Optimize lineup') !== -1
  && html.indexOf('disabled title="Already optimized"') === -1));
""")
    assert vals["CHANGES"] == "[]", "no swaps when every delta is negative"
    assert vals["NOTE"] == "true", "no-gain optimize must say the lineup is already optimized"
    assert vals["BTN"] == "true", "no-gain optimize must park the button disabled"
    assert vals["FLASH"] == "false", "nothing changed, so nothing flashes"
    assert vals["REENABLED"] == "true", "a lineup change must clear the parked state"


def test_scroll_respects_reduced_motion(lab_js):
    vals = _run_node(lab_js, _LAB_SETUP + r"""
window.matchMedia = function(q) { return { matches: true }; };
wvLabSwapDelta = function(si, b) { return 0.02; };
wvLabEvaluate = function(lineup) { return { winPct: 0.70, median: 125, p10: 105, p90: 145 }; };
wvLabLineup[1].bench = [__mkEntry('p5', 'Bench E', 'RB', 14, 6, 30, [])];
wvLabChaseUpside();
console.log('SCROLLINTOVIEW=' + JSON.stringify(__scrolledIntoView));
""")
    assert vals["SCROLLINTOVIEW"] == '{"block":"nearest","behavior":"auto"}', \
        "reduced-motion users must get an instant scroll, not a smooth one"
