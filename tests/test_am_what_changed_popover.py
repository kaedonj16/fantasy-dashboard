"""What Changed as an anchored popover (approved mock treatment B).

The Advanced Metrics "What Changed" section (.am-movers) used to expand
IN the row: the head toggled a .collapsed class and the groups panel
opened in-flow, pushing the leaderboard down. Kaedon picked the popover
rework on 2026-11-06: the in-row element is permanently one slim bar,
and the groups open in a floating panel anchored under the bar (the
chip-editor popover pattern), closing on outside click, Escape, or a
second bar click. Data fetching and group rendering are unchanged.
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


def _run_node(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def _popover_block(js):
    start = js.index("let _amMoversOpen = false;")
    end = js.index("function _loadMovers()")
    return js[start:end]


# ── Markup: slim bar in the row, groups inside a hidden popover ───────

def test_bar_is_slim_and_groups_live_in_the_popover():
    body = _body(_html())
    host = body.index('id="amMovers"')
    head = body.index('id="amMoversHead"')
    pop = body.index('id="amMoversPop"')
    groups = body.index('id="amMoversGroups"')
    compare = body.index('id="amCompareModal"')
    assert host < head < pop < groups < compare
    # Empty/no-movers behavior unchanged: host hidden until JS fills it.
    host_tag = body[body.rindex("<div", 0, host):body.index(">", host)]
    assert 'style="display:none;"' in host_tag
    # The popover is the hidden floating container for the groups.
    pop_tag = body[body.rindex("<div", 0, pop):body.index(">", pop)]
    assert "am-movers-pop" in pop_tag and "hidden" in pop_tag
    # The detail note lives inside the popover now, not in the bar.
    sub = body.index('id="amMoversSub"')
    assert pop < sub < groups, "the sub note belongs inside the popover"
    head_span = body[head:pop]
    assert "amMoversSub" not in head_span
    # The bar itself carries bolt, label, one-line summary, chevron.
    assert "amMoversSummary" in head_span
    assert "amMoversChev" in head_span
    head_tag = body[body.rindex("<div", 0, head):body.index(">", head)]
    assert 'aria-expanded="false"' in head_tag


# ── CSS: popover floats, anchored; nothing expands in-flow ────────────

def test_popover_css_is_anchored_absolute_and_bar_never_expands():
    css = _style(_html())
    host = re.search(r"\.am-movers \{([^}]*)\}", css).group(1)
    assert "position:relative" in host.replace(" ", ""), \
        ".am-movers must be the popover's positioned anchor"
    pop = re.search(r"\.am-movers-pop \{([^}]*)\}", css).group(1)
    assert "position:absolute" in pop.replace(" ", "")
    assert "top:calc(100% + 6px)" in pop
    assert "z-index:60" in pop.replace(" ", "")
    assert "999px" not in pop, "pill radius is banned site-wide"
    pop_hidden = re.search(r"\.am-movers-pop\[hidden\] \{([^}]*)\}", css).group(1)
    assert "display:none" in pop_hidden.replace(" ", "")
    # The old in-flow expand/collapse model is gone entirely.
    assert ".am-movers.collapsed" not in css
    assert ".collapsed .am-movers-groups" not in css
    # Open state is the chevron rotation hook, not a layout change.
    assert ".am-movers.open .am-movers-chev" in css


def test_popover_stacks_groups_on_phone():
    css = _style(_html())
    phone = css[css.index("@media (max-width:760px)"):]
    popover_phone = phone[:phone.index("@media (min-width:761px)")] \
        if "@media (min-width:761px)" in phone else phone
    assert ".am-movers-pop { left:0; right:0; width:auto; }" in popover_phone
    assert ".am-movers-groups { flex-direction:column;" in popover_phone


# ── JS: toggle + outside-click / Escape close (chip-editor pattern) ──

def test_popover_js_toggle_and_close_handlers_exist():
    js = _page_js()
    block = _popover_block(js)
    assert "function amSetMoversPopover(open)" in block
    assert "pop.hidden = !_amMoversOpen;" in block
    assert "head.setAttribute('aria-expanded', String(_amMoversOpen));" in block
    # Bar click and Enter/Space toggle.
    assert "amSetMoversPopover(!_amMoversOpen)" in block
    # Document-level close handlers, mirroring the chip editor's.
    assert "document.addEventListener('click'" in block
    assert "e.target.closest('#amMovers')" in block
    assert "document.addEventListener('keydown'" in block
    assert "e.key === 'Escape'" in block
    # Starts closed on every load; no persisted open state.
    assert "amSetMoversPopover(false);" in block
    assert "amMoversCollapsed" not in js
    # Picking a mover chip closes the popover before filtering the table.
    chip = js[js.index("function _moverChip"):js.index("function _moverGroup")]
    assert "amCloseMoversPopover();" in chip
    # Render path is untouched: same fetch, same groups element.
    assert "fetch('/api/advanced-metrics/movers?' + params.toString())" in js
    assert "document.getElementById('amMoversGroups')" in js


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_popover_open_close_behavior_in_node():
    js = _page_js()
    block = _popover_block(js)
    script = (
        "function makeEl(id) {\n"
        "  var el = { id: id, hidden: id === 'amMoversPop', dataset: {}, attrs: {},\n"
        "    title: '', _classes: {}, _handlers: {},\n"
        "    setAttribute: function(k, v) { this.attrs[k] = v; },\n"
        "    addEventListener: function(t, f) { (this._handlers[t] = this._handlers[t] || []).push(f); } };\n"
        "  el.classList = { toggle: function(c, on) { el._classes[c] = !!on; },\n"
        "    contains: function(c) { return !!el._classes[c]; } };\n"
        "  return el;\n"
        "}\n"
        "var els = { amMovers: makeEl('amMovers'), amMoversPop: makeEl('amMoversPop'),\n"
        "  amMoversHead: makeEl('amMoversHead') };\n"
        "var docHandlers = {};\n"
        "var document = { getElementById: function(id) { return els[id] || null; },\n"
        "  addEventListener: function(t, f) { (docHandlers[t] = docHandlers[t] || []).push(f); } };\n"
        + block + "\n"
        "function check(name, cond) { if (!cond) { console.error('FAIL ' + name); process.exit(1); } }\n"
        "function fire(el, t, ev) { (el._handlers[t] || []).forEach(function(f) { f(ev || {}); }); }\n"
        "function fireDoc(t, ev) { (docHandlers[t] || []).forEach(function(f) { f(ev); }); }\n"
        "_initMoversPopover();\n"
        "check('starts closed', els.amMoversPop.hidden === true && _amMoversOpen === false);\n"
        "check('aria starts false', els.amMoversHead.attrs['aria-expanded'] === 'false');\n"
        "fire(els.amMoversHead, 'click');\n"
        "check('bar click opens', els.amMoversPop.hidden === false && _amMoversOpen === true);\n"
        "check('host marked open', els.amMovers.classList.contains('open'));\n"
        "check('aria true while open', els.amMoversHead.attrs['aria-expanded'] === 'true');\n"
        "fireDoc('click', { target: { closest: function(sel) { return sel === '#amMovers' ? {} : null; } } });\n"
        "check('inside click keeps it open', _amMoversOpen === true);\n"
        "fireDoc('click', { target: { closest: function() { return null; } } });\n"
        "check('outside click closes', els.amMoversPop.hidden === true && _amMoversOpen === false);\n"
        "fire(els.amMoversHead, 'click');\n"
        "fireDoc('keydown', { key: 'Escape' });\n"
        "check('Escape closes', _amMoversOpen === false && els.amMoversPop.hidden === true);\n"
        "fire(els.amMoversHead, 'click');\n"
        "fire(els.amMoversHead, 'click');\n"
        "check('second bar click toggles closed', _amMoversOpen === false);\n"
    )
    _run_node(script)
