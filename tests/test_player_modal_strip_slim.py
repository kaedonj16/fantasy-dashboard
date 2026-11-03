"""Player modal Start/Sit strip: the slim verdict line (declutter, 2026-11-03).

The old strip was a tinted, bordered box with slot + grade pills; with the
long reason it wrapped to ~61px (+12px margin) in the modal. The approved
declutter replaces it with one line of text on the modal surface: verdict
word in its tone color, the slot as plain bold text, the reason's facts as
separate segments divided by hairline rules, and the Compare deep-link as
a plain text link. No fill, no border box, no pills -- the grade chip is
dropped entirely (it was not in the approved mock). A typical two-fact
verdict measures 29.5px tall, 37.5px including its bottom margin, in a
local render of the real stylesheet (844px modal content width).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _strip_fn() -> str:
    start = JS.index("function _pmLoadStartSitStrip(")
    end = JS.index("\nfunction _pmLoadInLeague", start)
    return JS[start:end]


def _css_rule(selector: str) -> str:
    start = CSS.index(selector + " {")
    return CSS[start:CSS.index("}", start)]


# ---------------------------------------------------------------------------
# Static structure
# ---------------------------------------------------------------------------

def test_strip_builder_has_no_pills_or_grade():
    fn = _strip_fn()
    # The grade chip is gone from the strip entirely.
    assert "pm-ss-grade" not in fn
    assert "d.grade" not in fn
    # The reason's facts render as hairline-divided segments.
    assert "split(' · ')" in fn
    assert "pm-ss-sep" in fn
    # Verdict + plain-text slot + the Compare deep-link survive.
    assert "pm-ss-verdict" in fn
    assert "pm-ss-slot" in fn
    assert "Compare in Start/Sit" in fn


def test_strip_css_is_a_line_not_a_box():
    rule = _css_rule(".pm-ss-strip")
    # No fill, no radius, no border box -- a single bottom hairline only.
    assert "background" not in rule
    assert "border-radius" not in rule
    assert "border-bottom: 1px solid var(--border)" in rule
    # The slot is plain text now: no pill padding/border of its own.
    slot = _css_rule(".pm-ss-slot")
    assert "padding" not in slot and "border" not in slot
    # Dead pill styles are gone; the hairline divider exists.
    assert ".pm-ss-grade" not in CSS
    assert ".pm-ss-sep {" in CSS
    # Verdict tone colors are unchanged.
    assert ".pm-ss-start .pm-ss-verdict { color: var(--win); }" in CSS
    assert ".pm-ss-bench .pm-ss-verdict" in CSS


# ---------------------------------------------------------------------------
# Behavioral (node): the real loader against stub payloads
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_strip_renders_slim_line_node():
    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    harness = (
        "var el = { className: '', innerHTML: '', hidden: true };\n"
        "var document = { getElementById: function () { return el; },\n"
        "  body: { contains: function () { return true; } } };\n"
        "var CURRENT = null;\n"
        "function _pmTakeEarlyJson(url) { return Promise.resolve(CURRENT); }\n"
        + _strip_fn() + "\n"
        "var cases = [\n"
        "  { name: 'start', payload: { state: 'ok', tone: 'start', verdict: 'Start',\n"
        "      slot: 'WR1', grade: 92, compare: true,\n"
        "      reason: 'Projects 19.6 · top WR score in your lineup · bench WR projects 9.4 lower · IND allows the 2nd most points to WRs' } },\n"
        "  { name: 'would-start', payload: { state: 'ok', tone: 'start', verdict: 'Would start for you',\n"
        "      slot: 'WR2', compare: true,\n"
        "      reason: \"Projects 18.1 vs your WR2's 11.4 · KC allows the 3rd most points to WRs\" } },\n"
        "  { name: 'bench', payload: { state: 'ok', tone: 'bench', verdict: 'Bench',\n"
        "      slot: 'WR3', compare: true, reason: 'Projects 12.3 · your starters project more' } },\n"
        "  { name: 'unavailable', payload: { state: 'unavailable', text: 'On bye in Week 9', compare: true } }\n"
        "];\n"
        "var results = {};\n"
        "function run(i) {\n"
        "  if (i >= cases.length) { finish(); return; }\n"
        "  CURRENT = cases[i].payload;\n"
        "  el.className = ''; el.innerHTML = ''; el.hidden = true;\n"
        "  _pmLoadStartSitStrip('4242', { position: 'WR' }, 'lg9', 'sleeper', '2026');\n"
        "  setTimeout(function () {\n"
        "    results[cases[i].name] = { cls: el.className, html: el.innerHTML, hidden: el.hidden };\n"
        "    run(i + 1);\n"
        "  }, 30);\n"
        "}\n"
        "function finish() {\n"
        "  var fails = [];\n"
        "  function chk(c, m) { if (!c) fails.push(m); }\n"
        "  var s = results['start'];\n"
        "  chk(s && s.hidden === false, 'start: strip shows');\n"
        "  chk(s.cls === 'pm-ss-strip pm-ss-start', 'start: tone class');\n"
        "  chk(s.html.indexOf('>Start<') >= 0, 'start: verdict');\n"
        "  chk(s.html.indexOf('>WR1<') >= 0, 'start: slot');\n"
        "  chk(s.html.indexOf('Projects <b>19.6</b>') >= 0, 'start: projection fact');\n"
        "  chk(s.html.indexOf('top WR score in your lineup') >= 0, 'start: top-score fact');\n"
        "  chk(s.html.indexOf('bench WR projects <b>9.4</b> lower') >= 0, 'start: bench-gap fact');\n"
        "  chk(s.html.indexOf('IND allows the <b>2</b>nd most points to WRs') >= 0, 'start: matchup fact');\n"
        "  chk(s.html.indexOf('pm-ss-sep') >= 0, 'start: hairline separators');\n"
        "  chk(s.html.indexOf('Compare in Start/Sit') >= 0, 'start: compare link');\n"
        "  chk(s.html.indexOf('tab=startsit&guest=4242') >= 0, 'start: deep link');\n"
        "  chk(s.html.indexOf('pm-ss-grade') < 0, 'start: no grade pill');\n"
        "  chk(s.html.indexOf('>92<') < 0, 'start: grade value not rendered');\n"
        "  var w = results['would-start'];\n"
        "  chk(w.html.indexOf('Would start for you') >= 0, 'guest: verdict');\n"
        "  chk(w.html.indexOf(\"Projects <b>18.1</b> vs your WR<b>2</b>'s <b>11.4</b>\") >= 0, 'guest: vs fact');\n"
        "  chk(w.html.indexOf('KC allows the <b>3</b>rd most points to WRs') >= 0, 'guest: matchup fact');\n"
        "  var b = results['bench'];\n"
        "  chk(b.cls === 'pm-ss-strip pm-ss-bench', 'bench: tone class');\n"
        "  chk(b.html.indexOf('>Bench<') >= 0 && b.html.indexOf('>WR3<') >= 0, 'bench: verdict + slot');\n"
        "  var u = results['unavailable'];\n"
        "  chk(u.cls === 'pm-ss-strip pm-ss-na', 'unavailable: class');\n"
        "  chk(u.html.indexOf('On bye in Week 9') >= 0, 'unavailable: text kept');\n"
        "  chk(u.html.indexOf('Compare in Start/Sit') >= 0, 'unavailable: link kept');\n"
        "  if (fails.length) { console.error(fails.join('; ')); process.exit(3); }\n"
        "  process.exit(0);\n"
        "}\n"
        "run(0);\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "strip_slim_check.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=15)
        assert res.returncode == 0, (res.stderr or res.stdout)
