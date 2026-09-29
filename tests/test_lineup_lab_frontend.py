"""Lineup Lab frontend contract tests.

The Lab lives on the Start/Sit tab as an Advise/Lab toggle. The browser runs
2,000 Monte Carlo sims client-side after one payload fetch; swaps must never
trigger another request.
"""
import re

import pytest

from dashboard_services.pages.waivers_page import build_waivers_body


@pytest.fixture(scope="module")
def page():
    return build_waivers_body("sleeper", 2026, "12345", {})


@pytest.fixture(scope="module")
def script(page):
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    matches = [s for s in scripts if "function wvLoad(" in s]
    assert matches, "waivers inline script not found"
    return matches[0]


def test_lab_toggle_markup(page):
    assert 'id="wvSsModeAdvise"' in page
    assert 'id="wvSsModeLab"' in page
    assert "wvSetSsMode('advise')" in page
    assert "wvSetSsMode('lab')" in page
    assert 'id="wvLab"' in page
    assert 'id="wvLabBody"' in page


def test_lab_css_present(page):
    assert ".wv-lab-hero" in page
    assert ".wv-lab-winbar" in page


def test_lab_loading_skeleton_css(page):
    # Animated loading state: staggered dots + structured shimmer skeleton.
    assert "@keyframes wv-lab-blink" in page
    assert ".wv-lab-dots" in page
    assert ".wv-lab-sk-hero" in page
    assert ".wv-lab-sk-winbar" in page
    assert ".wv-lab-sk-trow" in page
    assert ".wv-lab-sk-modes" in page
    assert ".wv-lab-sk-slot" in page


def test_lab_skeleton_used_while_loading(script):
    assert "function wvLabSkeleton(" in script
    assert "body.innerHTML = wvLabSkeleton()" in script
    # The old bare-text loading state is gone from the load path.
    assert "Simulating 2,000 lineups...</div>" not in script


def test_lab_uses_two_thousand_sims(script):
    assert "var WV_LAB_SIMS = 2000" in script


def test_lab_single_fetch_no_per_swap_requests(script):
    # One payload fetch for the whole Lab session (URL built once, fetched once).
    refs = re.findall(r"/api/lineup-lab", script)
    assert len(refs) == 1, f"expected exactly one lineup-lab reference, found {len(refs)}"
    assert re.search(r"fetch\(url\)\.then", script), "Lab payload must be fetched once"
    # ...and swap/optimize paths never fetch.
    for fn in ("function wvLabSwap(", "function wvLabOptimize(", "function wvLabSwapDelta("):
        start = script.index(fn)
        nxt = script.find("\nfunction ", start + 1)
        body = script[start:nxt if nxt != -1 else len(script)]
        assert "fetch(" not in body, f"{fn} must not hit the network"


def test_lab_common_random_numbers(script):
    # Swap deltas are computed on one shared set of base draws.
    assert "function wvLabBuildBase(" in script
    assert "function wvLabEvaluate(" in script
    assert "wvLabBase" in script
    # Correlated draws via Cholesky (stack correlation), skew-normal margins.
    assert "function wvLabCholesky(" in script
    assert "function wvLabSkewParams(" in script


def test_lab_modes_and_actions(script):
    for fn in ("function wvSetSsMode(", "function wvLoadLab(", "function wvRenderLab(",
               "function wvLabToggleSlot(", "function wvLabSwap(", "function wvLabToggleUpside(",
               "function wvLabOptimize("):
        assert fn in script, f"missing {fn}"


def test_lab_no_top_level_lexical_declarations(script):
    bad = [ln.strip() for ln in script.splitlines()
           if re.match(r"^(let|const|class)\s", ln)]
    lab_bad = [b for b in bad if "wvLab" in b or "WV_LAB" in b]
    assert not lab_bad, f"Lab top-level lexical declarations break reexec: {lab_bad[:5]}"


def test_lab_skeleton_renders_structured_markup(script):
    # The skeleton must mirror the loaded layout (hero + lineup rows) so the
    # page does not repaint when results arrive. Render it under node.
    pytest.importorskip("subprocess")
    import subprocess, tempfile, os
    start = script.index("function wvLabSkeleton(")
    end = script.index("\nfunction wvLoadLab(", start)
    js = script[start:end]
    harness = js + """
var html = wvLabSkeleton();
var slots = (html.match(/wv-lab-sk-slot/g) || []).length;
if (slots !== 9) throw new Error('expected 9 slot skeletons, got ' + slots);
var need = ['wv-lab-sk-hero', 'wv-lab-sk-winbar', 'wv-lab-sk-trow',
            'wv-lab-sk-modes', 'wv-lab-dots', 'wv-lab-loadmsg', 'Your lineup',
            'skeleton'];
for (var k = 0; k < need.length; k++) {
  if (html.indexOf(need[k]) < 0) throw new Error('skeleton missing ' + need[k]);
}
var open = (html.match(/<div/g) || []).length;
var close = (html.match(/<\\/div>/g) || []).length;
if (open !== close) throw new Error('unbalanced divs: ' + open + ' vs ' + close);
console.log('SKELETON_OK slots=' + slots);
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        out = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    assert out.returncode == 0, f"node skeleton harness failed: {out.stderr[-2000:]}"
    assert "SKELETON_OK" in out.stdout


def test_lab_engine_runs_in_node(script):
    pytest.importorskip("subprocess")
    import subprocess, tempfile, os
    start = script.index("// ── Lineup Lab (Start/Sit tab) ──")
    end = script.index("if (!window.__brctx)", start)
    js = script[start:end]
    harness = js + """
var WV_LAB_SIMS = 500;
wvLabData = { corr: {} };
var pids = ['1','2','3','4'];
wvLabBase = wvLabBuildBase(pids, WV_LAB_SIMS, 42);
function mk(pid, mean, std) { return { player_id: pid, profile: { mean: mean, std: std, skew_alpha: 2, dud_risk: 0 } }; }
var lineup = [mk('1', 20, 9), mk('2', 15, 8), mk('3', 12, 7)];
wvLabLineup = lineup;
var sp = wvLabSkewParams(42, 15, 2.0);
var rand = wvLabRng(49);
wvLabOppDraws = new Float64Array(WV_LAB_SIMS);
for (var s = 0; s < WV_LAB_SIMS; s++) {
  var a = rand() + 1e-12, b = rand();
  var r = Math.sqrt(-2 * Math.log(a)), ang = 2 * Math.PI * b;
  wvLabOppDraws[s] = Math.max(0, sp.xi + sp.omega * (sp.delta * Math.abs(r*Math.cos(ang)) + sp.w2 * r * Math.sin(ang)));
}
var res = wvLabEvaluate(lineup);
wvLabResult = res;
var d1 = wvLabSwapDelta(2, mk('4', 16, 8));
var d2 = wvLabSwapDelta(2, mk('4', 16, 8));
if (!(res.winPct > 0.5 && res.winPct < 0.8)) throw new Error('winPct out of range: ' + res.winPct);
if (d1 !== d2) throw new Error('CRN deltas not stable');
if (!(d1 > 0)) throw new Error('better player should raise win%: ' + d1);
console.log('LAB_ENGINE_OK winPct=' + res.winPct.toFixed(3) + ' delta=' + d1.toFixed(4));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        out = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    assert out.returncode == 0, f"node harness failed: {out.stderr[-2000:]}"
    assert "LAB_ENGINE_OK" in out.stdout
