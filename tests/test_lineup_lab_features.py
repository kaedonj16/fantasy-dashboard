"""Lineup Lab v2 feature contracts (shared-sample sims, best moves, why
panel, live mode) driven through node over the real Lab script section.

- The shared-sample evaluator must reproduce the old per-call evaluator
  exactly on the uncorrelated case (identity copula, no injuries), build
  its samples once per payload, and price swap deltas off cached totals
  identically to a full evaluate.
- The best-moves card must surface the true best swap and say honestly
  when no swap helps.
- The why panel must render only fields the payload carries.
- A final live player locks at their actual points; an in-progress player
  keeps their full distribution.
"""
import re
import subprocess
import tempfile
import os

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


def _run_node(js, body):
    pytest.importorskip("subprocess")
    harness = js + body
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        out = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    assert out.returncode == 0, f"node harness failed: {out.stderr[-2000:]}"
    return out.stdout


_FAKE_DOM = r"""
var __fakeBody = { innerHTML: '' };
function __makeEl() {
  var cls = {};
  return {
    classList: {
      toggle: function(c) { cls[c] = !cls[c]; return !!cls[c]; },
      contains: function(c) { return !!cls[c]; }
    },
    scrollIntoView: function(opts) {}
  };
}
var __els = {};
var window = { pageYOffset: 0, scrollTo: function(x, y) {}, __brctx: {} };
var document = {
  getElementById: function(id) { return id === 'wvLabBody' ? __fakeBody : null; },
  querySelector: function(sel) {
    if (!__els[sel]) __els[sel] = __makeEl();
    return __els[sel];
  },
  documentElement: { scrollTop: 0 }
};
"""


def test_shared_samples_match_reference_evaluator(lab_js):
    # With no correlations and injuries off, the shared-sample evaluator
    # must equal the old per-call evaluator EXACTLY (same draws, same
    # summation order), and it must build its samples only once no matter
    # how many lineups and deltas are evaluated afterwards.
    out = _run_node(lab_js, r"""
var WV_LAB_SIMS = 2000;
wvLabData = { corr: {} };
wvLabInjuryOnset = { QB: 0, RB: 0, WR: 0, TE: 0, K: 0, DEF: 0 };
function refEvaluate(lineup) {
  var totals = new Float64Array(WV_LAB_SIMS);
  for (var i = 0; i < lineup.length; i++) {
    var pf = lineup[i].profile;
    var sp = wvLabSkewParams(pf.mean, pf.std, pf.skew_alpha);
    var bu = wvLabBase[lineup[i].player_id];
    for (var s = 0; s < WV_LAB_SIMS; s++) {
      totals[s] += wvLabDrawPlayer(sp, pf.dud_risk || 0, bu.dud[s], bu.u0[s], bu.u1[s]);
    }
  }
  var wins = 0;
  for (var s = 0; s < WV_LAB_SIMS; s++) if (totals[s] > wvLabOppDraws[s]) wins++;
  var sorted = Array.prototype.slice.call(totals).sort(function(a, b) { return a - b; });
  return { winPct: wins / WV_LAB_SIMS, median: sorted[Math.floor(WV_LAB_SIMS / 2)],
           p10: sorted[Math.floor(WV_LAB_SIMS * 0.1)], p90: sorted[Math.floor(WV_LAB_SIMS * 0.9)] };
}
function mk(pid, mean, std, dud, bench) {
  return { player_id: pid, pos: 'RB', slot: 'RB',
           profile: { mean: mean, std: std, skew_alpha: 2, dud_risk: dud },
           bench: bench || [] };
}
var b4 = mk('4', 16, 8, 0);
var lineup = [mk('1', 20, 9, 0.1), mk('2', 15, 8, 0), mk('3', 12, 7, 0, [b4])];
wvLabLineup = lineup;
wvLabBase = wvLabBuildBase(['1', '2', '3', '4'], WV_LAB_SIMS, 42);
var sp = wvLabSkewParams(42, 15, 2.0);
var rand = wvLabRng(49);
wvLabOppDraws = new Float64Array(WV_LAB_SIMS);
for (var s = 0; s < WV_LAB_SIMS; s++) {
  var a = rand() + 1e-12, b = rand();
  var r = Math.sqrt(-2 * Math.log(a)), ang = 2 * Math.PI * b;
  wvLabOppDraws[s] = Math.max(0, sp.xi + sp.omega * (sp.delta * Math.abs(r * Math.cos(ang)) + sp.w2 * r * Math.sin(ang)));
}
var __builds = 0;
var __origBuild = wvLabBuildSamples;
wvLabBuildSamples = function() { __builds++; return __origBuild(); };
var res = wvLabEvaluate(lineup);
wvLabResult = res;
var ref = refEvaluate(lineup);
if (res.winPct !== ref.winPct || res.median !== ref.median ||
    res.p10 !== ref.p10 || res.p90 !== ref.p90) {
  throw new Error('shared-sample evaluate differs from reference: '
    + JSON.stringify(res) + ' vs ' + JSON.stringify(ref));
}
if (__builds !== 1) throw new Error('samples built ' + __builds + ' times, expected 1');
// More evaluations and deltas must not rebuild.
wvLabEvaluate(lineup);
var dFast = wvLabSwapDelta(2, b4);
wvLabSwapDelta(2, b4);
if (__builds !== 1) throw new Error('samples rebuilt during deltas: ' + __builds);
// Fast-path delta equals a full evaluate of the trial lineup.
var trial = lineup.slice(); trial[2] = b4;
var dFull = wvLabWinPct(trial) - res.winPct;
if (Math.abs(dFast - dFull) > 1 / WV_LAB_SIMS + 1e-9) {
  throw new Error('fast delta ' + dFast + ' != full delta ' + dFull);
}
if (!(dFast > 0)) throw new Error('better bench player should raise win%: ' + dFast);
console.log('SHARED_SAMPLES_OK');
""")
    assert "SHARED_SAMPLES_OK" in out


def test_best_moves_card(lab_js):
    out = _run_node(lab_js, _FAKE_DOM + r"""
var WV_LAB_SIMS = 2000;
wvLabData = { opponent: { name: 'Test Opp' }, corr: {} };
function mk(pid, name, mean, std, bench) {
  return { player_id: pid, name: name, pos: 'RB', slot: 'RB', proj: mean,
           floor: mean - 5, ceiling: mean + 5, matchup: 'vs DEN', tags: [],
           profile: { mean: mean, std: std, skew_alpha: 2, dud_risk: 0 },
           eligible: ['RB'], usage_stat: null, usage_avg: null, bench: bench || [] };
}
var star = mk('9', 'Bench Star', 20, 5, []);
var weak = mk('1', 'Weak Starter', 8, 5, [star]);
var strongBench = mk('8', 'Weak Bench', 6, 5, []);
var strong = mk('2', 'Strong Starter', 19, 5, [strongBench]);
var lineup = [weak, strong];
wvLabLineup = lineup;
wvLabBase = wvLabBuildBase(['1', '2', '8', '9'], WV_LAB_SIMS, 11);
wvLabOppDraws = new Float64Array(WV_LAB_SIMS);
for (var s = 0; s < WV_LAB_SIMS; s++) wvLabOppDraws[s] = 26;
wvLabResult = wvLabEvaluate(lineup);
var moves = wvLabBestMoves();
if (moves.length !== 1) throw new Error('expected exactly 1 best move, got ' + moves.length);
if (moves[0].inn !== 'Bench Star' || moves[0].out !== 'Weak Starter' ||
    moves[0].si !== 0 || moves[0].bi !== 0) {
  throw new Error('wrong best move: ' + JSON.stringify(moves[0]));
}
if (Math.round(moves[0].delta * 100) < 1) throw new Error('delta too small: ' + moves[0].delta);
var html = wvLabRenderBestMoves();
if (html.indexOf('Best moves') < 0) throw new Error('card header missing');
if (html.indexOf('Bench Star in for Weak Starter') < 0) throw new Error('move row missing: ' + html);
if (html.indexOf('wvLabSwap(0,0)') < 0) throw new Error('move row must apply the swap');
// After any action the changes summary owns the spot: no card.
wvLabLastChanges = [{ si: 0, out: 'Weak Starter', inn: 'Bench Star' }];
if (wvLabRenderBestMoves() !== '') throw new Error('card must hide after an action');
wvLabLastChanges = [];
// Honest empty state: starters already beat every bench option.
wvLabLineup = [strong, weak];
wvLabLineup[0].bench = [strongBench];
wvLabLineup[1].bench = [mk('7', 'Worse Bench', 4, 5, [])];
wvLabBase = wvLabBuildBase(['1', '2', '7', '8'], WV_LAB_SIMS, 11);
wvLabResult = wvLabEvaluate(wvLabLineup);
if (wvLabBestMoves().length !== 0) throw new Error('expected no moves for a maxed lineup');
var empty = wvLabRenderBestMoves();
if (empty.indexOf('already the best single-swap option') < 0) {
  throw new Error('honest empty state missing: ' + empty);
}
console.log('BEST_MOVES_OK');
""")
    assert "BEST_MOVES_OK" in out


def test_why_panel(lab_js):
    out = _run_node(lab_js, _FAKE_DOM + r"""
var WV_LAB_SIMS = 500;
wvLabData = { opponent: { name: 'Test Opp' }, corr: { '1:2': 0.42 } };
function mk(pid, name, extra) {
  var e = { player_id: pid, name: name, pos: 'WR', slot: 'WR', proj: 15,
            floor: 5, ceiling: 25, matchup: '', tags: [],
            profile: { mean: 15, std: 9.5, skew_alpha: 2, dud_risk: 0.12 },
            eligible: ['WR'], usage_stat: null, usage_avg: null, bench: [] };
  for (var k in (extra || {})) e[k] = extra[k];
  return e;
}
var partner = mk('2', 'Partner Player');
var hero = mk('1', 'Hero Player', { matchup: 'vs SF', usage_stat: 'targets', usage_avg: 8.5 });
wvLabLineup = [hero, partner];
var why = wvLabWhyHtml(hero);
if (why.indexOf('vs SF') < 0) throw new Error('matchup line missing: ' + why);
if (why.indexOf('8.5 targets/g') < 0 || why.indexOf('season average') < 0) {
  throw new Error('usage line missing: ' + why);
}
if (why.indexOf('Std 9.5') < 0 || why.indexOf('dud risk 12%') < 0) {
  throw new Error('spread line missing: ' + why);
}
if (why.indexOf('Partner Player (+0.42)') < 0) {
  throw new Error('correlation line missing: ' + why);
}
// Missing data is omitted, never invented.
var bare = mk('3', 'Bare Player', { profile: {} });
var bareWhy = wvLabWhyHtml(bare);
if (bareWhy.indexOf('No extra detail for this player this week.') < 0) {
  throw new Error('bare player should get the fallback line: ' + bareWhy);
}
if (bareWhy.indexOf('Matchup') >= 0 || bareWhy.indexOf('Correlation') >= 0) {
  throw new Error('bare player invented detail: ' + bareWhy);
}
// Rendered rows carry the panel and a tappable range line.
wvLabResult = { winPct: 0.5, p10: 90, p90: 150, median: 120 };
wvLabOppDraws = new Float64Array(WV_LAB_SIMS);
wvLabBase = wvLabBuildBase(['1', '2'], WV_LAB_SIMS, 3);
var slots = wvLabRenderSlots();
if (slots.indexOf('wv-lab-why') < 0) throw new Error('why panel not rendered');
if (slots.indexOf('wvLabToggleWhy(0)') < 0) throw new Error('range line not tappable');
// One panel open at a time.
wvLabToggleWhy(0);
var el0 = document.querySelector('.wv-lab-slot[data-si="0"]');
var el1 = document.querySelector('.wv-lab-slot[data-si="1"]');
if (!el0.classList.contains('why-open')) throw new Error('panel 0 did not open');
wvLabToggleWhy(1);
if (el0.classList.contains('why-open')) throw new Error('panel 0 stayed open');
if (!el1.classList.contains('why-open')) throw new Error('panel 1 did not open');
wvLabToggleWhy(1);
if (el1.classList.contains('why-open') || wvLabWhySlot !== -1) {
  throw new Error('panel 1 did not close');
}
console.log('WHY_PANEL_OK');
""")
    assert "WHY_PANEL_OK" in out


def test_live_locking_in_engine(lab_js):
    out = _run_node(lab_js, r"""
var WV_LAB_SIMS = 2000;
wvLabData = { corr: {} };
function mk(pid, mean, std, live) {
  var e = { player_id: pid, pos: 'RB', slot: 'RB',
            profile: { mean: mean, std: std, skew_alpha: 2, dud_risk: 0 }, bench: [] };
  if (live) e.live = live;
  return e;
}
wvLabOppDraws = new Float64Array(WV_LAB_SIMS);
for (var s = 0; s < WV_LAB_SIMS; s++) wvLabOppDraws[s] = 20;
// Final: the locked player contributes exactly their actual points.
var locked = [mk('1', 12, 9, { status: 'final', points: 24.3 })];
wvLabBase = wvLabBuildBase(['1'], WV_LAB_SIMS, 5);
var res = wvLabEvaluate(locked);
if (res.median !== 24.3 || res.p10 !== 24.3 || res.p90 !== 24.3) {
  throw new Error('locked player must be constant at actuals: ' + JSON.stringify(res));
}
if (res.winPct !== 1) throw new Error('locked 24.3 vs 20 must always win: ' + res.winPct);
// In progress: display points do NOT pin the sim; distribution survives.
var live = [mk('2', 20, 5, { status: 'live', points: 3.0 })];
wvLabBase = wvLabBuildBase(['2'], WV_LAB_SIMS, 5);
var res2 = wvLabEvaluate(live);
if (!(res2.median > 15 && res2.median < 25)) {
  throw new Error('in-progress player must keep its distribution, median ' + res2.median);
}
if (res2.p90 - res2.p10 < 5) throw new Error('in-progress range collapsed: ' + JSON.stringify(res2));
console.log('LIVE_LOCK_OK');
""")
    assert "LIVE_LOCK_OK" in out


def test_chase_upside_net_gain_rule(lab_js):
    # Net gain = ceiling gained minus projection given up. A volatile
    # dart-throw must not displace a steadier, better-projected starter
    # on ceiling alone; a swap that costs no projection still goes.
    out = _run_node(lab_js, _FAKE_DOM + r"""
wvLabEvaluate = function(lineup) { return {}; };
wvLabRenderLab = function() {};
wvLabScrollChangedIntoView = function(si) {};
function mk(name, proj, ceiling, bench) {
  return { player_id: name, name: name, pos: 'WR', slot: 'WR',
           proj: proj, floor: 0, ceiling: ceiling,
           matchup: '', tags: [], profile: {}, eligible: ['WR'],
           usage_stat: null, usage_avg: null, bench: bench || [] };
}
function reset(lineup) {
  wvLabLineup = lineup;
  wvLabOpenSlots = {};
  wvLabNoGain = {optimize: false, upside: false};
  wvLabLastChanges = [];
  wvLabLastNote = '';
}
// Case 1: Jamo-type bench (proj 9, ceiling 24) vs steady starter
// (proj 16, ceiling 19). Old rule swaps (+5 ceiling); net rule blocks
// (+5 ceiling, -7 projection = -2 net).
reset([mk('Steady', 16, 19, [mk('Jamo', 9, 24)])]);
wvLabChaseUpside();
if (wvLabLastChanges.length !== 0)
  throw new Error('dart-throw swap should be blocked, got ' + JSON.stringify(wvLabLastChanges));
if (wvLabLastNote !== 'Already your highest-ceiling lineup.')
  throw new Error('expected no-gain note, got ' + wvLabLastNote);
// Case 2: close projections (bench proj 15, ceiling 24): +5 ceiling,
// -1 projection = +4 net, so the swap still happens.
reset([mk('Steady', 16, 19, [mk('Wr2', 15, 24)])]);
wvLabChaseUpside();
if (wvLabLastChanges.length !== 1)
  throw new Error('expected 1 swap, got ' + wvLabLastChanges.length);
if (wvLabLastChanges[0].out !== 'Steady' || wvLabLastChanges[0].inn !== 'Wr2')
  throw new Error('wrong swap: ' + JSON.stringify(wvLabLastChanges[0]));
if (wvLabLineup[0].name !== 'Wr2')
  throw new Error('lineup not updated: ' + wvLabLineup[0].name);
// Case 3: bench beats the starter on projection too (proj 16 vs 14):
// no projection cost, pure ceiling gain swaps.
reset([mk('Steady', 14, 18, [mk('Stud', 16, 22)])]);
wvLabChaseUpside();
if (wvLabLastChanges.length !== 1 || wvLabLastChanges[0].inn !== 'Stud')
  throw new Error('projection-upgrade swap blocked: ' + JSON.stringify(wvLabLastChanges));
console.log('CHASE_UPSIDE_NET_GAIN_OK');
""")
    assert "CHASE_UPSIDE_NET_GAIN_OK" in out
