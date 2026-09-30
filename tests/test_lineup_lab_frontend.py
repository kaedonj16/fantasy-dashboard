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
               "function wvLabToggleSlot(", "function wvLabSwap(", "function wvLabChaseUpside(",
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


def test_lab_retry_card_when_opponent_missing(script):
    # A payload whose opponent failed to load must render an explicit retry
    # state, never the degenerate 50/50 hero.
    assert "data.opponent && data.opponent.missing" in script
    assert "no matchup to simulate" in script
    assert 'onclick="wvLabData=null;wvLoadLab()">Try again' in script


def _lab_engine_js(script):
    start = script.index("// ── Lineup Lab (Start/Sit tab) ──")
    end = script.index("if (!window.__brctx)", start)
    return script[start:end]


def _run_node(js, body):
    pytest.importorskip("subprocess")
    import subprocess, tempfile, os
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


def test_lab_injury_replacement_wiring(script):
    # In-game injury must replace the starter with the best eligible bench
    # player's draw — never a zero. Single RB starter + one bench RB:
    # onset 1.0 forces the injury every sim, so the median must track the
    # bench (~8); with injuries off it must track the starter (~20).
    js = _lab_engine_js(script)
    out = _run_node(js, """
var WV_LAB_SIMS = 2000;
wvLabData = { corr: {} };
wvLabInjuryOnset = {QB:0, RB:1, WR:1, TE:1, K:0, DEF:0};
function mkE(pid, pos, mean, std, bench) {
  return { player_id: pid, pos: pos, slot: pos,
    profile: { mean: mean, std: std, skew_alpha: 2, dud_risk: 0 },
    bench: bench || [] };
}
var benchRB = mkE('b1', 'RB', 8, 1, []);
var lineup = [mkE('s1', 'RB', 20, 1, [benchRB])];
wvLabBase = wvLabBuildBase(['s1','b1'], WV_LAB_SIMS, 7);
wvLabOppDraws = new Float64Array(WV_LAB_SIMS);
var hurt = wvLabEvaluate(lineup);
if (!(hurt.median > 6 && hurt.median < 10))
  throw new Error('injured median should track bench ~8, got ' + hurt.median);
wvLabInjuryOnset = {QB:0, RB:0, WR:0, TE:0, K:0, DEF:0};
var healthy = wvLabEvaluate(lineup);
if (!(healthy.median > 18 && healthy.median < 22))
  throw new Error('healthy median should track starter ~20, got ' + healthy.median);
// Empty bench: waiver-wire replacement (~45% of the starter), not a zero.
var lone = [mkE('s2', 'RB', 20, 1, [])];
wvLabBase = wvLabBuildBase(['s2'], WV_LAB_SIMS, 7);
wvLabInjuryOnset = {QB:0, RB:1, WR:0, TE:0, K:0, DEF:0};
var wire = wvLabEvaluate(lone);
if (!(wire.median > 7 && wire.median < 11))
  throw new Error('waiver-wire median should be ~9 (45% of 20), got ' + wire.median);
console.log('LAB_INJURY_WIRING_OK');
""")
    assert "LAB_INJURY_WIRING_OK" in out


def test_lab_injury_swap_determinism(script):
    # At real (fallback) injury rates, rebuilding the base with the same
    # seed must reproduce the evaluate output and every swap delta exactly —
    # the injury draws are part of the common random numbers.
    js = _lab_engine_js(script)
    out = _run_node(js, """
var WV_LAB_SIMS = 2000;
wvLabData = { corr: {} };
function mkE(pid, pos, mean, std, bench) {
  return { player_id: pid, pos: pos, slot: pos,
    profile: { mean: mean, std: std, skew_alpha: 2, dud_risk: 0.05 },
    bench: bench || [] };
}
var bench = [mkE('b1','RB',8,4,[]), mkE('b2','WR',7,4,[])];
var lineup = [
  mkE('q1','QB',20,9,[]), mkE('r1','RB',15,8,bench), mkE('r2','RB',14,8,bench),
  mkE('w1','WR',14,7,bench), mkE('w2','WR',12,7,bench), mkE('t1','TE',9,5,[]),
  mkE('f1','WR',11,7,bench), mkE('k1','K',8,3,[]), mkE('d1','DEF',9,4,[])
];
var pids = ['q1','r1','r2','w1','w2','t1','f1','k1','d1','b1','b2'];
wvLabOppDraws = new Float64Array(WV_LAB_SIMS);
for (var s = 0; s < WV_LAB_SIMS; s++) wvLabOppDraws[s] = 110;
function trial() {
  var t = lineup.slice();
  t[3] = mkE('b2','WR',7,4,[]); t[3].bench = bench;
  return t;
}
wvLabBase = wvLabBuildBase(pids, WV_LAB_SIMS, 99);
var r1 = wvLabEvaluate(lineup);
var d1 = wvLabEvaluate(trial()).winPct - r1.winPct;
wvLabBase = wvLabBuildBase(pids, WV_LAB_SIMS, 99);
var r2 = wvLabEvaluate(lineup);
var d2 = wvLabEvaluate(trial()).winPct - r2.winPct;
if (JSON.stringify(r1) !== JSON.stringify(r2))
  throw new Error('evaluate not deterministic across identical rebuilds');
if (d1 !== d2) throw new Error('swap delta not deterministic: ' + d1 + ' vs ' + d2);
console.log('LAB_INJURY_DETERMINISM_OK');
""")
    assert "LAB_INJURY_DETERMINISM_OK" in out


def test_lab_injury_rate_shifts_distribution(script):
    # Sanity on the injury draw itself: with onset 0.5 on every starter the
    # team median must sit clearly below the no-injury median, and the base
    # injury uniforms must hit the configured rate.
    js = _lab_engine_js(script)
    out = _run_node(js, """
var WV_LAB_SIMS = 4000;
wvLabData = { corr: {} };
function mkE(pid, pos, mean, std, bench) {
  return { player_id: pid, pos: pos, slot: pos,
    profile: { mean: mean, std: std, skew_alpha: 2, dud_risk: 0 },
    bench: bench || [] };
}
var bench = [mkE('b1','RB',6,2,[]), mkE('b2','WR',5,2,[])];
var lineup = [mkE('r1','RB',16,2,bench), mkE('w1','WR',14,2,bench), mkE('t1','TE',10,2,[])];
var pids = ['r1','w1','t1','b1','b2'];
wvLabBase = wvLabBuildBase(pids, WV_LAB_SIMS, 5);
wvLabOppDraws = new Float64Array(WV_LAB_SIMS);
var hits = 0;
for (var s = 0; s < WV_LAB_SIMS; s++) if (wvLabBase['r1'].inj[s] < 0.5) hits++;
var rate = hits / WV_LAB_SIMS;
if (!(rate > 0.45 && rate < 0.55)) throw new Error('inj uniform rate off: ' + rate);
wvLabInjuryOnset = {QB:0, RB:0.5, WR:0.5, TE:0.5, K:0, DEF:0};
var hurtMed = wvLabEvaluate(lineup).median;
wvLabInjuryOnset = {QB:0, RB:0, WR:0, TE:0, K:0, DEF:0};
var okMed = wvLabEvaluate(lineup).median;
if (!(hurtMed < okMed - 2)) throw new Error('injuries should drag the median: ' + hurtMed + ' vs ' + okMed);
console.log('LAB_INJURY_RATE_OK');
""")
    assert "LAB_INJURY_RATE_OK" in out


def test_lab_js_fallback_matches_python_table(script):
    # The JS fallback onset table must mirror the Python single source of
    # truth (data_building/injury_rates.py); the payload ships the same
    # values, so a drift here would silently retune the browser engine.
    from data_building.injury_rates import injury_onset_rate
    m = re.search(r"var WV_LAB_INJURY_ONSET_FALLBACK = (\{[^}]*\});", script)
    assert m, "fallback onset table not found in Lab script"
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        pm = re.search(pos + r"\s*:\s*([0-9.]+)", m.group(1))
        assert pm, f"{pos} missing from JS fallback table"
        assert float(pm.group(1)) == pytest.approx(
            round(injury_onset_rate(pos), 4), abs=1e-4
        ), f"{pos}: JS fallback {pm.group(1)} != Python table"


@pytest.fixture
def lab_payload(monkeypatch):
    # Minimal mocks to build a real Lab payload (mirrors the fixture in
    # test_lineup_lab.py): viewer roster 7 vs opponent roster 3.
    import dashboard_services.api as api_mod
    import data_building.lineup_lab as lab_mod
    import data_building.player_distributions as pd_mod
    import utils.fantasy_scoring as fs_mod
    import utils.utils as utils_mod

    PROJ = {"1": 22.5, "2": 18.0, "3": 16.9, "4": 14.2, "5": 9.0,
            "6": 12.0, "7": 3.0, "8": 8.0, "9": 22.5, "10": 15.0}
    POS = {"1": "QB", "2": "RB", "3": "WR", "4": "WR", "5": "TE",
           "6": "RB", "7": "K", "8": "DEF", "9": "QB", "10": "RB"}
    TEAM = {"1": "LAR", "2": "DET", "3": "LAR", "4": "CIN", "5": "KC",
            "6": "SF", "7": "DAL", "8": "BUF", "9": "BUF", "10": "MIA"}
    players_index = {
        pid: {"position": POS[pid], "pos": POS[pid], "team": TEAM[pid],
              "full_name": f"Player {pid}", "injury_status": ""}
        for pid in PROJ
    }
    ctx = {
        "current_week": 4,
        "rosters": [
            {"roster_id": 7, "owner_id": "u1",
             "players": ["1", "2", "3", "4", "5", "6", "7", "8"],
             "reserve": [], "taxi": []},
            {"roster_id": 3, "owner_id": "u2", "players": ["9", "10"],
             "reserve": [], "taxi": []},
        ],
        "users": [{"user_id": "u2", "display_name": "Pittsburgh Pilots"}],
        "players_index": players_index,
        "players": {},
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN"],
        "raw_scoring_settings": {},
        "scoring_settings": {},
    }

    def _matchups(league_id, week):
        return [
            {"roster_id": 7, "matchup_id": 1,
             "starters": ["1", "2", "3", "5", "6", "7", "8"],
             "players": ["1", "2", "3", "4", "5", "6", "7", "8"]},
            {"roster_id": 3, "matchup_id": 1,
             "starters": ["9", "10"], "players": ["9", "10"]},
        ]

    def _profiles(requests, season, week):
        out = {}
        for req in requests:
            pid = str(req["player_id"])
            mean = float(req["mean"])
            out[pid] = {
                "player_id": pid, "pos": req["pos"], "mean": mean,
                "std": round(2.0 + 0.42 * mean, 2), "skew_alpha": 2.0,
                "dud_risk": 0.0, "n_games": 3.0, "factors": {},
            }
        return out

    monkeypatch.setattr(api_mod, "get_matchups", _matchups)
    monkeypatch.setattr(pd_mod, "build_profiles", _profiles)
    monkeypatch.setattr(pd_mod, "correlation_pairs", lambda pids, season, ctx=None: {})
    monkeypatch.setattr(utils_mod, "load_week_projection", lambda season, week: {})
    monkeypatch.setattr(utils_mod, "load_week_sched", lambda season, week: [])
    monkeypatch.setattr(fs_mod, "weekly_projection_points",
                        lambda raw, pid, scoring, pos="": PROJ.get(str(pid)))
    return lab_mod.build_lineup_lab_payload(
        ctx=ctx, league_id="123", viewer_roster_id=7, season=2026, week=4)


def test_lab_payload_ships_injury_onset_and_opponent_haircut(lab_payload):
    from data_building.injury_rates import (
        expected_injury_loss_per_week, injury_onset_rate,
    )
    onset = lab_payload["injury_onset"]
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        assert onset[pos] == pytest.approx(round(injury_onset_rate(pos), 4))
    opp = lab_payload["opponent"]
    # Opp starters are pid 9 (QB 22.5) and pid 10 (RB 15.0): the haircut is
    # the expected injury loss, and the shipped mean is net of it.
    adj = (expected_injury_loss_per_week(22.5, "QB")
           + expected_injury_loss_per_week(15.0, "RB"))
    assert opp["injury_adj"] == pytest.approx(round(adj, 1))
    assert opp["mean"] == pytest.approx(round(37.5 - adj, 1))
    assert opp["injury_adj"] > 0


def test_lab_prefetch_wiring(script):
    # One shared URL builder: prefetch and toggle load request the same thing.
    assert "function wvLabUrl(" in script
    assert "function wvLabKey(" in script
    assert "function wvPrefetchLab(" in script
    start = script.index("function wvPrefetchLab(")
    body = script[start:script.index("\nfunction ", start + 1)]
    assert "fetch(wvLabUrl())" in body
    assert "wvLabPrefetch = { key: key, promise: p }" in body
    # Failures clear the stash silently (no UI side effects in the catcher).
    assert "wvLabPrefetch = null" in body
    # The toggle load consumes a matching stash instead of fetching again.
    start = script.index("function wvLoadLab(")
    body = script[start:script.index("\nfunction ", start + 1)]
    assert "wvLabPrefetch.key === wvLabKey()" in body
    assert "req = wvLabPrefetch.promise" in body
    assert "var url = wvLabUrl();" in body
    assert "req = fetch(url).then" in body
    # The prefetch fires from the Start/Sit success path, right after the
    # data (and its current_week) lands — the same week the Lab load reads.
    start = script.index("function wvFetchStartSit(")
    body = script[start:script.index("\nfunction ", start + 1)]
    assert body.index("wvStartSitData = d;") < body.index("wvPrefetchLab();")


def test_lab_prefetch_fires_and_toggle_consumes(script):
    # Behavioral, under node: the prefetch fires once with the exact Lab
    # URL; the toggle load consumes the stash (no second fetch) and renders
    # from it; a week switch ignores the stash; a failed prefetch is silent
    # and the real load fetches fresh with its normal behavior.
    js = _lab_engine_js(script)
    out = _run_node(js, """
var WV_PLATFORM = 'sleeper';
var WV_LEAGUE_ID = '12345';
var WV_SEASON = 2026;
var wvStartSitData = { current_week: 4 };
var fetchCalls = [];
var fetchMode = 'ok';
var payload = { state: 'needs_team', message: 'pick a team first' };
function fetch(url) {
  fetchCalls.push(url);
  if (fetchMode === 'fail') return Promise.reject(new Error('boom'));
  return Promise.resolve({ json: function() { return Promise.resolve(payload); } });
}
var bodyEl = { innerHTML: '' };
var document = { getElementById: function(id) { return id === 'wvLabBody' ? bodyEl : null; } };
function flush(cb) { setTimeout(cb, 20); }
var want = '/api/lineup-lab?platform=sleeper&league_id=12345&season=2026&week=4';
wvPrefetchLab();
if (fetchCalls.length !== 1)
  throw new Error('prefetch should fire exactly one fetch, got ' + fetchCalls.length);
if (fetchCalls[0] !== want) throw new Error('prefetch URL mismatch: ' + fetchCalls[0]);
if (wvLabUrl() !== want) throw new Error('wvLabUrl mismatch: ' + wvLabUrl());
flush(function() {
  wvLoadLab();
  flush(function() {
    if (fetchCalls.length !== 1)
      throw new Error('toggle must consume the stash, fetches=' + fetchCalls.length);
    if (bodyEl.innerHTML.indexOf('pick a team first') < 0)
      throw new Error('stashed payload not rendered: ' + bodyEl.innerHTML);
    if (wvLabPrefetch !== null) throw new Error('stash must be consumed by the load');
    wvStartSitData = { current_week: 5 };
    payload = { state: 'needs_team', message: 'week five' };
    wvPrefetchLab();
    if (fetchCalls.length !== 2) throw new Error('week-5 prefetch should fetch');
    if (fetchCalls[1].indexOf('week=5') < 0)
      throw new Error('week-5 URL wrong: ' + fetchCalls[1]);
    wvStartSitData = { current_week: 6 };
    wvLoadLab();
    flush(function() {
      if (fetchCalls.length !== 3)
        throw new Error('mismatched week must fetch fresh, fetches=' + fetchCalls.length);
      if (fetchCalls[2].indexOf('week=6') < 0)
        throw new Error('week-6 URL wrong: ' + fetchCalls[2]);
      fetchMode = 'fail';
      wvStartSitData = { current_week: 7 };
      wvPrefetchLab();
      flush(function() {
        if (wvLabPrefetch !== null)
          throw new Error('failed prefetch must clear the stash');
        fetchMode = 'ok';
        payload = { state: 'needs_team', message: 'recovered' };
        var before = fetchCalls.length;
        wvLoadLab();
        flush(function() {
          if (fetchCalls.length !== before + 1)
            throw new Error('load after a failed prefetch must fetch fresh');
          if (bodyEl.innerHTML.indexOf('recovered') < 0)
            throw new Error('fresh load not rendered: ' + bodyEl.innerHTML);
          console.log('LAB_PREFETCH_OK');
        });
      });
    });
  });
});
""")
    assert "LAB_PREFETCH_OK" in out
