"""Class C audit: staged-state parity for URL/query deep links.

Every URL staging path must register its values in the SAME state store
and run the SAME render/recalc funnel as the equivalent interactive
selection. These tests pin the divergences found in the audit:

1. Advanced Metrics: a state-bearing deep link (?metric=&pos=&team=,
   no ?preset=) was run over by the default landing preset, so the
   staged metric was replaced by the preset primary and the URL was
   rewritten; ?team= was additionally dead on arrival because the
   state literal carried a duplicate `team` key whose empty second
   value won. The ?graph= share path set the axis selects by bare
   .value assignment, leaving the search-picker labels (which sync on
   'change') naming the default metrics.
2. Compare: a lone ?p1= deep link filled slot 1 but never ran
   _renderSuggestions(), the registration step an interactive pick
   runs, so the tier-benchmark / similar-player suggestions for slot 2
   never appeared (the code comment promised they would).
3. Schedule Assistant: ?add= unshifted into selPids but never
   persisted the list, unlike an interactive add (persist()).
4. Trade Database: ?q= resolved against the limit=100 starter pool
   instead of the server-side search an interactive pick uses, so a
   name outside that pool silently staged nothing.
"""
import subprocess
import tempfile
import os

import pytest

from dashboard_services.pages.advanced_metrics_page import _AM_JS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _run_node(js):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js)
        path = f.name
    try:
        out = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    assert out.returncode == 0, f"node harness failed: {out.stderr[-2000:]}"
    return out.stdout


# ── Advanced Metrics ─────────────────────────────────────────────────────────

def test_am_state_literal_registers_team_param_once():
    start = _AM_JS.index("const state = {")
    end = _AM_JS.index("// Seven displayed metrics", start)
    literal = _AM_JS[start:end]
    assert literal.count("team:") == 1, literal
    assert "team: _initParams.get('team') || ''" in literal


def test_am_state_literal_behavioral_team_and_pos():
    start = _AM_JS.index("const state = {")
    end = _AM_JS.index("// Seven displayed metrics", start)
    literal = _AM_JS[start:end]
    out = _run_node(
        "var metricSel = { value: 'target_share' };\n"
        "var seasonSel = null;\n"
        "var _initParams = new URLSearchParams('?metric=target_share&pos=WR&team=KC');\n"
        "function _loadPins() { return new Set(); }\n"
        + literal
        + "\nif (state.team !== 'KC') throw new Error('team not staged: ' + JSON.stringify(state.team));"
        + "\nif (state.position !== 'WR') throw new Error('pos not staged');"
        + "\nif (state.metric !== 'target_share') throw new Error('metric not staged');"
        + "\nconsole.log('ok');"
    )
    assert "ok" in out


def test_am_deep_link_is_not_run_over_by_landing_preset():
    # Behavioral: run the shipped init chain (URL-state gate, persisted
    # view restore, landing fallback) with stubbed dependencies.
    start = _AM_JS.index("const _AM_URL_STATE_KEYS")
    end = _AM_JS.index("updateSortBtn(); updatePosButtons();", start)
    chain = _AM_JS[start:end]

    def run(search, label):
        prelude = (
            "var calls = [];\n"
            "var state = { metric: 'target_share', position: 'WR', season: '' };\n"
            "var metricSel = { value: 'target_share' };\n"
            "var searchEl = null;\n"
            "var cfg = { hasPremium: true, metrics: { target_share: {} } };\n"
            "var _PRESETS = { key_metrics: { metrics: ['expected_ppr_per_game'] },"
            " receiving: { metrics: ['target_share'] } };\n"
            "var _initParams = new URLSearchParams(" + repr(search) + ");\n"
            "var localStorage = { getItem: function() { return null; }, setItem: function() {} };\n"
            "function _loadAmView() { return null; }\n"
            "function defaultVol() { return ''; }\n"
            "function amLoadPreset(id) { calls.push(id); }\n"
        )
        return _run_node(
            prelude + chain
            + "\nconsole.log(JSON.stringify({ calls: calls, loaded: _presetLoaded }));"
        )

    import json

    # Assistant-style deep link: staged board must survive untouched
    # (no preset load; _presetLoaded stays false so fetchData() runs
    # with the URL-seeded state).
    out = json.loads(run("?metric=target_share&pos=WR&team=KC", "deep"))
    assert out == {"calls": [], "loaded": False}, out
    # Plain load still lands on the default preset.
    out = json.loads(run("", "plain"))
    assert out == {"calls": ["key_metrics"], "loaded": True}, out
    # An explicit ?preset= still wins.
    out = json.loads(run("?preset=receiving", "preset"))
    assert out == {"calls": ["receiving"], "loaded": True}, out


def test_am_graph_deep_link_syncs_picker_labels():
    start = _AM_JS.index("window.addEventListener('load', function() {")
    end = _AM_JS.index("window.amRenderGraph();", start)
    block = _AM_JS[start:end]
    assert "xSel.value = _gxInit" in block
    # Labels sync on 'change' (amInitSearchPicker); the staged values
    # must dispatch it, the way amOpenGraph does for its defaults.
    assert "dispatchEvent(new Event('change'))" in block


# ── Compare ──────────────────────────────────────────────────────────────────

def _compare_harness(search):
    app = _read("static/app.js")
    start = app.index("function initComparePage()")
    end = app.index("function openCompareSearch(", start)
    fn = app[start:end]
    prelude = """
function el(id) {
  return { id: id, value: '', hidden: false, innerHTML: '', style: {}, dataset: {},
    classList: { add: function() {}, remove: function() {}, toggle: function() {} },
    setAttribute: function() {}, getAttribute: function() { return null; },
    addEventListener: function() {}, focus: function() {},
    querySelectorAll: function() { return []; }, querySelector: function() { return null; },
    insertAdjacentHTML: function() {}, closest: function() { return null; } };
}
var _els = {};
var document = {
  querySelector: function(sel) {
    if (sel === '[data-page="compare"]') {
      return { querySelectorAll: function() { return []; },
               querySelector: function() { return null; } };
    }
    return null;
  },
  getElementById: function(id) { return _els[id] || (_els[id] = el(id)); },
  addEventListener: function() {},
};
var window = { location: { search: SEARCH, pathname: '/compare',
  href: 'https://x/compare' + SEARCH }, brErrorState: function() {} };
var location = window.location;
var localStorage = { getItem: function() { return null; }, setItem: function() {} };
var history = { replaceState: function() {} };
var fetches = [];
var fetch = function(url) {
  fetches.push(String(url));
  var payload = {};
  var m = String(url).match(/details\\/(.+)$/);
  if (m) payload = { player_id: m[1], name: 'Player ' + m[1], position: 'WR', team: 'KC' };
  else if (String(url).includes('similar')) payload = { players: [] };
  else if (String(url).includes('trending')) payload = { players: [] };
  return Promise.resolve({ ok: true, json: function() { return Promise.resolve(payload); } });
};
function _cmpEnsureBaselines() { return Promise.resolve(); }
function _cmpIsBaseline() { return false; }
function _cmpMatchBaselines() { return []; }
var _cmpBaselineById = {
  'avg-WR-1': { player_id: 'avg-WR-1', name: 'Avg WR1', position: 'WR', stats: { pos_rank_label: 'WR1' } },
  'avg-WR-2': { player_id: 'avg-WR-2', name: 'Avg WR2', position: 'WR', stats: { pos_rank_label: 'WR2' } },
};
var _cmpBaselineList = [];
function _cmpPlayerDetailsUrl(pid) { return '/details/' + pid; }
function _cmpPlayersSearchUrl(q) { return '/search?q=' + q; }
var renderedPairs = [];
function renderCompareInline(d1, d2) { renderedPairs.push([String(d1.player_id), String(d2.player_id)]); }
function renderCompareTriple() {}
function _renderCompareSkeleton() {}
function _wlEsc(s) { return String(s == null ? '' : s); }
function _deferInit() {}
""".replace("SEARCH", repr(search))
    return prelude + fn


def test_compare_lone_p1_deep_link_renders_suggestions():
    js = _compare_harness("?p1=111") + """
initComparePage();
setTimeout(function() {
  var sug = document.getElementById('cmpSuggest2');
  if (document.getElementById('cmpPick1').value !== 'Player 111')
    throw new Error('slot 1 not staged');
  if (!sug.innerHTML.includes('Benchmark'))
    throw new Error('suggestions not rendered for staged slot 1: ' + sug.innerHTML);
  if (!fetches.some(function(u) { return u.includes('/api/compare/similar'); }))
    throw new Error('similar-player fetch (interactive suggestion funnel) never fired');
  console.log('ok');
}, 100);
"""
    assert "ok" in _run_node(js)


def test_compare_pair_deep_link_renders_and_stages_trade_link():
    js = _compare_harness("?p1=111&p2=222") + """
initComparePage();
setTimeout(function() {
  if (JSON.stringify(renderedPairs) !== JSON.stringify([['111', '222']]))
    throw new Error('pair not rendered via renderCompareInline: ' + JSON.stringify(renderedPairs));
  var href = document.getElementById('cmpTradeLink').href;
  if (href !== '/trade?a=111&b=222')
    throw new Error('trade link not staged from the pair: ' + href);
  console.log('ok');
}, 100);
"""
    assert "ok" in _run_node(js)


# ── Schedule Assistant ───────────────────────────────────────────────────────

def test_schedule_url_add_persists_like_interactive_add():
    src = _read("dashboard_services/pages/schedule_page.py")
    start = src.index("var addPid = new URLSearchParams")
    block = src[start:src.index("} catch (e) {}", start)]
    assert "addedPids.unshift(addPid)" in block
    assert "persistAdded();" in block
    # The saved list both paths share, written by persistAdded():
    assert "localStorage.setItem(LS_ADDED, JSON.stringify(addedPids))" in src
    # The interactive path it must match (removing a staged player):
    assert "persistAdded(); render();" in src


# ── Trade Database ───────────────────────────────────────────────────────────

def test_trade_database_q_resolves_via_interactive_search():
    src = _read("routes/trade_bp.py")
    start = src.index("if (initQ) {{")
    block = src[start:src.index("}});", start)]
    assert "tdbSearchPlayers(initQ)" in block
    # The limit=100 starter pool is only the empty-search fallback now.
    assert block.index("tdbSearchPlayers(initQ)") < block.index("ensureTDBPlayers()")
    # Registration + recalc match the interactive dropdown pick.
    assert "selectedA = [{{ id: String(match.player_id), name: match.name }}]" in block
    assert "renderTDBChips('A')" in block
    assert "loadTDBPage(1)" in block
