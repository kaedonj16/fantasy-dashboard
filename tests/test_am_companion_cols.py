"""Companion stat columns + custom-filter load states (Kaedon, 2026-11-06).

Job 1: the stat a rate divides by is always visible beside the G column.
Per-snap rates sit beside Snaps (their denominator), not the numerator the
old category heuristic picked (targets for Receiving, carries for Rushing),
and General-category per-game rates got no companion at all. touches_per_game
sits beside Touches; First Down Rate (targets + carries denominator) shows
both. A metric that arrives as a custom FILTER column brings its companion
too, not just added compare columns. Payloads carry the new context fields:
season rows emit snp/tch from total_snaps/total_touches, weekly-range rows
emit snp/tch from SUM(snaps)/SUM(touches).

Job 2: adding touches/g >= 5 as a custom filter showed "No players pass
these filters" even though many players pass. The combo-filter stage reads
state.extraData, which loads async: until the filter key's data lands its
value reads null for every player, so the board collapses to the
filter-empty verdict before the filter has been evaluated at all, and if
both fetch attempts fail it stays there permanently (the failed column's
retry lives inside the table the empty state hides). render() now reports
the load state honestly: pending while in flight, failed (with a working
Retry) when the fetch errored. A settled filter that genuinely matches no
one keeps the honest verdict.
"""
import contextlib
import json
import os
import re
import shutil
import subprocess

import pytest

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _page_js():
    html = _html()
    return html[html.index("<script>"):html.index("</script>")]


def _cfg():
    m = re.search(r"const AM_CFG = (\{.*?\});", _html(), re.S)
    return json.loads(m.group(1))


def _extract_fn(js, name):
    m = re.search(r"function %s\([^)]*\) \{.*?\n  \}" % re.escape(name), js, re.S)
    assert m, f"{name} not found in page script"
    return m.group(0)


def _extract_fn_opt(js, name):
    m = re.search(r"function %s\([^)]*\) \{.*?\n  \}" % re.escape(name), js, re.S)
    return m.group(0) if m else ""


def _run_node(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr or proc.stdout


# ── Job 1a: cfg sample mapping is denominator-aware ────────────────────

def test_per_snap_metrics_map_to_snaps_not_the_numerator():
    metrics = _cfg()["metrics"]
    for key in ("pts_per_snap", "yards_per_snap",
                "targets_per_snap", "touches_per_snap"):
        assert metrics[key]["sample"] == "snaps", key


def test_touches_per_game_and_first_down_rate_samples():
    metrics = _cfg()["metrics"]
    assert metrics["touches_per_game"]["sample"] == "touches"
    # First Down Rate divides by targets + carries: both are its context.
    assert metrics["first_down_rate"]["sample"] == ["targets", "carries"]


def test_category_fallback_samples_unchanged():
    metrics = _cfg()["metrics"]
    assert metrics["target_share"]["sample"] == "targets"        # Receiving
    assert metrics["carries_per_game"]["sample"] == "carries"    # Rushing
    assert metrics["epa_per_play"]["sample"] == "dropbacks"      # Passing set
    assert metrics["yards_after_catch_per_reception"]["sample"] == "receptions"


# ── Job 1b: payloads carry snp/tch ─────────────────────────────────────

def _weekly_rows(monkeypatch, metric="targets_per_snap"):
    import data_building.advanced_metrics as metrics

    captured = {}

    class Conn:
        def execute(self, sql, params=()):
            captured["sql"] = sql
            captured["params"] = params
            return self

        def fetchall(self):
            return [{
                "player_id": "p1", "position": "RB", "weeks_played": 8,
                "value": 0.15, "ctx_receptions": 30, "ctx_targets": 40,
                "ctx_carries": 50, "ctx_snaps": 280, "ctx_touches": 95,
                "_wvol": 280,
            }]

    @contextlib.contextmanager
    def conn():
        yield Conn()

    monkeypatch.setattr(metrics, "get_conn", conn)
    rows = metrics.get_weekly_range_leaderboard(
        metric, position="RB", season=2026,
        week_start=1, week_end=8, min_vol=50)
    return rows, captured


def test_weekly_payload_emits_snaps_and_touches(monkeypatch):
    rows, captured = _weekly_rows(monkeypatch)
    assert "SUM(snaps) AS ctx_snaps" in captured["sql"]
    assert "SUM(touches) AS ctx_touches" in captured["sql"]
    assert len(rows) == 1
    assert rows[0]["snp"] == 280
    assert rows[0]["tch"] == 95
    # Existing context fields ride along unchanged.
    assert rows[0]["tgt"] == 40 and rows[0]["car"] == 50


def test_season_payload_emits_snaps_and_touches(monkeypatch):
    import data_building.advanced_metrics as metrics

    cols = ["games", "total_receptions", "total_targets", "total_carries",
            "total_pass_att", "total_snaps", "total_touches", "completion_pct"]
    captured = []

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=()):
            captured.append(sql)
            self._sql = sql
            return self

        def fetchall(self):
            if "information_schema" in self._sql:
                return [{"column_name": c} for c in cols]
            return [{
                "player_id": "p1", "position": "RB", "games": 9,
                "value": 12.5, "ctx_receptions": 40, "ctx_targets": 55,
                "ctx_carries": 70, "ctx_attempts": None,
                "ctx_completions": None, "ctx_snaps": 500, "ctx_touches": 110,
            }]

        def fetchone(self):
            return {"x": 1}

    monkeypatch.setattr(metrics, "get_conn", lambda: Conn())
    metrics._METRIC_LEADERBOARD_CACHE.clear()
    rows = metrics.get_metric_leaderboard(
        "touches_per_game", position="RB", season=2026)
    assert len(rows) == 1
    assert rows[0]["snp"] == 500
    assert rows[0]["tch"] == 110
    assert rows[0]["tgt"] == 55 and rows[0]["car"] == 70
    main_sql = next(s for s in captured if "DISTINCT ON" in s)
    assert "ctx_snaps" in main_sql and "total_snaps" in main_sql
    assert "ctx_touches" in main_sql and "total_touches" in main_sql


# ── Job 1c: contextColsFor / table schema (node) ───────────────────────

_CTX_FNS = ("amParseSeasons", "amSelectedSeasons", "amIsMultiSeason",
            "amIsCombine", "amIsEachYear", "amRowKey",
            "_mLabel", "_mLocked", "contextColsFor", "contextValue",
            "schemaValue", "tableColumnSchema")

_CTX_PRELUDE = r"""
var cfg = { hasPremium: true, proMetricInfo: {}, metrics: {
  pts_per_snap: { label: 'Pts/Snap', sample: 'snaps' },
  targets_per_snap: { label: 'Targets/Snap', sample: 'snaps' },
  touches_per_game: { label: 'Touches/G', sample: 'touches' },
  first_down_rate: { label: 'First Down Rate', sample: ['targets', 'carries'] },
  target_share: { label: 'Target Share', sample: 'targets' },
} };
var _activePresetId = null, _PRESETS = {};
function _row(over) {
  return Object.assign({ player_id: '1', name: 'P', position: 'RB', games: 9,
    value: 1, season: null, tgt: 40, car: 30, rec: 25, att: 0, db: 0,
    snp: 300, tch: 75 }, over || {});
}
function _state(over) {
  return Object.assign({ metric: 'pts_per_snap', season: '2026', combine: false,
    extraMetrics: [], extraData: {}, filterColKeys: new Set(),
    rows: [_row()], showTrends: false }, over || {});
}
var state = _state();
function check(name, cond) { if (!cond) { console.error('FAIL ' + name); process.exit(1); } }
function ids() { return tableColumnSchema().map(function(c) { return c.id; }); }
"""


def _ctx_script(tail):
    js = _page_js()
    fns = "\n".join(_extract_fn(js, n) for n in _CTX_FNS)
    return _CTX_PRELUDE + fns + "\n" + tail


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_snaps_column_sits_beside_games_for_pts_per_snap():
    script = _ctx_script(r"""
var got = ids();
check('snaps sample column present', got.indexOf('sample:snaps') !== -1);
check('snaps directly beside the G/Wks block',
  got.indexOf('sample:snaps') === got.indexOf('weeks') + 1);
var col = tableColumnSchema().find(function(c) { return c.id === 'sample:snaps'; });
check('snaps column reads the snp row field', col.key === 'snp' && col.label === 'Snaps');
check('context value resolves', contextValue(state.rows[0], 'snp') === 300);
""")
    _run_node(script)


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_targets_per_snap_gets_snaps_not_just_targets():
    script = _ctx_script(r"""
state = _state({ metric: 'targets_per_snap' });
var got = ids();
check('snaps column present for targets/snap', got.indexOf('sample:snaps') !== -1);
check('no duplicate targets sample column', got.indexOf('sample:targets') === -1);
""")
    _run_node(script)


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_touches_column_for_touches_per_game_and_first_down_rate_pair():
    script = _ctx_script(r"""
state = _state({ metric: 'touches_per_game' });
var got = ids();
check('touches column present', got.indexOf('sample:touches') !== -1);
var col = tableColumnSchema().find(function(c) { return c.id === 'sample:touches'; });
check('touches column reads the tch row field', col.key === 'tch' && col.label === 'Touches');
state = _state({ metric: 'first_down_rate' });
got = ids();
check('first down rate shows targets', got.indexOf('sample:targets') !== -1);
check('first down rate shows carries', got.indexOf('sample:carries') !== -1);
""")
    _run_node(script)


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_extra_column_and_filter_column_bring_their_companions():
    script = _ctx_script(r"""
// Added compare column: pts_per_snap as an extra on a target_share board.
// Primary rows carry no snaps; the extra column's payload context does.
state = _state({ metric: 'target_share', rows: [_row({ snp: null })],
  extraMetrics: ['pts_per_snap'],
  extraData: { pts_per_snap: { byId: { '1': 0.4 },
    context: { '1': { snp: 300 } }, maxAbs: 1 } } });
var got = ids();
check('extra column brings snaps companion', got.indexOf('sample:snaps') !== -1);
check('primary still brings its targets companion', got.indexOf('sample:targets') !== -1);
check('context value resolves from extra payload', contextValue(state.rows[0], 'snp') === 300);

// Custom filter column: touches_per_game used only as a filter still
// surfaces its Touches companion beside G, next to the filter column.
state = _state({ metric: 'target_share', rows: [_row({ tch: null })],
  filterColKeys: new Set(['touches_per_game']),
  extraData: { touches_per_game: { byId: { '1': 8.3 },
    context: { '1': { tch: 75 } }, maxAbs: 10 } } });
got = ids();
check('filter column brings touches companion', got.indexOf('sample:touches') !== -1);
check('filter column itself still renders', got.indexOf('filter:touches_per_game') !== -1);
check('context value resolves from filter payload', contextValue(state.rows[0], 'tch') === 75);
""")
    _run_node(script)


def test_extra_data_context_mapping_carries_snp_tch():
    """Both extra-data load paths (single + batch) must carry the new
    context fields off the payload rows, or companion columns sourced
    from a column payload stay blank."""
    js = _page_js()
    assert js.count("snp:r.snp, tch:r.tch") == 2


# ── Job 2: custom filter load states (node) ────────────────────────────

_JOB2_FNS = ("amParseSeasons", "amSelectedSeasons", "amIsMultiSeason",
             "amIsCombine", "amIsEachYear", "amRowKey", "amSeasonLabel",
             "_amCmpFetch", "defaultVol", "availableWeeks",
             "isMetricWeeklyCapable", "resolveWeekRange", "fetchExtraData",
             "_amMissing", "_amCmpVal", "contextColsFor", "contextValue",
             "schemaValue", "_posList", "_posLabel", "_mLabel", "_mLocked",
             "relevantPositions", "posColor", "fmtVal",
             "_amMinUnitWord", "_amMinChipLabel", "_amWeekChipLabel",
             "_amCustomChipLabel", "amActiveFilterChips", "_amEsc",
             "_amLowerMinOption", "amRenderFilterEmpty", "render")
_JOB2_NEW_FNS = ("_amFilterLoadState", "amRenderFilterLoad")

_JOB2_PRELUDE = r"""
var _els = {};
function _mkEl() {
  return { style: {}, innerHTML: '', textContent: '', hidden: false,
    classList: { add(){}, remove(){}, toggle(){} },
    setAttribute(){}, getAttribute(){ return null; }, append(){}, appendChild(){},
    addEventListener(){}, querySelector(){ return null; }, querySelectorAll(){ return []; } };
}
var loading = _mkEl(), empty = _mkEl(), tbody = _mkEl(), tableWrap = _mkEl(),
    avgNote = _mkEl(), avgNoteTxt = _mkEl(), paginationEl = _mkEl(),
    minGamesSel = { value: '', options: [{value:''}] }, volLabel = { textContent: 'Min' };
var document = {
  getElementById: function(id) { return _els[id] || (_els[id] = _mkEl()); },
  querySelector: function() { return null; },
  querySelectorAll: function() { return []; },
  createElement: function() { return _mkEl(); },
  createTextNode: function(t) { return { textContent: t }; },
  addEventListener: function() {},
  documentElement: { getAttribute: function() { return null; } },
};
var window = { brEmptyState: function(el, opts) { el.innerHTML = 'EMPTY:' + (opts && opts.title); },
               brTableSortSwap: null, __amAvgOpen: false };
var localStorage = { getItem: function() { return null; }, setItem: function() {} };
function percentileBadge() { return ''; }
function trendArrow() { return ''; }
function amTeamLabel() { return ''; }
function _weeklyChipHtml() { return ''; }
function trendCellHtml() { return '<td></td>'; }
function verifyRenderedSchema() {}
function updateComparePinnedBtn() {}
var _PIN_SVG = '', _LOCK_SVG = '', _AM_MAX_PINS = 5, PAGE_SIZE = 25;
var _activePresetId = null, _PRESETS = {};
var ownedIds = new Set();
var _primaryRows = [], _extraRows = [];
(function buildPayloads() {
  var pos = ['RB','WR','TE','QB'];
  for (var i = 0; i < 40; i++) {
    var id = String(4000 + i);
    _primaryRows.push({ player_id: id, name: 'Player ' + i, position: pos[i % 4],
      team: 'KC', games: 9, value: 0.30 - i * 0.005, tgt: 40, car: 30, rec: 25,
      att: 0, db: 0, snp: 300, tch: 75 });
    _extraRows.push({ player_id: id, name: 'Player ' + i, position: pos[i % 4],
      team: 'KC', games: 9, value: (i % 2 === 0) ? 6.5 : 3.25, tgt: 40, car: 30,
      rec: 25, att: 0, db: 0, snp: 300, tch: 75 });
  }
})();
var cfg = { platform: 'sleeper', leagueId: '1312067280816832512', hasPremium: true,
  seasons: [2026, 2025], availableWeeksBySeason: { '2026': [1,2,3,4,5,6,7,8,9] },
  proMetricInfo: {},
  metrics: {
    pts_per_snap: { label: 'Pts/Snap', positions: ['QB','RB','WR','TE'], category: 'General', sample: 'snaps', weeklyCapable: false },
    touches_per_game: { label: 'Touches/G', positions: ['RB','WR','TE'], category: 'General', sample: 'touches', weeklyCapable: true },
  } };
var state = { metric: 'pts_per_snap', season: '2026', combine: false,
  rows: _primaryRows, extraMetrics: [], extraData: {}, extraPrevData: {}, prevData: {},
  playerPos: {}, comboFilters: [], filterColKeys: new Set(), pinnedIds: new Set(),
  requestToken: 1, fetching: false, sortBy: 'pts_per_snap', sortDir: 'desc', page: 0,
  position: 'ALL', search: '', team: '', ageMin: '', ageMax: '', rosterOnly: false,
  showTrends: false, weekRange: '', weekStart: null, weekEnd: null, minVol: '',
  responseWeekFiltered: false, volCol: 'games', trendsBySeason: {}, trendsFailed: {} };
function check(name, cond) { if (!cond) { console.error('FAIL ' + name); process.exit(1); } }
function _applyFilter() {
  // The filterApply handler's exact sequence for a metric key.
  var fkey = 'touches_per_game';
  state.comboFilters.push({ key: fkey, op: 'gte', val: '5' });
  var p = null;
  if (fkey !== 'primary' && fkey !== 'age' && cfg.metrics[fkey]) {
    if (!state.extraMetrics.includes(fkey) && (!state.filterColKeys || !state.filterColKeys.has(fkey))) {
      if (!state.filterColKeys) state.filterColKeys = new Set();
      state.filterColKeys.add(fkey);
      p = fetchExtraData(fkey);
    }
  }
  state.page = 0;
  render();
  return p;
}
"""


def _job2_script(fetch_impl, tail):
    js = _page_js()
    fns = "\n".join(_extract_fn(js, n) for n in _JOB2_FNS)
    fns += "\n" + "\n".join(_extract_fn_opt(js, n) for n in _JOB2_NEW_FNS)
    return _JOB2_PRELUDE + fetch_impl + "\n" + fns + "\n" + tail


_OK_FETCH = r"""
function fetch(url) {
  var payload = { players: [] };
  if (url.indexOf('metric=touches_per_game') >= 0) payload = { players: _extraRows };
  return Promise.resolve({ ok: true, status: 200,
    json: function() { return Promise.resolve(payload); } });
}
"""


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_filter_add_shows_loading_not_a_false_verdict_while_fetch_is_in_flight():
    """Regression: with the extra fetch unsettled, every row's filter value
    reads null and the board used to settle into 'No players pass these
    filters' even though the filter had not been evaluated yet."""
    script = _job2_script(
        "function fetch(url) { return new Promise(function() {}); }\n",
        r"""
render();
check('board renders before the filter', state._exportRows.length === 40);
_applyFilter();
check('empty state is shown while loading', empty.style.display !== 'none');
check('loading copy, not the pass/fail verdict',
  empty.innerHTML.indexOf('Loading filter data') !== -1);
check('verdict absent while in flight',
  empty.innerHTML.indexOf('No players pass these filters') === -1);
process.exit(0);
""")
    _run_node(script)


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_filter_add_reports_failure_with_retry_when_the_fetch_errors():
    """Regression: when both fetch attempts fail, the entry settles
    {failed:true}; the board used to keep claiming the filter matched no
    one, with the only retry hidden inside the (hidden) table."""
    script = _job2_script(
        "function fetch(url) { return Promise.resolve({ ok: false, status: 500,"
        " json: function() { return Promise.resolve({}); } }); }\n",
        r"""
(async function() {
  render();
  var p = _applyFilter();
  await p;
  await new Promise(function(r) { setTimeout(r, 300); });
  var ed = state.extraData['touches_per_game'];
  check('entry settled failed', !!(ed && ed.failed));
  check('failure copy shown', empty.innerHTML.indexOf('Filter data failed to load') !== -1);
  check('retry wired to the failed key',
    empty.innerHTML.indexOf("amRetryExtra('touches_per_game')") !== -1);
  check('verdict absent on failure',
    empty.innerHTML.indexOf('No players pass these filters') === -1);
  process.exit(0);
})();
""")
    _run_node(script)


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_settled_filter_that_matches_no_one_keeps_the_honest_verdict():
    script = _job2_script(
        r"""
function fetch(url) {
  var rows = _extraRows.map(function(r) { return Object.assign({}, r, { value: 3.25 }); });
  var payload = { players: [] };
  if (url.indexOf('metric=touches_per_game') >= 0) payload = { players: rows };
  return Promise.resolve({ ok: true, status: 200,
    json: function() { return Promise.resolve(payload); } });
}
""",
        r"""
(async function() {
  render();
  var p = _applyFilter();
  await p;
  await new Promise(function(r) { setTimeout(r, 50); });
  check('empty state shown for a genuine no-match', empty.style.display !== 'none');
  check('honest verdict shown for a genuine no-match',
    empty.innerHTML.indexOf('No players pass these filters') !== -1);
  process.exit(0);
})();
""")
    _run_node(script)


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_settled_filter_with_matches_renders_the_passing_rows():
    script = _job2_script(_OK_FETCH, r"""
(async function() {
  render();
  var p = _applyFilter();
  await p;
  await new Promise(function(r) { setTimeout(r, 50); });
  check('passing rows render once data lands', state._exportRows.length === 20);
  check('empty state hidden', empty.style.display === 'none');
  process.exit(0);
})();
""")
    _run_node(script)


def test_render_consults_filter_load_state_before_the_verdict():
    js = _page_js()
    assert "function _amFilterLoadState(" in js
    assert "function amRenderFilterLoad(" in js
    empty_branch = js.index("if (!displayRows.length) {")
    load_check = js.index("_fLoad.pending.length", empty_branch)
    verdict = js.index("amRenderFilterEmpty(empty)", empty_branch)
    assert load_check < verdict, \
        "the load-state branch must precede the filter-empty verdict"
