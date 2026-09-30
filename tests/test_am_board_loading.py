"""Advanced Metrics board loading reliability.

Regression tests for the board-emptying bug Kaedon hit live: re-applying
the Key Metrics preset could blank the whole table even though the
primary data was healthy. Three client defects compounded:

1. render()'s "hide rows that lack data in most extra columns" filter
   counted PRO-locked and failed extra columns as loaded. Those columns
   settle with an empty byId by construction, so they inflated the hit
   budget past what the free, healthy columns could ever reach and every
   row filtered out.
2. A failed extra column rendered as a plain dash (the failed flag was
   written by fetchExtraData but never read), indistinguishable from a
   player genuinely having no data, with no way to retry.
3. fetchData() rendered only inside Promise.all([main, prevSeason]), so
   a hung YoY fetch held the whole board on skeletons; and every repeat
   trigger (e.g. re-applying the active preset) tore the board down and
   re-fired the primary + all extra requests, the burst that made the
   columns time out in the first place.

Plus the server half: the leaderboard route swallowed builder
exceptions into HTTP 200 with players=[], which the client rendered as
the "No data yet" empty state for what was really a server fault.
"""
import json
import subprocess

from dashboard_services.pages.advanced_metrics_page import _AM_JS


def _extract_from(start):
    """Brace-match from an index into _AM_JS; return the balanced block."""
    brace = _AM_JS.index("{", start)
    depth = 0
    for i in range(brace, len(_AM_JS)):
        ch = _AM_JS[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _AM_JS[start : i + 1]
    raise AssertionError("unbalanced braces during extraction")


def _extract_fn(name):
    return _extract_from(_AM_JS.index(f"function {name}("))


def _extract_row_filter():
    """The 'hide sparse rows' block inside render(), anchored on its comment."""
    anchor = _AM_JS.index("Hide rows that lack data")
    start = _AM_JS.index("if (state.extraMetrics.length > 0)", anchor)
    return _extract_from(start)


_FETCH_DATA_JS = _extract_fn("fetchData")
_ROW_FILTER_JS = _extract_row_filter()


def _run_node(script):
    out = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# 1. Row filter: locked/failed columns must not consume the hit budget.
# ---------------------------------------------------------------------------

def _run_filter_scenario(extra_data, rows):
    keys = list(extra_data)
    script = (
        "var state = { extraMetrics: %s, extraData: %s };\n"
        "function amRowKey(r) { return String(r.player_id); }\n"
        "var displayRows = %s;\n"
        "%s\n"
        "console.log(JSON.stringify(displayRows.map(function(r) {\n"
        "  return r.player_id; })));\n"
        % (
            json.dumps(keys),
            json.dumps(extra_data),
            json.dumps(rows),
            _ROW_FILTER_JS,
        )
    )
    return _run_node(script)


def test_row_filter_ignores_locked_and_failed_columns():
    # A logged-out Key Metrics style board: 8 extras, 3 PRO-locked, 1
    # failed under load, 4 healthy free columns. The QB has values in 2
    # of the 4 healthy columns, the RB in all 4. Counting the dead
    # columns (old behavior) demands 4 hits, so the QB vanishes; once a
    # second free column fails the whole board empties.
    extra_data = {
        "pro_a": {"byId": {}, "maxAbs": 1, "proLocked": True},
        "pro_b": {"byId": {}, "maxAbs": 1, "proLocked": True},
        "pro_c": {"byId": {}, "maxAbs": 1, "proLocked": True},
        "fail_a": {"byId": {}, "maxAbs": 1, "failed": True},
        "opp": {"byId": {"qb1": 1, "rb1": 1}},
        "snap": {"byId": {"qb1": 1, "rb1": 1}},
        "rz": {"byId": {"rb1": 1}},
        "tgt": {"byId": {"rb1": 1}},
    }
    rows = [{"player_id": "qb1"}, {"player_id": "rb1"}]
    assert _run_filter_scenario(extra_data, rows) == ["qb1", "rb1"]


def test_row_filter_still_hides_genuinely_sparse_rows():
    # Guard: the filter itself is wanted (a fullback with dashes across
    # rushing-efficiency columns should not clutter the board). With 4
    # healthy columns, a row with a value in only 1 is still hidden.
    extra_data = {
        "a": {"byId": {"fb1": 1, "wr1": 1}},
        "b": {"byId": {"wr1": 1}},
        "c": {"byId": {"wr1": 1}},
        "d": {"byId": {}},
    }
    rows = [{"player_id": "fb1"}, {"player_id": "wr1"}]
    assert _run_filter_scenario(extra_data, rows) == ["wr1"]


# ---------------------------------------------------------------------------
# 2. fetchData: YoY must not gate the board; identical reloads are no-ops.
# ---------------------------------------------------------------------------

_FETCH_PREAMBLE = (
    "var window = {};\n"
    "var document = { getElementById: function() { return null; } };\n"
    "var cfg = { platform: 'sleeper', leagueId: '', seasons: [2026, 2025],\n"
    "  metrics: { expected_ppr_per_game: { weeklyCapable: true } } };\n"
    "var state = { metric: 'expected_ppr_per_game', season: '2026',\n"
    "  combine: false, minVol: '', rows: [], extraMetrics: ['target_share'],\n"
    "  extraData: {}, extraPrevData: {}, prevData: {}, playerPos: {},\n"
    "  filterColKeys: new Set(), requestToken: 0, fetching: false,\n"
    "  _boardSig: null, volCol: 'games', page: 0, responseWeekFiltered: false };\n"
    "var loading = { style: {} }, empty = { style: {}, innerHTML: '' },\n"
    "  avgNote = { style: {} }, tbody = { style: {}, innerHTML: '' },\n"
    "  tableWrap = { style: {} }, metricSel = { value: '' };\n"
    "var renderLog = [];\n"
    "function render() { renderLog.push({ rows: state.rows.length,\n"
    "  prevKeys: Object.keys(state.prevData).length }); }\n"
    "function resolveWeekRange() { return { ws: null, we: null }; }\n"
    "function amSelectedSeasons() { return [2026]; }\n"
    "function amIsMultiSeason() { return false; }\n"
    "function syncTableHeader() {}\n"
    "function schemaSkeletonRows() { return ''; }\n"
    "function syncURL() {}\n"
    "function updateWeekNote() {}\n"
    "function updateVolHeader() {}\n"
    "function populateTeamFilter() {}\n"
    "function showAgeCtrl() {}\n"
    "var extrasLoaded = [];\n"
    "function _loadExtras(keys) { extrasLoaded.push(keys); }\n"
    "var fetchCalls = [];\n"
    "function fetch(url) {\n"
    "  fetchCalls.push(String(url));\n"
    "  return Promise.resolve({ status: 200, ok: true, json: function() {\n"
    "    return Promise.resolve({ players: [{ player_id: 'p1',\n"
    "      position: 'RB', value: 10, games: 3 }], vol_col: 'games',\n"
    "      is_week_filtered: false }); } });\n"
    "}\n"
)


def _run_fetch_scenario(prev_stub, body):
    script = (
        _FETCH_PREAMBLE
        + "function _amCmpFetch(url, ms) { return %s; }\n" % prev_stub
        + _FETCH_DATA_JS
        + "\n"
        + body
    )
    return _run_node(script)


def test_board_renders_without_waiting_for_yoy_fetch():
    # The YoY fetch hangs (its 15s timeout was the old board-wide stall).
    # The primary data is back immediately, so the board must render.
    res = _run_fetch_scenario(
        "new Promise(function() {})",
        "fetchData();\n"
        "setTimeout(function() {\n"
        "  console.log(JSON.stringify({ renders: renderLog }));\n"
        "  process.exit(0);\n"
        "}, 150);\n",
    )
    assert res["renders"], "board never rendered while the YoY fetch hung"
    assert res["renders"][0]["rows"] == 1


def test_yoy_arrows_merge_in_when_prev_fetch_lands_late():
    res = _run_fetch_scenario(
        "new Promise(function(res) { setTimeout(function() {\n"
        "    res({ players: [{ player_id: 'p1', value: 9 }] }); }, 60); })",
        "fetchData();\n"
        "setTimeout(function() {\n"
        "  console.log(JSON.stringify({ renders: renderLog }));\n"
        "  process.exit(0);\n"
        "}, 200);\n",
    )
    # First render: primary only. Second render: YoY merged in.
    assert len(res["renders"]) >= 2
    assert res["renders"][0]["prevKeys"] == 0
    assert res["renders"][-1]["prevKeys"] == 1


def test_identical_reload_does_not_refetch():
    # Re-applying the active preset calls fetchData() with unchanged
    # state. The board is already showing exactly that: it must just
    # re-render, not re-fire the primary (and, downstream, every extra).
    res = _run_fetch_scenario(
        "Promise.resolve({ players: [{ player_id: 'p1', value: 9 }] })",
        "fetchData();\n"
        "setTimeout(function() {\n"
        "  var fetchesAfterFirst = fetchCalls.length;\n"
        "  var rendersAfterFirst = renderLog.length;\n"
        "  fetchData();\n"
        "  setTimeout(function() {\n"
        "    console.log(JSON.stringify({ fetches: fetchCalls.length,\n"
        "      fetchesAfterFirst: fetchesAfterFirst,\n"
        "      renders: renderLog.length,\n"
        "      rendersAfterFirst: rendersAfterFirst }));\n"
        "    process.exit(0);\n"
        "  }, 100);\n"
        "}, 150);\n",
    )
    assert res["fetchesAfterFirst"] == 1
    assert res["fetches"] == 1, "identical reload re-fired the primary request"
    assert res["renders"] > res["rendersAfterFirst"]


# ---------------------------------------------------------------------------
# 3. Failed columns: explicit error cell + retry wiring (source contracts).
# ---------------------------------------------------------------------------

def test_failed_extra_column_is_consumed_by_render():
    # fetchExtraData has always WRITTEN { failed: true }; render must READ
    # it (row filter + cell), not just the writer.
    assert _AM_JS.count("ed.failed") >= 2


def test_failed_extra_column_has_retry_affordance():
    assert "window.amRetryExtra" in _AM_JS
    assert "data-am-retry" in _AM_JS


# ---------------------------------------------------------------------------
# 4. Server: a builder exception is a 500, not a fake empty board.
# ---------------------------------------------------------------------------

def test_leaderboard_builder_error_is_500_not_empty_board(offline_client, monkeypatch):
    import data_building.advanced_metrics as am

    def _boom(*a, **k):
        raise RuntimeError("builder exploded")

    monkeypatch.setattr(am, "get_metric_leaderboard", _boom)
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=opportunity_share&season=2026")
    assert resp.status_code == 500, resp.get_data(as_text=True)
    assert resp.get_json()["error"] == "leaderboard_failed"
