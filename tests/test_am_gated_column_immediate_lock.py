"""Gated extra columns on Advanced Metrics settle locked immediately.

Regression tests for the free-viewer bug where PRO-gated extra columns
(e.g. the Key Metrics preset's FPOE/G, Usage Trend, xFP Trend) sat on the
animated skeleton shimmer indefinitely: fetchExtraData only locked the
column after a leaderboard request came back 403, so any stall in that
round trip left the cells shimmering forever. The client already knows
the metric is gated (_mLocked), so a free viewer's column must settle
into the locked state synchronously, with no fetch at all. PRO viewers
and free metrics keep the normal fetch path.
"""
import json
import subprocess

from dashboard_services.pages.advanced_metrics_page import _AM_JS


def _extract_fn(name):
    """Pull a JS function definition out of the page bundle by brace matching."""
    start = _AM_JS.index("function %s(" % name)
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
    raise AssertionError("unbalanced braces extracting %s" % name)


_MLOCKED_JS = _extract_fn("_mLocked")
_AMROWKEY_JS = _extract_fn("amRowKey")
_FETCH_EXTRA_JS = _extract_fn("fetchExtraData")

FREE_CFG = {
    "hasPremium": False,
    "platform": "sleeper",
    "leagueId": None,
    "seasons": [2026, 2025],
    "metrics": {"target_share": {"label": "Target Share"}},
    "proMetricInfo": {
        "ppr_over_expected_per_game": {"label": "FPOE/G"},
        "opportunity_trend": {"label": "Usage Trend"},
        "xfp_trend": {"label": "xFP Trend"},
    },
}
PRO_CFG = dict(FREE_CFG, hasPremium=True)

# _amCmpFetch stub behaviours, keyed by name in the scenario payload.
_FETCH_STUBS = {
    # What the real helper returns for a gated metric: HTTP 403 -> proLocked.
    "pro403": "Promise.resolve({ proLocked: true })",
    # A response that never arrives (the stall behind the endless shimmer).
    "hang": "new Promise(function() {})",
    # A normal leaderboard payload.
    "data": (
        "Promise.resolve({ players: "
        "[{ player_id: 'p1', position: 'RB', value: 3.5 }] })"
    ),
}


def _run_fetch_scenario(cfg, key, stub):
    """Run fetchExtraData in Node with stubbed page globals; report the state."""
    script = (
        "var cfg = %s;\n"
        "var state = { requestToken: 0, season: '2026', combine: false,\n"
        "  extraMetrics: ['%s'], extraData: {}, extraPrevData: {},\n"
        "  playerPos: {}, filterColKeys: new Set() };\n"
        "var renderCalls = 0;\n"
        "function render() { renderCalls++; }\n"
        "function defaultVol() { return ''; }\n"
        "function resolveWeekRange() { return { ws: null, we: null }; }\n"
        "function amIsMultiSeason() { return false; }\n"
        "function amIsEachYear() { return false; }\n"
        "function amSelectedSeasons() { return [2026]; }\n"
        "var fetchCalls = [];\n"
        "function _amCmpFetch(url, ms, mode) {\n"
        "  fetchCalls.push({ url: url, mode: mode || null });\n"
        "  return %s;\n"
        "}\n"
        "%s\n%s\n%s\n"
        "fetchExtraData('%s');\n"
        "setTimeout(function() {\n"
        "  console.log(JSON.stringify({ extraData: state.extraData,\n"
        "    fetchCalls: fetchCalls, renderCalls: renderCalls }));\n"
        "}, 50);\n"
        % (
            json.dumps(cfg),
            key,
            _FETCH_STUBS[stub],
            _MLOCKED_JS,
            _AMROWKEY_JS,
            _FETCH_EXTRA_JS,
            key,
        )
    )
    out = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip())


def test_free_viewer_gated_column_locks_without_any_fetch():
    # Even with the 403 path available, a known gated column must not
    # spend a request on it: no fetch, locked state set, render fired.
    res = _run_fetch_scenario(FREE_CFG, "xfp_trend", "pro403")
    assert res["fetchCalls"] == []
    assert res["extraData"]["xfp_trend"]["proLocked"] is True
    assert res["renderCalls"] >= 1


def test_free_viewer_gated_column_locks_even_when_request_would_hang():
    # The reported bug: the gated fetch never settles, so the column
    # never left the skeleton. The lock must not depend on the request.
    res = _run_fetch_scenario(FREE_CFG, "opportunity_trend", "hang")
    assert res["fetchCalls"] == []
    assert res["extraData"]["opportunity_trend"]["proLocked"] is True


def test_pro_viewer_gated_metric_still_fetches_real_values():
    res = _run_fetch_scenario(PRO_CFG, "xfp_trend", "data")
    assert len(res["fetchCalls"]) >= 1
    assert res["fetchCalls"][0]["mode"] == "pro"
    ed = res["extraData"]["xfp_trend"]
    assert not ed.get("proLocked")
    assert ed["byId"]["p1"] == 3.5


def test_free_viewer_free_metric_still_fetches_real_values():
    res = _run_fetch_scenario(FREE_CFG, "target_share", "data")
    assert len(res["fetchCalls"]) >= 1
    ed = res["extraData"]["target_share"]
    assert not ed.get("proLocked")
    assert ed["byId"]["p1"] == 3.5
