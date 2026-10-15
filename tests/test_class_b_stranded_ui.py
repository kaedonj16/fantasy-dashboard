"""Class B audit: one-shot fetch failures stranding UI surfaces.

Follow-up to the trade-calculator incident fixed in PR #2625: a surface
whose only load fires once (deep link, tab lazy-load, page init) and whose
failure path renders an error/empty state with no retry and no re-trigger
is stranded forever, even though the data exists. This file pins the
recovery wiring for every surface the audit found stranded:

- Compare page (static/app.js): deep-link and picker loads rendered
  brErrorState with a null retry; the metrics fetches cached failures
  ({} / []) into _cmpAdvCache/_cmpWeeklyCache/_cmpSideResultCache, so
  even a manual re-render served the poisoned empty result.
- Trade intel panels (static/app.js): fetchSimilarTrades had a null
  retry; fetchPlayoffImpact treated a non-200 as "not available" and its
  catch had no retry; fetchTradeIntel hid the panel when every request
  failed; fetchPackages/fetchSendPackages had no res.ok guard, so a
  429/500 JSON error body rendered the false "no trade data" empty
  state; loadTradeTargets rendered bare text with no recovery.
- History sections (static/app.js): the one-shot sections fetch put a
  null retry on all three error cards.
- Teams page (static/teams.js): the Value Tracker, Schedule Strength,
  and Roster Intel panels set their lazy-load flag before fetching and
  rendered errors with no retry, so tab re-clicks returned early
  forever; the shared roster-intel promise stayed rejected once failed.
- Player modal trends (static/player_modal.js): the weekly-trends
  toggle set dataset.loaded before fetching and its catch had a null
  retry, so reopening the panel returned early forever.
- Advanced Metrics trends (advanced_metrics_page.py): a failed
  /api/weekly-trends fetch cached {} for the season, and the toggle
  handler only fetched when the cache was missing, so the column
  showed dashes for that season with no refetch.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
TEAMS_JS = (ROOT / "static" / "teams.js").read_text(encoding="utf-8")
MODAL_JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
AM_PY = (ROOT / "dashboard_services" / "pages" / "advanced_metrics_page.py").read_text(encoding="utf-8")


def _between(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


# -- Compare page -------------------------------------------------------------


def test_compare_deep_link_retry_reruns_the_load():
    body = _between(APP_JS, "const _loadDeepLink = () => {", "_loadDeepLink();")
    assert "Promise.all(qs.map(q => _fetchDetails(q)))" in body
    assert "window.brErrorState(resultEl, 'Could not load that comparison.', _loadDeepLink" in body


def test_compare_picker_failure_retry_reruns_maybe_compare():
    assert ("window.brErrorState(resultEl, 'Could not load one of the players.', "
            "() => _maybeCompare()") in APP_JS


def test_compare_metrics_fetches_do_not_cache_failures():
    body = _between(APP_JS, "async function _cmpFetchSideUncached(side) {",
                    "// Render one side's selector")
    # Full-season advanced-metrics, week-range advanced-metrics, and the
    # weekly-series fetches all throw on !ok instead of resolving {} / []
    # into _cmpAdvCache/_cmpWeeklyCache.
    assert body.count("if (!r.ok) throw new Error('HTTP ' + r.status); return r.json();") == 3
    assert ".catch(() => ({}))" not in body
    assert "catch (_) { weeks = []; }" not in body
    # The render-level catch offers a genuine retry.
    assert ("window.brErrorState('compareMetricsRows', 'Could not load advanced metrics.', "
            "() => cmpRenderMetrics()") in APP_JS


def test_compare_weekly_trends_failure_not_cached_and_retryable():
    body = _between(APP_JS, "async function cmpRenderWeekly(which) {",
                    "// One side's SEASON")
    assert "_cmpWeeklyCache[cacheKey] = weeks;" in body
    catch = body.split("catch (_) {", 1)[1]
    assert "_cmpWeeklyCache[cacheKey]" not in catch.split("return;")[0], (
        "the failure path must not write the weekly cache")
    assert "() => cmpRenderWeekly(which)" in catch


# -- Trade intel panels ---------------------------------------------------------


def test_similar_trades_retry_reruns_fetch():
    assert ("window.brErrorState(listEl, 'Trade data unavailable.', "
            "() => fetchSimilarTrades(_tradeGeneration, null)") in APP_JS


def test_playoff_impact_non_ok_is_failure_not_unavailable():
    body = _between(APP_JS, "async function fetchPlayoffImpact(gen, signal) {",
                    "// ------------------------------------------------------------\n  // fetchTradeIntel")
    assert 'if (!res.ok) throw new Error("Playoff impact failed (" + res.status + ")");' in body
    assert "const data = res.ok ? await res.json() : null;" not in body


def test_playoff_impact_bounded_signature_retry():
    assert "let _piRetryTimer = null;" in APP_JS
    assert "let _piRetrySig = \"\";" in APP_JS
    assert "let _piRetryAttempts = 0;" in APP_JS
    body = _between(APP_JS, "async function fetchPlayoffImpact(gen, signal) {",
                    "// ------------------------------------------------------------\n  // fetchTradeIntel")
    catch = body.split("} catch (e) {", 1)[1]
    assert 'const _piSig = giveIds.join(",") + "|" + getIds.join(",") + "|" + rosterId;' in catch
    assert "_piRetryAttempts < 2" in catch
    assert "if (_piRetryGen === _tradeGeneration) fetchPlayoffImpact(_piRetryGen, null);" in catch
    assert "1500" in catch and "4000" in catch
    # A landed sim clears the budget.
    assert "_piRetryAttempts = 0;" in body.split("state._lastPiData")[0]


def test_trade_intel_all_failed_bounded_retry():
    assert "let _tiRetryTimer = null;" in APP_JS
    body = _between(APP_JS, "async function fetchTradeIntel(gen, signal) {",
                    "// ------------------------------------------------------------\n  // loadTradeTargets")
    assert "results.every(r => r === null)" in body
    assert "_tiRetryAttempts < 2" in body
    assert "if (_tiRetryGen === _tradeGeneration) fetchTradeIntel(_tiRetryGen, null);" in body


def test_packages_fetches_throw_on_non_ok():
    # Server error JSON has no packages/options; without the guard it
    # rendered the false "no data" empty state.
    assert 'if (!res.ok) throw new Error("Packages request failed (" + res.status + ")");' in APP_JS
    assert 'if (!res.ok) throw new Error("Send packages request failed (" + res.status + ")");' in APP_JS


def test_load_trade_targets_error_state_retries():
    assert ("window.brErrorState(body, 'Could not load targets.', "
            "() => loadTradeTargets(containerEl)") in APP_JS


# -- History sections -----------------------------------------------------------


def test_history_sections_failure_retry_reruns_load():
    assert "const loadHistorySections = () => {" in APP_JS
    assert "loadHistorySections();" in APP_JS
    for msg in ("Error loading season awards.", "Error loading standings.",
                "Error loading season chart."):
        assert f"'{msg}', loadHistorySections" in APP_JS


# -- Teams page panels ------------------------------------------------------------


def test_teams_btm_error_has_genuine_retry():
    assert "_panelError(panel, 'Could not load data.', function () { fetchBtm(days); });" in TEAMS_JS
    assert "retry: function () { fetchBtm(days); }" in TEAMS_JS


def test_teams_sos_failure_clears_loaded_flag_and_retries():
    assert "delete _loaded.sos;" in TEAMS_JS
    assert "_panelError(panel, 'Could not load data.', function () { loadSos(); });" in TEAMS_JS
    assert "retry: function () { loadSos(); }" in TEAMS_JS


def test_teams_roster_intel_failure_clears_flag_and_shared_promise():
    assert "delete _loaded.rosterIntel;" in TEAMS_JS
    assert "_panelError(panel, 'Could not load roster intel.', function () { loadRosterIntel(); });" in TEAMS_JS
    # The shared roster-intel promise is dropped on rejection so the next
    # caller refetches instead of inheriting the failure.
    assert re.search(
        r"_riP\.catch\(function \(\) \{\s*if \(_riDataPromise === _riP\) "
        r"\{ _riDataPromise = null; _riDataKey = ''; \}",
        TEAMS_JS,
    )


# -- Player modal trends ------------------------------------------------------------


def test_modal_weekly_trends_failure_clears_loaded_and_retries():
    body = _between(MODAL_JS, "function pmToggleWeeklyTrends(playerId) {",
                    "const _ADV_METRIC_DESCS")
    catch = body.split(".catch(function() {", 1)[1]
    assert "wrap.dataset.loaded = '';" in catch
    assert "pmToggleWeeklyTrends(playerId);" in catch
    assert "window.brErrorState(body, 'Could not load trends.', null" not in body
    # Non-OK responses throw into that catch instead of parsing as data.
    assert "if (!r.ok) throw new Error('HTTP ' + r.status); return r.json();" in body


# -- Advanced Metrics trends ------------------------------------------------------------


def test_am_trends_failure_marked_and_refetched_on_toggle():
    assert "trendsFailed: {}," in AM_PY
    assert "state.trendsFailed[trendSeasonKey()] = true;" in AM_PY
    assert "delete state.trendsFailed[trendSeasonKey()];" in AM_PY
    assert ("(!state.trendsBySeason[trendSeasonKey()] || "
            "state.trendsFailed[trendSeasonKey()])") in AM_PY
