"""Trade Hub suggestions: the first paint stays up while sims fill in.

Kaedon (2026-11-23): on the trade calc page the suggestions "loaded in,
then went away and started to load again." The strategy loader paints a
fast analytical slate, then resolves each player group's Monte Carlo
numbers per player. Three re-entry paths could restart that loader over
the painted slate and throw it back to skeletons:

1. loadStrategyView re-entered for the identical view while its slate
   fetch or its per-player sim fill-in was still running (page-restore
   load, tab click, chip re-click).
2. _onContextChangePatch treated a programmatic league/season change
   event whose values matched the load already on screen as a real
   switch, cleared the memory cache and reloaded from skeletons.
3. _strategySimFinalize reset the reader to page 1 when the last sim
   landed, so the finished list jumped away mid-read.

Pins: re-entry for the shown view re-renders in place (or defers to the
in-flight load) instead of aborting/restarting; a same-context change
event is a no-op for the suggestions tab; a genuine refetch of the
shown view keeps the paint up (no skeleton write); finalize keeps the
reader's page.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _slice(start_marker: str, end_marker: str) -> str:
    start = APP_JS.index(start_marker)
    end = APP_JS.index(end_marker, start)
    return APP_JS[start:end]


def test_shown_view_state_is_tracked():
    assert "let _strategyShownKey" in APP_JS
    assert "let _lastStrategyCtx" in APP_JS
    # The slate paint and the cache-hit paint both record the shown key.
    assert APP_JS.count("_strategyShownKey = _sCacheKey;") == 2


def test_reentry_with_live_sim_job_rerenders_in_place():
    guard = _slice(
        "// Re-entry for the view already on screen",
        "// Newest-request-wins: bump the token",
    )
    assert "_strategySimJob.cacheKey === _sCacheKey" in guard
    assert "_renderImpactTable(_strategyData);" in guard
    assert "_renderStrategyCards(_strategyData, _strategyFilter);" in guard
    # The guard runs before the newest-request-wins abort block, so the
    # running sim job is never aborted by a same-view re-entry.
    assert APP_JS.index("// Re-entry for the view already on screen") < APP_JS.index(
        "// Newest-request-wins: bump the token"
    )


def test_reentry_during_slate_fetch_defers_to_inflight_load():
    guard = _slice(
        "// Re-entry for the view already on screen",
        "// Newest-request-wins: bump the token",
    )
    assert "_strategyInflight[_sCacheKey] === _strategyReqSeq" in guard


def test_same_context_change_event_is_a_noop_for_suggestions():
    patch = _slice(
        "function _onContextChangePatch()",
        "const leagueInputEl2",
    )
    assert "_ctxNow === _lastStrategyCtx" in patch
    assert patch.index("_ctxNow === _lastStrategyCtx") < patch.index("_strategyCache = {};")


def test_refetch_of_shown_view_keeps_paint_no_skeleton():
    loader = _slice(
        "const _keepPaint =",
        "_strategyInflight[_sCacheKey] = _mySeq;",
    )
    assert "_strategyShownKey === _sCacheKey && _strategyData.length > 0" in loader
    assert "if (!_keepPaint) {" in loader


def test_sim_finalize_keeps_reader_page():
    finalize = _slice(
        "function _strategySimFinalize(job)",
        "// Render a strategy result",
    )
    assert "_strategyPage = 0;" not in finalize
    assert "_strategyCache[job.cacheKey]" in finalize
