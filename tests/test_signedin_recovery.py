"""Signed-in recovery fixes (PR C): modal + waivers sections must always
land on a retryable state, never a dead end.

Item 1: the player modal's top-level "Try again" used to close and reopen
the modal; closePlayerModal() defers to history.back(), and the pending
popstate then tore down the freshly reopened modal, so the button read as
"close". The retry now re-runs the details load in place.

Item 2: the modal's RECENT NEWS section gets its own in-section error +
Retry (its bounded fetch layer shipped in #2704), and the VALUE HISTORY
chart can no longer fail silently into a grey box when the lazy Plotly
load rejects / is absent / draws nothing.

Item 3: the waivers "Best moves" candidates loader gained its 20s abort
in #2704; this adds the sibling sections' 25s DOM failsafe so a skeleton
that somehow survives the abort still lands on unavailable + Try again.

The _pmSmallFetch bounded-wait behavior itself is covered behaviorally by
tests/signedin_recovery_harness.mjs (run below) and by
tests/dupfetch_harness.mjs.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODAL = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
WAIVERS = (ROOT / "dashboard_services" / "pages" / "waivers_page.py").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")


def _block(src: str, start_marker: str, end_marker: str) -> str:
    start = src.index(start_marker)
    end = src.index(end_marker, start)
    return src[start:end]


# ── Item 1: modal Try again retries in place ─────────────────────────────

def test_details_load_is_a_named_rerunnable_step():
    assert "function _pmLoadDetails()" in MODAL
    assert "window.pmPlayerDetails.load(playerId, opts).then(data => {" in MODAL
    # Invoked once for the initial open.
    assert "\n  _pmLoadDetails();\n}" in MODAL


def test_details_retry_no_longer_closes_and_reopens():
    # The broken pattern (both the data.error path and the .catch path used
    # it) is gone from the whole file.
    assert "closePlayerModal();\n            openPlayerModal(playerId, playerName, opts);" not in MODAL
    assert "closePlayerModal();\n          openPlayerModal(playerId, playerName, opts);" not in MODAL
    # Both error paths now show the in-modal loading state and re-run the load.
    assert MODAL.count("_pmShowDetailsLoading();\n            _pmLoadDetails();") == 1
    assert MODAL.count("_pmShowDetailsLoading();\n          _pmLoadDetails();") == 1


def test_details_loading_helper_targets_the_modal_body():
    block = _block(MODAL, "function _pmShowDetailsLoading()", "_pmLoadDetails();")
    assert "getElementById('playerModalBody')" in block
    assert "player-modal-loading" in block
    assert "loading-spinner" in block


def test_other_modal_error_retries_still_in_place():
    # Regression guard: the tab/section retries that already re-ran their
    # loaders in place were not touched.
    assert "() => pmSwitchTab('breakout')" in MODAL
    assert "_pmLoadTeamPanel(panel, playerId, season);" in MODAL
    assert "_pmFetchTradesInto(panel, playerId, season, ctx);" in MODAL
    assert "_pmLoadScheduleGame(panel, item, true);" in MODAL


# ── Item 2a: news section error + Retry ──────────────────────────────────

def test_news_load_is_named_and_invoked_for_overview():
    assert "function _pmLoadNews()" in MODAL
    assert "if (data.position && data.position !== 'PICK') _pmLoadNews();" in MODAL


def test_news_failure_lands_on_section_error_with_retry():
    block = _block(MODAL, "function _pmNewsLoadFailed()", "function _pmLoadNews()")
    assert "getElementById('pmNewsBody')" in block
    assert "window.brErrorState(nb, 'Could not load news.'" in block
    assert "_pmLoadNews();" in block  # the retry re-runs the load
    assert "{ compact: true }" in block
    # Both failure routes (null payload and rejection) use it.
    assert "if (!nd) { _pmNewsLoadFailed(); return; }" in MODAL
    assert ".catch(() => { _pmNewsLoadFailed(); });" in MODAL


# ── Item 2b: value-history chart guard ───────────────────────────────────

def test_chart_draw_failures_land_on_error_with_redraw():
    block = _block(MODAL, "const _vhDrawFailed = function ()", "if (window.ensurePlotly)")
    assert "chartDiv.isConnected" in block
    assert "window.brErrorState(chartDiv, 'Could not load value history.'" in block
    assert "renderChart(_vhLastHistory)" in block


def test_chart_draw_is_verified_and_never_swallowed():
    # The draw result is returned and verified to have produced a plot.
    assert "return Plotly.newPlot('playerValueChart', traces, layout, {" in MODAL
    assert "if (!chartDiv.querySelector('.main-svg') && !chartDiv._fullLayout) _vhDrawFailed();" in MODAL
    assert "}).catch(_vhDrawFailed);" in MODAL
    # Missing ensurePlotly also lands on the guard, and the old silent
    # swallow after the Plotly load is gone.
    chart_region = _block(MODAL, "const _vhDrawFailed = function ()", "// Time-range filter")
    assert "} else {" in chart_region and "_vhDrawFailed();" in chart_region
    assert "}).catch(function () {});" not in chart_region
    # The redraw target tracks the last rendered subset (range switches).
    assert "let _vhLastHistory = fullHistory;" in MODAL
    assert "_vhLastHistory = history;" in MODAL


# ── Item 3: candidates DOM failsafe (on top of #2704's abort timer) ─────

def test_candidates_has_25s_dom_failsafe():
    block = _block(WAIVERS, "function wvLoadCandidates()", "// ── Start/Sit guests")
    assert "const candFailsafe = setTimeout(() => {" in block
    assert "25000" in block
    assert "if (requestSeq !== wvCandidateRequestSeq) return;" in block
    assert "list.querySelector('.skeleton')" in block
    assert "window.brErrorState('wvWaiverList', 'Unable to load waiver data.', wvLoadCandidates);" in block
    # Settling clears the failsafe on both paths (success and catch).
    assert block.count("clearTimeout(candFailsafe);") == 2
    # #2704's abort timer is still the first line of defense.
    assert "candTimedOut" in block and "20000" in block


# ── SW ───────────────────────────────────────────────────────────────────

def test_sw_cache_bumped():
    assert "br-fantasy-v80" in SW


# ── Behavioral harness ───────────────────────────────────────────────────

def test_small_fetch_hang_behavior():
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("node not available")
    proc = subprocess.run(
        [node, str(ROOT / "tests" / "signedin_recovery_harness.mjs")],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CHECKS PASSED (5)" in proc.stdout
