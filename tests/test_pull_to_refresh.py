"""Pull-to-refresh (PWA, v1) contracts.

The gesture controller lives in static/app.js before the
@public-js:core-end marker (so it ships in the core/public bundles),
delimited by /* br-ptr:start */ .. /* br-ptr:end */ so the Node harness
can extract and drive the REAL controller with fake DOM/touch events.

Pages opt in by registering window.brPageRefresh() returning a promise:
ScoreZone (its proven _manualRefresh path) and the Weekly Hub (which
re-invokes its existing #hubWeek week-change handler). Pages without a
hook get no gesture and no reload fallback.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
SCOREZONE_JS = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
WEEKLY_HUB_PY = (
    ROOT / "dashboard_services" / "pages" / "weekly_hub_page.py"
).read_text(encoding="utf-8")


# ── Controller placement + shape ─────────────────────────────────────────────
def test_controller_is_in_the_core_bundle_before_the_marker():
    start = APP_JS.index("/* br-ptr:start */")
    end = APP_JS.index("/* br-ptr:end */")
    core_end = APP_JS.index("// @public-js:core-end")
    assert start < end < core_end


def test_controller_is_touch_only_and_hook_gated():
    start = APP_JS.index("/* br-ptr:start */")
    end = APP_JS.index("/* br-ptr:end */")
    block = APP_JS[start:end]
    assert "addEventListener('touchstart'" in block
    assert "typeof window.brPageRefresh !== 'function'" in block
    # No reload fallback for pages without a hook.
    assert "location.reload" not in block


def test_overscroll_contain_stays_so_android_cannot_double_fire():
    assert "overscroll-behavior-y: contain;" in DASHBOARD_CSS


def test_indicator_css_lives_before_the_merged_icons_marker():
    ptr = DASHBOARD_CSS.index(".br-ptr {")
    merged = DASHBOARD_CSS.index("MERGED: icons.css")
    assert ptr < merged
    assert ".br-ptr-spinner" in DASHBOARD_CSS
    assert "@keyframes br-ptr-spin" in DASHBOARD_CSS
    # Spinner is a CSS border ring; the indicator block has no icon font/emoji hook.
    block = DASHBOARD_CSS[ptr:merged]
    assert "border-top-color" in block
    assert "font-awesome" not in block.lower()


# ── ScoreZone hook ────────────────────────────────────────────────────────────
def test_scorezone_manual_refresh_returns_its_promise():
    assert "return _refreshUserStream({ manual: true })" in SCOREZONE_JS
    assert "return _refresh({ manual: true })" in SCOREZONE_JS


def test_scorezone_registers_br_page_refresh_with_detach_guard():
    assert "window.brPageRefresh = _ptrPageRefresh;" in SCOREZONE_JS
    block = SCOREZONE_JS[
        SCOREZONE_JS.index("function _ptrPageRefresh()"):SCOREZONE_JS.index(
            "window.brPageRefresh = _ptrPageRefresh;"
        )
    ]
    assert "root.isConnected === false" in block
    assert "delete window.brPageRefresh" in block
    assert "return _manualRefresh();" in block


# ── Weekly Hub hook ───────────────────────────────────────────────────────────
def test_weekly_hub_registers_br_page_refresh():
    assert "window.brPageRefresh = function()" in WEEKLY_HUB_PY


def test_weekly_hub_hook_reinvokes_the_existing_week_change_handler():
    _h = WEEKLY_HUB_PY.index("window.brPageRefresh = function()")
    hook = WEEKLY_HUB_PY[_h:WEEKLY_HUB_PY.index("}})();", _h)]
    # Same handler, same week: dispatch the change event on #hubWeek
    # instead of fetching /api/weekly-week from a parallel loader.
    assert "sel.dispatchEvent(new Event('change'))" in hook
    assert "fetch(" not in hook
    # Settle signal is the handler's own loading state (sel.disabled),
    # and the scroll position is restored on completion.
    assert "sel.disabled" in hook
    assert "window.scrollTo(0, y)" in hook
    assert "sel.isConnected === false" in hook


# ── Behavioral harness (Node) ─────────────────────────────────────────────────
def test_ptr_controller_behavioral_harness():
    """Covers: no hook = no trigger; below threshold = no trigger; above
    threshold at the top = hook called once and no re-entry while the
    promise is pending; not at top / modal open / inner-scroller target /
    form field / horizontal gesture = no trigger."""
    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not available for the JS behavioral harness")
    result = subprocess.run(
        [node, str(ROOT / "tests" / "pull_to_refresh_harness.mjs")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL PULL-TO-REFRESH HARNESS CHECKS PASSED" in result.stdout
