"""Animated win-probability + skeleton loading states: markup/wiring contracts.

Verifies the loading states that were converted from spinners/bare text to
content-shaped skeletons, the motion-layer helpers exist and are wired into
the matchup live-update path, and the new CSS classes are defined with a
reduced-motion opt-out.
"""
from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("flask")

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(rel: str) -> str:
    return open(os.path.join(_ROOT, rel), encoding="utf-8").read()


def test_motion_helpers_exist_in_app_js():
    src = _read("static/app.js")
    assert "window.brTweenWinBar" in src
    assert "window.brAnimateMatchupRefresh" in src
    # Display-only: the tween must not touch probability math.
    assert "compute_win_prob" not in src


def test_gameday_refresh_uses_animator():
    src = _read("dashboard_services/pages/weekly_hub_page.py")
    assert "brAnimateMatchupRefresh(matchupsContainer" in src


def test_week_change_has_no_matchups_loading_cover():
    # Kaedon (2026-11-08): no refresh cover on the Matchups page. The week
    # change / pull-to-refresh load draws nothing over the board; the old
    # content stays visible and swaps in when the fetch lands.
    src = _read("dashboard_services/pages/weekly_hub_page.py")
    assert "weeklyMatchupsLoading" not in src
    assert "matchups-loading" not in src
    assert "loadingOverlay" not in src
    css = _read("static/dashboard.css")
    assert ".matchups-loading" not in css
    # The loading/settle contract the pull-to-refresh hook polls survives:
    # showLoading/hideLoading still flip sel.disabled, with no cover.
    assert "sel.disabled = true" in src
    assert "sel.disabled = false" in src
    # And the pull-to-refresh spinner chip itself is kept, untouched.
    app_js = _read("static/app.js")
    assert "br-ptr" in app_js and "br-ptr-spinner" in app_js
    assert ".br-ptr" in css and ".br-ptr-spinner" in css


def test_league_tab_initial_view_is_skeleton():
    src = _read("dashboard_services/pages/weekly_hub_page.py")
    # The league view's initial content (before the JS fetch resolves) is a
    # content-shaped skeleton, not the old bare "Loading league scores..." text.
    i = src.find("data-ls-view=")
    assert i != -1 and "league" in src[i : i + 40], "league view initial markup not found"
    region = src[i : i + 1600]
    assert "sk-list" in region
    assert "skeleton" in region
    assert "ls-loading" not in region
    assert "Loading league scores..." not in region


def test_scorezone_feed_loading_uses_skeleton():
    src = _read("static/scorezone.js")
    assert '<span class="rz-feed-spinner"></span>Loading plays' not in src
    assert "sk-card-row" in src  # feed placeholder rows


def test_fade_swap_css_defined_with_reduced_motion_opt_out():
    css = _read("static/dashboard.css")
    assert ".br-fade-swap" in css
    assert "brFadeSwap" in css
    # The global reduced-motion block must neutralise the new fade.
    m = re.search(r"@media \(prefers-reduced-motion: reduce\) \{[^}]*\.br-fade-swap[^}]*\}", css, re.S)
    assert m, ".br-fade-swap missing from a prefers-reduced-motion block"


def test_ls_skeleton_constant_used_for_loading_states():
    src = _read("static/app.js")
    assert "LS_SKELETON" in src
    assert src.count("view.innerHTML = LS_SKELETON;") == 1  # initial
    # Pending retry keeps the skeleton and appends the slow note past
    # ~30s (GUI Fix 1); past ~90s it escapes to unavailable + Retry.
    assert "view.innerHTML = LS_SKELETON + (pendingMs > 30000" in src
    assert "'<div class=\"ls-loading\">Loading league scores...</div>'" not in src
