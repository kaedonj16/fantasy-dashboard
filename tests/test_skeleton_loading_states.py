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


def test_week_change_overlay_uses_skeleton_not_spinner():
    src = _read("dashboard_services/pages/weekly_hub_page.py")
    m = re.search(r'id="weeklyMatchupsLoading".*?</div>\s*</div>', src, re.S)
    assert m, "week-change loading overlay markup not found"
    overlay = m.group(0)
    assert "matchups-spinner" not in overlay
    assert "skeleton" in overlay
    assert "matchups-loading-skel" in overlay


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
    assert ".matchups-loading-skel" in css


def test_ls_skeleton_constant_used_for_loading_states():
    src = _read("static/app.js")
    assert "LS_SKELETON" in src
    assert src.count("view.innerHTML = LS_SKELETON;") == 2  # initial + pending retry
    assert "'<div class=\"ls-loading\">Loading league scores...</div>'" not in src
