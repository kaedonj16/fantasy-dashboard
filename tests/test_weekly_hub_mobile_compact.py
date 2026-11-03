"""Weekly Hub mobile compact redesign (Kaedon-approved, 2026-11-02).

Phone widths only (desktop rendering unchanged):
- slimmer hub header, global top nav hidden on the hub page,
- five tabs in the main row (Matchups, League, Scorers, Scout, Lineup)
  with the League Scores sub-tab row replaced by the League tab,
- swipe-only matchup carousel: no heading/Prev-Next, next card peeking,
  position dots underneath,
- Sleeper-density preview header (win % on the projection line),
- tighter board rows, smaller ScoreZone Moments card,
- leaders strip as one compact card under the board.
"""
from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("flask")

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _hub_source():
    return _read("dashboard_services", "pages", "weekly_hub_page.py")


def _css():
    return _read("static", "dashboard.css")


# --- tab row ---------------------------------------------------------------

def test_league_tab_button_ships_with_the_preview():
    src = _hub_source()
    assert "_wk_league_btn" in src
    assert 'data-tab="league"' in src
    assert "wk-league-tab" in src
    # Only when the matchup preview (and its ls machinery) exists.
    assert src.count("if _show_matchup_preview else") >= 2


def test_league_tab_routing_js():
    src = _hub_source()
    # wkActivateTab special-cases 'league' through the ls-tabs machinery.
    assert "if (tab === 'league')" in src
    assert "lsLeagueBtn.click()" in src
    # Capture-phase interception so the generic card-tab handler never
    # looks for a league panel that does not exist.
    assert "__wkLeagueTabBound" in src
    assert "e.stopPropagation()" in src


def test_league_tab_hidden_on_desktop_css():
    css = _css()
    assert "#weeklyLeftTabs .wk-league-tab { display: none; }" in css
    # .wk-desktop (reflow's desktop marker class) hides it too.
    assert "#weeklyLeftTabs.wk-desktop .wk-league-tab" in css


def test_sub_tab_row_and_swaps_pill_removed_on_mobile():
    css = _css()
    assert ".weekly-hub .ls-tabs { display: none; }" in css
    assert "#weeklyLeftTabs .wk-tab-cnt { display: none; }" in css


# --- page top ---------------------------------------------------------------

def test_top_nav_hidden_only_on_hub_mobile():
    css = _css()
    assert 'body:has(.page-shell[data-page="weekly"]) .top-nav.br-mnav { display: none; }' in css


def test_hub_header_card_is_thinner_on_mobile():
    css = _css()
    assert ".weekly-hub .card-header-row h2 { font-size: 14px; }" in css
    assert ".weekly-hub .weekly-rz-btn" in css


# --- carousel ---------------------------------------------------------------

def test_carousel_markup_has_dots_container():
    pytest.importorskip("pandas")
    from dashboard_services import matchups as m
    slide = "<div class='m-slide'>x</div>"
    html = m.render_matchup_carousel_weeks({7: slide}, False, active_week=7)
    assert 'class="m-dots"' in html


def test_carousel_mobile_is_swipe_only_with_peek_and_dots():
    css = _css()
    assert ".weekly-hub .matchup-carousel .m-nav { display: none; }" in css
    assert "flex: 0 0 calc(100% - 30px)" in css
    assert ".weekly-hub .m-dots" in css
    assert ".m-dots { display: none; }" in css  # hidden outside hub mobile


def test_carousel_js_syncs_dots_and_uses_slide_pitch():
    js = _read("static", "app.js")
    assert "function syncCarouselDots(card)" in js
    # Index from slide offsets so the peeking layout cannot drift.
    assert "slides[1].offsetLeft - slides[0].offsetLeft" in js


# --- preview header -----------------------------------------------------------

def test_slide_win_pct_rides_the_projection_line(monkeypatch):
    pytest.importorskip("pandas")
    from dashboard_services import matchups as m

    def team(name, proj, rid, pid):
        return {
            "name": name, "username": "u", "record": "2-4", "avatar": "",
            "pts_total": 0.0, "proj_total": proj, "roster_id": rid,
            "owner_id": f"u{rid}",
            "starters": [{"pid": pid, "name": "Some Guy", "pos": "QB",
                          "pts": 0.0, "nfl": "KC"}],
        }

    html = m.render_matchup_slide(
        "2026",
        {"left": team("A Team", 125.5, 1, "p1"),
         "right": team("B Team", 114.8, 2, "p2"), "h2h": {}},
        8, 7,
        status_by_pid={"p1": m.STATUS_NOT_STARTED,
                       "p2": m.STATUS_NOT_STARTED},
        projections={8: {"projections": {"p1": 20.0, "p2": 12.0}}},
        players={}, teams={}, team_game_lookup={}, compact=True,
    )
    spans = re.findall(r"class='m-espn-wp'[^>]*>&middot; (\d+)%", html)
    assert len(spans) == 2
    assert int(spans[0]) + int(spans[1]) == 100
    # The styled win bar still renders (desktop keeps its end labels).
    assert "m-win-bar" in html and "m-wp-pct" in html


def test_preview_header_density_css():
    css = _css()
    assert ".weekly-hub .m-espn-num { font-size: 24px; }" in css
    assert ".weekly-hub .m-espn-wp { display: inline; }" in css
    assert ".m-espn-wp { display: none; }" in css
    assert ".weekly-hub .m-wp-track { height: 4px; }" in css
    assert ".weekly-hub .m-win-bar .m-wp-pct { display: none; }" in css
    assert ".weekly-hub .m-espn-meta" in css


# --- board rows + moments -----------------------------------------------------

def test_board_rows_tighter_on_mobile_only():
    css = _css()
    assert ".weekly-hub .mb-info > .pname { font-size: 14.5px; }" in css
    assert ".weekly-hub .mb-a { font-size: 19px; }" in css


def test_moments_card_smaller_on_mobile():
    css = _css()
    assert ".weekly-hub .rzm-hub-launcher .rzm-row-title { font-size: 14px; }" in css
    assert ".weekly-hub .rzm-hub-launcher .rzm-row-sub { font-size: 11.5px; }" in css


# --- leaders card ---------------------------------------------------------------

def test_leaders_title_carries_the_week():
    src = _hub_source()
    assert "Week {default_week} Leaders" in src
    assert "window.wkSetLeadersTitle" in src


def test_leaders_band_moves_under_the_board_on_mobile():
    src = _hub_source()
    # Reflow: into the Matchups panel on mobile, back above the
    # two-column row on desktop.
    assert "mPanel.appendChild(band)" in src
    assert "main.insertBefore(band, two)" in src
    css = _css()
    # Compact card rows: label column + content column, no 2x2 grid.
    assert "grid-template-columns: 92px minmax(0, 1fr)" in css
    assert ".weekly-hub .wk-leaders-strip .whl-name" in css
