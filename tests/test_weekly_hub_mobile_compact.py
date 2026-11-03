"""Weekly Hub mobile compact redesign (Kaedon-approved, 2026-11-02).

Phone widths only (desktop rendering unchanged):
- slimmer hub header (the global top nav stays: Kaedon reversed its
  hub-only hiding on 2026-11-03),
- five tabs in the main row (Matchups, League, Scorers, Scout, Lineup)
  with the League Scores sub-tab row replaced by the League tab,
- swipe-only matchup carousel: no heading/Prev-Next, full-width
  slides (no next-card sliver at rest), position dots underneath,
- Sleeper-density preview header (win % on the projection line) with
  the H2H lines as one small muted line under the team records,
- board rows shrunk past the mock (13.5px names, 18px scores, ~6px
  cell padding, smaller slot chips), smaller ScoreZone Moments card,
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

def test_top_nav_stays_visible_on_hub_mobile():
    # Kaedon reversed the compact build's hub-only hiding (2026-11-03):
    # the site's actual app header (logo, league pill, bell, search)
    # belongs on the Weekly Hub like on every other page.
    css = _css()
    assert 'body:has(.page-shell[data-page="weekly"]) .top-nav' not in css


def test_hub_header_card_is_thinner_on_mobile():
    css = _css()
    assert ".weekly-hub .card-header-row h2 { font-size: 14px; }" in css
    assert ".weekly-hub .weekly-rz-btn" in css
    # Mock values: 6px vertical card padding, no bottom margin (the
    # tab row's own 6px top margin is the whole header-to-tabs gap).
    assert ".weekly-hub .page-main > .card {\n        padding: 6px;" in css
    assert "margin-bottom: 0;" in css


def test_hub_header_fits_one_row_with_wrapped():
    css = _css()
    # The ScoreZone / Weekly Wrapped / week selector row never wraps,
    # even with the Wrapped pill and the LIVE badge both present:
    # controls run at a compact 10.5px and the badge stacks under the
    # title instead of widening it.
    assert "flex-wrap: nowrap;" in css
    assert ".weekly-hub .week-selector .wrapped-launch {" in css
    assert ".weekly-hub .card-header-row h2 {\n        display: flex;\n        flex-direction: column;" in css
    assert ".weekly-hub .week-selector .csd-trigger {" in css


def test_gap_between_header_and_tabs_is_six_px():
    css = _css()
    assert ".weekly-hub .page-main { gap: 0; }" in css
    assert "#weeklyLeftTabs .tab-bar { margin: 6px 10px 6px; }" in css


def test_scorezone_button_is_white_outline_pill_on_mobile():
    css = _css()
    # Even the live state matches the mock on the mobile hub header:
    # white fill, light border, navy text; the red dot is dropped
    # (the header already carries the LIVE badge).
    assert "background: #fff;" in css
    assert "border: 1.5px solid #d5dce4;" in css
    assert "color: #3d5a7a;" in css
    assert ".weekly-hub .weekly-rz-btn-live" in css
    assert ".weekly-hub .weekly-rz-btn-dot { display: none; }" in css


# --- carousel ---------------------------------------------------------------

def test_carousel_markup_has_dots_container():
    pytest.importorskip("pandas")
    from dashboard_services import matchups as m
    # The carousel template no longer carries a track-level dots
    # container: with full-matchup slides that spot rendered under the
    # whole board, where the dots were effectively invisible.
    slide = "<div class='m-slide'>x</div>"
    html = m.render_matchup_carousel_weeks({7: slide}, False, active_week=7)
    assert 'class="m-dots"' not in html


def test_slide_carries_dots_under_its_head(monkeypatch):
    pytest.importorskip("pandas")
    from dashboard_services import matchups as m

    monkeypatch.setattr(m, "load_teams_index", lambda: {})
    monkeypatch.setattr(m, "build_offense_rankings", lambda *_: {})
    monkeypatch.setattr(m, "load_week_stats", lambda season, week: {})
    monkeypatch.setattr(m, "load_week_schedule", lambda *_: [])
    monkeypatch.setattr(m, "build_team_schedule_lookup", lambda *_: {})

    def team(name, rid):
        return {
            "name": name, "username": "u", "record": "2-4", "avatar": "",
            "pts_total": 10.0, "proj_total": 10.0, "roster_id": rid,
            "owner_id": f"u{rid}",
            "starters": [{"pid": f"p{rid}", "name": "Some Guy", "pos": "QB",
                          "pts": 1.0, "nfl": "KC"}],
        }

    html = m.render_matchup_slide(
        "2026",
        {"left": team("A Team", 1), "right": team("B Team", 2), "h2h": {}},
        8, 7,
        status_by_pid={"p1": m.STATUS_FINAL, "p2": m.STATUS_FINAL},
        projections={}, players={}, teams={}, team_game_lookup={},
    )
    assert 'class="m-dots"' in html
    # Directly under the preview head, before the board body.
    assert html.index('class="m-dots"') > html.index('class="m-head')
    assert html.index('class="m-dots"') < html.index('class="m-body"')


def test_turning_points_slim_and_board_carded_on_mobile():
    css = _css()
    # Turning Points: regular width (no 12px inset), single surface,
    # no nested inner card.
    assert ".weekly-hub .m-drama-history {\n        margin: 8px 0 2px;" in css
    assert ".weekly-hub .m-drama-tp-item {\n        background: transparent;\n        border: none;" in css
    # Board rows sit on a white card like the Moments card.
    assert ".weekly-hub .m-body {\n        background: var(--card);" in css
    assert "border-radius: var(--radius);" in css


def test_carousel_mobile_is_swipe_only_full_width_with_dots():
    css = _css()
    assert ".weekly-hub .matchup-carousel .m-nav { display: none; }" in css
    # Full-width slides: no part of the next matchup shows at rest
    # (the brief peek left a broken sliver of the next board visible
    # down the right edge; Kaedon killed it, 2026-11-02).
    assert ".weekly-hub .matchup-carousel .m-slide { flex: 0 0 100%; }" in css
    assert "flex: 0 0 calc(100% - 30px)" not in css
    assert ".weekly-hub .m-dots" in css
    assert ".m-dots { display: none; }" in css  # hidden outside hub mobile


def test_carousel_js_syncs_dots_and_uses_slide_pitch():
    js = _read("static", "app.js")
    assert "function syncCarouselDots(card)" in js
    # Every slide carries its own dots container; all sync together.
    assert 'card.querySelectorAll(".m-dots")' in js
    # Index from slide offsets so the track gap cannot drift the index.
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
    assert ".weekly-hub .m-espn-proj {\n        font-size: 11.5px;" in css


def test_h2h_lines_sit_under_the_records_on_mobile():
    css = _css()
    # The duplicate under-records line is hidden everywhere else.
    assert ".m-h2h-under { display: none; }" in css
    # On the mobile hub the badge-row centre slot goes away and the
    # under-records copy shows as one small muted centred line.
    assert ".weekly-hub .m-espn-badges-c { display: none; }" in css
    assert ".weekly-hub .m-h2h-under {" in css
    assert "color: var(--text-subtle);" in css


def test_slide_renders_h2h_under_copy(monkeypatch):
    pytest.importorskip("pandas")
    from dashboard_services import matchups as m

    monkeypatch.setattr(m, "load_teams_index", lambda: {})
    monkeypatch.setattr(m, "build_offense_rankings", lambda *_: {})
    monkeypatch.setattr(m, "load_week_stats", lambda season, week: {})
    monkeypatch.setattr(m, "load_week_schedule", lambda *_: [])
    monkeypatch.setattr(m, "build_team_schedule_lookup", lambda *_: {})

    def team(name, rid):
        return {
            "name": name, "username": "u", "record": "2-4", "avatar": "",
            "pts_total": 10.0, "proj_total": 10.0, "roster_id": rid,
            "owner_id": f"u{rid}",
            "starters": [{"pid": f"p{rid}", "name": "Some Guy", "pos": "QB",
                          "pts": 1.0, "nfl": "KC"}],
        }

    html = m.render_matchup_slide(
        "2026",
        {"left": team("A Team", 1), "right": team("B Team", 2),
         "h2h": {"left_wins": 1, "right_wins": 0}},
        8, 7,
        status_by_pid={"p1": m.STATUS_FINAL, "p2": m.STATUS_FINAL},
        projections={}, players={}, teams={}, team_game_lookup={},
    )
    assert "class='m-h2h-under'" in html
    # Both copies carry the lines; CSS picks which one is visible.
    assert html.count("m-rivalry") == 2
    assert html.count("H2H this season:") == 2


# --- board rows + moments -----------------------------------------------------

def test_board_rows_shrunk_on_mobile_only():
    css = _css()
    # Kaedon's shrink (2026-11-02), past the mock's 14.5px / 19px:
    # 13.5px names, 18px scores including the .mb-final variant
    # (its 20px base would otherwise win over the plain .mb-a rule),
    # ~6px cell padding, 10.5px sub / game / stat / projection lines.
    assert ".weekly-hub .mb-info > .pname { font-size: 13.5px; }" in css
    assert ".weekly-hub .mb-a { font-size: 18px; }" in css
    assert ".weekly-hub .mb-a.mb-final { font-size: 18px; }" in css
    assert ".weekly-hub .m-combo .mb-cell {" in css
    assert "padding-top: 6px;" in css
    assert ".weekly-hub .mb-team { font-size: 10.5px; }" in css
    assert ".weekly-hub .mb-pj { font-size: 10.5px; }" in css
    assert ".weekly-hub .mb-pos .pos-badge {" in css


def test_moments_card_smaller_on_mobile():
    css = _css()
    assert ".weekly-hub .rzm-hub-launcher .rzm-row-title {" in css
    assert ".weekly-hub .rzm-hub-launcher .rzm-row-sub { font-size: 11.5px; }" in css
    # Mock padding/chevron: 8px vertical on the row, 14px chevron.
    assert "padding: 8px 12px 8px 0;" in css
    assert ".weekly-hub .rzm-hub-launcher .rzm-row-chevron { font-size: 14px; }" in css


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
