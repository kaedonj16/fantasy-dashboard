"""ScoreZone dynamicity set (2026-11-15): static contract checks.

The set is presentation-layer: motion hooks in scorezone.js / app.js and
rz CSS in dashboard.css. These tests pin the hooks by content so a future
cleanup cannot silently drop a package.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SZ = (ROOT / "static" / "scorezone.js").read_text()
APP = (ROOT / "static" / "app.js").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()
CHANGELOG = (ROOT / "dashboard_services" / "changelog.py").read_text()


def test_package1_score_countup_and_team_flash():
    assert "_dynAfterRender" in SZ
    assert "rz-score-flash-team" in SZ and "rz-score-flash-team" in CSS
    assert "--rz-flash" in SZ
    assert "rz-changed-flash" in SZ  # clock / situation highlight


def test_package1_feed_td_enter_no_wash():
    assert "rz-td-enter" in SZ and "rz-td-enter" in CSS
    assert "rz-delta-pop" in CSS


def test_package2_field_depth_and_fd_pulse():
    assert "rz-fb-rzdeep" in APP and "rz-fb-rzdeep" in CSS
    assert 'data-depth="' in APP
    assert "rz-fb-fd-pulse" in APP and "rz-fb-fd-pulse" in CSS
    assert "rz-board-heat-" in APP


def test_package3_game_state_classes():
    assert "rz-board-nailbiter" in APP and "is-nailbiter" in SZ
    assert "rz-board-blowout" in APP and "is-blowout-final" in SZ
    assert "rz-badge-pop" in APP
    assert "--rz-acc-a" in APP  # ambient wash vars ride the board


def test_package4_lead_flip_hooks():
    assert "rz-lead-flip" in SZ and "rz-lead-flip" in CSS
    assert "rz-card-flash" in SZ
    assert "_dynFlashHero" in SZ


def test_change5_sticky_bar():
    assert "_stickyBarHtml" in SZ and "rz-sticky-bar" in CSS
    assert "position: sticky" in CSS


def test_change6_live_dark_scoped_toggle():
    assert "rz-live-dark" in SZ and "rz-live-dark" in CSS
    assert "rzLiveDark" in SZ  # localStorage pref, global theme untouched
    assert "rz-theme-fade" in SZ


def test_change7_moments_rail_uses_existing_clips():
    assert "_momentsRailHtml" in SZ
    assert "rz-moments-rail" in CSS
    assert "data-rail-clip" in SZ  # drives the feed's own clip toggle
    assert "clipPosterUrl" in SZ    # poster only when the payload carries one


def test_change8_ambient_wash_board_only():
    assert ".rz-nfl-board::before" in CSS
    assert "--rz-acc-a" in CSS and "--rz-acc-h" in CSS


def test_theme_transition_site_wide():
    assert "theme-transition" in APP and "theme-transition" in CSS
    assert "__themeTransitionTimer" in APP  # class is removed after the switch
    # No permanent universal transition rule was introduced.
    assert "* { transition: all" not in CSS


def test_reduced_motion_and_changelog():
    assert "prefers-reduced-motion: reduce" in CSS
    assert "ScoreZone now feels live" in CHANGELOG
