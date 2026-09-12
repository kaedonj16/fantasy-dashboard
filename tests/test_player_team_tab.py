"""Player modal Team tab: /api/player-team endpoint and UI wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _player_team_route_src() -> str:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def api_player_team")
    end = src.find("clean_nan_for_json = _sanitize_for_json", start)
    assert start > 0 and end > start
    return src[start:end]


def test_player_team_route_exists_and_uses_real_sources():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def api_player_team")
    assert start > 0
    body = src[start:start + 5500]
    assert "get_players_index_global()" in body
    assert "load_relevant_index()" in body
    assert "get_team_full_name" in body
    assert "load_teams_index()" in body
    assert "get_players_global()" in body
    assert "_compute_team_offense_ranks" in body
    assert "build_team_schedule" in body
    assert "resolve_team_for_season" in body
    assert "schedule" in body
    assert "def api_player_team_boxscore" in src
    assert "get_shaped_boxscore" in src
    helpers = src[src.find("# ── Player modal Team tab"):start + 500]
    assert "stats_player_reg_" in helpers
    assert "fetch_season_snap_counts" in helpers
    assert "normalize_name" in helpers
    assert "_TEAM_OFFENSE_RANKS_CACHE" in helpers
    assert "_canon_team_abbr" in helpers


def test_player_modal_team_tab_ui_wiring():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert 'id="pmTabTeam"' in js
    assert 'id="pm-panel-team"' in js
    assert "_pmBuildTeamHTML" in js
    assert "/api/player-team/" in js
    assert "pmHasTeam" in js
    assert "'team'" in js
    assert "_pmTeamAdvOpen" in js
    assert "teamNavigation:true" in js.replace(" ", "")
    assert "Back to ${_pmEsc(prev.playerName)}" in js
    assert "pmPickTeamSeason" in js
    assert "pm-team-season-pills" in js
    assert "available_seasons" in js
    assert "data_mode" in js
    # O-line lives in its own Team-tab section (not mixed into Offense Profile
    # volume ranks), with position-primary highlighting and short labels.
    assert "pm-oline-sec" in js
    assert "Offensive Line" in js
    assert "pm-tp-primary" in js
    assert "Pass Block Grade" not in js
    assert "Run Block Grade" not in js
    assert "mkOl('Pass Block'" in js
    assert "mkOl('Run Block'" in js
    assert "mkOl('Overall'" in js
    assert "oline_' + metric" in js
    # Must not append O-line rows into the Offense Profile volume block.
    assert "${profile}${olineRows}" not in js
    assert "${profile}" in js
    assert "${olineSec}" in js
    # Schedule accordion has a compact preview and remains collapsed initially.
    assert "_pmBuildScheduleHTML" in js
    assert "Schedule &amp; Results" in js
    assert "pm-schedule-preview" in js
    assert "Current week" in js
    assert "pm-team-schedule" in js
    assert "pm-team-sched-toggle" in js
    assert "pm-team-sched-body" in js
    assert "_pmTeamSchedOpen" in js
    assert "pm-schedule-toggle" in js
    assert "pm-boxscore" in js
    assert "/api/player-team-boxscore" in js
    assert "pmPickTeamSeason" in js
    assert "_pmCollapseAllSchedule" in js
    assert "${scheduleSec}" in js
    assert "Box score available once the game begins" in js
    # Placement: Role → Schedule → environment → line → other groups.
    role_i = js.find("Player's Role &amp; Competition")
    sched_i = js.find("${scheduleSec}")
    env_i = js.find("Offensive Environment")
    line_i = js.find("${olineSec}", env_i)
    depth_i = js.find("Other Position Groups")
    assert role_i > 0 and sched_i > role_i and env_i > sched_i
    assert line_i > env_i and depth_i > line_i
    assert "PPR PPG" in js
    assert "_pmTeamRequestSeq" in js
    assert "panel.dataset.pmTeamRequest !== requestId" in js
    assert "_pmLoadScheduleGame(panel, item, true)" in js
    assert "panel.onclick = function" in js
    assert "focus_pid: ''" in js


def test_player_modal_team_tab_css():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    for cls in (
        ".pm-team-header", ".pm-crest", ".pm-hero-stat", ".pm-team-depth",
        ".player-badge-inj-q", ".pm-team-adv-toggle", ".pm-team-usage",
        ".pm-team-season-pills", ".pm-team-season-pill",
        ".pm-tp-primary", ".pm-tp-for", ".pm-oline-link",
        ".pm-team-schedule", ".pm-schedule-row", ".pm-schedule-toggle",
        ".pm-boxscore", ".pm-boxscore-table", ".pm-boxscore-focus",
        ".pm-boxscore-team-pill", ".pm-team-sched-toggle", ".pm-team-sched-body",
    ):
        assert cls in css, cls

    # Schedule section reuses Team-tab section chrome (padding + top border).
    assert ".pm-team-sec { padding: 14px 18px; border-top: 1px solid var(--border); }" in css
    assert "font-variant-numeric: tabular-nums" in css
    assert "prefers-reduced-motion" in css
    # Native disclosure buttons must override the global navy button fill.
    adv_css = css[css.find(".pm-team-adv-toggle {"):css.find(".pm-team-adv-body", css.find(".pm-team-adv-toggle {"))]
    assert "background: transparent" in adv_css
    assert "color: inherit" in adv_css
