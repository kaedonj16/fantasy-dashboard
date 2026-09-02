"""Regression tests for render_teams_sidebar HTML structure.

The unit-test CI job only installs pytest/ruff (no numpy), so these tests read
``dashboard_services/service.py`` source instead of importing the module.
"""
from __future__ import annotations

import re
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1] / "dashboard_services" / "service.py"


def _render_teams_sidebar_source() -> str:
    text = _SERVICE.read_text(encoding="utf-8")
    start = text.index("def render_teams_sidebar")
    end = text.index("\ndef render_predraft_sidebar", start)
    return text[start:end]


def test_picks_section_closing_tags_only_emitted_when_picks_exist():
    """Fleaflicker bench-only rosters must not leak stray </div> tags."""
    block = _render_teams_sidebar_source()
    picks_block = block[block.index("picks = t.get") : block.index("body_html =")]
    assert "if picks:" in picks_block
    assert picks_block.index('picks_out.append("</div></div>")') > picks_block.index("if picks:")


def test_render_teams_sidebar_includes_ir_section():
    block = _render_teams_sidebar_source()
    assert 'render_player_list("IR"' in block
    assert 'render_player_list("Bench"' in block
    assert 'render_player_list("Starters"' in block


def test_render_teams_sidebar_wraps_all_panels_in_team_panels():
    block = _render_teams_sidebar_source()
    assert "panels_html = \"<div class='team-panels'>\" + \"\".join(panel_html_parts) + \"</div>\"" in block
    assert re.search(
        r"f\"<div class='team-panel\{active_class\}' data-team-id='\{t\['roster_id'\]\}'>\"",
        block,
    )


def test_predraft_sidebar_links_to_league_cheat_sheet():
    text = _SERVICE.read_text(encoding="utf-8")
    start = text.index("def render_predraft_sidebar")
    end = text.index("\ndef render_dashboard_teams_sidebar", start)
    block = text[start:end]
    assert "draft/cheat-sheet" in block
    assert "Open cheat sheet" in block
    assert "Draft Room" in block
    assert "os-draft-prep" in block
    assert "Rosters aren't set yet" in block


def test_dashboard_sidebar_swaps_to_cheat_sheet_when_undrafted():
    text = _SERVICE.read_text(encoding="utf-8")
    start = text.index("def render_dashboard_teams_sidebar")
    end = text.index("\ndef build_picks_by_roster", start)
    block = text[start:end]
    assert "startup_draft_pending" in block
    assert "render_predraft_sidebar" in block
    assert 'return html_out, "Cheat Sheet"' in block
    assert "render_teams_sidebar(teams)" in block
