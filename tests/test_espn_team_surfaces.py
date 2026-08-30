"""Team modal (/api/team-details) format handling for ESPN / redraft leagues."""
from __future__ import annotations

import re

from pathlib import Path


def test_team_details_skips_picks_and_uses_redraft_for_redraft_leagues():
    src = Path("app.py").read_text(encoding="utf-8")
    # Locate the team-details handler body through the next route.
    start = src.find('@app.route("/api/team-details/<roster_id>")')
    end = src.find('@app.route("/api/player-league-trades/<player_id>")', start)
    assert start > 0 and end > start
    body = src[start:end]

    assert "is_redraft = _league_is_redraft(" in body
    assert "_value_primary" in body
    assert "redraft_value" in body
    assert "if not is_redraft:" in body
    assert '"is_redraft": bool(is_redraft)' in body
    # Must not invent picks when redraft — the invent-default block lives under
    # the is_redraft guard.
    invent = body.find('"current_owner": int(roster_id)')
    guard = body.find("if not is_redraft:")
    assert guard > 0 and invent > guard


def test_team_modal_js_hides_picks_for_redraft():
    src = Path("static/app.js").read_text(encoding="utf-8")
    assert "if (!data.is_redraft)" in src
    assert "Roster Value vs Age" in src


def test_trade_suggestions_ai_prompt_forbids_redraft_picks():
    src = Path("dashboard_services/ai/prompts.py").read_text(encoding="utf-8")
    start = src.find("def generate_trade_suggestions_result")
    end = src.find("def generate_team_ai_result")
    assert start > 0 and end > start
    body = src[start:end].lower()
    assert "never suggest a draft pick" in body
    assert "draft picks cannot be traded" in body
    assert "this redraft team" in body
    assert "playoff_status" in body
    assert "redraft_honesty_rules" in body
    assert 'scoring_type == "redraft"' in src[start:end]


def test_league_context_skips_synthesized_picks_for_redraft():
    src = Path("app.py").read_text(encoding="utf-8")
    start = src.find("# Future draft capital is a dynasty asset.")
    assert start > 0
    chunk = src[start:start + 1200]
    assert "not _league_is_redraft" in chunk
    assert "build_picks_by_roster(" in chunk


def test_teams_page_uses_team_avatar_and_redraft_values():
    src = Path("dashboard_services/pages/teams_page.py").read_text(encoding="utf-8")
    assert "team_avatar(platform, r, users)" in src
    assert "avatar_from_users(platform, users, str(rid))" not in src
    assert "build_model_value_lookup" in src
    assert "if not _is_redraft:" in src
    assert "picks_by_roster = {}" in src
    assert "redraft_window_label" in src
    assert "This season" in src
    assert "Playoff favorite" in src
    # Dynasty window copy stays for dynasty leagues only.
    assert "Competitive Windows" in src
    assert '_sort_archetype_label = "Odds"' in src
