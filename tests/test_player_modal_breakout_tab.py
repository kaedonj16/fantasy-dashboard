"""Player modal Breakout tab is only shown for board breakout candidates."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_player_modal_breakout_tab_gated_on_authoritative_membership():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert 'id="pmTabBreakout"' in js
    assert 'id="pm-panel-breakout"' in js
    # Outside the Breakout page, tab visibility comes from the authoritative
    # player-specific result rather than the asynchronous global indicator list.
    assert "breakoutData.board_eligible === true" in js
    assert "tabBreakout.style.display = boardEligible ? '' : 'none'" in js
    assert "isBreakout(pid) ? '' : 'none'" not in js
    assert "/api/breakout/player/${encodeURIComponent(playerId)}" in js
    # Must not unconditionally reveal the tab for every player.
    assert "if (tabBreakout) tabBreakout.style.display = '';" not in js


def test_breakout_badge_uses_same_player_membership_result():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "if (boardEligible)" in js


def test_breakout_page_passes_authoritative_candidate_context():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "String(candidate.player_id)" in app
    assert "candidate.player_name" in app
    assert "isBreakoutCandidate: true" in app
    assert "breakoutCandidate: candidate" in app


def test_player_modal_honors_authoritative_page_context():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "opts.isBreakoutCandidate === true && opts.breakoutCandidate" in js
    assert "!!contextBreakoutCandidate || (breakoutData && breakoutData.board_eligible === true)" in js
    assert "...contextBreakoutCandidate, available: true, board_eligible: true" in js
    assert "breakoutPanel._breakoutData = resolvedBreakoutData" in js
