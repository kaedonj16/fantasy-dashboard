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


def test_breakout_initial_failure_retries_silently_before_showing_retry_chip():
    # A first-attempt failure on the initial eligibility fetch is usually a
    # cold-backend timeout. The modal must retry once silently in the
    # background and only show the "unavailable" retry chip if that also fails.
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "_settleInitialBreakout" in js
    assert "_breakoutInitialRetried" in js
    # Silent retry scheduled with a delay, guarded by the stale-overlay checks.
    assert "setTimeout(() => {" in js
    assert "_loadBreakoutEligibility().then(" in js
    # The retry chip path is only reached after the silent retry also fails:
    # inside the _settleInitialBreakout block, the silent retry is scheduled
    # first and applyBreakoutEligibility(null, true) comes last.
    block_start = js.index("const _settleInitialBreakout")
    block = js[block_start:block_start + 2000]
    assert "_breakoutInitialRetried = true;" in block
    assert "setTimeout(() => {" in block
    assert "_loadBreakoutEligibility().then(" in block
    silent_retry_idx = block.index("_breakoutInitialRetried = true;")
    chip_idx = block.index("applyBreakoutEligibility(null, true)")
    assert silent_retry_idx < chip_idx
