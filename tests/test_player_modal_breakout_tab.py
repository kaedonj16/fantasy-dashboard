"""Player modal Breakout tab is only shown for board breakout candidates."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_player_modal_breakout_tab_gated_on_is_breakout():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert 'id="pmTabBreakout"' in js
    assert 'id="pm-panel-breakout"' in js
    # Tab visibility matches the BREAKOUT board badge indicator set.
    assert "isBreakout(pid) ? '' : 'none'" in js
    # Must not unconditionally reveal the tab for every player.
    assert "if (tabBreakout) tabBreakout.style.display = '';" not in js
