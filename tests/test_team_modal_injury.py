"""The team modal roster table marks injured players (Option A: inline badge).

Injury status lives only on the full Sleeper feed, so it is attached to each
roster player server-side (`/api/team-details`) and rendered as the same
`player-badge-inj-*` chip the player-detail modal uses. These tests lock the two
halves of that contract against silent regressions.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).parents[1]
_APP_JS = (_REPO / "static" / "app.js").read_text()
_APP_PY = (_REPO / "app.py").read_text()


def _roster_render_block() -> str:
    """The team-modal roster injury render, anchored on its unique marker."""
    start = _APP_JS.index("const injRaw = String(player.injury_status")
    return _APP_JS[start:start + 900]


def test_roster_renders_injury_badge_from_status():
    block = _roster_render_block()
    # Reads the field the server now attaches, and reuses the existing chip.
    assert "player.injury_status" in block
    assert "fa-triangle-exclamation" in block
    for cls in ("player-badge-inj-q", "player-badge-inj-d", "player-badge-inj-out"):
        assert cls in block, cls


def test_injury_severity_buckets_match_detail_modal():
    block = _roster_render_block()
    # Out-tier statuses drive the red chip; doubtful drives orange; else amber.
    for out_status in ("IR", "OUT", "PUP", "SUSP", "NFI"):
        assert f"'{out_status}'" in block, out_status
    assert "'DOUBTFUL'" in block
    # The body part is surfaced on hover.
    assert "injury_body_part" in block


def test_team_details_payload_attaches_injury_fields():
    # Server pulls the full feed once and normalizes healthy states out.
    assert "full_players = get_players_global()" in _APP_PY
    assert re.search(r'inj_status\s*=\s*""\s+if\s+_raw_inj\.lower\(\)\s+in\s+\("",\s*"active",\s*"act"\)', _APP_PY)
    assert '"injury_status": inj_status,' in _APP_PY
    assert '"injury_body_part": inj_body,' in _APP_PY
