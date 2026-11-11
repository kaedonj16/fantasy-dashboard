"""Site-wide canonical position order sweep (companion to the team modal fix).

Canonical order everywhere positions are sequenced for the user:
QB, RB, WR, TE, FLEX, SUPER_FLEX, K, DEF. These source contracts lock
the surfaces that previously sequenced positions in a different order:
compare popular chips, the SEO compare chips, the cheat sheet By
Position columns, the player modal "Also competing for targets" rows,
and the draft room thin-position flags. ESPN slot-list ordering is
covered behaviorally in test_espn_scoring_normalization.py.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).parents[1]
_APP_JS = (_REPO / "static" / "app.js").read_text()
_CHEAT = (_REPO / "static" / "cheat_sheet.js").read_text()
_PM = (_REPO / "static" / "player_modal.js").read_text()
_DR = (_REPO / "static" / "draft_room.js").read_text()
_SEO = (_REPO / "routes" / "seo_pages_bp.py").read_text()


def test_compare_popular_chips_canonical():
    start = _APP_JS.find("cmpPopularChips")
    chunk = _APP_JS[start:start + 4000]
    assert "['QB', 'RB', 'WR', 'TE'].map(pos =>" in chunk
    assert "['WR', 'RB', 'QB', 'TE']" not in _APP_JS


def test_seo_compare_chips_canonical():
    assert 'order = ["QB", "RB", "WR", "TE"]' in _SEO
    assert 'order = ["RB", "WR", "QB", "TE"]' not in _SEO


def test_cheat_sheet_by_position_columns_canonical():
    start = _CHEAT.find("function renderPos")
    chunk = _CHEAT[start:start + 600]
    assert "var POS = ['QB', 'RB', 'WR', 'TE'];" in chunk


def test_player_modal_competing_rows_canonical():
    assert "['RB', 'WR', 'TE']" in _PM
    assert "['WR', 'TE', 'RB']" not in _PM


def test_draft_room_thin_flags_canonical():
    assert "['QB','RB','WR','TE'].forEach(function(pos){" in _DR
    assert "['RB','WR','TE','QB']" not in _DR
