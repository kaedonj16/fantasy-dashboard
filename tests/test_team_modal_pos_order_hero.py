"""Team modal: canonical position order + 4-tile header statbar.

Kaedon's screenshot (ESPN modal, phone width): the Roster rooms were
built as two .tm2-col wrappers (QB/WR/K/DEF | RB/TE/IR/Taxi), so when
the grid stacked on mobile K rendered third, before RB/TE. Rooms must
read QB, RB, WR, TE, K, DEF, IR, Taxi at every width. The header
statbar's mobile/container rules were written for 3 tiles
(repeat(3, 1fr)), so a 4th tile (Efficiency) wrapped alone; with 4
tiles the mobile layout must be a balanced 2x2 grid.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).parents[1]
_APP_JS = (_REPO / "static" / "app.js").read_text()
_CSS = (_REPO / "static" / "dashboard.css").read_text()


def _rooms_expr() -> str:
    start = _APP_JS.find("const _roomsHTML")
    assert start > 0
    return _APP_JS[start:_APP_JS.find(";", start)]


def test_rooms_render_in_canonical_order_without_columns():
    expr = _rooms_expr()
    # No column wrappers: rooms are direct children of .tm2-rooms so DOM
    # order (and therefore the stacked phone order) is canonical.
    assert "tm2-col" not in expr
    assert 'class="tm2-rooms"' in expr
    rooms = re.findall(r"_tmRoom\('([A-Z]+)'", expr)
    stashes = re.findall(r"_tmStash\('([a-z]+)'", expr)
    assert rooms == ["QB", "RB", "WR", "TE", "K", "DEF"]
    assert stashes == ["ir", "taxi"]
    # Full rendered order, stashes after every position room.
    calls = re.findall(r"_tm(?:Room|Stash)\('([A-Za-z]+)'", expr)
    assert calls == ["QB", "RB", "WR", "TE", "K", "DEF", "ir", "taxi"]


def test_positional_strength_rows_sorted_canonically():
    start = _APP_JS.find("let _strengthCard")
    chunk = _APP_JS[start:start + 1600]
    assert "['QB', 'RB', 'WR', 'TE', 'K', 'DEF']" in chunk
    assert "_strengthSorted" in chunk
    # The sorted copy, not the raw server order, drives the rows.
    assert "_strengthSorted.forEach" in chunk


def test_statbar_sets_tile_count_marker():
    modal = _APP_JS[_APP_JS.index("function renderTeamDetails("):]
    modal = modal[:modal.index("function ", 10)] if False else modal[:8000]
    assert "statbar.dataset.tileCount = String(tiles.length)" in modal
    # Empty/hidden state clears the marker so a later 3-tile team is
    # not stuck with the 4-tile layout.
    assert "delete statbar.dataset.tileCount" in modal


def test_four_tile_statbar_is_2x2_in_both_mobile_blocks():
    hits = re.findall(
        r'\.team-modal-statbar\[data-tile-count="4"\]\s*\{[^}]*'
        r'grid-template-columns:\s*repeat\(2,\s*1fr\)',
        _CSS,
    )
    # One in the @media (max-width: 768px) block, one in its
    # @container (max-width: 768px) twin.
    assert len(hits) == 2, hits
    # The base 3-up rule stays for 3-or-fewer tiles.
    assert _CSS.count("grid-template-columns: repeat(3, 1fr)") >= 2


def test_playoff_odds_tile_still_uses_one_decimal_formatter():
    modal = _APP_JS[_APP_JS.index("function renderTeamDetails("):]
    tile = modal[modal.index("data.playoff_odds") - 200: modal.index("data.playoff_odds") + 600]
    assert "_fmtPlayoffOddsDisplay(data.playoff_odds)" in tile
    assert "Math.round(parseFloat(data.playoff_odds))" not in modal[:8000]
