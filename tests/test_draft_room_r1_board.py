"""R1 board guards for the Draft Room (approved 2026-11-06).

The board is a flush hairline grid of 68px R1 cells: a 3px position stripe,
a 54px headshot flush at the cell bottom in a reserved 50px photo zone, and
all text strictly right of the zone. Live states (on the clock, your picks,
last pick, keepers, traded picks) keep their marks in the new grammar.
"""

import re
from pathlib import Path

from dashboard_services.pages.draft_room_page import build_draft_room_body


REPO = Path(__file__).resolve().parents[1]


def _body():
    return build_draft_room_body(None, None, None, is_guest=True)


def _source():
    return (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")


def _cell_inner(source):
    return source.split("function cellInner(pn){", 1)[1].split(
        "\n  function buildBoard()", 1
    )[0]


def test_board_grid_is_flush_with_connected_round_strip():
    body = _body()
    source = _source()

    # No gaps or per-cell cards: hairlines are shared borders.
    assert ".dr-board { display: grid; gap: 0; min-width: max-content; padding: 0; }" in body
    assert "border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);" in body
    assert "border-radius: 0; padding: 0; height: 68px; min-height: 68px;" in body
    # Round labels carry a hairline each, so the sticky column is one strip.
    assert ".dr-rowhead { position: sticky; left: 0; z-index: 5;" in body
    # Columns widen to the R1 minimum; the strip column is 34px.
    assert "'34px repeat(' + teams + ', minmax(168px, 1fr))'" in source
    assert "minmax(108px, 1fr)" not in source
    # The corner reads Rd like the mock strip head.
    assert '<div class="dr-colhead dr-rowhead dr-corner">Rd</div>' in source


def test_r1_cell_geometry_stripe_photo_zone_and_text_column():
    body = _body()

    assert ".dr-cell .stripe { position: absolute; left: 0; top: 0; bottom: 0; width: 3px; z-index: 3; }" in body
    # Photo zone: 50px wide, photo 54px tall flush to the cell bottom.
    assert ".dr-ph { position: absolute; left: 3px; bottom: 0; width: 50px; height: 54px;" in body
    assert ".dr-ph img { display: block; width: 50px; height: 54px; object-fit: cover; object-position: bottom;" in body
    # Text lives strictly right of the zone and the name wraps to two lines.
    assert ".dr-cell-body { margin-left: 58px;" in body
    name_rule = body.split(".dr-cell-name {", 1)[1].split("}", 1)[0]
    assert "font-size: 12.5px" in name_rule
    assert "max-height: 31px" in name_rule
    assert "overflow-wrap: anywhere" in name_rule
    assert "white-space: nowrap" not in name_rule
    # Position badge inside cells is tinted from the cell's --pos, not solid.
    assert ".dr-cell .dr-posbadge {" in body
    assert "color-mix(in srgb, var(--pos, #64748b) 13%, var(--card))" in body


def test_board_headshot_resolution_and_fallback():
    source = _source()

    assert "function drBoardHsHtml(pl){" in source
    helper = source.split("function drBoardHsHtml(pl){", 1)[1].split(
        "\n  function cellInner(pn){", 1
    )[0]
    # Picks carry no headshot field: resolve the full pool record first.
    assert "playersById[String(pl.id)]" in helper
    assert "full.espnHeadshot" in helper
    # ESPN id builds the canonical full headshot URL.
    assert "'https://a.espncdn.com/i/headshots/nfl/players/full/' + eid + '.png'" in helper
    # Existing builder remains the next fallback (Sleeper CDN / team logo).
    assert "playerImgUrl(full.id != null ? full : pl)" in helper
    # Graceful fallback: the initials disc is in the zone, the img hides it
    # on load and removes itself on error; DEF renders the disc alone.
    assert 'class="dr-hs dr-ph-hs' in helper
    assert "onerror=\"this.remove()\"" in helper
    assert "pos !== 'DEF'" in helper
    # Board cells use the R1 helper, not the old circular headshot.
    inner = _cell_inner(source)
    assert "drBoardHsHtml(pl)" in inner
    assert "drHsHtml(pl, '')" not in inner


def test_filled_cell_corners_and_top_right_slot_priority():
    source = _source()
    inner = _cell_inner(source)

    # Stripe element carries the position colour; pick number stays top-left.
    assert '<span class="stripe" style="background:\' + posColor(pl.position) + \'"></span>' in inner
    assert '<span class="dr-cell-num">\' + pn + \'</span>' in inner
    # One top-right occupant: Last pick flag, then KEEP, then the traded-away
    # owner, then a traded-in YOU, and only then the display value.
    filled = inner.split("var h = ", 1)[1]
    order = [
        filled.index("dr-cell-lastflag"),
        filled.index("dr-cell-keepflag"),
        filled.index("dr-cell-owner"),
        filled.index("dr-cell-mineflag"),
        filled.index("dr-cell-val"),
    ]
    assert order == sorted(order)
    assert '<span class="dr-cell-lastflag">Last pick</span>' in inner
    assert "pn === justPick" in inner
    # The badge no longer hardcodes a solid inline background in cells.
    assert '<span class="dr-posbadge">\' + esc(pl.position) + \'</span>' in inner

    body = _body()
    assert ".dr-cell-lastflag {" in body
    assert ".dr-cell-val { position: absolute; top: 3px; right: 6px;" in body
    assert ".dr-cell-num { position: absolute; top: 3px; left: 6px;" in body


def test_empty_and_on_the_clock_cells():
    source = _source()
    inner = _cell_inner(source)
    empty = inner.split("if (!pl){", 1)[1].split("return eh;", 1)[0]

    # On the clock: label inside the empty cell, and no photo markup at all
    # in the empty branch (the headshot helper is only called for picks).
    assert '<span class="dr-cell-otc">On the clock</span>' in empty
    assert "pn === state.current" in empty
    assert "drBoardHsHtml" not in empty
    # Upcoming empties show the pick number; the centered round.pick is kept
    # only for owned/traded empties, under their flags.
    assert '<span class="dr-cell-num">\' + pn + \'</span>' in empty
    assert "else if (isMyPick(pn) || _eown)" in empty
    assert '<span class="dr-cell-mineflag">YOU</span>' in empty
    assert "tradedOwnerLabel(pn)" in empty

    body = _body()
    assert ".dr-cell-otc {" in body
    # The pulsing ring still marks the current cell.
    assert ".dr-cell-current { box-shadow: inset 0 0 0 2px var(--accent,#38bdf8); z-index: 1; animation: drPulse" in body
    assert ".dr-cell-empty { background: var(--card); }" in body
    assert "border-style: dashed" not in body.split(".dr-cell-empty {", 1)[1].split("}", 1)[0]


def test_you_marker_only_on_viewer_column_and_owned_picks():
    source = _source()
    board = source.split("function buildBoard(){", 1)[1].split(
        "\n  // Per-cell CSS var", 1
    )[0]

    # The header pill renders only for a column the viewer fully owns, with
    # the seat name under it; partial ownership keeps name + star, no pill.
    assert "ownsAllInColumn(s) ? '<span class=\"dr-youpill\">YOU</span><span class=\"dr-colhead-name\">'" in board
    assert "ownsAny ? ' ★' : ''" in board
    assert board.count("dr-youpill") == 1
    # Filled home-slot picks carry no in-cell YOU flag (the header pill and
    # the mine wash mark them); the flag is for traded-in picks only.
    inner = _cell_inner(source)
    assert "slotOnClock(pn, state.teams, state.order) !== state.slot" in inner

    body = _body()
    assert ".dr-colhead-name {" in body
    assert ".dr-cell-mine { box-shadow: inset 3px 0 0 var(--accent,#38bdf8);" in body
    assert ".dr-cell-claimed { box-shadow: inset 3px 0 0 var(--warning); }" in body


def test_changelog_entry_for_r1_board():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        item for item in CHANGELOG
        if "draft room" in item.get("text", "").lower()
        and "headshot" in item.get("text", "").lower()
    )
    assert entry["tag"] == "update"
    assert entry["link"] == "/draft"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]
