"""Matchup stat-row gap (Kaedon's screenshots, 2026-10-19).

Two stacked causes put a blank between the FINAL scoreline and the
box-score line:

1. The stat line is a separate .mb-stat-row grid row under the drive-bar
   slot, so the player cell's 8px bottom padding, the .mb-row 3px grid
   gap and the stat row's own 2px top padding stacked into an ~18px
   blank -- even for final games, where the drive-bar mount is empty
   and collapses. The approved v8 contract (PR #2507) is that stats sit
   directly under the game line when no drive bar renders.
2. The legacy .m-combo .p-game-line rule re-enables wrapping on the
   game-line span (higher specificity than .mb-game's nowrap), so a
   long game line wraps to two lines and that player's cell grows a
   line taller. The stat row is shared across the row, so the OTHER
   player's stats drop that far below their scoreline.

Fix: when a stat row (or a filled drive bar) follows, the cell keeps no
bottom padding and the stat row cancels the grid gap with a negative
top margin (the only space left is line-height leading, the same rhythm
as the lines inside the cell); and in stat rows the game line is pinned
to the single line the compact design intends. Rows with no stats keep
the wrapping segmented pregame lines, untruncated.

Replica-measured (390px and desktop): scoreline-to-stats line-box gap
13px -> 1px, both cells equal height, full final line fits untruncated.
Final scorelines are additionally pinned single-line in EVERY row via
the server-side .mb-game--final tag (a stat-less final row could still
wrap before), verified one line with the full text inside the cell at
430/390/375/360px.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
MATCHUPS = (
    ROOT / "dashboard_services" / "matchups.py"
).read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")


def test_cell_padding_drops_only_when_a_lower_row_follows():
    assert ".mb-row:has(> .mb-stat-row) > .mb-cell," in CSS
    assert ".mb-row:has(> .mb-fld:not(:empty)) > .mb-cell" in CSS
    assert "{ padding-bottom: 0; }" in CSS
    # Rows with no stat row and an empty drive bar keep the base padding:
    # the compact cell rule itself is untouched.
    assert ".mb-cell { padding: 8px 12px; }" in CSS


def test_stat_row_no_longer_stacks_its_own_top_padding():
    assert ".mb-stat-row { min-width: 0; padding: 0 9px; margin-top: -3px; }" in CSS
    # Under a filled drive bar the stat row keeps 1px of the grid gap.
    assert ".mb-row:has(> .mb-fld:not(:empty)) .mb-stat-row { margin-top: -2px; }" in CSS
    assert "padding: 2px 9px 0" not in CSS


def test_game_line_stays_single_line_in_stat_rows():
    # Scoped to stat rows: pregame-only rows keep wrapping, untruncated.
    assert ".mb-row:has(> .mb-stat-row) .mb-game {" in CSS
    assert "display: block; white-space: nowrap; overflow: hidden;" in CSS


def test_final_scoreline_never_wraps_in_any_row():
    # Kaedon: the "Final W 31-23 @ ATL" line must not wrap -- the server
    # tags final game lines and the CSS pins them single-line everywhere,
    # with a narrow-phone step-down (specificity beating .m-combo .meta)
    # so the whole line fits instead of ellipsizing.
    assert '" mb-game--final" if (status == STATUS_FINAL and not is_bye)' in MATCHUPS
    assert ".mb-row .mb-game--final { display: block; white-space: nowrap; }" in CSS
    assert ".m-combo .mb-game--final { font-size: 8px; }" in CSS


def test_stat_row_still_renders_under_the_drive_bar():
    # Server order in the .mb-row: drive-bar mounts first, stat rows after.
    fld_at = MATCHUPS.index("{left_fld}")
    stat_at = MATCHUPS.index("{left_stat_row}")
    assert fld_at < stat_at


def test_service_worker_bumped_for_the_new_css():
    assert "br-fantasy-v116" in SW
