"""Player Rankings sort-switch fixes (2026-10-30, Kaedon's screenshots).

Bug 1: clicking the Value column header while another sort was active did
not sort by value. The trailing header cell carries data-pr-sort-col="sort"
and _prHeaderSortKey mapped that to prSortKey (the CURRENT key), so during
a PPG sort a click on "Value" just flipped PPG desc -> asc, surfacing the
-2.0 PPG players with their server ranks (#556, #555, ...) while the PPG
header kept the arrow. The header is the Value column: it must resolve to
'value' (flipping only when Value is already active, which prHeaderSort
handles). Sole exception: on phones a Total Points sort renames this
header to "Total" (the Total column is hidden there), so while it carries
that label it keeps carrying the Total Points sort.

Bug 2: switching Dynasty <-> Redraft moved the header controls. The
subtitle is longer in Redraft ("this-season redraft value"), and Row 1 was
a flex row, so the wider title block pushed the format segments and the
search sideways. Row 1 is now a grid with fixed tracks for the segments
and search; the title column is 1fr and the subtitle is one line, so
neither the horizontal positions nor the row height can change with the
subtitle text. Phones keep the approved flex stack.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = (ROOT / "dashboard_services" / "pages" / "players_page.py").read_text(encoding="utf-8")
JS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")


def _header_sort_key_body():
    m = re.search(r"function _prHeaderSortKey\(cell\) \{(.*?)\n  \}\n", JS, re.S)
    assert m, "_prHeaderSortKey not found"
    return m.group(1)


def test_value_header_resolves_to_value_not_current_key():
    body = _header_sort_key_body()
    # The legacy bug, verbatim: 'sort' mapped to whatever key was active.
    assert "col === 'sort' ? prSortKey" not in JS
    assert "? prSortKey : col" not in JS
    # The Value column header sorts by value.
    assert "'value'" in body
    assert "return col;" in body  # ppg / total_pts headers keep their own keys


def test_sort_header_total_exception_for_phone_total_sorts():
    body = _header_sort_key_body()
    # Only while the header literally reads "Total" (phone Total Points
    # sort, where the Total column is hidden) does it carry total_pts.
    assert "prSortKey === 'total_pts'" in body
    assert "textContent.trim() === 'Total'" in body
    assert "'total_pts' : 'value'" in body


def test_total_sort_arrow_lands_on_visible_header():
    # prUpdateSortHeaders: during a phone Total Points sort the arrow goes
    # on the trailing "Total" header, not the hidden Total Points column.
    m = re.search(r"function prUpdateSortHeaders\(\) \{(.*?)\n\}\n", JS, re.S)
    assert m, "prUpdateSortHeaders not found"
    body = m.group(1)
    assert "prSortKey === 'total_pts'" in body
    assert "textContent.trim() === 'Total') ? 'sort' : 'total_pts'" in body


def test_header_row1_fixed_tracks_so_controls_do_not_move():
    # Fixed tracks for the two format segments + search; title column 1fr.
    assert "grid-template-columns: minmax(0, 1fr) auto auto minmax(190px, 240px);" in PAGE
    m = re.search(r"\.pr-head-row1 \{(.*?)\}", PAGE, re.S)
    assert m and "display: grid;" in m.group(1)
    # The search no longer sizes off the title block via margin-left:auto.
    m = re.search(r"\.pr-head-row1 \.filter-search \{(.*?)\}", PAGE, re.S)
    assert m and "margin-left: 0;" in m.group(1)


def test_subtitle_one_line_and_phone_stack_preserved():
    # One line on the grid header: a wrap would grow the row and push
    # Row 2 down when the Redraft subtitle is longer.
    m = re.search(r"\.pr-titleblock \.page-head-sub \{(.*?)\}", PAGE, re.S)
    assert m and "white-space: nowrap;" in m.group(1)
    # Phone (<=640px) keeps the approved flex stack and wrapping subtitle.
    phone = PAGE.split("@media (max-width: 640px)")[1]
    assert ".pr-head-row1 { display: flex; flex-wrap: wrap; gap: 8px; }" in phone
    assert ".pr-titleblock .page-head-sub { white-space: normal;" in phone


def test_changelog_entry_present_by_content():
    from pin_helpers import changelog_entry

    entry = changelog_entry("Player Rankings sorting: clicking the Value column header", link="/players")
    assert entry is not None
