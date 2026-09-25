"""Sticky leading player column for the keeper table and the Front Office
Report roster table.

Both tables live in horizontal scroll wrappers; the identifying Player cell
must stay pinned at left:0 with an opaque, theme-correct background that
mirrors every row state (normal, hover) so scrolled cells slide under cleanly.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEEPER_SRC = (ROOT / "dashboard_services/pages/_keeper_render.py").read_text(encoding="utf-8")
FOR_SRC = (ROOT / "dashboard_services/ai/front_office_report.py").read_text(encoding="utf-8")
FOR_CSS = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")


def _rendered(src: str) -> str:
    # _keeper_render.py is an f-string template; collapse the doubled CSS
    # braces the way the f-string render would.
    return src.replace("{{", "{").replace("}}", "}")


def _flat(s: str) -> str:
    return s.replace(" ", "").replace("\n", "")


def test_keeper_player_column_is_sticky():
    css = _rendered(KEEPER_SRC)
    # Both the header cell and every body player cell pin to the left edge.
    assert ".kpr-tbl thead th:first-child" in css
    assert ".kpr-tbl tbody td:first-child" in css
    assert "position:sticky;left:0" in _flat(css)
    sticky_block = _flat(css[css.find(".kpr-tbl thead th:first-child"):][:600])
    # Opaque background matching the wrapping .card (var(--card)) so cells
    # scrolling underneath do not show through.
    assert "background:var(--card)" in sticky_block
    # Separation edge between the pinned column and the scrolled columns.
    assert "border-right:1pxsolidvar(--border)" in sticky_block


def test_keeper_sticky_cell_mirrors_row_hover():
    css = _rendered(KEEPER_SRC)
    # The row hover background must be replicated on the pinned cell, or the
    # sticky cell would keep the card background while the row highlights.
    assert ".kpr-tbl tbody tr:hover td:first-child" in css
    hover_block = css[css.find(".kpr-tbl tbody tr:hover td:first-child"):][:160]
    assert "var(--card-soft" in hover_block


def test_keeper_player_cell_is_first_column():
    # keeper.js renders the player cell (badge + name + round/years controls)
    # as the first <td> of each row; the sticky selectors depend on that.
    js = (ROOT / "static/keeper.js").read_text(encoding="utf-8")
    assert "'<td><div class=\"kpr-nm-line\">" in js
    # And it immediately follows the opening <tr> of the row template.
    assert "'<tr data-pid=\"' + esc(row.p.id) + '\">' +\n        '<td><div class=\"kpr-nm-line\">" in js


def test_front_office_player_cell_has_name_class():
    # The sticky CSS keys off .for-td-name, so the player cell must keep it.
    assert "class='for-td-name'" in FOR_SRC
    assert "<th>Player</th>" in FOR_SRC


def test_front_office_sticky_css_dropin():
    # Drop-in file for static/dashboard.css (kept separate to avoid edit
    # collisions during the sticky-columns branch).
    css = FOR_CSS
    assert ".for-table td.for-td-name" in css
    assert ".for-table thead th:first-child" in css
    assert "position: sticky" in css
    assert "left: 0" in css
    assert "border-right: 1px solid var(--border)" in css
    # Opaque backgrounds: light theme matches .for-modal (var(--card)),
    # dark theme matches the explicit #1a2535 modal background.
    assert "background: var(--card)" in css
    assert '[data-theme="dark"] .for-table td.for-td-name' in css
    assert "#1a2535" in css
    # The header cell keeps its existing var(--card-soft) background from the
    # base .for-table th rule; only stacking/separation are added here.
    assert "var(--card-soft)" in css
