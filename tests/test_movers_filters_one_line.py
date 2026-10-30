"""Player Insights Movers controls must sit on ONE row.

Regression test for Kaedon's 2026-10-30 correction. Her original ask
("fix this and make it one line", with a screenshot of the trade
calculator's Player Insights panel) meant the FILTER CONTROLS: the
time range (7d / 14d / 30d) and the Dynasty / Redraft toggle stacked
as two separate chip rows. (An earlier PR, #2908, fixed the scope
subtitle instead; that change is merged and stays. This test pins the
controls row she actually meant.)

Fix:
- The template wraps both chip groups in a single
  .otc-movers-filter-row container.
- CSS makes that row a nowrap flex row carrying the panel spacing the
  first group used to carry; the inner groups drop their own panel
  margin/padding and never wrap, and the format group pins right with
  margin-left: auto.
- Tab switching in app.js toggles the row as one unit (the two groups
  always showed and hid together anyway).

These tests read the sources directly (no Flask app import).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()
PAGE = (ROOT / "dashboard_services" / "pages" / "trade_calculator_page.py").read_text()


def _rule(selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r" \{\n(.*?)\n\}", CSS, re.S)
    assert m, f"no CSS rule found for {selector}"
    return m.group(1)


def test_template_wraps_both_groups_in_one_filter_row():
    start = PAGE.index('<div class="otc-movers-filter-row">')
    end = PAGE.index('<div id="moversTabContent"', start)
    block = PAGE[start:end]
    # Time range first, format second, both inside the row.
    assert 'data-days="7"' in block and 'data-days="30"' in block
    assert 'data-movers-format="dynasty"' in block
    assert 'data-movers-format="redraft"' in block
    assert block.index('data-days="7"') < block.index('data-movers-format="dynasty"')
    # The format group keeps its JS/ARIA hooks.
    assert 'otc-movers-format-filters' in block
    assert 'aria-label="Movers value format"' in block


def test_filter_row_is_a_single_nowrap_flex_row():
    rule = _rule(".otc-movers-panel .otc-movers-filter-row")
    assert re.search(r"display\s*:\s*flex", rule)
    assert re.search(r"flex-wrap\s*:\s*nowrap", rule), (
        "the movers filter row must never wrap: one line is the whole point"
    )


def test_inner_groups_drop_panel_spacing_and_never_wrap():
    rule = _rule(".otc-movers-panel .otc-movers-filter-row .otc-day-filters")
    assert re.search(r"margin-top\s*:\s*0", rule), (
        "inner groups must not stack their own 8px panel margin inside the row"
    )
    assert re.search(r"padding\s*:\s*0", rule), (
        "inner groups must not double the row's 12px side padding"
    )
    assert re.search(r"flex-wrap\s*:\s*nowrap", rule)


def test_format_group_pins_to_the_right_edge():
    rule = _rule(".otc-movers-panel .otc-movers-filter-row .otc-movers-format-filters")
    assert re.search(r"margin-left\s*:\s*auto", rule), (
        "Dynasty/Redraft should pin right, opposite the time range"
    )


def test_tab_switching_toggles_the_row_as_one_unit():
    assert 'root.querySelector(".otc-movers-panel .otc-movers-filter-row")' in APP_JS
    assert 'filterRow.style.display = "flex"' in APP_JS
    assert APP_JS.count('filterRow.style.display = "none"') == 2, (
        "Breakouts and Targets tabs must both hide the whole filter row"
    )
    assert 'formatFilters' not in APP_JS, (
        "the old per-group show/hide wiring must not come back"
    )
