"""Focused tests for the trade-database unified toolbar (2026-10-22).

Kaedon's screenshot of the live desktop toolbar: a flex:1 search card
beside a bottom-aligned filter column of three unlabeled, unequal-width
chip rows (3/4/4) with ragged right edges, dead space above the chips and
inside the search card, and Sort dangling on its own row.

The approved mock replaces the two blocks with ONE toolbar card: the
search row on top (now chrome-less), a hairline divider, then a single
row of labeled groups (Format / League type / Trades within) with Sort
as a fourth labeled group pinned right. All filter behavior (classes,
data attrs, onclick handlers, the tdbSort select) is unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE_SRC = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")


def _rule(selector: str) -> str:
    m = re.search(re.escape(selector) + r"\s*\{\{(.*?)\}\}", PAGE_SRC, re.S)
    assert m, f"no inline CSS rule found for {selector}"
    return m.group(1)


def test_toolbar_is_one_card():
    rule = _rule(".tdb-toolbar")
    assert "display: block" in rule
    assert "background: var(--card)" in rule
    assert "border: 1px solid var(--border)" in rule
    assert "border-radius: 14px" in rule


def test_search_row_carries_no_card_chrome_of_its_own():
    rule = _rule(".tdb-sides-row.tdb-search-card")
    assert "border: none" in rule
    assert "padding: 0" in rule


def test_old_filter_column_is_gone():
    assert "tdb-filter-col" not in PAGE_SRC
    assert "tdb-sort-label" not in PAGE_SRC


def test_filter_row_is_labeled_groups_with_hairline_divider():
    rule = _rule(".tdb-filter-row")
    assert "display: flex" in rule
    assert "border-top: 1px solid" in rule
    assert "align-items: flex-end" in rule
    for label in ("Format", "League type", "Trades within", "Sort"):
        assert f">{label}<" in PAGE_SRC
    # Sort is a labeled group pinned to the right edge of the row.
    sort_rule = _rule(".tdb-fgroup-sort")
    assert "margin-left: auto" in sort_rule
    assert '<label class="tdb-fgroup-label" for="tdbSort">Sort</label>' in PAGE_SRC


def test_filter_behavior_is_unchanged():
    # Same chip classes, data attrs and handlers as before the rework.
    for val in ("all", "1qb", "sf"):
        assert f'data-lt="{val}"' in PAGE_SRC
        assert f"tdbFilter('{val}')" in PAGE_SRC
    for val in ("all", "dynasty", "redraft", "keeper"):
        assert f'data-lf="{val}"' in PAGE_SRC
        assert f"tdbFormatFilter('{val}')" in PAGE_SRC
    for val in ("all", "7d", "30d", "90d"):
        assert f'data-tr="{val}"' in PAGE_SRC
        assert f"tdbRangeFilter('{val}')" in PAGE_SRC
    assert 'id="tdbSort"' in PAGE_SRC
    assert 'onchange="tdbSortChange(this.value)"' in PAGE_SRC


def test_phone_stacks_groups_full_width():
    # The toolbar's phone block is the max-600 block that restyles
    # .tdb-toolbar (a separate one-liner block handles the card list).
    start = PAGE_SRC.find("@media(max-width: 600px) {{\n        .tdb-toolbar")
    assert start != -1, "toolbar phone media block not found"
    block = PAGE_SRC[start:start + 1600]
    assert ".tdb-fgroup {{ width: 100%; }}" in block
    assert ".tdb-fgroup-sort {{ margin-left: 0; }}" in block
