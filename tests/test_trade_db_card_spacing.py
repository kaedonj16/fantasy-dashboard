"""Focused tests for the trade-database card grid spacing.

Regression test for Kaedon's 2026-10-08 report: some trade cards rendered a
large empty white area below the "Fair deal" footer while row-mates were
compact. Root cause: `.tdb-list` is a CSS grid whose items stretch to the
row height by default, and `.tdb-card` lays its head/body/foot out in normal
flow from the top, so a shorter card's stretched remainder showed as dead
space below the footer.

Her revision (2026-10-08): keep the cards equal height per row, but pin the
footer to the bottom of each card so the leftover whitespace sits ABOVE the
footer inside the card and all footers line up in a flat row. Implemented as:
grid stretch (no `align-items: start`) + `.tdb-card` as a column flexbox +
`margin-top: auto` on `.tdb-card-foot`.

These tests read the page's inline CSS from the route source (no Flask app
import, no live database).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "routes" / "trade_bp.py").read_text()


def _rule(selector: str) -> str:
    # The page CSS lives inside an f-string, so braces are doubled in source.
    m = re.search(re.escape(selector) + r"\s*\{\{(.*?)\}\}", SRC, re.S)
    assert m, f"no inline CSS rule found for {selector}"
    return m.group(1)


def test_tdb_list_stretches_cards_to_equal_row_height():
    rule = _rule(".tdb-list")
    assert not re.search(r"align-items\s*:\s*start", rule), (
        "tdb-list grid must NOT use align-items: start; cards in a row must "
        "stretch to equal height so footers can line up in a flat row"
    )


def test_tdb_card_is_column_flex():
    rule = _rule(".tdb-card")
    assert re.search(r"display\s*:\s*flex", rule), "tdb-card must be display: flex"
    assert re.search(r"flex-direction\s*:\s*column", rule), (
        "tdb-card must be flex-direction: column so the footer can pin to the bottom"
    )


def test_tdb_card_foot_pinned_to_bottom():
    rule = _rule(".tdb-card-foot")
    assert re.search(r"margin-top\s*:\s*auto", rule), (
        "tdb-card-foot must use margin-top: auto so the footer pins to the "
        "card bottom and leftover space sits above the footer, not below it"
    )


def test_tdb_card_has_no_forced_height():
    rule = _rule(".tdb-card")
    assert not re.search(r"min-height\s*:", rule), "tdb-card must not force a min-height"
    assert not re.search(r"height\s*:\s*100%", rule), "tdb-card must not force height: 100%"
