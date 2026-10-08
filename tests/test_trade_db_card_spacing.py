"""Focused tests for the trade-database card grid spacing.

Regression test for Kaedon's 2026-10-08 report: some trade cards rendered a
large empty white area below the "Fair deal" footer while row-mates were
compact. Root cause: `.tdb-list` is a CSS grid whose items stretch to the
row height by default, and `.tdb-card` lays its head/body/foot out in normal
flow from the top, so a shorter card's stretched remainder showed as dead
space below the footer. The fix is `align-items: start` on the grid so each
card hugs its own content.

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


def test_tdb_list_does_not_stretch_cards():
    rule = _rule(".tdb-list")
    assert "display: grid" in rule.replace(";", "; ").replace("  ", " ") or "display:grid" in rule.replace(" ", "")
    assert re.search(r"align-items\s*:\s*start", rule), (
        "tdb-list grid must use align-items: start so shorter cards do not "
        "stretch to the row height and show dead space below the footer"
    )


def test_tdb_card_has_no_forced_height():
    rule = _rule(".tdb-card")
    assert not re.search(r"min-height\s*:", rule), "tdb-card must not force a min-height"
    assert not re.search(r"height\s*:\s*100%", rule), "tdb-card must not force height: 100%"
