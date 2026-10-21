"""Contract tests for the desktop standings Value Share table stretch.

On the desktop standings page the left card stretches to the two-column
row height (set by the power card), and the Value Share tab's table
fills that height: a flex chain runs card -> tab panels -> shares
panel -> scroll wrapper -> table, and the table's rows distribute the
leftover height instead of leaving dead space under the last row.
That behavior is pure CSS in static/dashboard.css (a min-width: 1025px
block scoped to .std-rework); these tests lock the rules in place.
Layout itself is verified in a browser harness; these guard the
stylesheet contract.
"""
import os
import re

_CSS_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "dashboard.css")


def _css():
    with open(_CSS_PATH, encoding="utf-8") as fh:
        return fh.read()


def _stretch_block(css):
    """The @media (min-width: 1025px) block holding the stretch rules."""
    idx = css.find("Value Share table stretch (desktop standings two-col)")
    assert idx != -1, "value share stretch block is missing"
    start = css.find("@media (min-width: 1025px)", idx)
    assert start != -1, "stretch rules are not inside a min-width: 1025px block"
    depth = 0
    for i in range(css.index("{", start), len(css)):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[start:i + 1]
    raise AssertionError("unbalanced braces in stretch block")


def test_stretch_block_scoped_to_desktop_and_standings():
    block = _stretch_block(_css())
    assert "@media (min-width: 1025px)" in block
    # Every rule in the block is scoped to the standings page body so
    # other surfaces reusing the shares table markup are untouched.
    block = re.sub(r"/\*.*?\*/", "", block, flags=re.S)
    selectors = re.findall(r"([^{}]+)\{", block)
    scoped = [s.strip() for s in selectors if s.strip() and not s.strip().startswith("@media")]
    assert scoped, "no rules found in stretch block"
    for sel in scoped:
        for part in sel.split(","):
            assert ".std-rework" in part, f"unscoped selector in stretch block: {part.strip()}"


def test_card_and_tab_chain_flexes():
    block = _stretch_block(_css())
    assert ".standings-col > .card {" in block and "flex-direction: column" in block
    assert ".card > .card-tabs { flex: 1 1 auto;" in block
    assert ".card .tab-panels {" in block and "flex: 1 1 auto;" in block
    assert ".tab-panel.standings-shares-panel.active {" in block


def test_shares_view_and_table_fill():
    block = _stretch_block(_css())
    assert "#stSharesInner {" in block and "flex: 1 1 auto;" in block
    assert ".standings-shares {" in block
    assert ".standings-shares-note { flex: 0 0 auto; }" in block
    assert ".standings-shares .st-tblscroll {" in block
    assert ".standings-shares-table { flex: 1 1 auto; }" in block


def test_stretch_rules_do_not_leak_outside_block():
    css = _css()
    block = _stretch_block(css)
    outside = css.replace(block, "")
    assert ".standings-shares-table { flex: 1 1 auto; }" not in outside
    assert ".tab-panel.standings-shares-panel.active" not in outside
