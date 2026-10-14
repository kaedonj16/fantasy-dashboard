"""Contract tests for the desktop standings power-panel stretch.

On the desktop standings page the Power Rankings card fills the standings
card's height and each tab view distributes its rows through the leftover
space. That behavior is pure CSS in static/dashboard.css (a min-width:
1025px block scoped to .std-rework); these tests lock the rules in place
so the panel cannot silently fall back to natural height with dead space
under the last row. Layout itself is verified in a browser harness; these
guard the stylesheet contract.
"""
import os
import re

_CSS_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "dashboard.css")


def _css():
    with open(_CSS_PATH, encoding="utf-8") as fh:
        return fh.read()


def _stretch_block(css):
    """The @media (min-width: 1025px) block holding the stretch rules."""
    idx = css.find("Power panel stretch (desktop standings two-col)")
    assert idx != -1, "power panel stretch block is missing"
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
    # Every rule in the block is scoped to the standings page body so other
    # surfaces reusing the power card markup are untouched.
    block = re.sub(r"/\*.*?\*/", "", block, flags=re.S)
    selectors = re.findall(r"([^{}]+)\{", block)
    scoped = [s.strip() for s in selectors if s.strip() and not s.strip().startswith("@media")]
    assert scoped, "no rules found in stretch block"
    for sel in scoped:
        for part in sel.split(","):
            assert ".std-rework" in part, f"unscoped selector in stretch block: {part.strip()}"


def test_layout_row_stretches_columns():
    block = _stretch_block(_css())
    assert ".standings-main.two-col-standings { align-items: stretch; }" in block
    assert ".standings-col > .card { flex: 1 1 auto;" in block
    assert "#stPowerInner" in block


def test_power_card_flex_chain_and_row_distribution():
    block = _stretch_block(_css())
    assert ".card.power {" in block and "flex-direction: column" in block
    assert ".card.power .tab-panel.active" in block
    assert ".card.power .pwr-lead { flex: 0 0 auto; }" in block
    assert ".card.power .pwr-list { flex: 1 1 auto;" in block
    assert ".card.power .pwr-row { flex: 1 1 0; }" in block


def test_other_tab_views_fill_too():
    block = _stretch_block(_css())
    assert ".card.power .bracket { flex: 1 1 auto;" in block
    assert ".card.power .po-wrap" in block
    assert ".card.power .po-table { flex: 1 1 auto; }" in block


def test_stretch_rules_do_not_leak_outside_block():
    css = _css()
    block = _stretch_block(css)
    outside = css.replace(block, "")
    assert ".card.power .pwr-row { flex: 1 1 0; }" not in outside
    assert ".card.power .tab-panel.active" not in outside
