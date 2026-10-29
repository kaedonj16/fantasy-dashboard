"""Usage risers row layout: stats and bars sit below the name.

Kaedon's screenshot (2026-10-29): on the home Usage risers card the
trend block sat beside the name and squeezed it into truncation
("Ollie Gordo...", "Drake Lon..."). The row is now a grid: name and
delta chip on the first line, the usage detail + mini bars on a second
line below the name, right-aligned, so the name gets the full row
width minus the chip. These tests lock that layout in the shipped CSS.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_CSS_PATH = os.path.join(_ROOT, "static", "dashboard.css")


def _read_css():
    with open(_CSS_PATH, encoding="utf-8") as fh:
        return fh.read()


def _rule_body(css, selector):
    """Return the declaration body of the first rule for `selector`."""
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, f"rule not found: {selector}"
    return re.sub(r"\s+", " ", match.group(1)).strip().lower()


def test_row_is_grid_with_trend_below_name():
    body = _rule_body(_read_css(), ".usage-mover")
    assert "display: grid" in body
    areas = re.search(r"grid-template-areas:\s*\"([^\"]+)\"\s*\"([^\"]+)\"", body)
    assert areas, f"row must declare two stacked grid areas: {body}"
    assert areas.group(1).split() == ["main", "delta"]
    assert areas.group(2).split() == ["trend", "trend"]


def test_name_area_keeps_full_width():
    body = _rule_body(_read_css(), ".usage-mover .um-main")
    assert "grid-area: main" in body
    assert "min-width: 0" in body


def test_trend_area_is_below_the_name():
    body = _rule_body(_read_css(), ".usage-mover .um-trend")
    assert "grid-area: trend" in body
    # No side-by-side flex sizing survives: the trend must not reserve
    # width next to the name again.
    assert "flex:" not in body


def test_detail_readout_stays_right_aligned():
    body = _rule_body(_read_css(), ".usage-mover .um-detail")
    assert "text-align: right" in body
    assert "font-weight: 600" in body


def test_bars_are_labelled_season_and_last_3():
    css = _read_css()
    row = _rule_body(css, ".usage-mover .um-bar-row")
    assert "display: flex" in row
    lab = _rule_body(css, ".usage-mover .um-bar-lab")
    assert "text-transform: uppercase" in lab
    with open(os.path.join(_ROOT, "app.py"), encoding="utf-8") as fh:
        app_src = fh.read()
    assert '<span class="um-bar-lab">Season</span>' in app_src
    assert '<span class="um-bar-lab">Last 3</span>' in app_src


def test_delta_chip_stays_on_the_name_line():
    body = _rule_body(_read_css(), ".usage-mover .um-delta")
    assert "grid-area: delta" in body
