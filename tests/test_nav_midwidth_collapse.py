"""Top nav: mid-width (<=1400px) pill collapse behind the hamburger.

Regression test: the 8 league pills need ~1370px to sit on one row. Between
the phone dock (<=768px) and a wide desktop they used to flex-wrap into
ragged rows (e.g. "My Leagues" alone on a row). Instead they must collapse
into the hamburger dropdown -- the same pattern the <=768px menu uses.
"""
from __future__ import annotations

import re
from pathlib import Path

_CSS = Path(__file__).resolve().parents[1] / "static" / "dashboard.css"


def _media_blocks(max_width_px: int) -> list[str]:
    """Return the bodies of every @media (max-width: Npx) block in the CSS."""
    css = _CSS.read_text(encoding="utf-8")
    blocks: list[str] = []
    for m in re.finditer(
        r"@media\s*\(\s*max-width:\s*" + str(max_width_px) + r"px\s*\)\s*\{", css
    ):
        depth = 1
        i = m.end()
        while i < len(css) and depth:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        blocks.append(css[m.end() : i - 1])
    return blocks


def _nav_block() -> str:
    """The 1400px block that carries the mid-width nav collapse rules."""
    cands = [b for b in _media_blocks(1400) if ".top-nav.br-mnav .nav-hamburger" in b]
    assert cands, "expected a 1400px media block with the mid-width nav rules"
    return cands[0]


def test_hamburger_shows_at_mid_widths():
    block = _nav_block()
    assert re.search(
        r"\.top-nav\.br-mnav\s+\.nav-hamburger\s*\{[^}]*display:\s*flex",
        block,
    ), "hamburger must become visible at <=1400px"


def test_pills_collapse_into_dropdown_at_mid_widths():
    block = _nav_block()
    m = re.search(
        r"\.top-nav\.br-mnav\s+\.nav-pills-container\s*\{([^}]*)\}", block
    )
    assert m, "pills container must be restyled at <=1400px"
    body = m.group(1)
    assert "position: absolute" in body, "collapsed menu must drop below the bar"
    assert "flex-direction: column" in body, "collapsed menu must stack pills"
    assert "max-height: 0" in body, "collapsed menu must start hidden"
    assert "overflow: hidden" in body


def test_pills_expand_when_menu_opens():
    block = _nav_block()
    m = re.search(
        r"\.top-nav\.br-mnav\s+\.nav-pills-container\.nav-open\s*\{([^}]*)\}",
        block,
    )
    assert m, ".nav-open must expand the collapsed menu"
    assert "max-height: 70vh" in m.group(1)


def test_submenus_expand_inline_in_collapsed_menu():
    block = _nav_block()
    m = re.search(
        r"\.top-nav\.br-mnav\s+\.nav-pill-dropdown-menu\s*\{([^}]*)\}", block
    )
    assert m, "dropdown submenus need mid-width rules too"
    assert "position: static" in m.group(1), (
        "submenu must expand inline instead of floating over the menu"
    )
