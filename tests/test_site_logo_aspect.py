"""Top-nav logo must never render distorted at smaller window widths.

Regression test: at <=1024px the CSS set only ``max-height: 42px`` on
``.site-logo``. The nav logo keeps ``width: 180px`` from the more specific
``.nav-left .site-logo`` rule, so the rendered box became 180x42 (ratio
4.29) instead of the intrinsic 512x211 (ratio 2.43) of Website_Logo.png:
the logo looked stretched/squashed in the 769-1024px window band (it was
fine wider, and fine at <=768px where the width drops to 92px). The fix
shrinks the logo proportionally (~102px wide for ~42px tall) with
selectors specific enough to beat the .nav-left width, and the base rule
carries object-fit: contain so a future one-sided cap cannot distort it.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CSS = _ROOT / "static" / "dashboard.css"

# Website_Logo.png intrinsic size (verified from the shipped asset).
_INTRINSIC_W, _INTRINSIC_H = 512, 211


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


def _logo_rule_at_1024() -> str:
    """The <=1024px block that restyles the top nav / logo."""
    cands = [b for b in _media_blocks(1024) if ".top-nav" in b and ".site-logo" in b]
    assert cands, "expected a 1024px media block with the site-logo rule"
    block = cands[0]
    m = re.search(r"\.site-logo[^{]*\{([^}]*)\}", block)
    assert m, "site-logo rule missing from the 1024px block"
    return m.group(1)


def test_1024_logo_rule_does_not_cap_height_alone():
    body = _logo_rule_at_1024()
    assert "max-height: 42px" not in body, (
        "capping height alone while width stays 180px squashes the logo"
    )


def test_1024_logo_shrinks_proportionally_to_about_42px_tall():
    body = _logo_rule_at_1024()
    m = re.search(r"width:\s*(\d+(?:\.\d+)?)px", body)
    assert m, "the <=1024px logo rule must set a proportional width"
    width = float(m.group(1))
    height = width * _INTRINSIC_H / _INTRINSIC_W
    assert abs(height - 42) <= 1.5, (
        f"width {width}px renders {height:.1f}px tall, not the intended ~42px"
    )
    assert "height: auto" in body
    assert "max-height: none" in body


def test_1024_logo_width_beats_nav_left_specificity():
    block = [b for b in _media_blocks(1024) if ".top-nav" in b and ".site-logo" in b][0]
    assert ".nav-left .site-logo" in block, (
        "a bare .site-logo width loses to .nav-left .site-logo (180px), "
        "so the <=1024px rule must repeat the higher-specificity selector"
    )


def test_base_logo_rule_guards_aspect_ratio():
    css = _CSS.read_text(encoding="utf-8")
    m = re.search(r"(?m)^\.site-logo\s*\{([^}]*)\}", css)
    assert m, "base .site-logo rule missing"
    assert "object-fit: contain" in m.group(1), (
        "base rule must letterbox rather than distort if a height cap returns"
    )


def test_logo_link_cannot_be_squeezed_by_league_chip():
    css = _CSS.read_text(encoding="utf-8")
    m = re.search(r"\.nav-left\s*>\s*a\s*\{([^}]*)\}", css)
    assert m, ".nav-left > a rule missing"
    assert "flex-shrink: 0" in m.group(1), (
        "the logo link is a flex item beside the league chip; it must not shrink"
    )
