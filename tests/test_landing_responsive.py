"""Pin the landing page's phone stacking rules.

The A+ landing grids (hero, PRO, free tools, demos) are multi-column on
desktop and must collapse inside the max-width:1024px media blocks. A
connect-card CSS rework once deleted those lines from the shared media
block (they lived in the same block as the connect stack), which left
the hero board and PRO grid crushed into slivers on phones while every
desktop check stayed green. These assertions keep the block whole.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    ".home-hero.ld-hero { grid-template-columns: 1fr; }",
    ".ld-pro { grid-template-columns: 1fr; }",
    ".ld-pro-grid { grid-template-columns: 1fr; }",
    ".ld-free-grid { grid-template-columns: 1fr; }",
    ".ld-demo-opts { grid-template-columns: 1fr 1fr; }",
    ".ld-hero-card { padding: 24px 22px; }",
    ".ld-card { padding: 22px 20px; }",
    ".ld-connect-grid { grid-template-columns: 1fr; }",
]


def _media_1024_text(css: str) -> str:
    """Concatenate every @media (max-width: 1024px) block's body."""
    out = []
    for m in re.finditer(r"@media \(max-width: 1024px\) \{", css):
        depth = 1
        i = m.end()
        while i < len(css) and depth:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        out.append(css[m.end():i - 1])
    return "\n".join(out)


@pytest.mark.parametrize("sheet", ["static/landing_lite.css", "static/dashboard.css"])
def test_landing_stacks_on_phones(sheet):
    css = (ROOT / sheet).read_text()
    block = _media_1024_text(css)
    assert block, f"no max-width:1024px block in {sheet}"
    for rule in REQUIRED:
        assert rule in block, f"{sheet} lost its phone stacking rule: {rule}"


# The signed-in home kept the old dark-card palette after the card went
# white, so the greeting and league rows rendered white-on-white, and its
# author display:flex beat the [hidden] attribute, so card states stacked
# instead of switching. Pin the light re-skin and the hidden enforcement.
SIGNED_IN_LIGHT_RULES = [
    'html:not([data-theme="dark"]) .home-card .signed-in-home-greeting { color: #64748b; }',
    'html:not([data-theme="dark"]) .home-card .signed-home-league { background: #f1f5f9; border-color: #e2e8f0; color: #122d4b; }',
    'html:not([data-theme="dark"]) .home-card .signed-home-league-open { color: #2563eb; }',
    'html:not([data-theme="dark"]) .home-card .connection-attention { color: #dc2626; }',
    'html:not([data-theme="dark"]) .home-card .home-reset-user { color: #64748b; }',
    'html:not([data-theme="dark"]) .home-card #signedInAddLeague { background: #122d4b; border-color: #122d4b; color: #fff; }',
    'html:not([data-theme="dark"]) .home-card .home-connect-back { color: #2563eb; }',
]


@pytest.mark.parametrize("sheet", ["static/landing_lite.css", "static/dashboard.css"])
def test_signed_in_home_readable_on_white_card(sheet):
    css = (ROOT / sheet).read_text()
    for rule in SIGNED_IN_LIGHT_RULES:
        assert rule in css, f"{sheet} lost its signed-in light rule: {rule}"


@pytest.mark.parametrize("sheet", ["static/landing_lite.css", "static/dashboard.css"])
def test_reconnect_button_keeps_league_name_visible(sheet):
    css = (ROOT / sheet).read_text()
    start = css.index(".reconnect-home-espn {")
    block = css[start:css.index("}", start)]
    assert "width: auto !important" in block, (
        f"{sheet}: the Reconnect button must not take the row's full width "
        "or the league name beside it is squeezed out"
    )


@pytest.mark.parametrize("sheet", ["static/landing_lite.css", "static/dashboard.css"])
def test_home_card_state_regions_hide_when_hidden(sheet):
    css = (ROOT / sheet).read_text()
    for selector in ("#signedInHome[hidden]", ".home-account-entry[hidden]",
                     ".saved-viewer-cta[hidden]", "#connectLeagueFlow[hidden]"):
        assert selector in css, f"{sheet} does not force-hide {selector}"
    assert css.count("display: none !important") >= 2, (
        f"{sheet} lost a display:none !important hidden enforcement"
    )


# The hero board stretches to the text card's height on desktop; centered
# in that height it was mostly dead space. Pin the top-aligned fill, the
# header strip, and the shifted row-animation delays (the strip is the
# board's new first child, so the first row is now :nth-child(4)).
BOARD_DESKTOP_RULES = [
    ".hp-mb-strip { display: none; }",
    ".ld-board-wrap .hp-mb { justify-content: flex-start; }",
    ".hp-mb-strip { display: flex; align-items: center; justify-content: space-between; padding: 14px 14px 0; }",
    ".hp-mb .mb-row { flex: 1 1 0; }",
    ".hp-mb-score { font-size: 36px; }",
    ".ld-board-wrap .mb-row:nth-child(4) { animation-delay: .5s; }",
    ".ld-board-wrap .mb-row:nth-child(8) { animation-delay: .98s; }",
]


@pytest.mark.parametrize("sheet", ["static/landing_lite.css", "static/dashboard.css"])
def test_desktop_board_fills_hero_height(sheet):
    css = (ROOT / sheet).read_text()
    for rule in BOARD_DESKTOP_RULES:
        assert rule in css, f"{sheet} lost its desktop board rule: {rule}"
    assert ".mb-row:nth-child(3)" not in css, (
        f"{sheet}: a row delay still targets :nth-child(3), which is the "
        "win bar now that the strip leads the board"
    )
