"""Modal bodies scroll vertically only (Kaedon, 2026-11-10).

Inside the player modal body (the Overview tab is where it showed) she
could scroll horizontally; the same applied to the other modal, sheet
and drawer bodies. A body with only `overflow-y: auto` computes its
`overflow-x` to `auto` too, so any stray wide child turned the whole
body into a horizontal scroller. Every modal-family body now pins
`overflow-x: hidden`, matching the team modal body's existing rule.

Intentional inner scrollers keep their own horizontal scroll: the tab
bars and the wrapped wide tables/charts inside a panel are separate
scroll containers and are NOT covered by this pin.
"""
from pathlib import Path

ROOT = Path(__file__).parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text()
APP_JS = (ROOT / "static" / "app.js").read_text()
TI_PAGE = (ROOT / "dashboard_services" / "pages" / "trade_intel_page.py").read_text()


def _block(css, selector, start=0):
    """Return the declaration block body for the first `selector { ... }`."""
    i = css.index(selector, start)
    j = css.index("{", i)
    k = css.index("}", j)
    return css[j + 1:k]


def _any_block_has(css, selector, needle):
    """True when any `selector { ... }` block in the sheet contains needle."""
    start = 0
    while True:
        try:
            i = css.index(selector, start)
        except ValueError:
            return False
        j = css.index("{", i)
        k = css.index("}", j)
        if needle in css[j + 1:k]:
            return True
        start = k + 1


def test_player_modal_body_never_scrolls_horizontally():
    body = _block(CSS, ".player-modal-body {")
    assert "overflow-y: auto" in body
    assert "overflow-x: hidden" in body


def test_team_modal_body_never_scrolls_horizontally():
    body = _block(CSS, ".team-modal-body {")
    assert "overflow-y: auto" in body
    assert "overflow-x: hidden" in body


def test_other_modal_bodies_never_scroll_horizontally():
    for selector in (
        ".for-modal-body {",
        ".rzm-modal-body {",
        ".td-body {",
        ".br-asst-body {",
        ".br-sheet-panel {",
    ):
        body = _block(CSS, selector)
        assert "overflow-y: auto" in body, selector
        assert "overflow-x: hidden" in body, selector


def test_trade_intel_trades_body_never_scrolls_horizontally():
    body = _block(TI_PAGE, ".ti-trades-body {{")
    assert "overflow-y: auto" in body
    assert "overflow-x: hidden" in body


def test_rookie_modal_fallback_shell_never_scrolls_horizontally():
    # createRkModalIfMissing builds #rkModalContent as its own scroll
    # container (the .rk-modal-body inside it does not scroll).
    assert "overflow-y:auto;overflow-x:hidden;" in APP_JS


def test_intentional_inner_scrollers_keep_horizontal_scroll():
    # The pin applies only where the body itself should not scroll: every
    # wide element inside a pinned body carries its own horizontal
    # scroller, and those scrollers are untouched.
    # Tab bars stay swipeable; they are not modal bodies.
    assert "overflow-x: auto" in _block(CSS, ".pm-tab-bar {")
    assert "overflow-x: auto" in _block(CSS, ".tm-tab-bar {")
    # Player modal: game log and box score scroll inside their wrappers.
    # (.table-wrap is declared twice; the later block carries the axis pin.)
    assert _any_block_has(CSS, ".table-wrap {", "overflow-x: auto")
    assert "overflow-x: auto" in _block(CSS, ".pm-boxscore-scroll {")
    # Front Office report: its min-width table scrolls inside its wrapper.
    assert "overflow-x: auto" in _block(CSS, ".for-table-wrap {")


def test_rookie_season_table_keeps_its_own_scroller():
    rookies = (ROOT / "dashboard_services" / "pages" / "rookies_page.py").read_text()
    assert "overflow-x: auto" in _block(rookies, ".rk-table-scroll {")
