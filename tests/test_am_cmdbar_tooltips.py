"""Advanced Metrics command-bar tooltip direction contract.

Bug (2026-10-13, screenshot): hovering the command-bar icon buttons
(Search / Graph / Filters / More) showed only the bottom sliver of the
tooltip bubble, text hidden.

Mechanism, proven in a Playwright harness with the real dashboard.css:
the app.js title upgrader turns each button's title= into a
[data-tooltip] bubble that renders ABOVE its trigger (bottom:100% +
6px). The bar sits at the top of the first card, flush under the
sticky .top-nav (63px tall, opaque, z-index 100): the bubble needs
~41px above the button, only ~19px is clear, and the bar's own sticky
stacking context (z-index 60) traps the bubble's --z-tooltip under
the nav. Measured on the Graph button: bubble 109x35, top 22px of it
behind the nav, 13px sliver visible - matching the screenshot.

Fix: command-bar bubbles open DOWNWARD (top:100% + 6px), inside the
card, the same direction the bar's overflow menu already opens. The
flip is scoped to .am-cmdbar .am-cmd-btn so every other [data-tooltip]
surface keeps the shared above-trigger placement.

Follow-up (2026-10-14, screenshot): the same bubble clipped
HORIZONTALLY on the overflow button. #amMoreToggle hugs the card's
right edge (its center sits ~29px from it), so the centered 102px
"More actions" bubble spilled ~21px past the card and the viewport
(harness-measured at 1280 and 900px wide) and the text cut off
mid-word. Fix: the More bubble right-aligns to the button's right
edge (left:auto; right:0; transform:none). The other three buttons
sit far enough inboard that centering never reaches the edge, so
they keep the shared centered placement.
"""
import re

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _style(html):
    return html[html.index("<style>"):html.index("</style>")]


def _tag(html, el_id):
    m = re.search(r'<[^>]*\bid="%s"[^>]*>' % re.escape(el_id), html)
    assert m, f"#{el_id} missing from page"
    return m.group(0)


def test_cmdbar_buttons_keep_titles_for_the_bubble():
    html = _html()
    for el_id in ("amSearchToggle", "amGraphBtn", "amFilterToggle", "amMoreToggle"):
        tag = _tag(html, el_id)
        assert 'class="am-cmd-btn"' in tag
        assert 'title="' in tag, f"#{el_id} lost the title the tooltip upgrade reads"


def test_cmdbar_tooltips_flip_below_the_button():
    css = _style(_html())
    m = re.search(
        r"\.am-cmdbar \.am-cmd-btn\[data-tooltip\]:hover::after,\s*"
        r"\.am-cmdbar \.am-cmd-btn\[data-tooltip\]:focus-visible::after\s*\{([^}]*)\}",
        css,
    )
    assert m, "command-bar tooltip flip rule missing (hover + focus-visible)"
    body = m.group(1)
    assert "bottom:auto" in body
    assert "top:100%" in body
    assert "margin-top:6px" in body
    assert "margin-bottom:0" in body


def test_more_actions_bubble_right_aligns_to_the_button():
    css = _style(_html())
    m = re.search(
        r"#amMoreToggle\[data-tooltip\]:hover::after,\s*"
        r"#amMoreToggle\[data-tooltip\]:focus-visible::after\s*\{([^}]*)\}",
        css,
    )
    assert m, "overflow-button tooltip right-alignment rule missing"
    body = m.group(1)
    assert "left:auto" in body
    assert "right:0" in body
    assert "transform:none" in body


def test_flip_stays_scoped_to_the_command_bar():
    css = _style(_html())
    # No page-wide [data-tooltip] repositioning: other surfaces on this
    # page (and the shared rule in dashboard.css) keep opening upward.
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selectors, body = m.group(1), m.group(2)
        if "data-tooltip" in selectors and "top:100%" in body:
            assert ".am-cmdbar" in selectors, (
                f"unscoped tooltip flip would move other bubbles: {selectors.strip()}"
            )
