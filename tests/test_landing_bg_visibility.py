"""Pins for the landing background graph's visibility contract.

The graph's lines and bars used to be invisible in their natural state
(stroke-dashoffset: 100, scaleY(0)) and only an animation could reveal
them, so any browser whose animation clock never advanced showed no
graph at all. The drawn state is now the natural state; the keyframes
carry the hidden start state and 'backwards' fill covers the delay, so
the entrance is unchanged where animations run and the graph is static
where they don't.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LITE = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")
DASH = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_graph_layer_cannot_be_covered_by_body_background():
    """html carries its own background color, so body's opaque white
    background (carrying the grid pattern) does NOT propagate to the
    canvas: it paints as a normal layer, ABOVE negative z-index
    descendants of the root stacking context. An .ld-bg at z-index -1
    under a non-stacking-context .home-page is therefore painted over
    by the body background and never shows. .home-page must create its
    own stacking context (z-index: 0) so the graph paints above every
    ancestor background and below the page content."""
    for css in (LITE, DASH):
        assert re.search(
            r"\.home-page \{ display: block; max-width: 1180px;"
            r" padding: 28px 20px 72px; position: relative; z-index: 0; \}", css)


def test_graph_visible_without_animation_in_both_sheets():
    for css in (LITE, DASH):
        assert re.search(r"\.ld-line1 \{ stroke-dasharray: 100; stroke-dashoffset: 0;", css)
        assert re.search(r"\.ld-line2 \{ stroke-dasharray: 100; stroke-dashoffset: 0;", css)
        assert "@keyframes ldDraw { from { stroke-dashoffset: 100; } to { stroke-dashoffset: 0; } }" in css
        assert "@keyframes ldBar { from { transform: scaleY(0); } to { transform: scaleY(1); } }" in css
        # No natural-state hiding may come back.
        assert not re.search(r"\.ld-line[12] \{[^}]*stroke-dashoffset: 100;", css)
        assert not re.search(r"\.ld-bar \{[^}]*transform: scaleY\(0\);", css)
        assert "backwards" in css


def test_bold_graph_treatment_and_publisher_card():
    """The approved bold treatment lives on the graph's own CSS classes:
    thicker, stronger lines, a deeper area fill, and the layer framed
    inside the viewport below the nav (full width) so the line cannot
    run off the top of the screen. The Strategy & resources section
    sits on a card."""
    for css in (LITE, DASH):
        assert (".ld-bg { position: fixed; top: 84px; left: 0; width: 100%;"
                " height: calc(100vh - 84px);") in css
        assert re.search(r"\.ld-line1 \{[^}]*stroke-width: 4\.25; opacity: \.55;", css)
        assert re.search(r"\.ld-line2 \{[^}]*stroke-width: 2\.75; opacity: \.30;", css)
        assert "#ldAfill stop:first-child { stop-opacity: .17; }" in css
        inner = re.search(r"\.home-publisher-inner \{([^}]*)\}", css)
        assert inner and "background: var(--card);" in inner.group(1)
        assert inner and "border: 1px solid var(--border);" in inner.group(1)
        assert inner and "border-radius: 20px;" in inner.group(1)
