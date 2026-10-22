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
