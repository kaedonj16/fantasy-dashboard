"""Regression guards for the mobile Value vs ADP timeline."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mobile_timeline_is_compact_and_touch_accessible():
    script = (ROOT / "static" / "draft_room.js").read_text(encoding="utf-8")
    page = (ROOT / "dashboard_services" / "pages" / "draft_room_page.py").read_text(
        encoding="utf-8"
    )

    assert "Math.max(720, 56 + (pts.length - 1) * 72)" in script
    assert "compact ? 8 : 5" in script
    assert "compact ? 11 : 9" in script
    assert "Tap a pick for details" in script
    assert "dot.addEventListener('click'" in script
    assert "tabindex: '0'" in script
    assert ".dd-chart-hint { display:block" in page
    assert "-webkit-overflow-scrolling:touch" in page
    assert "touch-action:pan-x" in page
