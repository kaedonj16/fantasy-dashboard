"""Responsive layout contracts for trade teams on the activity page."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
ACTIVITY_PAGE = (ROOT / "dashboard_services" / "pages" / "activity_page.py").read_text(
    encoding="utf-8"
)


def test_activity_trade_teams_use_two_columns_on_desktop():
    assert "class='act-trade-body'" in ACTIVITY_PAGE
    assert "class='teams'" in ACTIVITY_PAGE
    assert re.search(
        r"\.activity-page \.act-trade-body > \.teams\s*\{[^}]*"
        r"display:\s*grid;[^}]*grid-template-columns:\s*1fr 1fr;",
        CSS,
        re.DOTALL,
    )


def test_activity_trade_teams_stack_on_mobile():
    selector = ".activity-page .act-trade-body > .teams"
    selector_pos = CSS.rfind(selector)
    media_pos = CSS.rfind("@media (max-width: 900px)", 0, selector_pos)
    assert media_pos >= 0
    assert re.search(
        r"\.activity-page \.act-trade-body > \.teams\s*\{[^}]*"
        r"grid-template-columns:\s*1fr;",
        CSS[media_pos:selector_pos + 200],
        re.DOTALL,
    )
