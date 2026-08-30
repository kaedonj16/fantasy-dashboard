"""Team Values / manager-pill carousel must keep every roster reachable on mobile."""
from __future__ import annotations

from pathlib import Path

CSS = (Path(__file__).resolve().parents[1] / "static" / "dashboard.css").read_text(encoding="utf-8")
APP_JS = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")


def test_manager_pills_do_not_hide_inactive_on_mobile():
    """Hiding .manager-pill:not(.active) collapsed scrollWidth and hid arrows."""
    assert ".manager-pills-row .manager-pill:not(.active)" not in CSS
    assert "stuck on a single roster" in CSS or "every manager visible" in CSS


def test_manager_pill_arrows_show_when_multiple_teams():
    fn = APP_JS[APP_JS.index("function initManagerPills"): APP_JS.index("function initCardTabs")]
    assert "pills.length > 1" in fn
    assert "scrollWidth > pillsRow.clientWidth" in fn
