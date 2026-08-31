from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nav_is_best_ball_helper_exists():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _nav_is_best_ball(" in source
    assert '_waiver_label = "Waivers" if _bb else "Waivers & Start/Sit"' in source


def test_waivers_page_hides_startsit_for_best_ball():
    source = (ROOT / "dashboard_services" / "pages" / "waivers_page.py").read_text(encoding="utf-8")
    assert "is_best_ball(" in source
    assert "startsit_tab_html" in source
    assert "Best Ball league" in source


def test_dashboard_best_ball_badge():
    source = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert "Best Ball</span>" in source
    assert "_bb_badge" in source
