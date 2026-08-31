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
    # H1 must stay exact — badge is a sibling only (league chrome tests).
    assert '<h1 class="os-hero-title">Season Hub</h1>{_bb_badge}' in source


def test_dashboard_best_ball_thin_outlook():
    source = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert "Season outlook (thin)" in source
    assert "no weekly lineup" in source
    assert "_bb_outlook_html" in source
    assert "dashBestBallOutlook" in source


def test_draft_room_best_ball_skips_bye_and_applies_preset():
    room = (ROOT / "static" / "draft_room.js").read_text(encoding="utf-8")
    assert "cfg.isBestBall" in room
    assert "Best Ball has no weekly lineup" in room or "not Best Ball" in room
    assert "_rosterPreset = 'bestball'" in room
    assert "ROSTER_PRESETS.bestball" in room


def test_draft_room_auction_grades_disabled():
    room = (ROOT / "static" / "draft_room.js").read_text(encoding="utf-8")
    assert "if (cfg.isAuction) return null;" in room or "cfg.isAuction) return null" in room
    assert "Auction draft grades" in room
    assert "suggestAuctionBid" in (ROOT / "static" / "draft_board_core.js").read_text(encoding="utf-8")
