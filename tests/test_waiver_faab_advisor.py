"""FAAB bands, roster-full drop gate, and schedule urgency (roadmap R05)."""
from utils.waiver_score import faab_bid_bands, roster_needs_drop, schedule_urgency


def test_faab_bid_bands_shape_and_order():
    top = faab_bid_bands(100, 0, 100, need_mult=1.0, handcuff_upside=0.0)
    assert top["faab_low"] <= top["faab_target"] <= top["faab_high"]
    assert top["faab_high"] <= 50
    assert "top target" in top["faab_rationale"] or "solid" in top["faab_rationale"] or "flier" in top["faab_rationale"]

    need = faab_bid_bands(50, 0, 100, need_mult=1.2)
    base = faab_bid_bands(50, 0, 100, need_mult=1.0)
    assert need["faab_target"] >= base["faab_target"]
    assert "fills a roster need" in need["faab_rationale"]


def test_roster_needs_drop():
    assert roster_needs_drop(15, 15) is True
    assert roster_needs_drop(16, 15) is True
    assert roster_needs_drop(14, 15) is False
    assert roster_needs_drop(10, 0) is False


def test_schedule_urgency():
    assert schedule_urgency(30, 32) == "Claim before tough stretch"
    assert schedule_urgency(22, 32) == "Schedule turns harder soon"
    assert schedule_urgency(5, 32) is None
    assert schedule_urgency(None, 32) is None


def test_waivers_ui_shows_faab_bands_and_urgency():
    from dashboard_services.pages.waivers_page import build_waivers_body
    body = build_waivers_body("sleeper", 2026, "league", {})
    assert "faab_target" in body
    assert "faab_rationale" in body
    assert "schedule_urgency" in body
    assert "low · target · stretch" in body
