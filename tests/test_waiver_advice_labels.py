"""Waiver recommendation metrics should explain themselves without tooltips."""

from dashboard_services.pages.waivers_page import build_waivers_body


def test_waiver_advice_metrics_have_visible_labels():
    body = build_waivers_body("sleeper", 2026, "league", {})
    for label in ("Why add", "Data quality", "FAAB bid", "Value"):
        assert label in body


def test_unexplained_likely_rank_is_not_rendered():
    body = build_waivers_body("sleeper", 2026, "league", {})
    assert "Likely rank" not in body
    assert "${p.rank_low}" not in body


def test_faab_checkbox_controls_bid_visibility():
    body = build_waivers_body("sleeper", 2026, "league", {})
    assert 'id="wvShowFaab"' in body
    assert 'onchange="wvToggleFaab(this.checked)"' in body
    assert "window.wvFaabEnabled && wvShowFaab && p.faab_high" in body


def test_advice_pills_use_canonical_chip_styles():
    body = build_waivers_body("sleeper", 2026, "league", {})
    assert 'chip chip--sm chip--accent' in body
    assert 'chip chip--sm chip--neutral' in body
