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


def test_mobile_stacks_metrics_below_player():
    """On phones the metric columns (up to five with FAAB) drop onto their own
    full-width row instead of crowding the player name off the card."""
    body = build_waivers_body("sleeper", 2026, "league", {})
    assert ".wv-player-row { flex-direction: column; align-items: stretch;" in body
    # The metric strip spans the row and separates from the player above it.
    assert "border-top: 1px solid var(--border);" in body
