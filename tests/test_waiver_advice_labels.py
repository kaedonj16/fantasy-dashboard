"""Waiver recommendation metrics should explain themselves without tooltips."""

from dashboard_services.pages.waivers_page import build_waivers_body


def test_waiver_advice_metrics_have_visible_labels():
    body = build_waivers_body("sleeper", 2026, "league", {})
    for label in ("Why add", "FAAB bid", "Value"):
        assert label in body


def test_data_quality_metric_removed():
    body = build_waivers_body("sleeper", 2026, "league", {})
    assert "Data quality" not in body
    assert "confChip" not in body


def test_unexplained_likely_rank_is_not_rendered():
    body = build_waivers_body("sleeper", 2026, "league", {})
    assert "Likely rank" not in body
    assert "${p.rank_low}" not in body


def test_faab_checkbox_controls_bid_visibility():
    body = build_waivers_body("sleeper", 2026, "league", {})
    assert 'id="wvShowFaab"' in body
    assert 'onchange="wvToggleFaab(this.checked)"' in body
    assert "window.wvFaabEnabled && wvShowFaab && (p.faab_high || p.faab_target)" in body


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


def test_mobile_stacks_start_sit_actions_below_name():
    """START/FLEX badges and Open player must not share one cramped mobile row."""
    body = build_waivers_body("espn", 2026, "league", {})
    assert ".wv-ss-top { flex-direction: column; align-items: stretch;" in body
    assert ".wv-ss-actions { flex-wrap: wrap; flex-shrink: 1; width: 100%; }" in body
    assert ".wv-ss-player .wv-player-name" in body


def test_waiver_fetch_surfaces_api_errors():
    """A 500/error payload must not render as an empty 'No waiver targets' list."""
    body = build_waivers_body("espn", 2026, "league", {})
    assert "r.json().then(d => ({ ok: r.ok, d }))" in body
    assert "if (!ok || d.error)" in body
    assert "Unable to load waiver data." in body
