"""Guest home page: product tour, quotes, and reworked PRO pitch.

Source-contract tests for the Kaedon-approved guest home page implementation.
The tour, trust strip, quotes, free-tools band, how-it-works, FAQ, indie strip,
and final CTA render for guests only. The PRO section moved after the quotes
with reworked copy.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _guest_region() -> str:
    """Everything from the trust strip through the final CTA."""
    start = APP_PY.index('<div class="trust-strip">')
    end = APP_PY.index('</section>  {% endif %}',
                       APP_PY.index('<section class="final-cta">')) + len('</section>')
    return APP_PY[start:end]


def _pro_section() -> str:
    start = APP_PY.index('<section class="home-pro"')
    return APP_PY[start:APP_PY.index('</section>', start) + len('</section>')]


def test_tour_covers_five_tools_with_real_component_markup():
    region = _guest_region()
    for tool in ("Matchup Board", "Redzone", "Start/Sit", "Wrapped", "Trade Strategy"):
        assert tool in region
    # Real product class hooks, not generic placeholders.
    assert "mb-row" in region
    assert "hp-rz" in region
    assert "wv-cx" in region
    assert "hm-slide" in region
    assert "otc-sugg" in region
    # The old generic mock classes are gone.
    assert "hm-match-teams" not in region
    assert "hm-gamepill" not in region


def test_guest_only_gating_wraps_new_sections():
    region = _guest_region()
    assert "{% if not session.get('account_id') %}" in APP_PY
    # Trust strip, tour, and quotes sit between hero and PRO inside a guest block.
    hero_end = APP_PY.index('<section class="home-hero">')
    pro_start = APP_PY.index('<section class="home-pro"')
    between = APP_PY[hero_end:pro_start]
    assert between.count("{% if not session.get('account_id') %}") >= 1
    assert "{% endif %}" in between
    # The tour itself is inside the guest block, not visible when logged in.
    assert '<section class="home-previews">' in between


def test_pro_section_moved_after_quotes_with_reworked_copy():
    quotes_start = APP_PY.index('<section class="quotes">')
    pro_start = APP_PY.index('<section class="home-pro"')
    freeband_start = APP_PY.index('<section class="freeband">')
    assert quotes_start < pro_start < freeband_start
    pro = _pro_section()
    assert "From $5 a year." in pro
    assert "Less than $1 a month" not in pro
    assert "$10/year. Cancel anytime." in pro
    assert "From $5/year" not in pro
    assert "See exactly how each trade shifts your playoff odds before you send it" in pro
    assert "Next week's breakouts, flagged before your league notices" in pro


def test_single_tour_cta_replaces_repeated_links():
    region = _guest_region()
    assert region.count("Connect your league to see yours") == 0
    assert region.count('class="tour-cta-btn"') == 2  # tour end + final CTA
    assert 'href="#homeCardTitle"' in region


def test_real_testimonials_present():
    region = _guest_region()
    assert "THATS ACTUALLY SO SICK BRO" in region
    assert "Jayden Waddell" in region
    assert "outperformed all my projections in 3 leagues" in region
    assert "can't wait to see what it looks like in season" in region
    assert "Manager name" not in region


def test_free_tools_band_links_real_public_pages():
    region = _guest_region()
    assert '<section class="freeband">' in region
    for path in ("/rankings/dynasty", "/dynasty-trade-value-chart", "/guides", "/nfl-teams"):
        assert path in region


def test_faq_how_indie_sections_present():
    region = _guest_region()
    assert '<section class="how">' in region
    assert '<section class="faq">' in region
    assert "Built by a fantasy manager, not a media company." in region


def test_wrapped_deck_autoplay_wired():
    region = _guest_region()
    assert 'id="tourDeck"' in region
    assert 'id="tourDots"' in region
    assert "getElementById('tourDeck')" in region


def test_no_mock_artifacts_or_em_dashes():
    region = _guest_region()
    assert "mock-banner" not in region
    assert "themeToggle" not in region
    assert "—" not in region
    assert "—" not in _pro_section()


def test_guest_home_css_present():
    for selector in (".home-previews {", ".quotes-grid {", ".freeband {",
                     ".faq {", ".tour-cta {", ".tour-cta-btn {",
                     ".home-pro-lead strong {"):
        assert selector in DASH_CSS, selector
