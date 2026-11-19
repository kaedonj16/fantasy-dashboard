"""Guest home page: the approved reimagine (hero, gallery, connect card,
what-you-get, quote, PRO card, FAQ, final CTA, indie sign-off).

Source-contract tests for Kaedon-approved home page. The 2026-11-19 extras
removal retired the tools strips, live demos, freeband, feature list and
publisher sections; the page is the mock and nothing more. The PRO section
follows the quote. The connect card keeps the site's existing flows and
handlers; only its layout and theme changed.
"""
import html as _html
import re as _re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
LITE_CSS = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")


def _guest_region() -> str:
    """Hero through the final CTA (the whole guest section stack)."""
    start = APP_PY.index('<section class="home-hero')
    end = APP_PY.index('<section class="ld-indie">', start)
    return APP_PY[start:end]


def _pro_section() -> str:
    start = APP_PY.index('<section class="home-pro')
    return APP_PY[start:APP_PY.index('</section>', start) + len('</section>')]


def _plain(text: str) -> str:
    return _html.unescape(_re.sub(r"<[^>]+>", "", text))


def test_hero_board_uses_real_component_markup():
    region = _guest_region()
    # The gallery's matchup preview is the site's real .hp-mb matchup
    # component, in the approved reimagine's form: You/Opp labels and the
    # QB/RB/WR rows of the classic home preview card.
    assert "hp-mb" in region
    assert "mb-row" in region
    assert "m-win-bar" in region
    assert "pos-badge" in region
    assert '<div class="hp-mb-team">You</div>' in region
    assert '<div class="hp-mb-team">Opp</div>' in region
    assert '<div class="hp-mb-score">132.4</div>' in region
    assert '<div class="hp-mb-score">128.9</div>' in region
    # The final CTA is the in-page connect anchor (the old hero CTA row
    # was removed with the 2026-11-19 extras).
    assert 'href="#homeCardTitle"' in region


def test_guest_only_gating_wraps_new_sections():
    assert "{% if not session.get('account_id') %}" in APP_PY
    hero_end = APP_PY.index('<section class="home-hero')
    pro_start = APP_PY.index('<section class="home-pro')
    between = APP_PY[hero_end:pro_start]
    assert between.count("{% if not session.get('account_id') %}") >= 1
    assert "{% endif %}" in between
    assert '<section class="ld-sec ld-get-sec"' in between


def test_pro_section_after_quote_with_reworked_copy():
    assert '<section class="quotes">' not in APP_PY
    quote_start = APP_PY.index('class="home-testimonials"')
    pro_start = APP_PY.index('<section class="home-pro')
    faq_start = APP_PY.index('<section class="faq">')
    assert quote_start < pro_start < faq_start
    pro = _pro_section()
    assert "From $10 a year." in pro
    assert "$10/year. Cancel anytime." in pro
    assert "Less than $1 a month" not in pro
    assert "From $10/year" not in pro
    assert "See exactly how each trade shifts your playoff odds before you send it" in pro
    assert "breakouts, flagged before your league notices" in pro
    assert "data-home-pro-open" in pro


def test_real_testimonial_present_once():
    region = _guest_region()
    assert _plain(region).count("THATS ACTUALLY SO SICK BRO") == 1
    assert "Jayden Waddell" in region
    assert "Name coming soon" not in region
    assert "Blackedraw manager" not in region
    assert "home-hero-proof" not in region
    # The reimagine restores the classic testimonials section (the ld-quote
    # dark figure is retired).
    assert '<section class="home-testimonials"' in region
    assert "ld-quote" not in region


def test_faq_present_and_retired_sections_gone():
    region = _guest_region()
    assert '<section class="faq">' in region
    assert '<section class="how">' not in APP_PY
    assert '<section class="indie">' not in APP_PY
    assert '<section class="final-cta">' not in APP_PY
    assert '<div class="trust-strip">' not in APP_PY
    assert '<section class="home-previews">' not in APP_PY


def test_connect_card_flows_and_finish_intact():
    region = _guest_region()
    for flow in ("sleeperFlow", "espnFlow", "yahooFlow", "mflFlow", "fleaflickerFlow"):
        assert f'id="{flow}"' in region
    for plat in ("sleeper", "espn", "yahoo", "mfl", "fleaflicker"):
        assert f'data-platform="{plat}"' in region
    assert 'class="plat-desc"' in region
    assert 'id="leagueSelectForm"' in region
    assert 'id="googleContinueBtn"' in region
    assert "Continue without account" in region


def test_no_mock_artifacts_or_em_dashes():
    region = _guest_region()
    assert "mock-banner" not in region
    assert "themeToggle" not in region
    assert "—" not in region
    assert "—" not in _pro_section()


def test_guest_home_css_present():
    # The landing block must exist in BOTH sheets: signed-in visitors get
    # dashboard.css, guests on / get landing_lite.css only.
    for selector in (".ld-hero {", ".ld-card {", ".ld-quote {", ".ld-pro {",
                     ".ld-free-row {", ".ld-bg {", ".faq {"):
        assert selector in DASH_CSS, selector
        assert selector in LITE_CSS, selector
    # The lite sheet also mirrors the board component rules.
    assert ".hp-mb {" in LITE_CSS
    assert ".pos-badge {" in LITE_CSS
