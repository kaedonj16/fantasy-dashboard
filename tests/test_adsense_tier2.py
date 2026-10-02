"""AdSense Tier 2 regression tests (2026-10-02).

Tier 2 addressed the "Low value content" finding with three changes:
1. Guides deepened into real long-form articles with section headings.
2. Homepage DOM order: hero/editorial content precedes the connect card.
3. Player trade-value pages carry noindex (already out of the sitemap).

These tests are intentionally source/data-level so they run without the full
app import (which needs optional third-party packages in some environments).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _guide_words(body: str) -> int:
    text = re.sub(r"<[^>]+>", " ", body)
    return len(re.sub(r"\s+", " ", text).split())


def test_guides_are_long_form_with_sections():
    from routes.guides_content import GUIDE_ORDER, GUIDES, GUIDE_UPDATED

    assert GUIDE_UPDATED == "2026-10-02"
    assert len(GUIDE_ORDER) == 14
    assert set(GUIDE_ORDER) == set(GUIDES)
    for slug in GUIDE_ORDER:
        body = GUIDES[slug]["body"]
        words = _guide_words(body)
        assert words >= 900, f"{slug} too thin for Tier 2 ({words} words)"
        h2 = body.count('<h2 class="static-section-title">')
        assert h2 >= 5, f"{slug} needs real section headings (found {h2})"
        assert '<div class="static-section-title">' not in body, (
            f"{slug} still uses div section titles instead of h2"
        )
        assert "\u2014" not in body and "&mdash;" not in body, (
            f"{slug} contains an em dash"
        )


def test_homepage_editorial_precedes_connect_card_in_dom():
    source = (ROOT / "app.py").read_text()
    hero = source.index("home-hero-left")
    previews = source.index('<section class="home-previews">')
    faq = source.index('<section class="faq">')
    card = source.index('<div class="home-hero-right">')
    # Hero intro first, then editorial sections, then the account machinery.
    assert hero < previews < faq < card


def test_player_trade_value_page_is_noindex():
    source = (ROOT / "routes" / "seo_pages_bp.py").read_text()
    start = source.index("def page_player_trade_value")
    end = source.index("\ndef ", start + 1)
    region = source[start:end]
    assert "noindex=True" in region
