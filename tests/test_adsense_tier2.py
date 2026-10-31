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


# Per-article metadata for every guide, keyed by slug:
# (category, published, updated). Dates are the honest ones, not varied
# for show: the first six guides carry the site's declared published date
# of 2026-06-15 (the GUIDE_PUBLISHED value in place since the guides were
# first committed), the Tier 2 guides were published at the deepening,
# and every guide's updated date is 2026-10-02, the Tier 2 date itself.
EXPECTED_GUIDE_META = {
    "dynasty-trade-value": ("Inside the Model", "2026-06-15", "2026-10-02"),
    "superflex-vs-1qb": ("Roster Construction", "2026-06-15", "2026-10-02"),
    "reading-advanced-metrics": ("Inside the Model", "2026-06-15", "2026-10-02"),
    "rookie-draft-strategy": ("Roster Construction", "2026-06-15", "2026-10-02"),
    "buy-low-sell-high": ("Market Movement", "2026-06-15", "2026-10-02"),
    "evaluating-a-trade": ("Trade Case Studies", "2026-06-15", "2026-10-02"),
    "two-for-one-consolidation-trade-case": ("Trade Case Studies", "2026-10-30", "2026-10-30"),
    "dynasty-rebuild-strategy": ("Roster Construction", "2026-10-02", "2026-10-02"),
    "contending-in-dynasty": ("Roster Construction", "2026-10-02", "2026-10-02"),
    "te-premium-leagues": ("Roster Construction", "2026-10-02", "2026-10-02"),
    "startup-draft-guide": ("Roster Construction", "2026-10-02", "2026-10-02"),
    "waiver-wire-and-faab": ("Weekly Decisions", "2026-10-02", "2026-10-02"),
    "adp-vs-trade-value": ("Market Movement", "2026-10-02", "2026-10-02"),
    "positional-aging-curves": ("Inside the Model", "2026-10-02", "2026-10-02"),
    "using-the-trade-calculator": ("Guides", "2026-10-02", "2026-10-02"),
}


def test_guides_are_long_form_with_sections():
    from routes.guides_content import (
        GUIDE_CATEGORY_ORDER,
        GUIDE_ORDER,
        GUIDES,
        guide_read_minutes,
    )

    assert len(GUIDE_ORDER) == 15
    assert set(GUIDE_ORDER) == set(GUIDES) == set(EXPECTED_GUIDE_META)
    for slug in GUIDE_ORDER:
        category, published, updated = EXPECTED_GUIDE_META[slug]
        guide = GUIDES[slug]
        # Per-article structure: every guide carries its own category and
        # its own published/updated dates, and a computable read time.
        assert guide["category"] == category, f"{slug} category drifted"
        assert guide["category"] in GUIDE_CATEGORY_ORDER
        assert guide["published"] == published, f"{slug} published drifted"
        assert guide["updated"] == updated, f"{slug} updated drifted"
        assert guide_read_minutes(guide["body"]) > 0
        body = guide["body"]
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
    hero = source.index('<section class="home-hero')
    previews = source.index("Built for the whole week")
    faq = source.index('<section class="faq">')
    card = source.index('<div class="home-hero-right">')
    # Hero intro first, then the connect card directly (owner's call: the
    # card must be findable in the first scroll; it previously sat after
    # the FAQ and signed-in visitors never received it at all), then the
    # editorial sections and the FAQ.
    assert hero < card < previews < faq


def test_player_trade_value_page_is_noindex():
    source = (ROOT / "routes" / "seo_pages_bp.py").read_text()
    start = source.index("def page_player_trade_value")
    end = source.index("\ndef ", start + 1)
    region = source[start:end]
    assert "noindex=True" in region
