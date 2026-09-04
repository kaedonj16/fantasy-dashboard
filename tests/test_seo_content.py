"""Render tests for the AdSense/SEO content additions.

These pages are the "value" layer that lifts the site above the "low value
content" bar: unique on-page prose plus structured data (FAQPage, DefinedTermSet)
that search engines reward. They render offline (no live league data), so they go
through the real Flask stack via the ``offline_client`` fixture and assert the
contract the SEO relies on — the schema blocks and the human-readable copy.

Skipped automatically when Flask/pandas aren't installed; they run in CI where
the full stack is present.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _html(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    return r.get_data(as_text=True)


def test_faq_has_faqpage_schema(offline_client):
    html = _html(offline_client, "/faq")
    # FAQPage structured data is generated from the same source as the accordion.
    assert '"@type":"FAQPage"' in html
    assert '"@type":"Question"' in html
    assert "How are dynasty trade values calculated?" in html
    # Links back into the site (internal linking) survive into the answers.
    assert "/guides/dynasty-trade-value" in html


def test_glossary_page_and_schema(offline_client):
    html = _html(offline_client, "/glossary")
    assert "Fantasy Football Glossary" in html
    # DefinedTermSet with individual DefinedTerm entries.
    assert '"@type":"DefinedTermSet"' in html
    assert '"@type":"DefinedTerm"' in html
    # A few representative terms must render as cards.
    for term in ("Superflex", "TE Premium", "Zero-RB", "Target Share"):
        assert term in html


def test_glossary_in_sitemap(offline_client):
    xml = _html(offline_client, "/sitemap.xml")
    assert "/glossary" in xml


@pytest.mark.parametrize("path,needle", [
    ("/rankings/dynasty-qb", "How to read dynasty QB value"),
    ("/rankings/dynasty-rb", "How to read dynasty RB value"),
    ("/rankings/dynasty-wr", "How to read dynasty WR value"),
    ("/rankings/dynasty-te", "How to read dynasty TE value"),
    ("/rankings/dynasty",    "How dynasty rankings work"),
])
def test_position_pages_have_unique_analysis(offline_client, path, needle):
    """Each dynasty ranking page carries its own analysis block, so the four
    position pages are no longer near-duplicate templates."""
    html = _html(offline_client, path)
    assert 'class="rnk-analysis"' in html
    assert needle in html


def test_ranking_analysis_blocks_are_distinct(offline_client):
    """The QB and RB analyses must not be the same copy (the whole point)."""
    qb = _html(offline_client, "/rankings/dynasty-qb")
    rb = _html(offline_client, "/rankings/dynasty-rb")
    assert "How to read dynasty QB value" in qb and "How to read dynasty QB value" not in rb
    assert "How to read dynasty RB value" in rb and "How to read dynasty RB value" not in qb


def test_trade_value_chart_has_methodology_and_faq(offline_client):
    html = _html(offline_client, "/dynasty-trade-value-chart")
    # Methodology prose + FAQ (with schema) below the tool on the flagship page.
    assert "How these dynasty values are built" in html
    assert '"@type":"FAQPage"' in html
    assert "Should I use 1QB or Superflex values?" in html


def test_top_movers_has_updated_stamp(offline_client):
    html = _html(offline_client, "/top-movers")
    assert "refreshed daily" in html
    assert 'class="rf-updated"' in html


def test_publisher_transparency_pages_are_complete(offline_client):
    privacy = _html(offline_client, "/privacy")
    about = _html(offline_client, "/about")
    contact = _html(offline_client, "/contact")
    terms = _html(offline_client, "/terms")
    assert "Last updated:" in privacy
    assert "Google's partner-sites policy" in privacy
    assert "adssettings.google.com" in privacy
    assert "Children's Privacy" in privacy
    assert "Your Choices" in privacy
    assert "Funding Choices" in privacy
    assert "Editorial Standards &amp; Corrections" in about
    assert "mailto:admin@brfantasy.com" in contact
    assert "Last updated:" in terms
    assert "never ask users to click ads" in terms


def test_ad_placements_are_explicitly_disclosed(offline_client):
    html = _html(offline_client, "/")
    assert 'aria-label="Advertisement"' in html
    assert 'class="ad-disclosure">Advertisement</span>' in html


@pytest.mark.parametrize("path", [
    "/privacy", "/terms", "/about", "/contact", "/support", "/faq", "/pricing",
])
def test_thin_pages_do_not_serve_ads(offline_client, path):
    """AdSense forbids ad units on pages without enough publisher content."""
    html = _html(offline_client, path)
    assert "adsbygoogle" not in html
    assert 'aria-label="Advertisement"' not in html


def test_ads_txt_is_cacheable_and_lists_publisher(offline_client):
    r = offline_client.get("/ads.txt")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "pub-9164153092633845" in body
    assert "google.com" in body
    # Long cache so CDN can keep serving ads.txt when origin is slow.
    assert "max-age=86400" in (r.headers.get("Cache-Control") or "")


def test_robots_allows_adsense_crawlers(offline_client):
    body = _html(offline_client, "/robots.txt")
    assert "Mediapartners-Google" in body
    assert "AdsBot-Google" in body
    assert "Allow: /" in body


def test_homepage_has_crawlable_publisher_content(offline_client):
    """AdSense reviewers land on / first. The connect-league card is not enough
    publisher content; the homepage must also ship original HTML articles/links."""
    html = _html(offline_client, "/")
    assert "home-publisher" in html
    assert "home-hero-editorial" in html
    assert 'href="/guides"' in html
    assert 'href="/rankings/dynasty"' in html
    assert 'href="/dynasty-trade-value-chart"' in html
    assert "How Dynasty Trade Value Works" in html
    # Full-screen splash must not hide the page from no-JS reviewers.
    assert "#appSplash{display:none!important}" in html


def test_guides_are_substantial_articles_with_schema(offline_client):
    from routes.guides_content import GUIDE_ORDER, GUIDES

    assert len(GUIDE_ORDER) >= 12
    xml = _html(offline_client, "/sitemap.xml")
    for slug in GUIDE_ORDER:
        g = GUIDES[slug]
        words = len(__import__("re").sub(r"<[^>]+>", " ", g["body"]).split())
        assert words >= 400, f"{slug} is too thin ({words} words)"
        assert f"/guides/{slug}" in xml
        html = _html(offline_client, f"/guides/{slug}")
        assert '"@type":"Article"' in html
        assert '"@type":"BreadcrumbList"' in html
        assert '"name":"Home"' in html
        assert '"name":"Guides"' in html
        assert g["title"] in html
        assert "hoodiekj" in html
        assert "adsbygoogle" in html


def test_guest_nav_exposes_learn_pages():
    import app
    with app.app.test_request_context("/"):
        nav = app.build_nav(None, "home", "sleeper", 2026)
        more = app._mobile_nav_guest("home")
    assert "href='/guides'" in nav
    assert "Learn <span" in nav
    assert "href='/guides'" in more
    assert "href='/glossary'" in more
    assert "Strategy Guides" in more


def test_adsense_script_loads_immediately_for_google_crawlers():
    import app
    assert "mediapartners-google" in app._AD_INIT
    assert "adsbot" in app._AD_INIT
