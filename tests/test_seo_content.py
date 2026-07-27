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
