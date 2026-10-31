"""Guides index + article template tests.

The guides index groups the articles by category with a chip, date,
and read time per card; each article page opens with a header carrying
its category chip, its own published/updated dates, the site byline, a
computed read time, and section-jump pills generated from the article's
actual h2 sections. Read time is computed from the article's word count
at READ_TIME_WPM. The sitemap carries each guide's own updated date as
its lastmod.

Data-level tests run without the app import; render tests go through
the real Flask stack via the ``offline_client`` fixture.
"""

import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _html(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    return r.get_data(as_text=True)


# ── Read time ────────────────────────────────────────────────────────────────

def test_read_time_computed_from_word_count():
    from routes.guides_content import (
        READ_TIME_WPM,
        guide_read_minutes,
        guide_word_count,
    )

    assert READ_TIME_WPM == 200
    body_400 = "<p>" + " ".join(["word"] * 400) + "</p>"
    assert guide_word_count(body_400) == 400
    assert guide_read_minutes(body_400) == 2
    # Markup does not count as words, and the floor is 1 minute.
    assert guide_word_count("<p>short <strong>guide</strong></p>") == 2
    assert guide_read_minutes("<p>short guide</p>") == 1
    assert guide_read_minutes("") == 1


def test_every_guide_has_a_real_read_time():
    from routes.guides_content import GUIDE_ORDER, GUIDES, guide_read_minutes

    for slug in GUIDE_ORDER:
        minutes = guide_read_minutes(GUIDES[slug]["body"])
        # The legacy guides run 1,000+ words, so 5-6 minutes at 200 wpm;
        # the pin is the honest range, not a decorative number.
        assert 4 <= minutes <= 8, f"{slug} read time {minutes} min"


# ── Index grouping ───────────────────────────────────────────────────────────

def test_index_groups_guides_by_category(offline_client):
    from routes.guides_content import (
        GUIDE_CATEGORY_CHIPS,
        GUIDE_CATEGORY_ORDER,
        GUIDE_ORDER,
        GUIDES,
    )

    html = _html(offline_client, "/guides")

    # Category headers render once each, in the canonical order.
    header_at = []
    for category in GUIDE_CATEGORY_ORDER:
        marker = f'<div class="guide-cathead"><h3>{category}</h3>'
        assert marker in html, f"missing category header: {category}"
        header_at.append(html.index(marker))
    assert header_at == sorted(header_at)

    # Every guide card sits inside its own category's segment, with the
    # category chip, the updated date, and a read time on the card.
    segments = {}
    for i, category in enumerate(GUIDE_CATEGORY_ORDER):
        start = header_at[i]
        end = header_at[i + 1] if i + 1 < len(header_at) else len(html)
        segments[category] = html[start:end]
    for slug in GUIDE_ORDER:
        guide = GUIDES[slug]
        category = guide["category"]
        segment = segments[category]
        assert f'href="/guides/{slug}"' in segment, (
            f"{slug} card is not under {category}"
        )
        chip_label = GUIDE_CATEGORY_CHIPS[category][0]
        assert f">{chip_label}</span>" in segment
        assert "min read" in segment
    # No guide is listed twice across the groups; the Latest strip above
    # the groups repeats its 3 cards, so the page total is groups + 3.
    assert len(GUIDE_ORDER) == 15
    assert html.count('class="guide-card"') == len(GUIDE_ORDER) + 3


# ── Article header ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("slug", ["dynasty-trade-value", "waiver-wire-and-faab"])
def test_article_header_chip_dates_byline_and_jump_pills(offline_client, slug):
    from routes.guides_content import GUIDES, guide_read_minutes, guide_sections

    guide = GUIDES[slug]
    html = _html(offline_client, f"/guides/{slug}")

    # Full category name on the article chip, per-article date line, the
    # site byline, and the computed read time.
    assert f'<span class="guide-chip"' in html
    assert f'>{guide["category"]}</span>' in html
    published_long = "Jun 15, 2026" if slug == "dynasty-trade-value" else "Oct 2, 2026"
    assert f"Published {published_long}" in html
    assert "Updated Oct 2, 2026" in html
    assert "By BR Fantasy" in html
    assert f"{guide_read_minutes(guide['body'])} min read" in html

    # Jump pills are generated from the article's actual h2 sections:
    # one pill per section, pill text matches the heading text, and each
    # pill target exists as an id in the rendered body.
    sections = guide_sections(guide["body"])
    assert len(sections) >= 5
    toc = html.split('<div class="guide-toc">', 1)[1].split("</div>", 1)[0]
    pills = re.findall(r'<a href="#guide-sec-([^"]+)">([^<]+)</a>', toc)
    assert pills == [(anchor, title) for anchor, title in sections]
    for anchor, _title in sections:
        assert f'id="guide-sec-{anchor}"' in html


# ── Sitemap ──────────────────────────────────────────────────────────────────

def test_sitemap_lastmod_comes_from_each_guides_updated_date(offline_client):
    from routes.guides_content import GUIDE_ORDER, GUIDES

    xml = _html(offline_client, "/sitemap.xml")
    for slug in GUIDE_ORDER:
        block = re.search(
            rf"<url><loc>[^<]*/guides/{slug}</loc>(.*?)</url>", xml
        )
        assert block, f"sitemap is missing /guides/{slug}"
        assert f"<lastmod>{GUIDES[slug]['updated']}</lastmod>" in block.group(1)
    # Static entries carry no lastmod; only guides gained one.
    privacy = re.search(r"<url><loc>[^<]*/privacy</loc>(.*?)</url>", xml)
    assert privacy and "<lastmod>" not in privacy.group(1)


# ── Latest strip + category chip nav + first case study ─────────────────────

def test_index_latest_strip_leads_with_newest_article(offline_client):
    html = _html(offline_client, "/guides")

    latest_at = html.index('<div class="guide-cathead"><h3>Latest</h3>')
    first_group_at = html.index(
        '<div class="guide-cathead"><h3>Trade Case Studies</h3>'
    )
    assert latest_at < first_group_at
    latest = html[latest_at:first_group_at]
    # The newest article (published 2026-10-30) is the first Latest card,
    # and the strip holds exactly 3 cards.
    assert latest.count('class="guide-card"') == 3
    first_card = latest.index('class="guide-card"')
    a1_card = latest.index('href="/guides/two-for-one-consolidation-trade-case"')
    assert first_card < a1_card < first_card + 200


def test_index_category_chips_link_to_existing_section_anchors(offline_client):
    from routes.guides_content import GUIDE_CATEGORY_ORDER

    html = _html(offline_client, "/guides")
    nav = html.split('<nav class="guide-catnav">', 1)[1].split("</nav>", 1)[0]
    for category in GUIDE_CATEGORY_ORDER:
        anchor = "guide-cat-" + category.lower().replace(" ", "-")
        assert f'href="#{anchor}"' in nav, f"chip missing for {category}"
        assert f'id="{anchor}"' in html, f"section anchor missing: {anchor}"


def test_first_case_study_renders_with_both_tables(offline_client):
    html = _html(
        offline_client, "/guides/two-for-one-consolidation-trade-case"
    )
    assert "Two Receivers for an Elite Running Back" in html
    assert "Published Oct 30, 2026" in html
    # Both value tables and the closing bottom line render.
    assert html.count("<table") == 2
    for needle in ("Bijan Robinson", "Drake London", "Tetairoa McMillan",
                   "Justin Fields", "1,101", "604", "494", "1,011", "529",
                   "438", "1,108", "988"):
        assert needle in html, f"case study is missing {needle}"
    assert '<div class="highlight-box">' in html
    assert "Bottom line:" in html
    # The quotable finding and the method/data note are part of the article.
    assert "The finding:" in html
    assert "Data and method:" in html
    assert "October 30," in html
