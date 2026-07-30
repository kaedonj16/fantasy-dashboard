"""Unit tests for news blending/dedupe (dashboard_services.news).

These are pure-Python (no network, no httpx) — news.py imports httpx lazily so
it can be imported in the CI test environment.
"""
from dashboard_services.news import (
    _blend_dedupe,
    _blend_sources,
    _norm_url,
    _norm_title,
    _parse_reddit_children,
    _parse_gnews_xml,
)


def _child(**data):
    # Sensible defaults for a "good" vetted link post; override per test.
    base = {
        "title": "Reporter: Player X is active",
        "score": 100,
        "is_self": False,
        "stickied": False,
        "over_18": False,
        "url": "https://espn.com/story/1",
        "domain": "espn.com",
        "created_utc": 1_800_000_000,
    }
    base.update(data)
    return {"data": base}


def _item(headline, url, published, source="ESPN"):
    return {"headline": headline, "url": url, "published": published, "source": source}


def test_norm_url_canonicalizes():
    a = _norm_url("https://www.espn.com/nfl/story/_/id/1/foo?utm=x#top")
    b = _norm_url("http://espn.com/nfl/story/_/id/1/foo/")
    assert a == b == "espn.com/nfl/story/_/id/1/foo"


def test_norm_title_strips_punctuation_and_case():
    assert _norm_title("Bijan Robinson: OUT (ankle)!") == "bijan robinson out ankle"


def test_dedupe_by_url_prefers_primary():
    # Same article surfaced by ESPN and linked on Reddit -> one item, ESPN kept.
    espn = [_item("ESPN headline", "https://espn.com/a/1", "2026-07-30T10:00:00Z", "ESPN")]
    reddit = [_item("Redditor phrasing", "http://www.espn.com/a/1/", "2026-07-30T11:00:00Z", "r/nfl")]
    out = _blend_dedupe(espn, reddit, limit=10)
    assert len(out) == 1
    assert out[0]["source"] == "ESPN"


def test_dedupe_by_title():
    espn = [_item("Same Story Title", "https://espn.com/a/1", "2026-07-30T10:00:00Z")]
    reddit = [_item("same story title", "https://nfl.com/b/2", "2026-07-30T09:00:00Z", "r/nfl")]
    out = _blend_dedupe(espn, reddit, limit=10)
    assert len(out) == 1


def test_blend_sorts_by_recency_desc():
    espn = [_item("older", "https://espn.com/a/1", "2026-07-28T10:00:00Z")]
    reddit = [_item("newer", "https://nfl.com/b/2", "2026-07-30T10:00:00Z", "r/nfl")]
    out = _blend_dedupe(espn, reddit, limit=10)
    assert [i["headline"] for i in out] == ["newer", "older"]


def test_blend_respects_limit():
    espn = [_item(f"e{i}", f"https://espn.com/a/{i}", f"2026-07-3{i}T10:00:00Z") for i in range(5)]
    reddit = [_item(f"r{i}", f"https://nfl.com/b/{i}", f"2026-07-2{i}T10:00:00Z", "r/nfl") for i in range(5)]
    out = _blend_dedupe(espn, reddit, limit=4)
    assert len(out) == 4


def test_missing_url_still_deduped_by_title_and_kept():
    # Items without a URL should not collide with each other on empty URL.
    a = [_item("Alpha", "", "2026-07-30T10:00:00Z")]
    b = [_item("Beta", "", "2026-07-30T09:00:00Z", "r/nfl")]
    out = _blend_dedupe(a, b, limit=10)
    assert len(out) == 2


# ── Reddit "verified, not random people" filter ───────────────────────────────

def test_reddit_keeps_vetted_link_post():
    out = _parse_reddit_children([_child(subreddit="nfl")])
    assert len(out) == 1
    assert out[0]["source"] == "espn.com · r/nfl"
    assert out[0]["url"] == "https://espn.com/story/1"


def test_reddit_drops_self_post():
    assert _parse_reddit_children([_child(is_self=True)]) == []


def test_reddit_drops_low_score():
    assert _parse_reddit_children([_child(score=5)]) == []


def test_reddit_drops_stickied_and_nsfw():
    assert _parse_reddit_children([_child(stickied=True)]) == []
    assert _parse_reddit_children([_child(over_18=True)]) == []


def test_reddit_drops_reddit_hosted_media():
    assert _parse_reddit_children([_child(domain="i.redd.it", url="https://i.redd.it/x.jpg")]) == []
    assert _parse_reddit_children([_child(domain="v.redd.it", url="https://v.redd.it/x")]) == []


def test_reddit_require_substr_filters_by_title():
    posts = [
        _child(title="Bijan Robinson dominates", url="https://espn.com/a"),
        _child(title="Some other RB news", url="https://espn.com/b"),
    ]
    out = _parse_reddit_children(posts, require_substr="robinson")
    assert [p["headline"] for p in out] == ["Bijan Robinson dominates"]


def test_reddit_uses_url_overridden_by_dest_when_present():
    out = _parse_reddit_children([
        _child(url="https://reddit.com/r/nfl/comments/x",
               url_overridden_by_dest="https://nfl.com/real-article",
               domain="nfl.com")
    ])
    assert out[0]["url"] == "https://nfl.com/real-article"


# ── Google News RSS parsing ───────────────────────────────────────────────────

_GNEWS_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Parker Washington impresses at Jaguars camp - Jaguars Wire</title>
    <link>https://news.google.com/rss/articles/AAA</link>
    <pubDate>Wed, 29 Jul 2026 15:00:00 GMT</pubDate>
    <source url="https://jaguarswire.usatoday.com">Jaguars Wire</source>
  </item>
  <item>
    <title>Some unrelated headline about another team - ESPN</title>
    <link>https://news.google.com/rss/articles/BBB</link>
    <pubDate>Wed, 29 Jul 2026 12:00:00 GMT</pubDate>
    <source url="https://espn.com">ESPN</source>
  </item>
</channel></rss>"""


def test_gnews_parses_and_strips_source_suffix():
    items = _parse_gnews_xml(_GNEWS_RSS)
    assert len(items) == 2
    top = items[0]
    assert top["headline"] == "Parker Washington impresses at Jaguars camp"  # " - Source" stripped
    assert top["source"] == "Jaguars Wire"
    assert top["url"] == "https://news.google.com/rss/articles/AAA"
    assert top["published"].startswith("2026-07-29")


def test_gnews_require_substr_filters_by_headline():
    items = _parse_gnews_xml(_GNEWS_RSS, require_substr="washington")
    assert [i["source"] for i in items] == ["Jaguars Wire"]


def test_gnews_bad_xml_returns_empty():
    assert _parse_gnews_xml("not xml at all") == []


def test_blend_sources_three_way_priority_and_dedupe():
    espn = [_item("Shared Story", "https://espn.com/x", "2026-07-30T10:00:00Z", "ESPN")]
    gnews = [_item("shared story", "https://gnews/x", "2026-07-30T11:00:00Z", "Jaguars Wire")]
    reddit = [_item("Fresh take", "https://nfl.com/y", "2026-07-30T12:00:00Z", "r/nfl")]
    out = _blend_sources([espn, gnews, reddit], limit=10)
    # Dupe collapses to the ESPN copy (earlier list wins); two items remain.
    assert len(out) == 2
    assert any(i["source"] == "ESPN" for i in out)
    assert not any(i["source"] == "Jaguars Wire" for i in out)
