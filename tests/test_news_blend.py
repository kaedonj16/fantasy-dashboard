"""Unit tests for news blending/dedupe (dashboard_services.news).

These are pure-Python (no network, no httpx) — news.py imports httpx lazily so
it can be imported in the CI test environment.
"""
from dashboard_services.news import _blend_dedupe, _norm_url, _norm_title


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
