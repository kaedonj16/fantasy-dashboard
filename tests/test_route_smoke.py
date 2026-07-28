"""Smoke test: every important route renders (HTTP 200) through the real stack.

Most render paths aren't unit-tested because they need a full league context +
DB, and their HTML is cached - so a render break (a template typo, a helper that
lost an argument, a None that isn't guarded) can stay invisible until the cache
expires and the page 500s in production. This walks the public pages and the
league pages (via the seeded tour-demo league) through the offline_client, which
renders in-process with Sleeper HTTP mocked, and asserts each returns 200.

It's a broad safety net, not a correctness check: it catches "this page stopped
rendering", which is exactly the failure mode the cached render paths hide.

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

# Public pages (no league context) - policy, SEO/content, and the tool pages.
PUBLIC_ROUTES = [
    "/", "/faq", "/glossary", "/guides", "/guides/dynasty-trade-value",
    "/privacy", "/terms", "/about", "/contact", "/support", "/pricing",
    "/trade", "/top-movers", "/dynasty-trade-value-chart", "/players",
    "/rankings/dynasty", "/rankings/dynasty-qb", "/rankings/dynasty-rb",
    "/rankings/dynasty-wr", "/rankings/dynasty-te",
    "/robots.txt", "/sitemap.xml", "/ads.txt",
]

# League pages rendered from the seeded tour-demo league (no live data needed).
# These exercise the heavy, cache-hidden page builders.
LEAGUE_PAGES = [
    "dashboard", "standings", "teams", "weekly", "activity",
    "awards", "history", "graphs", "recap",
]
LEAGUE_ROUTES = [f"/sleeper/2026/tourdemo/{p}?tour=1" for p in LEAGUE_PAGES]


@pytest.mark.parametrize("path", PUBLIC_ROUTES)
def test_public_route_renders(offline_client, path):
    r = offline_client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"


@pytest.mark.parametrize("path", LEAGUE_ROUTES)
def test_league_route_renders(offline_client, path):
    r = offline_client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
