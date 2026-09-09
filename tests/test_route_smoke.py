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
    "/guides/dynasty-rebuild-strategy", "/guides/startup-draft-guide",
    "/privacy", "/terms", "/about", "/contact", "/support", "/pricing",
    "/trade", "/top-movers", "/dynasty-trade-value-chart", "/players",
    "/oline-rankings", "/oline-rankings?metric=pass_block", "/oline-rankings/2024",
    "/rankings/dynasty", "/rankings/dynasty-qb", "/rankings/dynasty-rb",
    "/rankings/dynasty-wr", "/rankings/dynasty-te",
    "/robots.txt", "/sitemap.xml", "/ads.txt",
]

# League pages rendered from the seeded tour-demo league (no live data needed).
# These exercise the heavy, cache-hidden page builders.
LEAGUE_PAGES = [
    "dashboard", "standings", "teams", "weekly", "activity",
    "awards", "history", "graphs", "recap",
    "waivers", "schedule", "compare",
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


def test_oline_rankings_api(offline_client):
    # The O-line rankings API returns the sorted ratings table (or an empty
    # rows list before the first cron build), never a 500.
    r = offline_client.get("/api/oline-rankings", query_string={"season": "2024"})
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert body.get("metric") == "composite"
    assert isinstance(body.get("rows"), list)
    if body["rows"]:
        top = body["rows"][0]
        assert top["rank"] == 1
        assert 0.0 <= top["composite"] <= 100.0
        # sorted best-to-worst
        comps = [row["composite"] for row in body["rows"]]
        assert comps == sorted(comps, reverse=True)


def test_oline_for_player_helper():
    # The player-modal O-line helper: position-aware primary metric, graceful
    # nulls, and season fallback to the newest built cache. DB-free.
    import app
    rb = app._oline_for_player(2026, "PHI", "RB")
    assert rb and rb["primary"] == "run_block"
    assert rb["primary_value"] == rb["run_block"]
    assert 1 <= rb["primary_rank"] <= rb["total_teams"]
    assert rb.get("composite_rank") is not None
    assert 1 <= rb["composite_rank"] <= rb["total_teams"]
    assert rb.get("pass_block_rank") is not None
    assert rb.get("run_block_rank") is not None
    qb = app._oline_for_player(2026, "BUF", "QB")
    assert qb and qb["primary"] == "pass_block"
    # 2026 isn't built in the seed; helper falls back to the newest season.
    assert rb["season"] <= 2026
    # No rating -> no section.
    assert app._oline_for_player(2026, "", "WR") is None
    assert app._oline_for_player(2026, "FA", "RB") is None


def test_prewarm_league_requires_league_id(offline_client):
    # The switcher's background prewarm endpoint must exist and reject a call
    # with no league_id (rather than 404/500), so the route stays wired up.
    r = offline_client.get("/api/prewarm-league")
    assert r.status_code == 400, r.status_code
    assert r.get_json().get("ok") is False


def test_prewarm_league_skips_espn(offline_client):
    # ESPN idle prewarm is a no-op: it contended with private-league ESPN traffic
    # and switch already refreshes the destination context.
    r = offline_client.get(
        "/api/prewarm-league",
        query_string={"platform": "espn", "league_id": "887776065", "season": "2026"},
    )
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert body.get("ok") is True
    assert body.get("skipped") is True
    assert body.get("reason") == "espn"
