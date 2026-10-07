"""Regression tests for ESPN viewer team resolution on the trade calculator.

Covers:
1. viewerRid falls back to the session-injected window._viewerRid (validated
   against the current league's roster list).
2. The "even it out" balancer hides instead of using the global player pool
   when the roster filter is on but her team is unknown.
3. /api/league-rosters never serves a cached per-user viewer_roster_id.
"""

from pathlib import Path

APP_JS = Path(__file__).parents[1] / "static" / "app.js"


def test_viewer_rid_falls_back_to_session_injected_value():
    source = APP_JS.read_text(encoding="utf-8")
    loader = source[source.index("async function initRosterFilter()"):]
    loader = loader[: loader.index("\n  function setupSearch(")]
    # window._viewerRid is the session-injected viewer used by the rest of the
    # site (teams page, etc.). It must be validated against this league's
    # rosters so a stale value from another league can never leak in.
    assert "window._viewerRid" in loader
    assert "rosterFilter.byRid[_sessRid]" in loader


def test_balancer_hides_when_viewer_unknown_under_roster_filter():
    source = APP_JS.read_text(encoding="utf-8")
    balancer = source[source.index("function renderTradeBalancer("):]
    balancer = balancer[: balancer.index("\n  function ", 10)]
    # When the roster filter is active and the light side is her team (B) but
    # viewerRid is empty, the balancer must hide rather than silently suggest
    # from the global player pool.
    assert 'lightSide === "B" && !rosterFilter.viewerRid' in balancer
    assert "return hide()" in balancer


def test_league_rosters_cache_never_stores_viewer_roster_id():
    """The /api/league-rosters shared cache must only hold per-team data.

    viewer_roster_id is per-user; caching it under the (platform, league, season)
    key would serve one user's viewer id (or empty string) to another user.
    """
    source = Path(__file__).parents[1] / "app.py"
    text = source.read_text(encoding="utf-8")
    fn = text[text.index("def api_league_rosters():"):]
    fn = fn[: fn.index("\n\n\n# /api/advanced-metrics/seasons")]
    # The cache write must only store teams, never the per-user viewer id.
    assert '_ROSTER_API_CACHE[_roster_key] = {"data": {"teams": teams}' in fn
    assert '"viewer_roster_id"' not in fn.split('_ROSTER_API_CACHE[_roster_key] =')[1].split("\n")[0]
    # The viewer is resolved fresh on every request, after the cache lookup.
    assert "viewer = get_viewer_session_for_league(" in fn
