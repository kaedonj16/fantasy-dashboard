"""Rankings freshness + hover resilience.

Three reported problems, one root each:

1. Total Points / PPG on the board progressed week by week instead of
   per player. The per-player PPG map was correct, but it is baked into
   the league-players payload, which was memoized on model values +
   ADP only, so totals froze until the daily value refresh even though
   the player modal (per-request) already counted a finished game.
   Fix: _lp_weekly_signature() (completed weeks, live-week final count,
   completed-week stat-file sizes) is folded into the payload memo key,
   the disk-tier filename, the overlay cache key (and therefore the
   serialized-body caches), and via overlay_key into the ETag version
   key, so a finished game invalidates every layer at once.

2. The hover card could not open until the players payload landed
   (SSR rows carry no data-pid). A hover that arrives early is now
   remembered by name and fulfilled by prQlTryPending after the first
   real render. The rankings view also serves its serialized body from
   cache before running the projection overlay it cannot affect.

3. Quick-look zones (trend / last 3 games / next matchup) could spin
   forever or render a fake empty state: the fetches had no timeout,
   and a failure marked the zone done with null data. Fetches now time
   out at 10s, failures are NOT marked done (the next hover retries),
   and an open card shows a retry message instead.
"""

import pathlib

from tests.pin_helpers import changelog_entry

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text()
RANKINGS_JS = (ROOT / "static" / "rankings.js").read_text()


def test_weekly_signature_captures_final_events():
    assert "def _lp_weekly_signature() -> tuple:" in APP
    # League-wide completed weeks + live-week final count drive it.
    assert "qualification_policy(season).completed_weeks" in APP
    assert "load_week_schedule(season, live_week)" in APP
    assert "_is_final(g)" in APP
    # Completed-week stat files contribute their sizes; the live week's
    # file is deliberately NOT sized (its stats grow during games).
    assert 'f"sleeper_stats_s{season}_w{w}.json"' in APP
    # Memoized briefly so per-request key building stays cheap.
    assert '_LP_WEEKLY_SIG_MEMO' in APP


def test_signature_threads_every_cache_layer():
    # Payload memo: hit requires the signature to match, and rebuilds
    # store it.
    assert 'cached.get("weekly_sig") == weekly_sig' in APP
    assert '"weekly_sig": weekly_sig, "payload": payload' in APP
    # Disk tier: read/write take it and the filename hashes it.
    assert "_lp_payload_read_disk(key, model_ts, adp_sig, weekly_sig)" in APP
    assert "_lp_payload_write_disk(key, model_ts, adp_sig, payload, weekly_sig)" in APP
    assert "weekly_part" in APP
    # Overlay key: serialized-body caches (board / trade / rankings)
    # miss when a game finishes; the ETag version key embeds
    # overlay_key, so browser 304s roll too.
    overlay = APP[APP.index("def _lp_overlay_cache_key"):APP.index("def _overlay_league_players_payload")]
    assert "_lp_weekly_signature()," in overlay


def test_rankings_warm_short_circuit_is_gated():
    route = APP[APP.index("Rankings warm path"):]
    route = route[:route.index("payload = _overlay_league_players_payload(")]
    # Serves the exact cache the rankings response fn writes.
    assert '_LP_BOARD_JSON_CACHE.get(\n                ("rankings", overlay_key, bool(_mi_is_sf)))' in APP
    assert 'view="rankings"' in route
    # Gates: historical season, explicit ADP source, and the Yahoo
    # league overlay all bypass the short-circuit.
    assert "_sc_adp not in _sc_labels" in route
    assert '_sc_plat == "yahoo" and _sc_lg' in route
    assert "_sc_season == _sc_cur" in route


def test_quicklook_fetches_time_out_and_retry():
    assert "var PR_QL_FETCH_TIMEOUT_MS = 10000;" in RANKINGS_JS
    assert "function prQlFetchJSON(url, timeoutMs)" in RANKINGS_JS
    assert "ctrl.abort()" in RANKINGS_JS
    fetch_src = RANKINGS_JS[RANKINGS_JS.index("function prQlFetch(p)"):]
    fetch_src = fetch_src[:fetch_src.index("function prQlBuild(p)")]
    # Failure must not mark a zone done (the old null-data path did).
    assert "entry.quickDone = true" in fetch_src
    assert fetch_src.count("entry.quickDone = true") == 1
    assert "entry.quickError = true" in fetch_src
    assert "entry.gamesError = true" in fetch_src
    # An open card shows the retry message rather than a fake empty.
    assert "Hover again to retry." in RANKINGS_JS
    assert "QL_RETRY_HTML" in RANKINGS_JS
    assert "prQlFillQuick(p, null, true)" in fetch_src
    assert "prQlFillGames(p, null, true)" in fetch_src


def test_pending_hover_before_data_lands():
    assert "var _prQlPending = null;" in RANKINGS_JS
    assert "function prQlTryPending()" in RANKINGS_JS
    # prRender fulfills the pending hover after real rows exist.
    render_src = RANKINGS_JS[RANKINGS_JS.index("function prRender()"):]
    render_src = render_src[:render_src.index("function prRenderPagination")]
    assert "prQlTryPending()" in render_src
    # It only opens while the cursor is still on the player's row.
    assert "row.matches(':hover')" in RANKINGS_JS
    # Leaving the row (or hiding) cancels it.
    assert "if (_prQlPending && _prQlPending.row === row) _prQlPending = null;" in RANKINGS_JS


def test_changelog_announces_freshness_fix():
    entry = changelog_entry("count each player's own finished games",
                            "time out and retry cleanly")
    assert entry["tag"] == "fix"
    assert entry["link"] == "/players"
    assert "—" not in entry["text"]
