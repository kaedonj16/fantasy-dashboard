"""Regression tests for the duplicate-data-fetch eliminations.

Client-side request sharing is covered behaviorally by
tests/dupfetch_harness.mjs (the SHIPPED functions extracted by source and
driven with a fake fetch / fake clock). The rest are source contracts and
behavioral server-side tests.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
MODAL_JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")


def _run_harness(name: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS behavioral harness")
    result = subprocess.run(
        [node, str(ROOT / "tests" / name)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKS PASSED" in result.stdout, result.stdout + result.stderr


# ── Client behavioral harness ────────────────────────────────────────────────
def test_request_sharing_behavioral_harness():
    """brGetLeaguePlayersData (in-flight dedup, 30min reuse, SWR, localStorage
    persistence, failure retry) and _pmSmallFetch (news/ADP: in-flight dedup,
    TTL reuse, expiry refetch, no failure caching, LRU bound) -- exercised
    against the shipped code."""
    _run_harness("dupfetch_harness.mjs")


# ── Client source contracts ──────────────────────────────────────────────────
def test_trade_count_not_fetched_by_client():
    """The server renders #tradeCount with a static fallback; the client only
    lazy-refreshes it from /api/trade-count via requestIdleCallback (never on
    the critical path)."""
    assert "setTradeCountLabel" not in APP_JS
    # The lazy fetch exists but must be deferred, not blocking page load.
    assert "requestIdleCallback" in APP_JS
    assert '_lazyTradeCount' in APP_JS


def test_league_players_fetch_shared_between_nav_and_trade():
    """One page-level /api/league-players promise serves both the nav search
    idle-preload and the trade calculator init. Uses the slim ?view=trade
    payload with default cache mode (ETag + max-age=1800)."""
    assert "function brGetLeaguePlayersData(" in APP_JS
    # The slim trade endpoint is fetched (inside the shared helper).
    assert "'/api/league-players?view=trade'" in APP_JS
    # No cache bypass: ETag + max-age=1800 must work.
    assert "cache: 'no-store'" not in APP_JS.split("function brGetLeaguePlayersData(")[1].split("}")[0]
    # Both callers go through the shared helper.
    assert APP_JS.count("brGetLeaguePlayersData(") >= 3  # def + 2 callers


def test_season_trends_single_call():
    """Season mode issues one metrics=all request instead of fetching the
    default series, discarding it, then requesting every series again."""
    body = MODAL_JS.split("function pmLoadSeasonAll(playerId, host) {", 1)[1]
    body = body.split("\n}\n", 1)[0]
    assert "pmSeasonTrendFetch(playerId, 'all')" in body
    assert body.count("pmSeasonTrendFetch(") == 1


def test_breakout_tab_joins_inflight_prefetch():
    """Tapping Breakout before the modal-open eligibility fetch resolves must
    join the in-flight promise instead of firing a duplicate request."""
    assert "_pmBreakoutInflight = { playerId: String(playerId), promise:" in MODAL_JS
    tab = MODAL_JS.split("if (tab === 'breakout' && panel && !panel.dataset.loaded) {", 1)[1]
    tab = tab.split("// ──", 1)[0]
    assert "_pmBreakoutInflight" in tab
    assert "_bkInflight" in tab


def test_trade_intel_fetches_abortable_per_generation():
    """Rapid recomputeTrade() calls abort the previous generation's intel
    fetches (trade intel, similar trades, playoff impact) and discard stale
    renders via the generation counter."""
    assert "let _tradeIntelAbort = null;" in APP_JS
    assert "if (_tradeIntelAbort) _tradeIntelAbort.abort();" in APP_JS
    assert "new AbortController()" in APP_JS
    for fn in ("fetchTradeIntel", "fetchSimilarTrades", "fetchPlayoffImpact"):
        sig = APP_JS.split(f"async function {fn}(", 1)[1].split(")", 1)[0]
        assert sig == "gen, signal", f"{fn} must take (gen, signal)"
        body = APP_JS.split(f"async function {fn}(", 1)[1]
        body = body.split(f"\n  // ---", 1)[0].split("\n  async function ", 1)[0]
        assert "gen !== _tradeGeneration" in body, f"{fn} must discard stale generations"
        assert "signal" in body, f"{fn} must thread the abort signal"


def test_news_and_adp_go_through_shared_cache():
    """Reopening the same player reuses the cached news/ADP payload instead of
    refetching; the ADP key includes the season."""
    assert "_pmSmallFetch('adp:' + playerId + ':' + season" in MODAL_JS
    assert "_pmSmallFetch('news:' + playerId" in MODAL_JS
    assert MODAL_JS.count("/api/player-news/${encodeURIComponent(playerId)}") == 1
    assert MODAL_JS.count("/api/player-adp/${encodeURIComponent(playerId)}") == 1


# ── Server behavioral tests ──────────────────────────────────────────────────
def test_hydrate_sleeper_draft_picks_reuses_prefetched_picks(monkeypatch):
    """_hydrate_sleeper_draft_picks must not call get_draft_picks for drafts
    whose picks were already fetched during the draft-history walk."""
    keeper = pytest.importorskip("dashboard_services.pages.keeper_page")
    api = pytest.importorskip("dashboard_services.api")

    calls: list[str] = []

    def _boom(draft_id):
        calls.append(draft_id)
        raise AssertionError("get_draft_picks should not be called for prefetched drafts")

    monkeypatch.setattr(api, "get_draft_picks", _boom)
    drafts = [{"draft_id": "d1"}, {"draft_id": "d2", "picks": [{"player_id": "x"}]}]
    out = keeper._hydrate_sleeper_draft_picks(
        drafts, picks_by_draft={"d1": [{"player_id": "p1", "round": 3}]}
    )
    assert calls == []
    assert out[0]["picks"] == [{"player_id": "p1", "round": 3}]
    assert out[1]["picks"] == [{"player_id": "x"}]


def test_hydrate_sleeper_draft_picks_fetches_when_not_prefetched(monkeypatch):
    """Drafts absent from picks_by_draft still hydrate via get_draft_picks."""
    keeper = pytest.importorskip("dashboard_services.pages.keeper_page")
    api = pytest.importorskip("dashboard_services.api")

    monkeypatch.setattr(api, "get_draft_picks", lambda did: [{"player_id": "p9"}])
    out = keeper._hydrate_sleeper_draft_picks([{"draft_id": "d3"}], picks_by_draft={})
    assert out[0]["picks"] == [{"player_id": "p9"}]


def test_load_relevant_index_parses_disk_once(monkeypatch, tmp_path):
    """The ~586KB relevant-player index must be parsed once per mtime, not
    once per endpoint (player-details, game-logs, player-team)."""
    utils = pytest.importorskip("utils.utils")

    idx = tmp_path / "relevant.json"
    idx.write_text('{"players": {"1": {"name": "A"}}}', encoding="utf-8")
    monkeypatch.setattr(utils, "path_relevant_index", lambda: str(idx))
    # Keep the test hermetic: no merged overlay.
    monkeypatch.setattr(utils, "_RELEVANT_INDEX_MERGED", {})
    monkeypatch.setattr(utils, "_overlay_players_index", lambda base, _m: base)

    parses: list[str] = []
    real_read_json = utils.read_json

    def _counting_read_json(path):
        parses.append(path)
        return real_read_json(path)

    monkeypatch.setattr(utils, "read_json", _counting_read_json)
    utils._JSON_CACHE.pop(str(idx), None)

    first = utils.load_relevant_index()
    second = utils.load_relevant_index()
    assert parses == [str(idx)], parses
    assert first == second
    assert first["players"]["1"]["name"] == "A"
