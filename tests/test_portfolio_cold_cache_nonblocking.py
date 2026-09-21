"""Best-effort portfolio readers must never trigger a cold synchronous league
build on the request path.

Regression guard for the log-observed failures:
  * ``GET /api/portfolio-actions`` blocked ~128s building every league serially
    on a cold cache.
  * ``GET /api/portfolio/matchup`` (one per My Leagues card) triggered the same
    cold build, which the card's 12s client abort always lost to -- so every
    matchup preview card stayed hidden.

The fix: ``get_league_ctx_from_cache`` grows an ``allow_build=False`` mode that
serves last-known-good or nothing and warms the cache in the background; the two
readers use it; the matchup client re-polls a ``pending`` league until warm.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ctx_cache_helper_has_no_build_mode_and_background_warm():
    source = (ROOT / "app.py").read_text()
    # The keyword-only opt-out exists and never runs the synchronous build path.
    assert "def get_league_ctx_from_cache(" in source
    assert "allow_build: bool = True" in source
    helper = source.split("def get_league_ctx_from_cache(")[1].split("\ndef ")[0]
    assert "if not allow_build:" in helper
    assert "_warm_league_ctx_async(" in helper
    # The no-build branch must return before acquiring the build lock.
    nobuild = helper.split("if not allow_build:")[1].split("_acquire_context_lock")[0]
    assert "build_league_context" not in nobuild

    # The background warm is deduplicated and concurrency-bounded so a page with
    # many leagues cannot spawn unbounded cold builds or spike worker memory.
    warm = source.split("def _warm_league_ctx_async(")[1].split("\ndef ")[0]
    assert "_LEAGUE_WARM_INFLIGHT" in warm
    assert "_LEAGUE_WARM_SEM" in warm
    assert "daemon=True" in warm


def test_matchup_endpoint_is_non_blocking_and_signals_pending():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    endpoint = source.split("def api_portfolio_matchup")[1].split("\n@user_pages_bp.route")[0]
    assert "allow_build=False" in endpoint
    # A cold miss is a fast, explicit "poll again", not a blocking build.
    assert '"pending": True' in endpoint


def test_portfolio_actions_skips_cold_leagues():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    fn = source.split("def api_portfolio_actions")[1].split("\n@user_pages_bp.route")[0]
    assert "allow_build=False" in fn
    # Cold leagues are skipped for this pass rather than built inline.
    assert "if not lctx:" in fn


def test_matchup_client_repolls_pending_slots():
    source = (ROOT / "app.py").read_text()
    # The live-matchup loader re-polls a pending league with backoff and keeps the
    # skeleton visible meanwhile instead of hiding the card.
    assert "d.pending" in source
    assert "slot._mAttempt" in source
