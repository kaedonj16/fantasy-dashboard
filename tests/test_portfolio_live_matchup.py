"""My Leagues live matchup score: endpoint wiring, gating, and card slot."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_live_matchup_endpoint_wired_and_gated():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    assert '@user_pages_bp.route("/api/portfolio/matchup")' in source
    endpoint = source.split("def api_portfolio_matchup")[1].split("\n@user_pages_bp.route")[0]
    # Signed-out visitors get a silent no-op, not an error.
    assert '"live": False' in endpoint
    # Only in-season / playoffs, and never offseason or an unlinked team.
    assert 'season_type' in endpoint
    # Gate must accept the NORMALIZED phase "reg" (get_nfl_state maps Sleeper's
    # "regular" onto off/pre/reg/post) -- a "regular"-only check never matched
    # in-season and hid every card.
    assert '"reg"' in endpoint
    assert '"post"' in endpoint
    assert 'offseason_mode' in endpoint
    assert 'viewer_roster_id' in endpoint
    # A Google-account viewer's team is linked via the account, not a Sleeper
    # session identity, so the roster must be resolved through the account (with a
    # session fallback) -- otherwise every card reads live:false for account users.
    assert 'resolve_account_viewer_for_league' in endpoint
    # The preview is shown all week (pre / live / final), matching the single-
    # league dashboard -- it is NOT suppressed until ~90 min before kickoff.
    assert "_week_scores_visible" not in endpoint
    assert "get_nfl_games_for_week" not in endpoint
    # Totals come from the shared live-total helper, oriented you/opp.
    assert "team_live_totals" in endpoint
    # Win probability from the shared model, only when there's an opponent.
    assert "compute_win_prob" in endpoint
    assert '"win_prob": win_prob' in endpoint


def test_live_matchup_backfills_bundle_misses_from_raw_weekly_file():
    """Starters missing from the precomputed bundle (ESPN/Yahoo roster pids, or a
    first paint before hydration) must fall back to the raw weekly projection file
    -- league-scored -- so the card's projection and win bar match the Matchups
    tab instead of zeroing those players. Guards the fix from silent deletion."""
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    endpoint = source.split("def api_portfolio_matchup")[1].split("\n@user_pages_bp.route")[0]
    # Same raw-file fallback the full matchup slide uses.
    assert "load_week_projection" in endpoint
    assert "_proj_value_for_pid" in endpoint
    assert "raw_week_map" in endpoint
    # Enrich a copy, never the TTL-cached bundle shared across requests.
    assert "dict(proj_map)" in endpoint


def test_live_matchup_fetch_is_cached_and_current_week_only():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    assert "_LIVE_MATCHUP_CACHE" in source
    assert "_LIVE_MATCHUP_TTL" in source
    builder = source.split("def _build_live_matchups")[1].split("\n@user_pages_bp.route")[0]
    # One week's matchup + statuses, not a full-season rebuild.
    assert "build_matchup_preview" in builder
    assert "build_status_for_week" in builder
    assert "week=week" in builder


def test_matchup_status_label():
    pytest.importorskip("flask")
    from routes.user_pages_bp import _matchup_status_label
    from dashboard_services.matchups import (
        STATUS_FINAL, STATUS_IN_PROGRESS, STATUS_NOT_STARTED,
    )

    assert _matchup_status_label({}, []) == "pre"
    assert _matchup_status_label({"a": STATUS_NOT_STARTED, "b": STATUS_NOT_STARTED},
                                 ["a", "b"]) == "pre"
    assert _matchup_status_label({"a": STATUS_IN_PROGRESS, "b": STATUS_NOT_STARTED},
                                 ["a", "b"]) == "in"
    assert _matchup_status_label({"a": STATUS_FINAL, "b": STATUS_NOT_STARTED},
                                 ["a", "b"]) == "in"
    assert _matchup_status_label({"a": STATUS_FINAL, "b": STATUS_FINAL},
                                 ["a", "b"]) == "final"
    # Int pids resolve through the str() fallback.
    assert _matchup_status_label({"7": STATUS_FINAL}, [7]) == "final"


def test_portfolio_card_has_no_matchup_block():
    # Per-league previews no longer show the matchup band: no My Matchup /
    # League Scores tabs, no score block, no RedZone Moments row. Those live
    # only on the league dashboard's matchup section now.
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    assert "data-lg-live" not in fn
    assert "pf-lg-live" not in fn
    assert "pf-live-skel" not in fn
    assert "pf-live-wp" not in fn
    client = (ROOT / "static" / "app.js").read_text()
    assert "function matchupHtml" not in client
    assert "function renderMatchup" not in client
    assert "data-rzm-row" not in client
    assert "ls-tabs" not in client
    # NOTE: the .rzm-row/.ls-tabs CSS stays in dashboard.css on purpose: the
    # Weekly Hub matchup page reuses it for its own RedZone Moments row/modal.
    # The summary hydration path is untouched: cards still carry fetch keys and
    # hydrate record/standing/streak from /api/portfolio/card.
    assert "data-summary-card" in fn
    assert "/api/portfolio/card" in client
    assert "document.hidden" in client


def test_live_slot_not_on_pending_or_error_cards():
    # Vacuous now that the live slot is gone everywhere: no branch of the card
    # renderer may emit matchup markup.
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    assert fn.count("data-lg-live") == 0


def test_matchup_endpoint_fantasy_final_exposes_result_and_margin():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    fn = source.split("def api_portfolio_matchup")[1].split("\n@user_pages_bp.route")[0]
    # Visibility remains bounded to the game window, but fantasy finality (and
    # therefore the result) comes from provider week data, not NFL statuses.
    assert 'ZoneInfo("America/New_York")' in fn
    assert '_finalized_fantasy_week' in fn
    assert 'status = "final" if fantasy_final' in fn
    assert 'team.get("pts_total")' in fn
    assert 'result' in fn
    assert 'margin' in fn
    # Tuesday may use the just-finished week, but completed current-week results
    # are not hidden based on an unrelated wall-clock weekday.
    assert "finalized_week == week - 1 and today.weekday() == 1" in fn
    assert "today.weekday() not in (6, 0, 1)" not in fn
    assert '"live": True' in fn


def test_finalized_fantasy_week_supports_tuesday_rollover():
    pd = pytest.importorskip("pandas")
    from routes.user_pages_bp import _finalized_fantasy_week

    ctx = {"df_weekly": pd.DataFrame([
        {"roster_id": 9, "week": 2, "finalized": True},
        {"roster_id": 9, "week": 3, "finalized": False},
    ])}
    assert _finalized_fantasy_week(ctx, "9", 3) == 2
    assert _finalized_fantasy_week(ctx, "other", 3) is None
