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
    assert '("regular", "post")' in endpoint
    assert 'offseason_mode' in endpoint
    assert 'viewer_roster_id' in endpoint
    # Totals come from the shared live-total helper, oriented you/opp.
    assert "team_live_totals" in endpoint


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


def test_portfolio_card_has_live_slot_and_hydration():
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    # A hidden, per-league slot on valid cards carrying the fetch keys.
    assert "data-lg-live" in fn
    assert "data-platform=" in fn
    assert "data-league-id=" in fn
    assert "pf-lg-live" in fn
    # Client hydration hits the endpoint, caps concurrency, refreshes live games.
    assert "/api/portfolio/matchup" in fn
    assert "document.hidden" in fn
    # Offseason cards do not get a live slot (odds/scores are meaningless there).
    live_block = fn.split("_lg_season_live")[1].split("league_rows +=")[0]
    assert 'lg.get("offseason")' in live_block


def test_live_slot_not_on_pending_or_error_cards():
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    # The slot markup is emitted once, right before the valid-card markup, so
    # pending and error branches (which `continue` earlier) never render it.
    # (The JS querySelectorAll also references the attribute, hence the exact
    # "data-lg-live hidden" match here rather than a bare attribute count.)
    assert fn.count("data-lg-live hidden") == 1
