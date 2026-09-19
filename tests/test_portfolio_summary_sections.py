import pytest

pd = pytest.importorskip("pandas")

from dashboard_services.portfolio_summary import first_non_null, recent_streak


def test_recent_streak_prefers_canonical_weekly_score_columns():
    weekly = pd.DataFrame([
        {"week": 1, "roster_id": "1", "points": 120, "points_against": 100, "finalized": True},
        {"week": 2, "roster_id": "1", "points": 98, "points_against": 110, "finalized": True},
        {"week": 3, "roster_id": "1", "points": 130, "points_against": 125, "finalized": True},
    ])
    assert recent_streak(weekly, "1") == ["W", "L", "W"]


def test_recent_streak_supports_legacy_aliases_and_real_zeroes():
    weekly = pd.DataFrame([
        {"week": 1, "roster_id": 7, "pts": 0.0, "opp_pts": 3.0, "finalized": True},
        {"week": 2, "roster_id": 7, "PF": 4.0, "PA": 0.0, "finalized": True},
    ])
    assert recent_streak(weekly, "7") == ["L", "W"]
    assert first_non_null({"points": 0.0, "pts": 99}, ("points", "pts")) == 0.0


def test_progressive_hydration_keeps_streak_slot_and_resets_manual_retry():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    # render() replaces the stats container, so its replacement must recreate
    # the streak target before the section patch runs.
    render = source[source.index("function render(c,d)"):source.index("function load(c)")]
    assert "data-summary-streak" in render
    assert "c._summaryAttempt=0;q.unshift(c)" in source


def test_cached_stale_summary_preserves_underlying_sync_timestamp(monkeypatch):
    import dashboard_services.accounts as accounts
    import dashboard_services.portfolio_summary as summaries

    synced = "2026-09-19T10:00:00+00:00"
    monkeypatch.setattr(accounts, "resolve_account_viewer_for_league",
                        lambda *a, **k: {"viewer_roster_id": "1"})
    monkeypatch.setattr(summaries, "_store_persistent", lambda *a: None)
    ctx = {
        "_cache_synced_at": synced,
        "_cache_stale": True,
        "rosters": [{"roster_id": 1, "owner_id": "u", "players": [], "settings": {}}],
        "users": [{"user_id": "u", "display_name": "Team"}],
        "league": {"name": "League"},
        "players_index": {},
    }
    result = summaries.build_league_summary(
        7, {"platform": "sleeper", "league_id": "L", "season": 2026},
        lambda *a: ctx,
    )
    assert result["generated_at"] != synced
    assert result["last_successful_sync_at"] == synced
    assert result["refreshed_at"] == synced
    assert result["stale"] is True and result["_cache_stale"] is True
    assert result["partial"] is True


def test_missing_context_freshness_stays_unknown(monkeypatch):
    import dashboard_services.accounts as accounts
    import dashboard_services.portfolio_summary as summaries

    monkeypatch.setattr(accounts, "resolve_account_viewer_for_league",
                        lambda *a, **k: {"viewer_roster_id": "1"})
    monkeypatch.setattr(summaries, "_store_persistent", lambda *a: None)
    result = summaries.build_league_summary(
        7, {"platform": "sleeper", "league_id": "L", "season": 2026},
        lambda *a: {"rosters": [{"roster_id": 1, "players": [], "settings": {}}],
                    "users": [], "league": {}, "players_index": {}},
    )
    assert result["last_successful_sync_at"] is None
    assert result["refreshed_at"] is None
