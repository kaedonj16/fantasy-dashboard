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

    source = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    # render() replaces the stats container, so its replacement must recreate
    # the streak target before the section patch runs.
    render = source[source.index("function renderSummary(card, data)"):source.index("function updateAggregateRecord()")]
    assert "data.streak" in render
    assert "schedule(owner, button.closest('.pf-lg-card'), 0, true)" in source


def test_render_summary_reapplies_server_side_styling_after_async_refresh():
    """renderSummary() rewrites data-summary-stats after the async summary
    fetch resolves. It must reproduce the exact classes/markup the server
    renders on first paint (record color, weak-standing flag, streak pills,
    the pf-lg-stat--streak alignment modifier), or those cues visibly
    disappear the moment the card hydrates."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    render = source[source.index("function renderSummary(card, data)"):source.index("function updateAggregateRecord()")]
    assert "color-win" in render and "color-loss" in render
    assert "pf-lg-v--weak" in render
    assert "pf-lg-stat--streak" in render
    assert "pf-s-pill" in render and "pf-s-w" in render and "pf-s-l" in render
    assert "pf-streak-empty" in render
    assert 'title="Regular-season standings: wins, then points for"' in render


def test_render_summary_rebuilds_weakest_room_line():
    """v25: the card's position data is one weakest-room line (the full rank
    grid lives in the insights heatmap). Cards that start in the 'loading'
    skeleton state only get the line once renderSummary rebuilds it."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    assert "function renderWeakest(card, data, totalN)" in source
    seg = source[source.index("function renderWeakest(card, data, totalN)"):source.index("function renderSummary(card, data)")]
    assert "pos_user_rank" in seg
    assert "data-weakest" in seg
    assert "Weakest room:" in seg
    render = source[source.index("function renderSummary(card, data)"):source.index("function updateAggregateRecord()")]
    assert "renderWeakest(card, data, totalN)" in render
    # Playoff odds came off the My Leagues cards 2026-10-27 (the stat never
    # populated): renderSummary must not rebuild an odds stat, and the sort
    # datum it stamps is now dataset.rank for the Standings sort.
    assert "playoff_odds_pct" not in render
    assert "card.dataset.rank" in render


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


def test_build_summary_carries_archetype_badge_strengths(monkeypatch):
    """The archetype badge ratios pos_user_vals against pos_league_avgs
    (league medians). build_league_summary is what populates the SWR cache
    every account card is served from; when it omitted both fields, every
    account portfolio card degraded to "Balanced" regardless of roster
    shape. Single-player rooms make weighted strength == value, so the
    expected numbers are exact."""
    import dashboard_services.accounts as accounts
    import dashboard_services.portfolio_summary as summaries

    monkeypatch.setattr(accounts, "resolve_account_viewer_for_league",
                        lambda *a, **k: {"viewer_roster_id": "1"})
    monkeypatch.setattr(
        "dashboard_services.ai.context_builders.portfolio_record_and_rank",
        lambda *a: (5, 2, 0, 500.0, 3))
    values = {
        "qb1": {"name": "QB One", "position": "QB", "value": 300},
        "qb2": {"name": "QB Two", "position": "QB", "value": 250},
        "qb3": {"name": "QB Three", "position": "QB", "value": 200},
        "te1": {"name": "TE One", "position": "TE", "value": 900},
        "te2": {"name": "TE Two", "position": "TE", "value": 100},
        "te3": {"name": "TE Three", "position": "TE", "value": 200},
    }
    monkeypatch.setattr(
        "dashboard_services.ai.context_builders.league_format_value_lookup",
        lambda ctx, _cache=None: values)
    stored = []
    monkeypatch.setattr(summaries, "_store_persistent",
                        lambda key, value: stored.append(value))
    ctx = {
        "rosters": [
            {"roster_id": 1, "owner_id": "u1", "players": ["qb1", "te1"], "settings": {}},
            {"roster_id": 2, "owner_id": "u2", "players": ["qb2", "te2"], "settings": {}},
            {"roster_id": 3, "owner_id": "u3", "players": ["qb3", "te3"], "settings": {}},
        ],
        "users": [{"user_id": "u1", "display_name": "My Team"}],
        "league": {"name": "League"},
        "players_index": {},
        "total_rosters": 3,
    }
    result = summaries.build_league_summary(
        7, {"platform": "sleeper", "league_id": "L", "season": 2026},
        lambda *a: ctx,
    )
    # Weighted strengths under the fixed denominator: a lone QB scores
    # 300 / 1.25 and a lone TE 900 / 1.20 (unfilled slots count as 0).
    assert result["pos_user_vals"] == {"QB": 240.0, "RB": 0.0, "WR": 0.0, "TE": 750.0}
    # Medians across the three rosters; an all-zero position falls back to 1
    # (the inline summary's `or 1`), so its ratio can never win the badge.
    assert result["pos_league_avgs"] == {"QB": 200.0, "RB": 1, "WR": 1, "TE": 166.66666666666669}
    # TE ratio 4.5x the median is the badge winner, not "Balanced".
    ratios = {p: result["pos_user_vals"][p] / result["pos_league_avgs"][p]
              for p in ("QB", "RB", "WR", "TE")}
    assert max(ratios, key=ratios.get) == "TE" and ratios["TE"] > 1.10
    # The cached copy the portfolio page actually serves carries them too.
    assert stored and stored[0]["pos_user_vals"] == result["pos_user_vals"]
    assert stored[0]["pos_league_avgs"] == result["pos_league_avgs"]


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
