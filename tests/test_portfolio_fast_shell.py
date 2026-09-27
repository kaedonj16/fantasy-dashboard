"""Account portfolios populate cards and cross-league insights on first paint."""


def test_account_portfolio_renders_complete_cross_league_data(offline_client, monkeypatch):
    import routes.user_pages_bp as pages
    from datetime import datetime, timezone

    seen = {}
    monkeypatch.setattr(pages, "get_nfl_state", lambda: {"season": 2026, "week": 3})
    monkeypatch.setattr(pages, "_games_scheduled_today", lambda season, week: False)
    def resolve(viewer_user_id, account_id, current_season, *, enrich_live=True):
        seen["kwargs"] = {"enrich_live": enrich_live}
        return ([{
            "league_id": "L1", "platform": "sleeper", "season": 2026,
            "name": "Fast League", "is_favorite": True,
        }], 2026)
    monkeypatch.setattr("dashboard_services.accounts.resolve_my_leagues", resolve)
    monkeypatch.setattr(
        "dashboard_services.accounts.schedule_account_league_reconciliation",
        lambda *a, **k: None,
    )
    # Account users are served from the Postgres summary cache
    # (stale-while-revalidate); the inline context computation is bypassed.
    monkeypatch.setattr(pages, "get_cached_summary", lambda *a: {
        "platform": "sleeper", "league_id": "L1", "season": 2026,
        "name": "Fast League", "state": "ready",
        "wins": 7, "losses": 4, "ties": 0, "record": "7-4", "rank": 2,
        "total_teams": 12, "pf": 1234.5,
        "streak": ["W", "L", "W"],
        "pos_user_rank": {"QB": 3, "RB": 5, "WR": 2, "TE": 8},
        "pos_user_pctile": {"QB": 70.0, "RB": 55.0, "WR": 85.0, "TE": 30.0},
        "all_players": {
            "p1": {"name": "Player One", "position": "WR", "value": 321.0,
                   "pos_rank": "WR12", "nfl_team": "BUF"},
        },
        "total_value": 321.0, "offseason": False, "urgency": 2.8,
        "team_name": "My Team",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 7
        sess["account_email"] = "a@example.com"
    response = offline_client.get("/portfolio")
    assert response.status_code == 200
    assert b"Fast League" in response.data
    assert b"7-4" in response.data
    assert b"Player Holdings" in response.data and b"Player One" in response.data
    assert b"NFL Exposure" in response.data
    assert b"Positional Strength" in response.data
    assert b"Record loading" not in response.data
    assert seen["kwargs"]["enrich_live"] is False


def test_summary_endpoint_replaces_shell_even_when_matchup_is_not_live(offline_client, monkeypatch):
    """Summary has its own endpoint; live:false can never gate the record."""
    monkeypatch.setattr(
        "dashboard_services.accounts.resolve_account_leagues",
        lambda account_id, current_season=None: [{
            "platform": "sleeper", "league_id": "L1", "season": 2026,
            "name": "Correct season", "team_id": "9",
        }],
    )
    monkeypatch.setattr(
        "dashboard_services.portfolio_summary.build_league_summary",
        lambda account_id, membership, loader: {
            "platform": "sleeper", "league_id": "L1", "season": 2026,
            "state": "ready", "team_name": "Viewer's Team", "record": "7-4-1",
            "rank": 3, "total_teams": 12, "refreshed_at": "2026-09-15T12:00:00+00:00",
        },
    )
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 7
    response = offline_client.get("/api/portfolio/summary?platform=sleeper&league_id=L1&season=2026")
    assert response.status_code == 200
    assert response.json["state"] == "ready"
    assert response.json["team_name"] == "Viewer's Team"
    assert response.json["record"] == "7-4-1"
    assert response.json["total_teams"] == 12


def test_summary_endpoint_rechecks_membership_and_season(offline_client, monkeypatch):
    monkeypatch.setattr(
        "dashboard_services.accounts.resolve_account_leagues",
        lambda *a, **k: [{"platform": "sleeper", "league_id": "L1", "season": 2025}],
    )
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 8
    response = offline_client.get("/api/portfolio/summary?platform=sleeper&league_id=L1&season=2026")
    assert response.status_code == 403


def test_fast_shell_keeps_card_seasons_in_navigation_and_pagination(offline_client, monkeypatch):
    import routes.user_pages_bp as pages
    monkeypatch.setattr(pages, "get_nfl_state", lambda: {"season": 2026})
    leagues = [{"league_id": f"L{i}", "platform": "yahoo", "season": 2025,
                "name": f"League {i}"} for i in range(1, 6)]
    monkeypatch.setattr("dashboard_services.accounts.resolve_my_leagues",
                        lambda *a, **k: (leagues, 2026))
    monkeypatch.setattr("dashboard_services.accounts.schedule_account_league_reconciliation",
                        lambda *a: None)
    monkeypatch.setattr(pages, "get_league_ctx_from_cache",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 9
    response = offline_client.get("/portfolio")
    assert response.status_code == 200
    assert b"data-lg-key='yahoo:L1'" in response.data
    assert b"Page '+(page+1)+' of '+pages" in response.data
    assert b"pager.hidden=ord.length<=PAGE" in response.data


def test_fast_shell_renders_all_cards_in_durable_order(offline_client, monkeypatch):
    import routes.user_pages_bp as pages
    monkeypatch.setattr(pages, "get_nfl_state", lambda: {"season": 2026})
    # Favorites first, then a stable secondary order.
    leagues = [
        {"league_id": "C", "platform": "sleeper", "season": 2026,
         "name": "League C", "is_favorite": True},
        {"league_id": "A", "platform": "sleeper", "season": 2026,
         "name": "League A", "is_favorite": True},
        {"league_id": "D", "platform": "espn", "season": 2026,
         "name": "League D", "is_favorite": False},
        {"league_id": "B", "platform": "espn", "season": 2026,
         "name": "League B", "is_favorite": False},
    ]
    monkeypatch.setattr("dashboard_services.accounts.resolve_my_leagues",
                        lambda *a, **k: (leagues, 2026))
    monkeypatch.setattr("dashboard_services.accounts.schedule_account_league_reconciliation",
                        lambda *a: None)
    monkeypatch.setattr(pages, "get_league_ctx_from_cache",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 10
    response = offline_client.get("/portfolio")
    assert response.status_code == 200
    # All cards are in the first paint, not appended as responses finish.
    # (Match the real card markup, not the hydration script's own
    # `[data-summary-card]` selector literals, which also contain the substring.)
    assert response.data.count(b"data-lg-key='") == 4
    # Order is locked to the durable membership.
    order = response.data.decode()
    c = order.index("data-lg-key='sleeper:C'")
    a = order.index("data-lg-key='sleeper:A'")
    d = order.index("data-lg-key='espn:D'")
    b = order.index("data-lg-key='espn:B'")
    assert a < c < b < d


def test_cold_league_renders_hydratable_shell_not_a_blocking_build(offline_client, monkeypatch):
    """A league with no cached summary must render a hydratable 'loading'
    shell -- the client /api/portfolio/card loader fills it -- instead of
    triggering a cold synchronous build inline. That inline build pinned a
    worker for ~80s on a multi-league portfolio and starved the page's own
    summary/refresh XHRs.
    """
    import routes.user_pages_bp as pages
    monkeypatch.setattr(pages, "get_nfl_state", lambda: {"season": 2026, "week": 3})
    monkeypatch.setattr(pages, "_games_scheduled_today", lambda season, week: False)
    monkeypatch.setattr(
        "dashboard_services.accounts.resolve_my_leagues",
        lambda *a, **k: ([{"league_id": "L1", "platform": "sleeper",
                           "season": 2026, "name": "Cold League"}], 2026),
    )
    monkeypatch.setattr("dashboard_services.accounts.schedule_account_league_reconciliation",
                        lambda *a, **k: None)
    # Cold cache: no summary row.
    monkeypatch.setattr(pages, "get_cached_summary", lambda *a: None)
    # The page render must not touch the league context at all on a cold
    # cache; the background refresh (kicked below) warms it separately.
    def _no_ctx(*a, **k):
        raise AssertionError("page must not read league context on a cold cache")
    monkeypatch.setattr(pages, "get_league_ctx_from_cache", _no_ctx)
    kicked = []
    monkeypatch.setattr(pages, "_kick_summary_refresh",
                        lambda account_id, membership: kicked.append(
                            (account_id, str(membership.get("league_id")))))
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 11
    response = offline_client.get("/portfolio")
    assert response.status_code == 200
    # A background refresh is kicked for the cold league.
    assert kicked == [(11, "L1")]
    # The cold league is a hydratable card the client loaders will fill.
    assert b"data-lg-key='sleeper:L1'" in response.data
    assert b"data-summary-card" in response.data
    assert b"Record loading" in response.data


def test_summary_endpoint_returns_pending_without_running_provider_work(offline_client, monkeypatch):
    """A passive cold-cache read schedules a warm and never invokes summary work."""
    monkeypatch.setattr(
        "dashboard_services.accounts.resolve_account_leagues",
        lambda account_id, current_season=None: [{
            "platform": "fleaflicker", "league_id": "92916", "season": 2026,
            "name": "Down League",
        }],
    )

    monkeypatch.setattr("routes.user_pages_bp.get_league_ctx_from_cache",
                        lambda *a, **k: {})
    monkeypatch.setattr("dashboard_services.portfolio_summary.build_league_summary",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("passive cold request performed summary work")))
    # A cold passive request returns pending before summary/provider work starts.
    monkeypatch.setattr("dashboard_services.portfolio_summary.get_cached_summary",
                        lambda *a, **k: None)
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 7
    response = offline_client.get(
        "/api/portfolio/summary?platform=fleaflicker&league_id=92916&season=2026")
    assert response.status_code == 200
    body = response.json
    assert body["ok"] is True
    assert body["state"] == "pending"
    assert body["pending"] is True
    assert body["retry_after_ms"] == 3000
