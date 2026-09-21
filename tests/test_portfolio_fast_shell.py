"""Account portfolios populate cards and cross-league insights on first paint."""


def test_account_portfolio_renders_complete_cross_league_data(offline_client, monkeypatch):
    import routes.user_pages_bp as pages

    seen = {}
    monkeypatch.setattr(pages, "get_nfl_state", lambda: {"season": 2026})
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
    monkeypatch.setattr(pages, "get_league_ctx_from_cache", lambda *a, **k: {
        "league": {"name": "Fast League"},
        "rosters": [{"roster_id": 9, "owner_id": "u1",
                     "players": ["p1", "p2", "p3", "p4", "p5"]}],
        "users": [{"user_id": "u1", "display_name": "My Team"}],
        "players_index": {"p1": {"name": "Player One", "pos": "WR", "team": "BUF"}},
        "total_rosters": 1, "roster_positions": ["QB", "RB", "WR", "TE"],
        "latest_draft": {"status": "complete"},
    })
    monkeypatch.setattr("dashboard_services.accounts.resolve_account_viewer_for_league",
                        lambda *a, **k: {"viewer_roster_id": "9"})
    monkeypatch.setattr("dashboard_services.ai.context_builders.league_format_value_lookup",
                        lambda ctx: {"p1": {"name": "Player One", "position": "WR",
                                              "team": "BUF", "value": 321,
                                              "pos_rank_label": "WR12"}})
    monkeypatch.setattr("dashboard_services.ai.context_builders.portfolio_record_and_rank",
                        lambda *a: (7, 4, 0, 1234.5, 2))
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
    """A league whose context is not cached must render a hydratable 'loading'
    shell -- the client summary/matchup loaders fill it -- instead of triggering a
    cold synchronous build inline. That inline build pinned a worker for ~80s on a
    multi-league portfolio and starved the page's own summary/refresh XHRs.
    """
    import routes.user_pages_bp as pages
    monkeypatch.setattr(pages, "get_nfl_state", lambda: {"season": 2026})
    monkeypatch.setattr(
        "dashboard_services.accounts.resolve_my_leagues",
        lambda *a, **k: ([{"league_id": "L1", "platform": "sleeper",
                           "season": 2026, "name": "Cold League"}], 2026),
    )
    monkeypatch.setattr("dashboard_services.accounts.schedule_account_league_reconciliation",
                        lambda *a, **k: None)
    calls = {}

    def loader(platform, league_id, season, *, allow_build=True):
        calls["allow_build"] = allow_build
        return {}  # cold: nothing cached

    monkeypatch.setattr(pages, "get_league_ctx_from_cache", loader)
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 11
    response = offline_client.get("/portfolio")
    assert response.status_code == 200
    # The page render only reads cache; it never asks to build inline.
    assert calls["allow_build"] is False
    # The cold league is a hydratable card the client loaders will fill.
    assert b"data-lg-key='sleeper:L1'" in response.data
    assert b"data-summary-card" in response.data
    assert b"Record loading" in response.data
