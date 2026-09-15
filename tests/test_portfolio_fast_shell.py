"""The account portfolio first paint never performs provider/context work."""
def test_account_portfolio_uses_durable_shell(offline_client, monkeypatch):
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
    monkeypatch.setattr(pages, "get_league_ctx_from_cache", lambda *a: (_ for _ in ()).throw(AssertionError("context loaded")))
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 7
        sess["account_email"] = "a@example.com"
    response = offline_client.get("/portfolio")
    assert response.status_code == 200
    assert b"Fast League" in response.data
    assert b"Record loading" in response.data
    assert b"/api/portfolio/summary" in response.data
    assert b"data-summary-stats" in response.data
    assert b"Summary timed out. Retry." in response.data
    assert b"var cards=" in response.data and b"MAX=3" in response.data
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
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 9
    response = offline_client.get("/portfolio")
    assert response.status_code == 200
    assert b"/yahoo/2025/L1/dashboard" in response.data
    assert b"Page '+(page+1)+' of '+pages" in response.data
    assert b"pager.hidden=ord.length<=PAGE" in response.data
