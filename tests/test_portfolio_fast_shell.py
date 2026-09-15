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
    assert seen["kwargs"]["enrich_live"] is False
