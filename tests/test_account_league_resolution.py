from dashboard_services import accounts


def test_google_account_uses_every_linked_sleeper_identity(monkeypatch):
    monkeypatch.setattr(
        accounts, "list_account_platform_ids", lambda account_id, platform: ["user-a", "user-b"]
    )
    monkeypatch.setattr(accounts, "list_user_leagues", lambda account_id: [])

    memberships = {
        "user-a": [{"league_id": "league-a", "name": "Alpha", "season": 2026}],
        "user-b": [{"league_id": "league-b", "name": "Beta", "season": 2026}],
    }
    monkeypatch.setattr(
        "dashboard_services.api.get_sleeper_user_leagues",
        lambda user_id, season: memberships.get(user_id, []),
    )

    leagues, season = accounts.resolve_my_leagues(None, 42, 2026)

    assert season == 2026
    assert {league["league_id"] for league in leagues} == {"league-a", "league-b"}


def test_active_viewer_and_linked_identity_are_deduplicated(monkeypatch):
    monkeypatch.setattr(
        accounts, "list_account_platform_ids", lambda account_id, platform: ["same-user"]
    )
    monkeypatch.setattr(accounts, "list_user_leagues", lambda account_id: [])
    calls = []

    def fetch(user_id, season):
        calls.append((user_id, season))
        return [{"league_id": "league-1", "name": "League", "season": season}]

    monkeypatch.setattr("dashboard_services.api.get_sleeper_user_leagues", fetch)

    leagues, _ = accounts.resolve_my_leagues("same-user", 42, 2026)

    assert calls == [("same-user", 2026)]
    assert [league["league_id"] for league in leagues] == ["league-1"]


def test_one_failed_sleeper_identity_does_not_hide_other_leagues(monkeypatch):
    monkeypatch.setattr(
        accounts, "list_account_platform_ids", lambda account_id, platform: ["stale", "working"]
    )
    monkeypatch.setattr(
        accounts,
        "list_user_leagues",
        lambda account_id: [{"platform": "espn", "league_id": "espn-1", "season": 2026}],
    )

    def fetch(user_id, season):
        if user_id == "stale":
            raise RuntimeError("provider unavailable")
        return [{"league_id": "sleeper-1", "season": season}]

    monkeypatch.setattr("dashboard_services.api.get_sleeper_user_leagues", fetch)

    leagues, _ = accounts.resolve_my_leagues(None, 42, 2026)

    assert {(league["platform"], league["league_id"]) for league in leagues} == {
        ("sleeper", "sleeper-1"),
        ("espn", "espn-1"),
    }


def test_google_account_returns_every_saved_platform_without_provider_sessions(monkeypatch):
    monkeypatch.setattr(accounts, "list_account_platform_ids", lambda *args: [])
    monkeypatch.setattr(
        accounts,
        "list_user_leagues",
        lambda account_id: [
            {"platform": "sleeper", "league_id": "sleeper-saved", "season": 2026},
            {"platform": "espn", "league_id": "espn-saved", "season": 2025},
            {"platform": "yahoo", "league_id": "yahoo-saved", "season": 2025},
            {"platform": "future", "league_id": "future-saved", "season": 2026},
        ],
    )

    leagues, _ = accounts.resolve_my_leagues(None, 42, 2026)

    assert {(league["platform"], league["league_id"]) for league in leagues} == {
        ("sleeper", "sleeper-saved"),
        ("espn", "espn-saved"),
        ("yahoo", "yahoo-saved"),
        ("future", "future-saved"),
    }
    assert next(league for league in leagues if league["platform"] == "espn")["season"] == 2026


def test_saved_sleeper_league_survives_live_provider_failure(monkeypatch):
    monkeypatch.setattr(accounts, "list_account_platform_ids", lambda *args: ["linked-user"])
    monkeypatch.setattr(
        accounts,
        "list_user_leagues",
        lambda account_id: [{
            "platform": "sleeper", "league_id": "saved-league", "season": 2026,
            "name": "Saved name", "team_id": "7",
        }],
    )

    def unavailable(*args):
        raise RuntimeError("Sleeper unavailable")

    monkeypatch.setattr("dashboard_services.api.get_sleeper_user_leagues", unavailable)

    leagues, _ = accounts.resolve_my_leagues(None, 42, 2026)

    assert leagues == [{
        "platform": "sleeper", "league_id": "saved-league", "season": 2026,
        "name": "Saved name", "team_id": "7",
    }]


def test_live_sleeper_metadata_enriches_saved_membership_without_duplicate(monkeypatch):
    monkeypatch.setattr(accounts, "list_account_platform_ids", lambda *args: ["linked-user"])
    monkeypatch.setattr(
        accounts,
        "list_user_leagues",
        lambda account_id: [{
            "platform": "sleeper", "league_id": "same-league", "season": 2025,
            "name": "Old name", "team_id": "3",
        }],
    )
    monkeypatch.setattr(
        "dashboard_services.api.get_sleeper_user_leagues",
        lambda user_id, season: [{
            "league_id": "same-league", "season": 2026, "name": "Current name",
        }],
    )

    leagues, _ = accounts.resolve_my_leagues(None, 42, 2026)

    assert leagues == [{
        "platform": "sleeper", "league_id": "same-league", "season": 2026,
        "name": "Current name", "team_id": "3",
    }]


def test_sleeper_username_lookup_imports_from_service_not_app():
    source = open("routes/league_meta_bp.py", encoding="utf-8").read()
    assert "from app import get_sleeper_user_by_username" not in source
