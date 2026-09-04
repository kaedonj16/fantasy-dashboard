"""MFL starter derivation and projected playoff brackets."""
from dashboard_services.providers.mfl_api import MFLProvider


def test_mfl_starters_from_weekly_results(monkeypatch):
    provider = MFLProvider()
    payloads = {
        "league": {"league": {"id": "123", "name": "Dynasty", "size": "2",
                    "lastRegularSeasonWeek": "14", "starters": "QB,RB,WR",
                    "franchises": {"franchise": [{"id": "0001", "name": "Owls"}]}}},
        "players": {"players": {"player": [{"id": "9", "name": "Known Player", "position": "QB"}]}},
        "rosters": {"rosters": {"franchise": [{"id": "0001", "player": [{"id": "9"}, {"id": "8"}]}]}},
        "weeklyResults": {"weeklyResults": {"matchup": [{"franchise": [
            {"id": "0001", "score": "101.5", "player": [
                {"id": "9", "status": "starter", "score": "20"},
                {"id": "8", "status": "nonstarter", "score": "4"},
            ]},
            {"id": "0002", "score": "99"},
        ]}]}},
    }
    monkeypatch.setattr(provider, "_export", lambda kind, *a, **k: payloads[kind])
    monkeypatch.setattr(provider, "_canonical_map", lambda *a: {"9": "canon-9", "8": "canon-8"})
    roster = provider.get_rosters("123", 2026)[0]
    assert roster["starters"] == ["canon-9"]
    assert roster["players"] == ["canon-9", "canon-8"]


def test_mfl_bracket_projects_from_seeds(monkeypatch):
    provider = MFLProvider()
    monkeypatch.setattr(provider, "get_league", lambda *a, **k: {
        "settings": {"playoff_week_start": 15, "playoff_teams": 4},
    })
    monkeypatch.setattr(provider, "get_matchups", lambda *a, **k: [])
    monkeypatch.setattr(provider, "_playoff_seeds", lambda *a, **k: [1, 2, 3, 4])
    games = provider.get_bracket("1", 2026, "winners")
    assert len(games) == 2
    assert all(g.get("projected") for g in games)
    assert provider.get_bracket("1", 2026, "losers") == []


def test_mfl_bracket_derives_from_playoff_matchups(monkeypatch):
    provider = MFLProvider()
    monkeypatch.setattr(provider, "get_league", lambda *a, **k: {
        "settings": {"playoff_week_start": 15, "playoff_teams": 4},
    })
    monkeypatch.setattr(provider, "get_matchups", lambda lid, season, week: (
        [
            {"roster_id": 1, "matchup_id": 1, "points": 120},
            {"roster_id": 4, "matchup_id": 1, "points": 80},
        ] if week == 15 else []
    ))
    monkeypatch.setattr(provider, "_playoff_seeds", lambda *a, **k: [1, 2, 3, 4])
    games = provider.get_bracket("1", 2026, "winners")
    assert games[0]["w"] == 1
    assert games[0]["l"] == 4
    assert games[0].get("derived")
