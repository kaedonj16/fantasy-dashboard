from __future__ import annotations

from utils.weekly_email import (
    compact_league_blurb,
    multi_league_sections_html,
    other_leagues_for_account,
)


def test_compact_league_blurb_minimal(monkeypatch):
    monkeypatch.setattr(
        "utils.weekly_email._canonical_standing",
        lambda *a, **k: (3, 5, 2),
    )
    html = compact_league_blurb(
        platform="espn", season=2025, league_id="99",
        roster_id="1", league_name="Second League",
        base_url="https://brfantasy.com",
    )
    assert "Second League" in html
    assert "#3" in html
    assert "5-2" in html
    assert "/espn/2025/99/dashboard" in html


def test_compact_league_blurb_empty_league_id():
    assert compact_league_blurb(platform="sleeper", season=2025, league_id="") == ""


def test_other_leagues_skips_primary(monkeypatch):
    monkeypatch.setattr(
        "dashboard_services.accounts.list_user_leagues",
        lambda aid: [
            {"platform": "sleeper", "league_id": "A", "season": 2025, "name": "Primary", "team_id": "1"},
            {"platform": "espn", "league_id": "B", "season": 2025, "name": "Other", "team_id": "2"},
            {"platform": "yahoo", "league_id": "C", "season": 2025, "name": "Third", "team_id": "3"},
        ],
    )

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k):
            class R:
                def fetchone(self): return None
            return R()

    monkeypatch.setattr("dashboard_services.db.get_conn", lambda: _Conn())
    rows = other_leagues_for_account(
        7, primary_platform="sleeper", primary_league_id="A", primary_season=2025, limit=2,
    )
    assert [r["league_id"] for r in rows] == ["B", "C"]
    assert all(r["league_id"] != "A" for r in rows)


def test_multi_league_sections_empty_when_no_others(monkeypatch):
    monkeypatch.setattr("utils.weekly_email.other_leagues_for_account", lambda *a, **k: [])
    assert multi_league_sections_html(
        1, primary_platform="sleeper", primary_league_id="x", primary_season=2025,
    ) == ""


def test_multi_league_sections_header(monkeypatch):
    monkeypatch.setattr(
        "utils.weekly_email.other_leagues_for_account",
        lambda *a, **k: [{"platform": "espn", "league_id": "9", "season": 2025, "roster_id": "", "name": "Alt"}],
    )
    monkeypatch.setattr(
        "utils.weekly_email.compact_league_blurb",
        lambda **kw: f"<div>{kw['league_name']}</div>",
    )
    html = multi_league_sections_html(
        1, primary_platform="sleeper", primary_league_id="x", primary_season=2025,
    )
    assert "Your other leagues" in html
    assert "Alt" in html
