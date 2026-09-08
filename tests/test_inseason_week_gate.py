"""In-season UI should turn on the week a regular-season game is scheduled,
not only after Sleeper flips to ``regular`` or the first week finalizes.
"""
from datetime import date

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")


WEEK1_OPENER = [{"gameDate": "20260909", "gameID": "20260909_NE@SEA"}]


def _schedules(monkeypatch, week1=None, other=None):
    import app as appmod

    week1 = week1 if week1 is not None else WEEK1_OPENER

    def _load(season, week):
        if int(week) == 1:
            return week1
        return other or []

    monkeypatch.setattr(appmod, "load_week_schedule", _load)


def test_nfl_week_window_is_tue_through_mon():
    import app as appmod

    start, end = appmod._nfl_week_window(date(2026, 9, 8))  # Tuesday
    assert start == date(2026, 9, 8)
    assert end == date(2026, 9, 14)

    start, end = appmod._nfl_week_window(date(2026, 9, 9))  # Wednesday opener
    assert start == date(2026, 9, 8)
    assert end == date(2026, 9, 14)

    start, end = appmod._nfl_week_window(date(2026, 9, 14))  # Monday
    assert start == date(2026, 9, 8)
    assert end == date(2026, 9, 14)


def test_game_tomorrow_counts_as_this_week(monkeypatch):
    import app as appmod
    _schedules(monkeypatch)

    today = date(2026, 9, 8)  # Tuesday; opener is Wednesday Sep 9
    assert appmod._regular_season_week_with_games(2026, 0, today=today) == 1
    assert appmod._games_scheduled_this_week(2026, 0, today=today) is True


def test_august_preseason_does_not_count_week1_slate(monkeypatch):
    import app as appmod
    _schedules(monkeypatch)

    today = date(2026, 8, 15)
    assert appmod._regular_season_week_with_games(2026, 3, today=today) is None
    assert appmod._games_scheduled_this_week(2026, 3, today=today) is False


def test_preseason_with_game_this_week_is_in_season(monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "_regular_season_week_with_games", lambda *a, **k: 1)
    nfl = {"season": 2026, "week": 0, "season_type": "pre"}
    assert appmod._nfl_offseason_mode(nfl, 2026) is False


def test_preseason_without_game_this_week_stays_offseason(monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "_regular_season_week_with_games", lambda *a, **k: None)
    nfl = {"season": 2026, "week": 3, "season_type": "pre"}
    assert appmod._nfl_offseason_mode(nfl, 2026) is True


def test_historical_season_is_never_offseason_mode():
    import app as appmod
    nfl = {"season": 2026, "week": 0, "season_type": "pre"}
    assert appmod._nfl_offseason_mode(nfl, 2025) is False


def _zero_team_stats():
    return pd.DataFrame([
        {"owner": "Alpha", "Wins": 0, "Losses": 0, "Ties": 0, "PF": 0.0},
        {"owner": "Bravo", "Wins": 0, "Losses": 0, "Ties": 0, "PF": 0.0},
    ])


def test_standings_show_inseason_when_game_this_week(monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "_games_scheduled_this_week", lambda *a, **k: True)
    ctx = {
        "season": 2026,
        "current_week": 1,
        "offseason_mode": True,
        "team_stats": _zero_team_stats(),
    }
    assert appmod._use_offseason_standings(ctx) is False


def test_standings_stay_value_board_before_opener_week(monkeypatch):
    """Sleeper can flip to regular a week early; keep the value board until
    a game is actually scheduled this NFL week."""
    import app as appmod
    monkeypatch.setattr(appmod, "_games_scheduled_this_week", lambda *a, **k: False)
    ctx = {
        "season": 2026,
        "current_week": 1,
        "offseason_mode": False,
        "team_stats": _zero_team_stats(),
    }
    assert appmod._use_offseason_standings(ctx) is True


def test_standings_page_renders_inseason_body_when_game_this_week(monkeypatch):
    import routes.league_pages_bp as pages

    ctx = {
        "season": 2026,
        "current_week": 1,
        "offseason_mode": True,
        "team_stats": _zero_team_stats(),
        "platform": "sleeper",
        "league_id": "L",
    }
    monkeypatch.setattr(pages, "get_league_ctx_from_cache", lambda *a, **k: ctx)
    monkeypatch.setattr(pages, "_use_offseason_standings", lambda _ctx: False)
    monkeypatch.setattr(pages, "build_standings_body", lambda _ctx: "INSEASON_STANDINGS")
    monkeypatch.setattr(pages, "_build_offseason_standings_body", lambda _ctx: "OFFSEASON_STANDINGS")
    monkeypatch.setattr(
        pages, "render_page",
        lambda title, league_id, active, body, platform, season: body,
    )
    assert pages.page_standings("sleeper", 2026, "L") == "INSEASON_STANDINGS"
