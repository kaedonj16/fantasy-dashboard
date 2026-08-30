"""Week 1 / pre-finalized standings should still list every team at 0-0."""
import pandas as pd

from dashboard_services.service import _seed_zero_standings, finalize_team_stats


def test_seed_zero_standings_from_avatars():
    avatars = {"Alpha": "a.png", "Bravo": None, "Charlie": "c.png"}
    df = _seed_zero_standings(avatars)
    assert list(df["owner"]) == ["Alpha", "Bravo", "Charlie"]
    assert (df["Wins"] == 0).all()
    assert (df["PF"] == 0.0).all()
    assert list(df["Record"]) == ["0-0", "0-0", "0-0"]


def test_finalize_empty_weeks_seeds_standings():
    empty = pd.DataFrame(columns=["owner", "week", "points", "points_against", "matchup_id"])
    avatars = {"Team A": "", "Team B": ""}
    stats = finalize_team_stats(empty, avatars, {}, [], last_week=1)
    assert not stats.empty
    assert set(stats["owner"]) == {"Team A", "Team B"}
    assert (stats["Wins"] == 0).all()


def test_seed_empty_avatar_map_returns_empty():
    assert _seed_zero_standings({}).empty
    assert _seed_zero_standings(None).empty
