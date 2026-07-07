"""Tests for utils.season_review."""
from utils.season_review import season_review


def test_empty():
    assert season_review([]) == {}
    assert season_review([{"week": 1, "points": None, "win": 0}]) == {}


def test_basic_record_and_scoring():
    weekly = [
        {"week": 1, "points": 100.0, "win": 1},
        {"week": 2, "points": 80.0, "win": 0},
        {"week": 3, "points": 120.5, "win": 1},
        {"week": 4, "points": 90.0, "win": 0.5},
    ]
    r = season_review(weekly)
    assert r["games"] == 4
    assert r["wins"] == 2
    assert r["losses"] == 1
    assert r["ties"] == 1
    assert r["record"] == "2-1-1"
    assert r["points_for"] == 390.5
    assert r["avg_points"] == round(390.5 / 4, 1)
    assert r["best_week"] == {"week": 3, "points": 120.5}
    assert r["worst_week"] == {"week": 2, "points": 80.0}


def test_longest_win_streak():
    weekly = [
        {"week": 1, "points": 100, "win": 1},
        {"week": 2, "points": 100, "win": 1},
        {"week": 3, "points": 50, "win": 0},
        {"week": 4, "points": 100, "win": 1},
        {"week": 5, "points": 100, "win": 1},
        {"week": 6, "points": 100, "win": 1},
    ]
    assert season_review(weekly)["longest_win_streak"] == 3


def test_all_play_passthrough():
    weekly = [{"week": 1, "points": 100, "win": 1}]
    ap = {"all_play_wins": 8.0, "all_play_losses": 2.0, "luck_delta": -1.5, "expected_seed": 2}
    r = season_review(weekly, all_play_entry=ap, finish_rank=5, num_teams=12, pf_rank=3)
    assert r["all_play_record"] == "8-2"
    assert r["luck_delta"] == -1.5
    assert r["expected_seed"] == 2
    assert r["finish_rank"] == 5
    assert r["num_teams"] == 12
    assert r["pf_rank"] == 3


def test_week_order_independent():
    # Rows out of order should still find the right best/worst and streak.
    weekly = [
        {"week": 3, "points": 120, "win": 1},
        {"week": 1, "points": 100, "win": 1},
        {"week": 2, "points": 80, "win": 0},
    ]
    r = season_review(weekly)
    assert r["best_week"]["week"] == 3
    assert r["worst_week"]["week"] == 2
    assert r["longest_win_streak"] == 1
