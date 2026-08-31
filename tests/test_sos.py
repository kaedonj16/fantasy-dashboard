"""Strength-of-schedule helpers: identity matching, week windows, indexing."""
import pytest

pytest.importorskip("pandas")
pd = pytest.importorskip("pandas")

from dashboard_services.service import (
    build_team_strength,
    compute_sos_by_team,
    compute_week_opponents,
    owner_pairs_from_weekly,
    regular_season_length,
)


def _stats(owners, avgs, win_pcts=None):
    n = len(owners)
    return pd.DataFrame({
        "owner": owners,
        "AVG": avgs,
        "Win%": win_pcts if win_pcts is not None else [0.5] * n,
    })


def _left_right(a, b, a_rid=None, b_rid=None):
    return {
        "left": {"name": a, "roster_id": a_rid or a},
        "right": {"name": b, "roster_id": b_rid or b},
    }


def test_regular_season_length_from_playoff_week_start():
    assert regular_season_length({"playoff_week_start": 15}) == 14
    assert regular_season_length({"playoff_week_start": 14}) == 13
    assert regular_season_length({"playoff_week_start": 0}) == 14
    assert regular_season_length({}) == 14


def test_compute_week_opponents_prefers_owner_name_over_roster_id():
    pairs = compute_week_opponents([
        _left_right("Alpha", "Delta", "1", "4"),
    ])
    assert pairs == [("Alpha", "Delta")]


def test_owner_pairs_from_weekly_uses_matchup_id():
    df = pd.DataFrame([
        {"owner": "Alpha", "week": 1, "matchup_id": 1, "opponent": "Delta"},
        {"owner": "Delta", "week": 1, "matchup_id": 1, "opponent": "Alpha"},
        {"owner": "Bravo", "week": 1, "matchup_id": 2, "opponent": "Charlie"},
        {"owner": "Charlie", "week": 1, "matchup_id": 2, "opponent": "Bravo"},
    ])
    pairs = owner_pairs_from_weekly(df)
    weeks = {w for w, _a, _b in pairs}
    assert weeks == {1}
    owners = {frozenset((a, b)) for _w, a, b in pairs}
    assert owners == {frozenset(("Alpha", "Delta")), frozenset(("Bravo", "Charlie"))}


def test_build_team_strength_ranks_by_scoring_not_powerscore():
    # PowerScore would invert the ranking (hot-but-weak vs steady scorer).
    ts = pd.DataFrame({
        "owner": ["HighAvg", "LowAvg"],
        "AVG": [140.0, 90.0],
        "Win%": [0.8, 0.2],
        "PowerScore": [-1.0, 2.0],
    })
    strength = build_team_strength(ts)
    assert strength["HighAvg"] > strength["LowAvg"]
    assert strength["HighAvg"] == pytest.approx(1.0)
    assert strength["LowAvg"] == pytest.approx(0.0)


def test_sos_tougher_past_opponent_scores_higher():
    owners = ["Alpha", "Bravo", "Charlie", "Delta"]
    strength = build_team_strength(_stats(owners, [130, 110, 95, 80], [0.8, 0.6, 0.4, 0.2]))
    matchups = {
        1: [_left_right("Alpha", "Delta"), _left_right("Bravo", "Charlie")],
        2: [_left_right("Alpha", "Bravo"), _left_right("Charlie", "Delta")],
    }
    out = compute_sos_by_team(matchups, strength, weeks_past=1, users=[], regular_season_weeks=2)
    # Alpha drew the weakest team in week 1; Delta drew the strongest.
    assert out["Delta"]["past_sos"] > out["Alpha"]["past_sos"]
    assert out["Delta"]["past_cnt"] == 1
    assert out["Alpha"]["ros_cnt"] == 1
    # League-average index is 100.
    assert abs(sum(v["past_sos"] for v in out.values()) / 4 - 100.0) < 1e-6


def test_sos_resolves_roster_ids_via_matchup_aliases():
    owners = ["Alpha", "Bravo", "Charlie", "Delta"]
    strength = build_team_strength(_stats(owners, [130, 110, 95, 80]))
    # Identities on the wire are roster ids; names still present for aliasing.
    matchups = {
        1: [
            {"left": {"name": "Alpha", "roster_id": "1"},
             "right": {"name": "Delta", "roster_id": "4"}},
            {"left": {"name": "Bravo", "roster_id": "2"},
             "right": {"name": "Charlie", "roster_id": "3"}},
        ],
    }
    # Force the old-shape path for past games: pairs of roster ids only.
    past_pairs = [(1, "1", "4"), (1, "2", "3")]
    out = compute_sos_by_team(
        matchups, strength, weeks_past=1, users=[],
        regular_season_weeks=1, past_pairs=past_pairs,
    )
    assert out["Delta"]["past_sos"] > out["Alpha"]["past_sos"]
    assert all(v["past_cnt"] == 1 for v in out.values())


def test_sos_display_name_users_map_to_team_name():
    strength = {"The Squad": 1.0, "Benchwarmers": 0.0}
    users = [
        {"display_name": "alice", "metadata": {"team_name": "The Squad"}},
        {"display_name": "bob", "metadata": {"team_name": "Benchwarmers"}},
    ]
    matchups = {
        1: [{"left": {"name": "alice"}, "right": {"name": "bob"}}],
    }
    out = compute_sos_by_team(matchups, strength, weeks_past=1, users=users, regular_season_weeks=1)
    assert out["The Squad"]["past_cnt"] == 1
    assert out["Benchwarmers"]["past_cnt"] == 1
    assert out["The Squad"]["past_sos"] < out["Benchwarmers"]["past_sos"]


def test_sos_ignores_playoff_weeks_even_when_weeks_past_is_high():
    owners = ["Alpha", "Bravo"]
    strength = {"Alpha": 1.0, "Bravo": 0.0}
    matchups = {
        1: [_left_right("Alpha", "Bravo")],
        14: [_left_right("Alpha", "Bravo")],
        15: [_left_right("Alpha", "Bravo")],  # playoff — must not count
    }
    out = compute_sos_by_team(
        matchups, strength, weeks_past=16, users=[], regular_season_weeks=14,
    )
    # Weeks 1 and 14 are past; week 15 is excluded. ROS is empty.
    assert out["Alpha"]["past_cnt"] == 2
    assert out["Bravo"]["past_cnt"] == 2
    assert out["Alpha"]["ros_cnt"] == 0
    assert out["Alpha"]["ros_sos"] == 0.0


def test_sos_past_pairs_from_weekly_do_not_need_matchups():
    owners = ["Alpha", "Bravo"]
    strength = {"Alpha": 1.0, "Bravo": 0.0}
    out = compute_sos_by_team(
        {}, strength, weeks_past=1, users=[], regular_season_weeks=2,
        past_pairs=[(1, "Alpha", "Bravo")],
    )
    assert out["Alpha"]["past_cnt"] == 1
    assert out["Bravo"]["past_sos"] > out["Alpha"]["past_sos"]
    assert out["Alpha"]["ros_cnt"] == 0  # no future matchups supplied


def test_equal_strength_indexes_to_one_hundred():
    strength = {"A": 0.5, "B": 0.5}
    matchups = {1: [_left_right("A", "B")]}
    out = compute_sos_by_team(matchups, strength, weeks_past=1, users=[], regular_season_weeks=1)
    assert out["A"]["past_sos"] == 100.0
    assert out["B"]["past_sos"] == 100.0
