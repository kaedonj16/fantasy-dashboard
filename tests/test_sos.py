"""Strength-of-schedule helpers: identity matching, week windows, indexing."""
import pytest

pytest.importorskip("pandas")
pd = pytest.importorskip("pandas")

from dashboard_services.service import (
    all_play_pct_by_owner,
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
    # Strength is a z-score blend now: two teams land at +1 / -1.
    ts = pd.DataFrame({
        "owner": ["HighAvg", "LowAvg"],
        "AVG": [140.0, 90.0],
        "Win%": [0.8, 0.2],
        "PowerScore": [-1.0, 2.0],
    })
    strength = build_team_strength(ts)
    assert strength["HighAvg"] > strength["LowAvg"]
    assert strength["HighAvg"] == pytest.approx(1.0)
    assert strength["LowAvg"] == pytest.approx(-1.0)


def test_all_play_pct_by_owner_counts_the_whole_week():
    df = pd.DataFrame([
        {"owner": "Alpha", "week": 1, "points": 130.0},
        {"owner": "Bravo", "week": 1, "points": 110.0},
        {"owner": "Charlie", "week": 1, "points": 90.0},
        {"owner": "Delta", "week": 1, "points": 110.0},
        {"owner": "Alpha", "week": 2, "points": 100.0},
        {"owner": "Bravo", "week": 2, "points": 120.0},
        {"owner": "Charlie", "week": 2, "points": 140.0},
        {"owner": "Delta", "week": 2, "points": 80.0},
    ])
    ap = all_play_pct_by_owner(df)
    # Week 1: Alpha beats all 3; Bravo beats Charlie, ties Delta -> 1.5/3.
    # Week 2: Alpha beats only Delta -> 1/3.
    assert ap["Alpha"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert ap["Bravo"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert ap["Delta"] == pytest.approx((0.5 + 0.0) / 2)


def test_build_team_strength_prefers_all_play_over_head_to_head_win_pct():
    # Mirror U won its only head-to-head game (against the weakest scorer)
    # while Mirror V lost to the strongest -- H2H Win% ranks U over V, but
    # all-play (and scoring) rank V over U, and strength must follow them.
    ts = pd.DataFrame({
        "owner": ["Top", "MirrorU", "MirrorV", "Bottom"],
        "AVG": [140.0, 100.0, 110.0, 90.0],
        "Win%": [1.0, 1.0, 0.0, 0.0],
    })
    all_play = {"Top": 1.0, "MirrorU": 0.25, "MirrorV": 0.75, "Bottom": 0.0}
    strength = build_team_strength(ts, all_play_pct=all_play)
    assert strength["MirrorV"] > strength["MirrorU"]
    h2h_only = build_team_strength(ts)
    assert h2h_only["MirrorU"] > h2h_only["MirrorV"]


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
    # Two teams only play each other, so the centered index is 100 for
    # both; the identity assertions above are what this test pins.
    assert out["The Squad"]["past_sos"] == 100.0
    assert out["Benchwarmers"]["past_sos"] == 100.0


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
    # Two teams only ever play each other: the schedules are identical, so
    # neither reads tougher -- the old raw average called Bravo's harder
    # purely because Alpha is the stronger team.
    assert out["Alpha"]["past_sos"] == 100.0
    assert out["Bravo"]["past_sos"] == 100.0
    assert out["Alpha"]["ros_cnt"] == 0  # no future matchups supplied


def test_sos_balanced_round_robin_is_100_for_everyone():
    # The headline flaw: in a perfectly balanced round robin (everyone
    # faces everyone once, zero schedule luck) the old formula still
    # graded the best team easiest (~84) and the worst hardest (~116)
    # because a team never plays itself. Centered SOS must read 100 for
    # all of them, however strong or weak they are.
    owners = ["Alpha", "Bravo", "Charlie", "Delta"]
    strength = build_team_strength(
        _stats(owners, [140, 120, 100, 80], [1.0, 0.67, 0.33, 0.0])
    )
    pairs = [
        (1, "Alpha", "Delta"), (1, "Bravo", "Charlie"),
        (2, "Alpha", "Charlie"), (2, "Bravo", "Delta"),
        (3, "Alpha", "Bravo"), (3, "Charlie", "Delta"),
    ]
    out = compute_sos_by_team(
        {}, strength, weeks_past=3, users=[], regular_season_weeks=3,
        past_pairs=pairs,
    )
    assert all(v["past_cnt"] == 3 for v in out.values())
    assert all(v["past_sos"] == pytest.approx(100.0) for v in out.values())


def test_sos_future_uses_future_strength_when_supplied():
    owners = ["Alpha", "Bravo", "Charlie", "Delta"]
    # Season strength: Alpha is the strongest. Future strength (e.g. its
    # stars are hurt): Alpha drops to weakest, Delta becomes strongest.
    season = {"Alpha": 2.0, "Bravo": 0.5, "Charlie": -0.5, "Delta": -2.0}
    future = {"Alpha": -2.0, "Bravo": -0.5, "Charlie": 0.5, "Delta": 2.0}
    matchups = {
        1: [_left_right("Alpha", "Bravo"), _left_right("Charlie", "Delta")],
        2: [_left_right("Alpha", "Delta"), _left_right("Bravo", "Charlie")],
    }
    out = compute_sos_by_team(
        matchups, season, weeks_past=1, users=[], regular_season_weeks=2,
        future_strength=future,
    )
    # Week 2: Alpha faces Delta -- the toughest future opponent -- while
    # Delta faces the weakest (Alpha). Season strength would invert this.
    assert out["Alpha"]["ros_sos"] > out["Delta"]["ros_sos"]
    plain = compute_sos_by_team(
        matchups, season, weeks_past=1, users=[], regular_season_weeks=2,
    )
    assert plain["Alpha"]["ros_sos"] < plain["Delta"]["ros_sos"]


def test_equal_strength_indexes_to_one_hundred():
    strength = {"A": 0.5, "B": 0.5}
    matchups = {1: [_left_right("A", "B")]}
    out = compute_sos_by_team(matchups, strength, weeks_past=1, users=[], regular_season_weeks=1)
    assert out["A"]["past_sos"] == 100.0
    assert out["B"]["past_sos"] == 100.0
