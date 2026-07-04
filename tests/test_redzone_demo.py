"""Tests for utils.redzone_demo (deterministic Redzone demo simulation)."""
from utils.redzone_demo import (
    DEMO_GAME_SECONDS,
    DEMO_SCORING,
    demo_fold,
    demo_pts,
    demo_rng,
    demo_script,
)


class TestDemoRng:
    def test_deterministic_per_seed(self):
        a, b = demo_rng(42), demo_rng(42)
        assert [a() for _ in range(5)] == [b() for _ in range(5)]

    def test_different_seeds_diverge(self):
        a, b = demo_rng(1), demo_rng(2)
        assert [a() for _ in range(5)] != [b() for _ in range(5)]

    def test_zero_seed_does_not_lock_up(self):
        r = demo_rng(0)
        vals = [r() for _ in range(5)]
        assert all(0.0 <= v < 1.0 for v in vals)
        assert len(set(vals)) > 1

    def test_output_range(self):
        r = demo_rng(123)
        assert all(0.0 <= r() < 1.0 for _ in range(100))


class TestDemoScript:
    def test_stable_across_calls(self):
        assert demo_script("p_abc", "WR") == demo_script("p_abc", "WR")

    def test_plays_ordered_and_within_game(self):
        plays = demo_script("p_qb1", "QB")
        times = [p["t"] for p in plays]
        assert times == sorted(times)
        assert all(0 < t < DEMO_GAME_SECONDS for t in times)

    def test_position_play_kinds(self):
        qb_kinds = {p["kind"] for p in demo_script("p_q", "QB")}
        assert qb_kinds <= {"pass", "rush", "int"}
        rb_kinds = {p["kind"] for p in demo_script("p_r", "RB")}
        assert rb_kinds <= {"rush", "rec", "target"}
        wr_kinds = {p["kind"] for p in demo_script("p_w", "WR")}
        assert wr_kinds <= {"rec", "target"}


class TestDemoFold:
    PLAYS = [
        {"t": 10, "kind": "rec", "yds": 12, "td": 0},
        {"t": 40, "kind": "target"},
        {"t": 90, "kind": "rec", "yds": 8, "td": 1},
    ]

    def test_folds_only_up_to_time(self):
        L = demo_fold(self.PLAYS, 50)
        assert L["rec"] == 1
        assert L["targets"] == 2  # catch counts as a target too
        assert L["rec_yds"] == 12
        assert L["rec_td"] == 0

    def test_full_game(self):
        L = demo_fold(self.PLAYS, DEMO_GAME_SECONDS)
        assert L["rec"] == 2
        assert L["targets"] == 3
        assert L["rec_yds"] == 20
        assert L["rec_td"] == 1

    def test_time_zero_is_empty(self):
        L = demo_fold(self.PLAYS, 0)
        assert all(v == 0 for v in L.values())

    def test_qb_stats(self):
        plays = [
            {"t": 5, "kind": "pass", "yds": 25, "td": 1},
            {"t": 30, "kind": "int"},
            {"t": 60, "kind": "rush", "yds": 9, "td": 0},
        ]
        L = demo_fold(plays, 100)
        assert L["pass_yds"] == 25 and L["pass_td"] == 1
        assert L["int"] == 1
        assert L["carries"] == 1 and L["rush_yds"] == 9


class TestDemoPts:
    def test_scoring_math(self):
        L = demo_fold(TestDemoFold.PLAYS, DEMO_GAME_SECONDS)
        # 2 rec (0.5 PPR) + 20 rec yds (0.1) + 1 rec TD (6.0)
        assert demo_pts(L) == 2 * 0.5 + 20 * 0.1 + 6.0

    def test_interception_is_negative(self):
        L = demo_fold([{"t": 1, "kind": "int"}], 10)
        assert demo_pts(L) == DEMO_SCORING["pass_int"]

    def test_points_monotonic_over_time_without_ints(self):
        plays = demo_script("p_wr9", "WR")  # WR scripts contain no INTs
        pts = [demo_pts(demo_fold(plays, t)) for t in range(0, DEMO_GAME_SECONDS + 1, 60)]
        assert pts == sorted(pts)
