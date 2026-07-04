"""Tests for utils.pick_slots (rookie draft order and pick labels)."""
from utils.pick_slots import (
    avg_pick_value_for_round,
    compute_pick_slots,
    pick_label,
    placements_from_bracket,
    slots_from_regular_season,
)


# A 6-team playoff bracket: champion rid 1, runner-up 2, third 3, fourth 4.
BRACKET = [
    {"t1": 1, "t2": 4, "w": 1, "l": 4},
    {"t1": 2, "t2": 3, "w": 2, "l": 3},
    {"t1": 1, "t2": 2, "w": 1, "l": 2, "p": 1},   # title game
    {"t1": 3, "t2": 4, "w": 3, "l": 4, "p": 3},   # third-place game
]


class TestPlacementsFromBracket:
    def test_collects_participants(self):
        rids, _ = placements_from_bracket(BRACKET)
        assert rids == {1, 2, 3, 4}

    def test_placements_from_p_field(self):
        _, placements = placements_from_bracket(BRACKET)
        assert placements == {1: 1, 2: 2, 3: 3, 4: 4}

    def test_ignores_tbd_references(self):
        bracket = [{"t1": {"w": 1}, "t2": 2, "w": None, "l": None}]
        rids, placements = placements_from_bracket(bracket)
        assert rids == {2}
        assert placements == {}

    def test_empty_bracket(self):
        assert placements_from_bracket([]) == (set(), {})
        assert placements_from_bracket(None) == (set(), {})


class TestComputePickSlots:
    def test_full_order_worst_first_champion_last(self):
        # 6 teams: rids 1-4 made the playoffs, 5 and 6 did not.
        # Regular-season rank (1 = best): rid 1 best ... rid 6 worst.
        reg_ranks = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        rids, placements = placements_from_bracket(BRACKET)
        slots = compute_pick_slots(reg_ranks, rids, placements)
        # Non-playoff: rid 6 (worst) slot 1, rid 5 slot 2.
        # Playoff: fourth place (4) slot 3 ... champion (1) slot 6.
        assert slots == {6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6}

    def test_champion_always_picks_last(self):
        reg_ranks = {1: 4, 2: 1, 3: 2, 4: 3}  # champion had the WORST record
        rids, placements = placements_from_bracket(BRACKET)
        slots = compute_pick_slots(reg_ranks, rids, placements)
        assert slots[1] == 4  # playoff finish overrides regular season

    def test_returns_empty_without_placements(self):
        assert compute_pick_slots({1: 1, 2: 2}, {1, 2}, {}) == {}


class TestSlotsFromRegularSeason:
    def test_worst_record_picks_first(self):
        slots = slots_from_regular_season({1: 1, 2: 2, 3: 3})
        assert slots == {1: 3, 2: 2, 3: 1}

    def test_total_teams_override(self):
        # Rank map may be missing teams; the override keeps slots stable.
        slots = slots_from_regular_season({1: 1, 3: 3}, total_teams=4)
        assert slots == {1: 4, 3: 2}

    def test_empty(self):
        assert slots_from_regular_season({}) == {}


class TestPickLabel:
    def test_exact_slot(self):
        assert pick_label(2026, 1, 3) == "2026 1.03"

    def test_mid_round_suffixes(self):
        assert pick_label(2026, 1) == "2026 1st (Mid)"
        assert pick_label(2026, 2) == "2026 2nd (Mid)"
        assert pick_label(2026, 3) == "2026 3rd (Mid)"
        assert pick_label(2026, 4) == "2026 4th (Mid)"

    def test_missing_year_or_round(self):
        assert pick_label(0, 1) == "Pick"
        assert pick_label(2026, 0) == "Pick"


class TestAvgPickValueForRound:
    BY_ID = {"2026_1_01": 100.0, "2026_1_02": 80.0, "2026_2_01": 40.0}

    def test_averages_matching_round(self):
        assert avg_pick_value_for_round(self.BY_ID, 2026, 1) == 90.0

    def test_round_prefix_does_not_cross_rounds(self):
        assert avg_pick_value_for_round(self.BY_ID, 2026, 2) == 40.0

    def test_no_matches(self):
        assert avg_pick_value_for_round(self.BY_ID, 2027, 1) == 0.0
