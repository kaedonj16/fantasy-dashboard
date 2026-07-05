"""Tests for utils.pick_slots (rookie draft order and pick labels)."""
from utils.pick_slots import (
    avg_pick_value_for_round,
    compute_pick_slots,
    is_pick_asset_id,
    parse_pick_asset,
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


class TestIsPickAssetId:
    def test_slot_and_bucket_forms(self):
        assert is_pick_asset_id("2026_1_01")
        assert is_pick_asset_id("2026_1_early")
        assert is_pick_asset_id("2026_2")

    def test_player_ids_rejected(self):
        assert not is_pick_asset_id("4046")       # bare Sleeper id
        assert not is_pick_asset_id("")
        assert not is_pick_asset_id(None)

    def test_malformed_rejected(self):
        assert not is_pick_asset_id("26_1_01")     # 2-digit year
        assert not is_pick_asset_id("year_1_01")   # non-numeric year
        assert not is_pick_asset_id("2026_x_01")   # non-numeric round


class TestParsePickAsset:
    def test_exact_slot(self):
        p = parse_pick_asset("2026_1_03")
        assert p["season"] == 2026 and p["round"] == 1
        assert p["slot"] == 3 and p["slot_raw"] == "03"
        assert p["bucket"] is None
        assert p["name"] == "2026 1.03"

    def test_bucket(self):
        p = parse_pick_asset("2027_2_early")
        assert p["bucket"] == "Early"
        assert p["slot"] is None and p["slot_raw"] == "early"
        assert p["name"] == "2027 2nd (Early)"

    def test_round_only(self):
        p = parse_pick_asset("2026_3")
        assert p["slot_raw"] == ""
        assert p["name"] == "2026 3rd"

    def test_fourth_round_suffix(self):
        assert parse_pick_asset("2026_4_mid")["name"] == "2026 4th (Mid)"

    def test_invalid_returns_none(self):
        assert parse_pick_asset("4046") is None
        assert parse_pick_asset(None) is None


# ---- bucket_for_slot / pick_value_from_table --------------------------------

from utils.pick_slots import bucket_for_slot, pick_value_from_table


class TestBucketForSlot:
    def test_twelve_team_thirds(self):
        assert bucket_for_slot(1, 12) == "early"
        assert bucket_for_slot(4, 12) == "early"
        assert bucket_for_slot(5, 12) == "mid"
        assert bucket_for_slot(8, 12) == "mid"
        assert bucket_for_slot(9, 12) == "late"
        assert bucket_for_slot(12, 12) == "late"

    def test_degenerate_league_size(self):
        assert bucket_for_slot(1, 0) == "mid"


class TestPickValueFromTable:
    TBL = {
        "2027_1_01": 900.0,
        "2027_1_02": 800.0,
        "2027_1_early": 850.0,
        "2027_1_mid": 500.0,
        "2027_2": 150.0,
    }

    def test_exact_slot_wins(self):
        assert pick_value_from_table(self.TBL, 2027, 1, slot=1, num_teams=12) == 900.0

    def test_bucket_fallback_when_slot_missing(self):
        # Slot 3 has no key; early bucket (slot 3 of 12) should be used.
        assert pick_value_from_table(self.TBL, 2027, 1, slot=3, num_teams=12) == 850.0

    def test_slot_average_when_no_slot_given(self):
        assert pick_value_from_table(self.TBL, 2027, 1) == 850.0  # avg(900, 800)

    def test_round_key_fallback(self):
        assert pick_value_from_table(self.TBL, 2027, 2) == 150.0

    def test_mid_bucket_before_round_key(self):
        tbl = {"2027_3_mid": 60.0, "2027_3": 40.0}
        assert pick_value_from_table(tbl, 2027, 3) == 60.0

    def test_unknown_pick_is_zero(self):
        assert pick_value_from_table(self.TBL, 2029, 1) == 0.0
        assert pick_value_from_table({}, 2027, 1) == 0.0

    def test_non_positive_values_skipped(self):
        tbl = {"2027_1_01": 0, "2027_1_mid": -5, "2027_1": 33.0}
        assert pick_value_from_table(tbl, 2027, 1, slot=1, num_teams=12) == 33.0
