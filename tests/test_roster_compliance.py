"""Tests for utils.roster_compliance."""
from utils.roster_compliance import (
    IR_SLOT_ELIGIBLE,
    effective_taxi_slots,
    roster_compliance_issues,
)

INFO = {
    "1": {"name": "Healthy Starter", "injury_status": "", "years_exp": 4},
    "2": {"name": "IR Guy", "injury_status": "IR", "years_exp": 2},
    "3": {"name": "Recovered Guy", "injury_status": "", "years_exp": 5},
    "4": {"name": "Rookie Bench", "injury_status": "", "years_exp": 0},
    "5": {"name": "Vet Bench", "injury_status": "", "years_exp": 7},
    "6": {"name": "PUP Guy", "injury_status": "PUP", "years_exp": 3},
    "7": {"name": "Questionable Guy", "injury_status": "Questionable", "years_exp": 2},
}


def _issues(**kw):
    defaults = dict(
        players=[], starters=[], reserve=[], taxi=[],
        player_info=INFO, reserve_slots=0, taxi_slots=0,
    )
    defaults.update(kw)
    return roster_compliance_issues(**defaults)


class TestEffectiveTaxiSlots:
    def test_zero_when_unset(self):
        assert effective_taxi_slots({}) == 0
        assert effective_taxi_slots({"taxi_slots": 0}) == 0

    def test_dynasty_keeps_slots(self):
        assert effective_taxi_slots({"type": 2, "taxi_slots": 4}) == 4
        assert effective_taxi_slots(
            {"league_type": "dynasty", "taxi_slots": 3},
        ) == 3

    def test_keeper_suppresses_leftover_slots(self):
        # Sleeper type=1 keeper can still carry dynasty taxi_slots leftovers.
        assert effective_taxi_slots({"type": 1, "taxi_slots": 5}) == 0
        assert effective_taxi_slots(
            {"league_type": "keeper", "taxi_slots": 2},
        ) == 0

    def test_redraft_suppresses_leftover_slots(self):
        assert effective_taxi_slots({"type": 0, "taxi_slots": 3}) == 0

    def test_reads_settings_from_league_when_omitted(self):
        league = {"settings": {"type": 1, "taxi_slots": 4}}
        assert effective_taxi_slots(league=league) == 0
        league_dyn = {"settings": {"type": 2, "taxi_slots": 4}}
        assert effective_taxi_slots(league=league_dyn) == 4


class TestIrStash:
    def test_ir_player_on_active_roster_with_open_slot(self):
        issues = _issues(players=["1", "2"], starters=["1"], reserve_slots=1)
        assert len(issues) == 1
        assert issues[0]["kind"] == "ir_stash"
        assert "IR Guy" in issues[0]["detail"]

    def test_no_flag_when_ir_slots_full(self):
        issues = _issues(players=["1", "2", "3"], starters=["1"],
                         reserve=["3"], reserve_slots=1)
        assert all(i["kind"] != "ir_stash" for i in issues)

    def test_no_flag_when_league_has_no_ir_slots(self):
        assert _issues(players=["1", "2"], starters=["1"], reserve_slots=0) == []

    def test_questionable_not_ir_eligible(self):
        issues = _issues(players=["1", "7"], starters=["1"], reserve_slots=1)
        assert issues == []
        assert "Questionable" not in IR_SLOT_ELIGIBLE

    def test_capped_by_free_slots(self):
        issues = _issues(players=["2", "6"], starters=[], reserve_slots=1)
        stash = [i for i in issues if i["kind"] == "ir_stash"]
        assert len(stash) == 1


class TestIrActivate:
    def test_recovered_player_in_ir_slot_flagged(self):
        issues = _issues(players=["1", "3"], starters=["1"],
                         reserve=["3"], reserve_slots=1)
        assert len(issues) == 1
        assert issues[0]["kind"] == "ir_activate"
        assert "Recovered Guy" in issues[0]["detail"]

    def test_still_injured_reserve_not_flagged(self):
        issues = _issues(players=["1", "2"], starters=["1"],
                         reserve=["2"], reserve_slots=1)
        assert issues == []

    def test_unknown_player_in_reserve_skipped(self):
        issues = _issues(players=["1", "999"], starters=["1"],
                         reserve=["999"], reserve_slots=1)
        assert issues == []


class TestTaxiStash:
    def test_rookie_on_bench_with_open_taxi_slot(self):
        issues = _issues(players=["1", "4"], starters=["1"], taxi_slots=2)
        assert len(issues) == 1
        assert issues[0]["kind"] == "taxi_stash"
        assert "Rookie Bench" in issues[0]["detail"]

    def test_starting_rookie_not_flagged(self):
        issues = _issues(players=["4"], starters=["4"], taxi_slots=2)
        assert issues == []

    def test_veteran_bench_not_flagged(self):
        issues = _issues(players=["1", "5"], starters=["1"], taxi_slots=2)
        assert issues == []

    def test_no_flag_when_taxi_full(self):
        issues = _issues(players=["1", "4"], starters=["1"],
                         taxi=["8"], taxi_slots=1)
        assert issues == []

    def test_taxi_players_not_treated_as_active(self):
        # An IR-designated player already on taxi shouldn't trigger ir_stash.
        issues = _issues(players=["1", "2"], starters=["1"],
                         taxi=["2"], reserve_slots=1, taxi_slots=1)
        assert all(i["kind"] != "ir_stash" for i in issues)

    def test_keeper_leftover_taxi_slots_do_not_surface_stash(self):
        # Call-site contract: effective_taxi_slots must zero keeper leftovers
        # before roster_compliance_issues runs, so no taxi tip fires.
        slots = effective_taxi_slots({"type": 1, "taxi_slots": 4})
        issues = _issues(players=["1", "4"], starters=["1"], taxi_slots=slots)
        assert issues == []
        assert all(i["kind"] != "taxi_stash" for i in issues)


def test_clean_roster_no_issues():
    assert _issues(players=["1", "5"], starters=["1"],
                   reserve_slots=2, taxi_slots=2) == []
