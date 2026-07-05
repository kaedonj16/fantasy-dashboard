"""Tests for utils.lineup_issues (lineup problem detection and summaries)."""
import pytest

from utils.lineup_issues import (
    SERIOUS_INJURY_STATUSES,
    find_lineup_issues,
    summarize_issues,
)


PLAYING = {"KC", "BUF", "CIN", "JAX"}

INFO = {
    "1": {"name": "Patrick Mahomes", "team": "KC", "injury_status": ""},
    "2": {"name": "Ja'Marr Chase", "team": "CIN", "injury_status": ""},
    "3": {"name": "Travis Etienne", "team": "JAX", "injury_status": "Out"},
    "4": {"name": "Bye Guy", "team": "DAL", "injury_status": ""},
    "5": {"name": "Hurt Bye Guy", "team": "DAL", "injury_status": "IR"},
}


def test_clean_lineup_has_no_issues():
    assert find_lineup_issues(["1", "2"], INFO, PLAYING) == []


def test_empty_slot_flagged():
    issues = find_lineup_issues(["1", "0"], INFO, PLAYING)
    assert len(issues) == 1
    assert issues[0]["kind"] == "empty"


def test_serious_injury_flagged():
    issues = find_lineup_issues(["3"], INFO, PLAYING)
    assert len(issues) == 1
    assert issues[0]["kind"] == "injury"
    assert "Out" in issues[0]["detail"]
    assert "Travis Etienne" in issues[0]["detail"]


def test_questionable_not_flagged():
    info = {"9": {"name": "Q Guy", "team": "KC", "injury_status": "Questionable"}}
    assert find_lineup_issues(["9"], info, PLAYING) == []
    assert "Questionable" not in SERIOUS_INJURY_STATUSES


def test_bye_flagged_when_schedule_known():
    issues = find_lineup_issues(["4"], INFO, PLAYING)
    assert len(issues) == 1
    assert issues[0]["kind"] == "bye"
    assert "on bye" in issues[0]["detail"]


def test_bye_skipped_without_schedule():
    # No schedule data must not flag every starter as on bye.
    assert find_lineup_issues(["4"], INFO, None) == []
    assert find_lineup_issues(["4"], INFO, set()) == []


def test_injury_supersedes_bye():
    issues = find_lineup_issues(["5"], INFO, PLAYING)
    assert len(issues) == 1
    assert issues[0]["kind"] == "injury"


def test_ordering_empty_then_injury_then_bye():
    issues = find_lineup_issues(["4", "3", "0"], INFO, PLAYING)
    assert [i["kind"] for i in issues] == ["empty", "injury", "bye"]


def test_unknown_player_not_flagged():
    # No data means no verdict; a missing player must not create noise.
    assert find_lineup_issues(["999"], INFO, PLAYING) == []


def test_summary_single_issue():
    issues = find_lineup_issues(["3"], INFO, PLAYING)
    assert summarize_issues(issues) == "Travis Etienne is listed Out"


def test_summary_combines_with_and():
    issues = find_lineup_issues(["0", "3"], INFO, PLAYING)
    s = summarize_issues(issues)
    assert s == "1 empty starting slot and Travis Etienne is listed Out"


def test_summary_pluralizes_empty_slots():
    issues = find_lineup_issues(["0", "0"], INFO, PLAYING)
    assert summarize_issues(issues) == "2 empty starting slots"


def test_summary_overflow_capped():
    info = {
        str(i): {"name": f"Player {i}", "team": "DAL", "injury_status": "Out"}
        for i in range(1, 7)
    }
    issues = find_lineup_issues([str(i) for i in range(1, 7)], info, PLAYING)
    s = summarize_issues(issues, max_names=3)
    assert "3 more issues" in s


def test_summary_empty_for_no_issues():
    assert summarize_issues([]) == ""


def test_no_em_dashes_or_emojis_in_output():
    issues = find_lineup_issues(["0", "3", "4"], INFO, PLAYING)
    text = summarize_issues(issues) + "".join(i["detail"] for i in issues)
    assert "—" not in text  # em dash
    assert all(ord(c) < 0x2600 for c in text)  # no emoji blocks


# ---- projection_upgrades ----------------------------------------------------

from utils.lineup_issues import projection_upgrades

SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
POS = {
    "q1": "QB", "r1": "RB", "r2": "RB", "r3": "RB",
    "w1": "WR", "w2": "WR", "w3": "WR", "t1": "TE", "t2": "TE",
}


def test_upgrade_suggested_for_outprojecting_bench_player():
    starters = ["q1", "r1", "r2", "w1", "w2", "t1", "r3"]
    proj = {"q1": 18, "r1": 12, "r2": 10, "r3": 9, "w1": 11, "w2": 4, "w3": 13, "t1": 7}
    swaps = projection_upgrades(starters, list(POS), proj, POS, SLOTS)
    assert swaps
    assert swaps[0]["in"] == "w3" and swaps[0]["out"] == "w2"
    assert swaps[0]["gain"] == 9.0


def test_no_upgrade_when_lineup_already_optimal():
    starters = ["q1", "r1", "r2", "w1", "w3", "t1", "r3"]
    proj = {"q1": 18, "r1": 12, "r2": 10, "r3": 9, "w1": 11, "w2": 4, "w3": 13, "t1": 7}
    assert projection_upgrades(starters, list(POS), proj, POS, SLOTS) == []


def test_small_gain_below_threshold_ignored():
    starters = ["q1", "r1", "r2", "w1", "w2", "t1", "r3"]
    proj = {"q1": 18, "r1": 12, "r2": 10, "r3": 9, "w1": 11, "w2": 12.0, "w3": 13.0, "t1": 7}
    assert projection_upgrades(starters, list(POS), proj, POS, SLOTS, min_gain=2.0) == []


def test_cross_position_flex_swaps_not_suggested():
    # Optimizer wants a second TE in the FLEX over an RB; like-for-like only,
    # so no suggestion should be produced for a TE-for-RB swap.
    starters = ["q1", "r1", "r2", "w1", "w2", "t1", "r3"]
    proj = {"q1": 18, "r1": 12, "r2": 10, "r3": 2, "w1": 11, "w2": 9, "w3": 1, "t1": 7, "t2": 15}
    swaps = projection_upgrades(starters, list(POS), proj, POS, SLOTS)
    assert all(POS[s["in"]] == POS[s["out"]] for s in swaps)


def test_empty_inputs_return_no_swaps():
    assert projection_upgrades([], list(POS), {"q1": 10}, POS, SLOTS) == []
    assert projection_upgrades(["q1"], list(POS), {}, POS, SLOTS) == []
    assert projection_upgrades(["q1"], list(POS), {"q1": 10}, POS, []) == []


def test_max_swaps_cap():
    starters = ["q1", "r1", "r2", "w1", "w2", "t1", "t2"]
    proj = {"q1": 1, "r1": 1, "r2": 1, "r3": 20, "w1": 1, "w2": 1, "w3": 20, "t1": 1, "t2": 1}
    swaps = projection_upgrades(starters, list(POS), proj, POS, SLOTS, max_swaps=1)
    assert len(swaps) == 1
