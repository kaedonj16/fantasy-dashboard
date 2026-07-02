"""Unit tests for utils.redzone_stats.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.redzone_stats import (
    rz_def_stat_line,
    rz_num,
    rz_safe_epoch,
    rz_stat_line_from_ps,
)


# ---- rz_num ---------------------------------------------------------------

def test_rz_num_coerces_numbers_and_strings():
    assert rz_num(5) == 5.0
    assert rz_num("12.5") == 12.5


def test_rz_num_bad_input_is_zero():
    assert rz_num(None) == 0.0
    assert rz_num("") == 0.0
    assert rz_num("abc") == 0.0


# ---- rz_safe_epoch --------------------------------------------------------

def test_rz_safe_epoch_valid():
    assert rz_safe_epoch("1700000000") == 1700000000.0
    assert rz_safe_epoch(1700000000.5) == 1700000000.5


def test_rz_safe_epoch_empty_and_bad():
    assert rz_safe_epoch(None) == 0.0
    assert rz_safe_epoch("") == 0.0
    assert rz_safe_epoch("not-a-number") == 0.0


# ---- rz_stat_line_from_ps -------------------------------------------------

def test_stat_line_maps_all_groups():
    ps = {
        "Passing": {"passYds": "250", "passTD": 2, "int": 1},
        "Rushing": {"carries": 10, "rushYds": 40, "rushTD": 1},
        "Receiving": {"receptions": 5, "recYds": 60, "recTD": 0, "targets": 8},
        "Kicking": {"fgMade": 3, "fgLong": 52, "xpMade": 2},
    }
    line = rz_stat_line_from_ps(ps)
    assert line["pass_yds"] == 250.0
    assert line["pass_td"] == 2.0
    assert line["int"] == 1.0
    assert line["carries"] == 10.0
    assert line["rec"] == 5.0
    assert line["targets"] == 8.0
    assert line["fgm"] == 3.0
    assert line["fg_long"] == 52.0
    assert line["xpm"] == 2.0


def test_stat_line_missing_groups_default_zero():
    line = rz_stat_line_from_ps({})
    assert set(line) >= {"pass_yds", "rush_yds", "rec", "fgm"}
    assert all(v == 0.0 for v in line.values())


def test_stat_line_none_input_does_not_raise():
    line = rz_stat_line_from_ps(None)
    assert line["pass_yds"] == 0.0


def test_stat_line_kicker_field_fallbacks():
    # fgm accepts either fgm or fgMade; fg_long accepts several spellings.
    a = rz_stat_line_from_ps({"Kicking": {"fgm": 4, "fgLng": 55, "xpm": 1}})
    assert a["fgm"] == 4.0
    assert a["fg_long"] == 55.0
    assert a["xpm"] == 1.0


# ---- rz_def_stat_line -----------------------------------------------------

def test_def_stat_line_maps_fields():
    side = {"Defense": {"totalSacks": 3, "interceptions": 2,
                        "fumblesRecovered": 1, "defTD": 1}}
    line = rz_def_stat_line(side)
    assert line == {"sacks": 3.0, "def_int": 2.0, "fum_rec": 1.0, "def_td": 1.0}


def test_def_stat_line_lowercase_key_and_missing():
    assert rz_def_stat_line({"defense": {"sacks": 2}})["sacks"] == 2.0
    empty = rz_def_stat_line({})
    assert empty == {"sacks": 0.0, "def_int": 0.0, "fum_rec": 0.0, "def_td": 0.0}


def test_def_stat_line_none_input():
    assert rz_def_stat_line(None)["sacks"] == 0.0
