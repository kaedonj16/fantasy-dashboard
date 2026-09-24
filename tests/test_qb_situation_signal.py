"""Tests for the start/sit backup/third-string QB signal (no Flask, no network)."""
from utils.qb_situation import qb_situation_chip
from utils.waiver_score import build_depth_index


def _players(*qbs):
    """Build a fake Sleeper players map. Each qb is (pid, order, status, name)."""
    return {
        pid: {
            "team": "ATL",
            "position": "QB",
            "depth_chart_order": order,
            "injury_status": status,
            "full_name": name,
        }
        for pid, order, status, name in qbs
    }


def _signal(players, team="ATL"):
    return qb_situation_chip(team, build_depth_index(players), players)


def test_qb1_healthy_no_signal():
    players = _players(
        ("1", 1, "", "Michael Penix"),
        ("2", 2, "", "Kirk Cousins"),
        ("3", 3, "", "Easton Stick"),
    )
    assert _signal(players) is None


def test_qb1_out_gives_backup_qb():
    players = _players(
        ("1", 1, "Out", "Michael Penix"),
        ("2", 2, "", "Kirk Cousins"),
        ("3", 3, "", "Easton Stick"),
    )
    chip = _signal(players)
    assert chip is not None
    assert chip["label"] == "Backup QB"
    assert chip["kind"] == "qb2"
    assert "Kirk Cousins" in chip["note"]


def test_qb1_qb2_out_gives_third_string():
    players = _players(
        ("1", 1, "Out", "Michael Penix"),
        ("2", 2, "Doubtful", "Kirk Cousins"),
        ("3", 3, "", "Easton Stick"),
    )
    chip = _signal(players)
    assert chip is not None
    assert chip["label"] == "3rd-string QB"
    assert chip["kind"] == "qb3"
    assert "Easton Stick" in chip["note"]


def test_questionable_qb1_still_blocks():
    """A questionable QB1 usually plays, so no downgrade signal."""
    players = _players(
        ("1", 1, "Questionable", "Michael Penix"),
        ("2", 2, "", "Kirk Cousins"),
    )
    assert _signal(players) is None


def test_missing_depth_chart_no_signal():
    players = _players(("1", 1, "", "Michael Penix"))
    assert qb_situation_chip("DAL", build_depth_index(players), players) is None


def test_team_case_insensitive():
    players = _players(
        ("1", 1, "Out", "Michael Penix"),
        ("2", 2, "", "Kirk Cousins"),
    )
    assert _signal(players, team="atl")["kind"] == "qb2"


def test_no_healthy_qb_no_signal():
    players = _players(
        ("1", 1, "Out", "Michael Penix"),
        ("2", 2, "IR", "Kirk Cousins"),
    )
    assert _signal(players) is None


def test_bad_depth_order_skipped():
    players = _players(
        ("1", 1, "Out", "Michael Penix"),
        ("2", None, "", "Kirk Cousins"),
        ("3", 3, "", "Easton Stick"),
    )
    chip = _signal(players)
    assert chip is not None
    assert chip["kind"] == "qb3"


def test_empty_inputs_no_signal():
    assert qb_situation_chip(None, {}, {}) is None
    assert qb_situation_chip("ATL", {}, {}) is None
    assert qb_situation_chip("", None, None) is None
