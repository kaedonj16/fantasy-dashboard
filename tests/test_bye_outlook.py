"""Tests for the forward-looking bye-week planner (utils/bye_outlook)."""
from utils.bye_outlook import build_bye_outlook, summarize_bye_outlook


# TEAM -> bye week
BYES = {
    "KC": 6, "BUF": 7, "MIA": 7, "PHI": 7, "SF": 9, "DAL": 7, "GB": 10,
}


def _p(pos, team):
    return {"pos": pos, "team": team}


def test_empty_when_no_schedule_data():
    assert build_bye_outlook({}, [_p("WR", "MIA")], {"WR": 2}) == []


def test_groups_byes_by_week_and_position():
    roster = [_p("WR", "MIA"), _p("WR", "BUF"), _p("WR", "PHI"), _p("RB", "KC")]
    out = build_bye_outlook(BYES, roster, {"WR": 2, "RB": 2})
    by_week = {w["week"]: w for w in out}
    assert by_week[7]["by_pos"] == {"WR": 3}
    assert by_week[7]["total"] == 3
    assert by_week[6]["by_pos"] == {"RB": 1}
    # sorted ascending by week
    assert [w["week"] for w in out] == [6, 7]


def test_tight_and_crunch_flags():
    # 3 WRs on bye in week 7, league starts 2 WRs -> tight/crunch.
    roster = [_p("WR", "MIA"), _p("WR", "BUF"), _p("WR", "PHI"), _p("RB", "KC")]
    out = build_bye_outlook(BYES, roster, {"WR": 2, "RB": 2})
    wk7 = next(w for w in out if w["week"] == 7)
    assert wk7["tight"] == ["WR"]
    assert wk7["crunch"] is True
    # single RB on bye in week 6 with 2 RB slots -> not tight.
    wk6 = next(w for w in out if w["week"] == 6)
    assert wk6["tight"] == []
    assert wk6["crunch"] is False


def test_from_week_skips_played_byes():
    roster = [_p("RB", "KC"), _p("WR", "BUF")]  # KC bye 6, BUF bye 7
    out = build_bye_outlook(BYES, roster, {"RB": 2, "WR": 2}, from_week=7)
    assert [w["week"] for w in out] == [7]


def test_kdef_only_tracked_when_started():
    roster = [_p("DEF", "MIA"), _p("K", "BUF")]
    # No K/DEF slots -> ignored entirely.
    assert build_bye_outlook(BYES, roster, {"WR": 2}) == []
    # With DEF slot -> DEF bye is tracked.
    out = build_bye_outlook(BYES, roster, {"WR": 2, "DEF": 1})
    assert any(w["by_pos"].get("DEF") == 1 for w in out)


def test_position_key_aliases_and_bad_rows():
    roster = [
        {"position": "WR", "nfl": "MIA"},   # alias keys
        {"pos": "WR", "team": ""},          # missing team -> skipped
        "not-a-dict",                       # skipped
        {"pos": "WR"},                       # no team -> skipped
    ]
    out = build_bye_outlook(BYES, roster, {"WR": 2})
    assert len(out) == 1
    assert out[0]["by_pos"] == {"WR": 1}


def test_summarize_prefers_crunches():
    roster = [_p("WR", "MIA"), _p("WR", "BUF"), _p("WR", "PHI"),  # wk7 crunch
              _p("RB", "KC")]                                     # wk6 single
    out = build_bye_outlook(BYES, roster, {"WR": 2, "RB": 2})
    summary = summarize_bye_outlook(out)
    assert "Week 7: 3 WRs on bye" in summary


def test_summarize_empty():
    assert summarize_bye_outlook([]) == ""
