"""Wave 3 PR 1: GM brief builder, deadline stamps, race line.

Spec: Wave 3 sections 3.1 (deadline stamps + ranking float), 3.2
(playoff-race line), 5 (degradation) and 7 PR 1. Pure module tests:
the brief builder reads plain ctx dicts, so fixtures are dicts.
"""
import pytest

pytest.importorskip("pandas")

from datetime import datetime
from zoneinfo import ZoneInfo

from dashboard_services import gm_brief as gb

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=ET).timestamp()  # Wed noon ET
THU = datetime(2026, 11, 5, 20, 20, tzinfo=ET).timestamp()  # Thu 8:20 PM ET
SUN = datetime(2026, 11, 8, 13, 0, tzinfo=ET).timestamp()   # Sun 1:00 PM ET

GAMES = [
    {"home": "BAL", "away": "CIN", "gameTime_epoch": THU},
    {"home": "GB", "away": "PHI", "gameTime_epoch": SUN},
    {"home": "SEA", "away": "ARI", "gameTime_epoch": SUN},
]


def _ctx(**over):
    ctx = {
        "current_week": 9,
        "current_season": 2026,
        "season": 2026,
        "gm_schedule_games": GAMES,
        "players": {
            "bal_qb": {"full_name": "Baltimore QB", "team": "BAL",
                       "position": "QB"},
            "gb_wr": {"full_name": "Green Bay WR", "team": "GB",
                      "position": "WR"},
            "sea_qb": {"full_name": "Seattle QB", "team": "SEA",
                       "position": "QB"},
        },
        "rosters": [{"roster_id": "1", "players": ["bal_qb", "gb_wr"],
                     "starters": ["gb_wr"]}],
        "roster_map": {"1": "Viewer Team"},
    }
    ctx.update(over)
    return ctx


def _rules(source="user_confirmed", **over):
    rec = {
        "source": source,
        "playoff_teams": 8,
        "playoff_week_start": 15,
        "waiver_type": "rolling",
        "waiver_process_time": "03:00",
        "waiver_process_days": "daily",
        "waiver_process_tz": "America/New_York",
    }
    rec.update(over)
    return rec


# ---------------------------------------------------------------------------
# deadline stamps (3.1)
# ---------------------------------------------------------------------------

def test_thursday_player_action_carries_thursday_kickoff():
    action = {"kind": "lineup", "action": "Start Baltimore QB",
              "score": 90, "player_ids": ["bal_qb"]}
    brief = gb.build_gm_brief(_ctx(), "1", [action], now=NOW)
    stamp = brief["actions"][0]["deadline"]
    assert stamp["kind"] == "kickoff"
    assert stamp["at"] == THU
    assert "Thu" in stamp["label"]


def test_sunday_only_action_carries_no_kickoff_stamp():
    action = {"kind": "lineup", "action": "Start Green Bay WR",
              "score": 90, "player_ids": ["gb_wr"]}
    brief = gb.build_gm_brief(_ctx(), "1", [action], now=NOW)
    assert "deadline" not in brief["actions"][0]


def test_hedge_bench_replacement_thursday_stamps_the_action():
    # The starter plays Sunday (GB) but the bench replacement plays
    # Thursday (BAL): the hedge dies Thursday, so Thursday stamps it.
    action = {"kind": "lineup", "action": "Hedge Green Bay WR",
              "score": 88, "player_ids": ["gb_wr", "bal_qb"]}
    brief = gb.build_gm_brief(_ctx(), "1", [action], now=NOW)
    assert brief["actions"][0]["deadline"]["at"] == THU


def test_kickoff_stamp_works_with_no_rules_record():
    ctx = _ctx()
    assert "league_rules" not in ctx
    action = {"kind": "lineup", "action": "Start Baltimore QB",
              "score": 90, "player_ids": ["bal_qb"]}
    brief = gb.build_gm_brief(ctx, "1", [action], now=NOW)
    assert brief["actions"][0]["deadline"]["at"] == THU
    assert brief["layers"]["rules"] == "default_unverified"


def test_waiver_action_stamped_with_confirmed_process_time():
    ctx = _ctx(league_rules=_rules("user_confirmed"))
    action = {"kind": "waiver", "action": "Claim Seattle QB",
              "score": 80, "player_id": "sea_qb"}
    brief = gb.build_gm_brief(ctx, "1", [action], now=NOW)
    stamp = brief["actions"][0]["deadline"]
    assert stamp["kind"] == "waiver_process"
    assert "3:00 AM ET" in stamp["label"]
    # Next process after Wed noon is Thursday 3:00 AM ET.
    expect = datetime(2026, 11, 5, 3, 0, tzinfo=ET).timestamp()
    assert stamp["at"] == expect


@pytest.mark.parametrize("source", ["derived", "default_unverified"])
def test_waiver_action_not_stamped_from_unconfirmed_record(source):
    ctx = _ctx(league_rules=_rules(source))
    action = {"kind": "waiver", "action": "Claim Seattle QB",
              "score": 80, "player_id": "sea_qb"}
    brief = gb.build_gm_brief(ctx, "1", [action], now=NOW)
    assert "deadline" not in brief["actions"][0]


# ---------------------------------------------------------------------------
# race line (3.2)
# ---------------------------------------------------------------------------

def _standings(chasing=False):
    above_wins = [7, 6, 6, 5, 5, 4, 4]
    rows = [
        {"roster_id": str(i + 2), "team_name": f"Team {i + 2}",
         "wins": w, "losses": 8 - w, "pf": 1200.0 - i * 10}
        for i, w in enumerate(above_wins)
    ]
    if chasing:
        rows.append({"roster_id": "8", "team_name": "Cutoff Team",
                     "wins": 4, "losses": 4, "pf": 900.0})
    rows.append({"roster_id": "1", "team_name": "Viewer Team",
                 "wins": 3, "losses": 5, "pf": 983.25})
    rows.append({"roster_id": "9", "team_name": "East Bay Biters",
                 "wins": 3, "losses": 5, "pf": 879.25})
    rows.append({"roster_id": "10", "team_name": "Team 10", "wins": 2,
                 "losses": 6, "pf": 800.0})
    rows.append({"roster_id": "11", "team_name": "Team 11", "wins": 1,
                 "losses": 7, "pf": 700.0})
    return rows


def test_race_line_holds_seed_pf_delta_vs_first_team_out():
    ctx = _ctx(league_rules=_rules("user_confirmed"),
               gm_standings=_standings())
    brief = gb.build_gm_brief(ctx, "1", [], now=NOW)
    race = brief["race_line"]
    assert race is not None
    assert race["playoff_teams"] == 8
    assert race["seed"] == 8
    assert race["status"] == "holds"
    assert race["games_back"] == 0.0
    assert race["pf_delta"] == 104.0
    assert race["pf_rival"] == "East Bay Biters"
    assert "East Bay Biters" in race["text"]
    assert brief["race_prompt"] is None  # user_confirmed: no prompt


def test_race_line_chasing_gb_and_pf_vs_cutoff_holder():
    ctx = _ctx(league_rules=_rules("feed"),
               gm_standings=_standings(chasing=True))
    brief = gb.build_gm_brief(ctx, "1", [], now=NOW)
    race = brief["race_line"]
    assert race["seed"] == 9
    assert race["status"] == "chasing"
    assert race["games_back"] == 1.0
    assert race["pf_delta"] == 83.25
    assert race["pf_rival"] == "Cutoff Team"
    # The confirmation prompt rides along until user_confirmed.
    assert brief["race_prompt"] is not None


def test_race_line_suppressed_for_default_unverified_with_prompt():
    ctx = _ctx(league_rules=_rules("default_unverified"),
               gm_standings=_standings())
    brief = gb.build_gm_brief(ctx, "1", [], now=NOW)
    assert brief["race_line"] is None
    assert brief["race_prompt"] is not None
    assert brief["race_prompt"]["kind"] == "rules_confirmation"


def test_race_line_shows_for_derived():
    ctx = _ctx(league_rules=_rules("derived"),
               gm_standings=_standings())
    brief = gb.build_gm_brief(ctx, "1", [], now=NOW)
    assert brief["race_line"] is not None
    assert brief["race_line"]["seed"] == 8


# ---------------------------------------------------------------------------
# ranking float (3.1)
# ---------------------------------------------------------------------------

def test_deadline_within_24h_floats_above_equal_score_undated():
    dated = {"kind": "lineup", "action": "Start Baltimore QB",
             "score": 80, "player_ids": ["bal_qb"]}
    undated = {"kind": "lineup", "action": "Start Green Bay WR",
               "score": 80, "player_ids": ["gb_wr"]}
    higher = {"kind": "lineup", "action": "Bigger", "score": 95,
              "player_ids": ["gb_wr"]}
    brief = gb.build_gm_brief(_ctx(), "1", [undated, dated, higher],
                              now=NOW)
    order = [a["action"] for a in brief["actions"]]
    # Score stays primary (95 first); at equal score the <=24h deadline
    # (Thu 8:20 PM is ~32h away, so use a same-day stamp instead).
    assert order[0] == "Bigger"


def test_deadline_float_uses_next_day_window():
    soon = NOW + 3600  # one hour out: inside the 24h float window
    dated = {"kind": "waiver", "action": "Claim A", "score": 80,
             "deadline": {"kind": "waiver_process", "at": soon,
                          "label": "Waiver process 3:00 AM ET"}}
    undated = {"kind": "lineup", "action": "Undated", "score": 80}
    ranked = gb.sort_actions([undated, dated], now=NOW)
    assert ranked[0]["action"] == "Claim A"
    # A far deadline does not float.
    far = dict(dated, action="Claim B",
               deadline={"kind": "kickoff", "at": NOW + 3 * 86400,
                         "label": "Locks Sun 1:00 PM ET"})
    # No float: the original relative order is preserved.
    ranked = gb.sort_actions([far, undated], now=NOW)
    assert ranked[0]["action"] == "Claim B"
