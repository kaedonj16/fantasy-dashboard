"""Wave 3 acceptance test: league 92916, Week 9 2026 (spec section 6).

Frozen fixture ctx for Tua in the Pink as of 2026-11-04: rosters and
standings as of that date, the rules record confirmed (8-team field,
rolling waivers, 3:00 AM ET process), and role states Lock promoted,
Brissett promoted, Fagnano starter, Milton promoted.

Scoring (spec section 6): elements 1, 2, 4, 6 are pass/fail on
substance; 3, 5, 7, 8 pass on presence + correct facts. An element
the brief cannot source from the fixture must be absent, not
approximated.

Cross-stack note: the "Dropped today" chip data and the starting-role
waiver override live on the #3009 / #3012 stacks, which are not in
this branch's ancestry. The fixture therefore carries Brissett's
dropped-today flag and both claims' role reasons as the collector
inputs those stacks will supply (the upgraded waiver collector in
this PR passes them through when present), and the brief degrades
honestly when they are absent: test_dropped_today_chip_absent_
without_cross_stack_data pins that degradation.
"""
import json

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from datetime import datetime
from zoneinfo import ZoneInfo

import dashboard_services.ai.context_builders as cb
from dashboard_services import gm_brief as gb

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=ET).timestamp()
THU = datetime(2026, 11, 5, 20, 20, tzinfo=ET).timestamp()
SUN = datetime(2026, 11, 8, 13, 0, tzinfo=ET).timestamp()

VIEWER = "1"


def _p(team, pos, name, bye=None, status=""):
    row = {"full_name": name, "team": team, "position": pos,
           "injury_status": status}
    if bye:
        row["byeWeek"] = bye
    return row


def _fixture_ctx():
    players = {
        # Viewer: Tua in the Pink.
        "fagnano": _p("BAL", "QB", "Joe Fagnano", bye=13),
        "milton": _p("PHI", "QB", "Joe Milton", bye=10),
        "mccarthy": _p("MIN", "QB", "J.J. McCarthy", bye=13),
        "cmc": _p("SF", "RB", "Christian McCaffrey", status="Questionable"),
        "kyren": _p("LAR", "RB", "Kyren Williams", bye=11),
        "dobbins": _p("DEN", "RB", "J.K. Dobbins", bye=10),
        "harvey": _p("DEN", "RB", "RJ Harvey", bye=10, status="IR"),
        "bigsby": _p("PHI", "RB", "Tank Bigsby", bye=10),
        "jsn": _p("SEA", "WR", "Jaxon Smith-Njigba", bye=11),
        "higgins": _p("CIN", "WR", "Tee Higgins"),
        "waddle": _p("MIA", "WR", "Jaylen Waddle", bye=10),
        "watson": _p("GB", "WR", "Christian Watson", status="Questionable"),
        "flowers": _p("BAL", "WR", "Zay Flowers"),
        "moore": _p("CHI", "WR", "DJ Moore"),
        "laporta": _p("DET", "TE", "Sam LaPorta"),
        "sadiq": _p("PHI", "TE", "Kenyon Sadiq"),
        "folk": _p("TEN", "K", "Nick Folk", bye=11),
        "little": _p("JAX", "K", "Cam Little"),
        "sea_dst": _p("SEA", "DEF", "Seattle Seahawks", bye=11),
        # Claim targets and the starters ahead of them.
        "darnold": _p("SEA", "QB", "Sam Darnold", status="OUT"),
        "lock": _p("SEA", "QB", "Drew Lock"),
        "lamar": _p("BAL", "QB", "Lamar Jackson", status="IR"),
        "brissett": _p("BAL", "QB", "Jacoby Brissett", bye=13),
        # Trade targets.
        "caleb": _p("CHI", "QB", "Caleb Williams"),
        "love": _p("GB", "QB", "Jordan Love"),
        # Opponents.
        "nix": _p("DEN", "QB", "Bo Nix", bye=10),
        "eb_rb": _p("HOU", "RB", "East Bay Runner"),
        "eb_wr": _p("HOU", "WR", "East Bay Receiver"),
        "jacobs": _p("GB", "RB", "Josh Jacobs", bye=11),
        "judkins": _p("CLE", "RB", "Quinshon Judkins", bye=11),
        "stevenson": _p("NE", "RB", "Rhamondre Stevenson", bye=11),
        "myers": _p("SEA", "K", "Jason Myers", bye=11),
        "taco_qb": _p("NYJ", "QB", "Taco QB"),
        "gf_rb": _p("DAL", "RB", "Goodfellas Runner"),
        "gf_wr": _p("DAL", "WR", "Goodfellas Receiver"),
        "ch_rb": _p("KC", "RB", "Cheeseheads Runner"),
        "ch_wr": _p("KC", "WR", "Cheeseheads Receiver"),
    }
    viewer_players = [
        "fagnano", "milton", "mccarthy", "cmc", "kyren", "dobbins",
        "harvey", "bigsby", "jsn", "higgins", "waddle", "watson",
        "flowers", "moore", "laporta", "sadiq", "folk", "little",
        "sea_dst",
    ]
    viewer_starters = [
        "fagnano", "mccarthy", "cmc", "kyren", "dobbins", "jsn",
        "higgins", "waddle", "watson", "laporta", "folk", "sea_dst",
    ]
    rosters = [
        {"roster_id": "1", "players": viewer_players,
         "starters": viewer_starters, "reserve": [],
         "settings": {"waiver_position": 5, "wins": 3, "losses": 5}},
        {"roster_id": "2", "players": ["caleb", "gf_rb", "gf_wr"],
         "starters": ["caleb", "gf_rb", "gf_wr"]},
        {"roster_id": "3", "players": ["love", "ch_rb", "ch_wr"],
         "starters": ["love", "ch_rb", "ch_wr"]},
        {"roster_id": "9", "players": ["nix", "eb_rb", "eb_wr"],
         "starters": ["nix", "eb_rb", "eb_wr"]},
        {"roster_id": "10",
         "players": ["jacobs", "judkins", "stevenson", "myers",
                     "taco_qb"],
         "starters": ["jacobs", "judkins", "stevenson", "myers",
                      "taco_qb"]},
    ]
    for rid in ("4", "5", "6", "7", "8", "11", "12"):
        rosters.append({"roster_id": rid, "players": [],
                        "starters": []})
    standings = [
        {"roster_id": "4", "team_name": "Team Four", "wins": 7,
         "losses": 1, "pf": 1200.0},
        {"roster_id": "5", "team_name": "Team Five", "wins": 6,
         "losses": 2, "pf": 1150.0},
        {"roster_id": "6", "team_name": "Team Six", "wins": 6,
         "losses": 2, "pf": 1100.0},
        {"roster_id": "7", "team_name": "Team Seven", "wins": 5,
         "losses": 3, "pf": 1050.0},
        {"roster_id": "8", "team_name": "Team Eight", "wins": 5,
         "losses": 3, "pf": 1000.0},
        {"roster_id": "11", "team_name": "Team Eleven", "wins": 4,
         "losses": 4, "pf": 990.0},
        {"roster_id": "12", "team_name": "Team Twelve", "wins": 4,
         "losses": 4, "pf": 985.0},
        {"roster_id": "1", "team_name": "Tua in the Pink", "wins": 3,
         "losses": 5, "pf": 983.25},
        {"roster_id": "9", "team_name": "East Bay Biters", "wins": 3,
         "losses": 5, "pf": 879.25},
        {"roster_id": "2", "team_name": "Goodfellas", "wins": 3,
         "losses": 5, "pf": 870.0},
        {"roster_id": "3", "team_name": "Cheeseheads", "wins": 2,
         "losses": 6, "pf": 800.0},
        {"roster_id": "10", "team_name": "Taco", "wins": 2,
         "losses": 6, "pf": 750.0},
    ]
    role_states = {
        pid: {"state": "starter", "reason": "", "week": 9}
        for pid in ("fagnano", "mccarthy", "cmc", "kyren", "dobbins",
                    "jsn", "higgins", "waddle", "watson", "flowers",
                    "moore", "laporta", "folk")
    }
    role_states.update({
        "milton": {"state": "promoted", "reason": "Hurts on IR",
                   "week": 9},
        "lock": {"state": "promoted", "reason": "Darnold out",
                 "week": 9},
        "brissett": {"state": "promoted", "reason": "Lamar on IR",
                     "week": 9},
        "harvey": {"state": "starter_out", "reason": "IR", "week": 9},
        "bigsby": {"state": "unknown", "reason": "", "week": 9},
        "sadiq": {"state": "unknown", "reason": "", "week": 9},
    })
    base_proj = {"cmc": 19.0, "kyren": 15.0, "dobbins": 12.0,
                 "jsn": 16.0, "higgins": 13.0, "waddle": 12.0,
                 "watson": 11.0, "flowers": 11.0, "moore": 10.0,
                 "laporta": 10.0}
    proj = {}
    for wk in range(9, 15):
        proj[wk] = dict(base_proj)
    proj[9].update({"fagnano": 15.0, "milton": 17.0, "mccarthy": 16.0})
    proj[10].update({"mccarthy": 16.0})
    proj[11].update({"milton": 17.0, "mccarthy": 16.0})
    proj[12].update({"milton": 17.0, "mccarthy": 16.0})
    proj[13].update({"milton": 17.0})
    proj[14].update({"milton": 17.0, "mccarthy": 16.0})
    return {
        "platform": "fleaflicker",
        "league_id": "92916",
        "season": 2026,
        "current_season": 2026,
        "current_week": 9,
        "players": players,
        "rosters": rosters,
        "roster_map": {r["roster_id"]: "" for r in rosters},
        "gm_standings": standings,
        "league": {"settings": {"reserve_slots": 1}},
        "league_rules": {
            "source": "user_confirmed",
            "playoff_teams": 8,
            "playoff_week_start": 15,
            "waiver_type": "rolling",
            "waiver_process_time": "03:00",
            "waiver_process_days": "daily",
            "waiver_process_tz": "America/New_York",
        },
        "role_states": role_states,
        "proj_by_week": {wk: {"projections": p}
                         for wk, p in proj.items()},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "WR", "TE",
                             "FLEX", "SUPER_FLEX", "K", "DEF"],
        "gm_schedule_games": [
            {"home": "BAL", "away": "CIN", "gameTime_epoch": THU},
            {"home": "GB", "away": "CHI", "gameTime_epoch": SUN},
            {"home": "PHI", "away": "NYG", "gameTime_epoch": SUN},
            {"home": "SF", "away": "LAR", "gameTime_epoch": SUN},
            {"home": "TEN", "away": "JAX", "gameTime_epoch": SUN},
        ],
        "matchups_by_week": {
            9: [{"left": {"roster_id": "1"},
                 "right": {"roster_id": "4"}}],
            10: [{"left": {"roster_id": "1"},
                  "right": {"roster_id": "9"}}],
            11: [{"left": {"roster_id": "1"},
                  "right": {"roster_id": "10"}}],
            12: [{"left": {"roster_id": "1"},
                  "right": {"roster_id": "2"}}],
        },
        "model_value_table": [
            {"id": "sadiq", "value": 40}, {"id": "little", "value": 60},
            {"id": "bigsby", "value": 120}, {"id": "harvey", "value": 300},
            {"id": "moore", "value": 700}, {"id": "flowers", "value": 800},
        ],
    }


def _queue_actions():
    return [
        {"kind": "lineup", "tag": "Lineup", "priority": "high",
         "action": "Start Joe Milton over Joe Fagnano",
         "why": "Joe Milton projects 17.0 vs Joe Fagnano's 15.0.",
         "cta_label": "Fix lineup", "cta_url": "#", "score": 103,
         "player_ids": ["milton", "fagnano"]},
        {"kind": "waiver", "tag": "Waivers", "priority": "high",
         "action": "Claim Drew Lock",
         "why": "Starting for SEA: Sam Darnold out.",
         "cta_label": "Claim now", "cta_url": "#", "score": 80,
         "player_id": "lock",
         "role_reason": "Starting for SEA: Sam Darnold out"},
        {"kind": "waiver", "tag": "Waivers", "priority": "high",
         "action": "Claim Jacoby Brissett",
         "why": "Starting for BAL: Lamar Jackson on IR.",
         "cta_label": "Claim now", "cta_url": "#", "score": 78,
         "player_id": "brissett",
         "role_reason": "Starting for BAL: Lamar Jackson on IR",
         "dropped_today": True},
        {"kind": "lineup", "tag": "Lineup", "priority": "med",
         "action": "Start Nick Folk over Cam Little",
         "why": "One week only: Folk projects ahead this week.",
         "cta_label": "Fix lineup", "cta_url": "#", "score": 65,
         "player_ids": ["folk", "little"]},
        {"kind": "lineup", "tag": "Lineup", "priority": "med",
         "action": "Watch Christian McCaffrey (Questionable, hamstring)",
         "why": "If McCaffrey sits, J.K. Dobbins is the plug-in.",
         "cta_label": "Fix lineup", "cta_url": "#", "score": 60,
         "player_ids": ["cmc", "dobbins"], "plugin_player_id": "dobbins"},
        {"kind": "lineup", "tag": "Lineup", "priority": "med",
         "action": "Hedge Christian Watson (Questionable, ribs) with Zay Flowers",
         "why": "Flowers is the replacement if Watson sits.",
         "cta_label": "Fix lineup", "cta_url": "#", "score": 59,
         "player_ids": ["watson", "flowers"]},
    ]


def _trade_actions(monkeypatch):
    """The upgraded collector fed by a mocked Wave 2 suggestions ctx."""
    import app as appmod

    sugg = {
        "viewer_needs": ["QB"],
        "viewer_surplus": [],
        "top_partners": [
            {"roster_id": "2", "team_name": "Goodfellas",
             "partner_record": "3-5", "partner_direction": "balanced",
             "acceptance_rate": 62, "fairness_label": "Fair",
             "targets_they_have": [
                 {"id": "caleb", "name": "Caleb Williams",
                  "position": "QB"}],
             "targets_viewer_sends": [
                 {"id": "watson", "name": "Christian Watson",
                  "position": "WR"},
                 {"id": "mccarthy", "name": "J.J. McCarthy",
                  "position": "QB"}]},
            {"roster_id": "3", "team_name": "Cheeseheads",
             "partner_record": "2-6", "partner_direction": "balanced",
             "acceptance_rate": 55, "fairness_label": "Fair",
             "bye_flag": {"week": 11, "position": "QB",
                          "label": "Bye hits your Week 11 crunch"},
             "targets_they_have": [
                 {"id": "love", "name": "Jordan Love",
                  "position": "QB"}],
             "targets_viewer_sends": [
                 {"id": "watson", "name": "Christian Watson",
                  "position": "WR"},
                 {"id": "dobbins", "name": "J.K. Dobbins",
                  "position": "RB"}]},
        ],
    }
    monkeypatch.setattr(cb, "build_trade_suggestions_context",
                        lambda ctx, rid: sugg)
    ctx = _fixture_ctx()
    with appmod.app.test_request_context():
        return appmod._next_steps_trade_actions(ctx, VIEWER)


def _brief(monkeypatch):
    ctx = _fixture_ctx()
    actions = _queue_actions() + _trade_actions(monkeypatch)
    return gb.build_gm_brief(ctx, VIEWER, actions, now=NOW)


# Element 1 (pass/fail): claims ordered Lock, Brissett, process stamp,
# priority 5, role whys, Harvey IR funding, Sadiq drop.
def test_element_1_claims(monkeypatch):
    brief = _brief(monkeypatch)
    claims = brief["claims"]
    assert [c["name"] for c in claims] == ["Drew Lock", "Jacoby Brissett"]
    lock_claim, brissett_claim = claims
    for claim in claims:
        assert claim["waiver_priority"] == 5
        assert claim["deadline"]["kind"] == "waiver_process"
        assert "3:00 AM ET" in claim["deadline"]["label"]
    assert "Darnold out" in lock_claim["why"]
    assert "Lamar" in brissett_claim["why"]
    assert "IR" in brissett_claim["why"]
    assert brissett_claim["dropped_today"] is True
    assert lock_claim["funding"]["kind"] == "ir_move"
    assert lock_claim["funding"]["name"] == "RJ Harvey"
    assert brissett_claim["funding"]["kind"] == "drop"
    assert brissett_claim["funding"]["name"] == "Kenyon Sadiq"


# Element 2 (pass/fail): Milton over Fagnano stamped with Thursday's
# BAL kickoff; Folk over Little present as a one-week action.
def test_element_2_lineup(monkeypatch):
    brief = _brief(monkeypatch)
    milton = next(a for a in brief["actions"]
                  if "Milton over" in a["action"])
    assert milton["deadline"]["kind"] == "kickoff"
    assert milton["deadline"]["at"] == THU
    assert "Thu" in milton["deadline"]["label"]
    folk = next(a for a in brief["actions"] if "Folk over" in a["action"])
    assert "One week" in folk["why"]


# Element 3 (presence + facts): watch items with named plug-ins, and
# the Watson hedge stamped Thursday (Flowers), not Sunday.
def test_element_3_watch_and_hedge(monkeypatch):
    brief = _brief(monkeypatch)
    watch = next(a for a in brief["actions"]
                 if a["action"].startswith("Watch Christian McCaffrey"))
    assert "Questionable" in watch["action"]
    assert "Dobbins" in watch["why"]
    hedge = next(a for a in brief["actions"]
                 if a["action"].startswith("Hedge Christian Watson"))
    assert "Flowers" in hedge["action"]
    assert hedge["deadline"]["at"] == THU


# Element 4 (pass/fail): step-1 Caleb deal, QB back in the package,
# partner Goodfellas, conditioned acceptance present.
def test_element_4_caleb_step_1(monkeypatch):
    brief = _brief(monkeypatch)
    caleb = next(a for a in brief["actions"]
                 if a.get("receive_ids") == ["caleb"])
    assert caleb["partner"] == "Goodfellas"
    assert set(caleb["send_ids"]) == {"watson", "mccarthy"}
    assert caleb["acceptance_rate"] == 62
    assert caleb["sequence"]["step"] == 1
    assert brief["layers"]["partner_model"] is True


# Element 5 (presence + facts): Love contingent on element 4, Week 11
# bye-collision flag, Watson spent only once among step-1 deals.
def test_element_5_love_contingent(monkeypatch):
    brief = _brief(monkeypatch)
    love = next(a for a in brief["actions"]
                if a.get("receive_ids") == ["love"])
    assert love["sequence"]["step"] == 2
    assert love["why"].startswith("If the Goodfellas deal dies:")
    assert (love.get("bye_flag") or {}).get("week") == 11
    step1_sends = [
        a for a in brief["actions"]
        if a.get("kind") == "trade"
        and (a.get("sequence") or {}).get("step") == 1
        and "watson" in (a.get("send_ids") or [])
    ]
    assert len(step1_sends) == 1


# Element 6 (pass/fail): race line on the true 8-team field, PF
# tiebreak vs East Bay named, no 6-field math anywhere.
def test_element_6_race_line(monkeypatch):
    brief = _brief(monkeypatch)
    race = brief["race_line"]
    assert race is not None
    assert race["seed"] == 8
    assert race["playoff_teams"] == 8
    assert race["pf_rival"] == "East Bay Biters"
    assert race["pf_delta"] == 104.0
    assert "East Bay Biters" in race["h2h_remaining"]
    blob = json.dumps(brief, default=str)
    assert "1 GB out" not in blob


# Element 7 (presence + facts): bye panel Week 10 East Bay QB hole,
# Week 11 both sides' losses with Taco's RBs named.
def test_element_7_bye_panel(monkeypatch):
    brief = _brief(monkeypatch)
    panel = brief["bye_panel"]
    assert panel is not None
    wk10 = next(w for w in panel["weeks"] if w["week"] == 10)
    assert wk10["opponent"]["team_name"] == "East Bay Biters"
    assert wk10["opponent"]["players_by_pos"]["QB"] == ["Bo Nix"]
    assert any("no QB left" in n for n in wk10["notes"])
    # Sadiq (PHI) is also on a Week 10 bye, so TE is in the losses too.
    assert wk10["viewer"]["by_pos"] == {"QB": 1, "RB": 3, "WR": 1,
                                        "TE": 1}
    wk11 = next(w for w in panel["weeks"] if w["week"] == 11)
    assert wk11["viewer"]["total"] == 5
    assert wk11["opponent"]["team_name"] == "Taco"
    assert wk11["opponent"]["players_by_pos"]["RB"] == [
        "Josh Jacobs", "Quinshon Judkins", "Rhamondre Stevenson"]


# Element 8 (presence + facts): coverage Weeks 9-14, Week 10
# Milton-bye and Week 13 BAL-bye holes flagged with causes.
def test_element_8_coverage(monkeypatch):
    brief = _brief(monkeypatch)
    cov = brief["qb_coverage"]
    assert cov is not None
    weeks = {w["week"]: w for w in cov["weeks"]}
    assert set(weeks) == {9, 10, 11, 12, 13, 14}
    wk9_names = [s["name"] for s in weeks[9]["qb"]["startable"]]
    assert "Joe Fagnano" in wk9_names
    assert "Joe Milton" in wk9_names
    hole10 = next(h for h in cov["holes"]
                  if h["week"] == 10 and h["position"] == "QB")
    assert "bye: Joe Milton" in hole10["causes"]
    hole13 = next(h for h in cov["holes"]
                  if h["week"] == 13 and h["position"] == "QB")
    causes13 = " ".join(hole13["causes"])
    assert "Joe Fagnano" in causes13
    assert "J.J. McCarthy" in causes13


def test_dropped_today_chip_absent_without_cross_stack_data(monkeypatch):
    # Degradation (spec section 5): with no dropped-today data sourced
    # (the #3009/#3012 stacks are not in this ancestry), the claim
    # carries no chip rather than an invented one.
    ctx = _fixture_ctx()
    actions = _queue_actions()
    for action in actions:
        action.pop("dropped_today", None)
    brief = gb.build_gm_brief(ctx, VIEWER, actions, now=NOW)
    assert all(c["dropped_today"] is False for c in brief["claims"])
