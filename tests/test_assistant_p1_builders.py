"""P1 builder tests: breakout board/track record/history, Teams, picks.

Everything external is monkeypatched at the evidence module's lazy shims,
the same pattern as tests/test_assistant_evidence.py: the breakout API,
the league-format value lookup, and the pick value table are faked; the
Teams machinery itself (rank_rosters_by_position, calculate_roster_grade,
pick_value_from_table, parse_pick_asset) runs for real.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose, evidence

# -- fixture league: 4 teams, Pilots strongest, Sacko weakest -------------------
# pid: (name, position, dynasty value, redraft value)
PLAYERS = {
    "a1": ("Pilots QB", "QB", 3000, 2800), "a2": ("Pilots RB", "RB", 2500, 2600),
    "a3": ("Pilots WR", "WR", 3200, 3000), "a4": ("Pilots TE", "TE", 1800, 1700),
    "a5": ("Zay Flowers", "WR", 4680, 4400),
    "b1": ("Couch QB", "QB", 2600, 2400), "b2": ("Woody Marks", "RB", 3400, 3200),
    "b3": ("Couch WR", "WR", 3500, 3300), "b4": ("Couch TE", "TE", 1200, 1300),
    "c1": ("Goblins QB", "QB", 2000, 1900), "c2": ("Goblins RB", "RB", 2100, 2000),
    "c3": ("Goblins WR", "WR", 2200, 2100), "c4": ("Goblins TE", "TE", 900, 950),
    "d1": ("Sacko QB", "QB", 1200, 1100), "d2": ("Sacko RB", "RB", 1300, 1200),
    "d3": ("Sacko WR", "WR", 1400, 1300), "d4": ("Sacko TE", "TE", 500, 550),
}
ROSTER_PLAYERS = {
    "1": ["a1", "a2", "a3", "a4", "a5"], "2": ["b1", "b2", "b3", "b4"],
    "3": ["c1", "c2", "c3", "c4"], "4": ["d1", "d2", "d3", "d4"],
}
RECORDS = {"1": (5, 7), "2": (9, 3), "3": (6, 6), "4": (2, 10)}
TEAM_NAMES = {"1": "Pittsburgh Pilots", "2": "Caleb's Casting Couch",
              "3": "Gridiron Goblins", "4": "Sacko Squad"}
# 2027 round 1: the viewer (roster 2) traded their own pick to the Pilots
# (roster 1) and holds the Goblins' pick. Slots by record, worst first:
# Sacko 1, Pilots 2, Goblins 3, Couch 4.
PICKS_BY_ROSTER = {
    "1": [{"season": 2027, "round": 1, "original_owner": "1"},
          {"season": 2027, "round": 1, "original_owner": "2"}],
    "2": [{"season": 2027, "round": 1, "original_owner": "3"}],
    "3": [],
    "4": [{"season": 2027, "round": 1, "original_owner": "4"}],
}
PICK_TABLE = {"2027_1_01": 2800.0, "2027_1_02": 2300.0, "2027_1_03": 1900.0,
              "2027_1_04": 1500.0, "2027_1_early": 2300.0, "2027_1_mid": 1500.0,
              "2027_1_late": 900.0}


def _by_id(force=None):
    key = 3 if force == "redraft" else 2
    return {pid: {"value": vals[key], "position": vals[1], "age": 25,
                  "redraft_value_1qb": vals[3], "name": vals[0]}
            for pid, vals in PLAYERS.items()}


def _ctx():
    rosters = [{"roster_id": rid, "owner_id": f"u{rid}", "players": ROSTER_PLAYERS[rid],
                "settings": {"wins": RECORDS[rid][0], "losses": RECORDS[rid][1]}}
               for rid in ("1", "2", "3", "4")]
    return {"rosters": rosters, "users": [], "roster_map": dict(TEAM_NAMES),
            "roster_positions": ["QB", "RB", "WR", "TE", "FLEX"],
            "model_value_table": [{"id": pid} for pid in PLAYERS],
            "picks_by_roster": PICKS_BY_ROSTER,
            "league": {"season": 2026, "settings": {"type": 2}},
            "scoring_settings": {}, "raw_scoring_settings": {}}


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Caleb's Casting Couch",
              "week": 7, "ctx": _ctx(),
              "roster": {"roster_id": "2", "players": list(ROSTER_PLAYERS["2"])},
              "roster_player_ids": set(ROSTER_PLAYERS["2"]), "is_paid": False}
    bundle.update(over)
    return bundle


@pytest.fixture()
def base(monkeypatch):
    index = {pid: {"name": vals[0], "team": "BUF", "position": vals[1]}
             for pid, vals in PLAYERS.items()}
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: dict(index))
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda *a, **k: [{"id": pid, "value": vals[2]}
                                          for pid, vals in PLAYERS.items()])
    monkeypatch.setattr(evidence, "league_format_values",
                        lambda ctx, force_scoring_type=None: _by_id(force_scoring_type))
    monkeypatch.setattr(evidence, "pick_value_table", lambda *a, **k: dict(PICK_TABLE))
    return monkeypatch


# -- Teams builder -----------------------------------------------------------------

def test_teams_builder_strongest_weakest_dynasty(base):
    pack, tools = evidence.build_teams(_bundle(), {}, [])
    assert pack["mode"] == "dynasty" and pack["mode_label"] == "Dynasty"
    assert pack["strongest"]["team_name"] == "Pittsburgh Pilots"
    assert pack["weakest"]["team_name"] == "Sacko Squad"
    assert "utils.trade.rank_rosters_by_position" in tools
    assert "context_builders.calculate_roster_grade" in tools
    mine = next(t for t in pack["teams"] if t["is_viewer"])
    assert mine["team_name"] == "Caleb's Casting Couch"
    assert mine["grade"] is not None and mine["grade_score"] is not None
    assert pack["by_position"]["TE"]["weakest"]["team_name"] == "Sacko Squad"


def test_teams_builder_position_focus(base):
    pack, _ = evidence.build_teams(_bundle(), {"position_hint": "TE"}, [])
    assert pack["position_focus"] == "TE"
    ranking = pack["position_ranking"]
    assert [r["rank"] for r in ranking] == [1, 2, 3, 4]
    assert ranking[0]["team_name"] == "Pittsburgh Pilots"
    assert ranking[-1]["team_name"] == "Sacko Squad"
    assert ranking[-1]["strength"] == 500.0


def test_teams_builder_this_season_mode(base):
    pack, _ = evidence.build_teams(_bundle(), {"teams_mode": "this_season"}, [])
    assert pack["mode"] == "this_season" and pack["mode_label"] == "This Season"
    # Grades still computed (on redraft values) and ordered the same here.
    assert pack["strongest"]["team_name"] == "Pittsburgh Pilots"
    assert pack["strongest"]["grade_score"] is not None


# -- breakout builders ----------------------------------------------------------------

BOARD = {"season": 2026, "as_of_week": 6, "data_available": True, "candidates": [
    {"player_id": f"p{i}", "player_name": f"Player {i}", "team": "BUF",
     "position": "WR", "breakout_score": 90.0 - i, "classification_label": "Breakout Call",
     "confidence": 70.0 - i} for i in range(1, 7)]}


def test_breakout_board_preview_rule(base, monkeypatch):
    monkeypatch.setattr(evidence, "breakout_board_candidates", lambda season=None: dict(BOARD))
    pack, _ = evidence.build_breakout_board(_bundle(is_paid=False), {}, [])
    # Non-premium mirror of the board route: first 3 plus the locked count,
    # and nothing beyond the preview exists in the pack to cite.
    assert pack["preview"] is True
    assert [c["name"] for c in pack["candidates"]] == ["Player 1", "Player 2", "Player 3"]
    assert pack["locked_count"] == 3
    assert pack["total_calls"] == 6
    assert pack["week_label"] == "Week 7"
    pack, _ = evidence.build_breakout_board(_bundle(is_paid=True), {}, [])
    assert pack["preview"] is False
    assert len(pack["candidates"]) == 6
    assert "locked_count" not in pack


def test_breakout_board_unavailable(base, monkeypatch):
    monkeypatch.setattr(evidence, "breakout_board_candidates",
                        lambda season=None: {"data_available": False, "candidates": []})
    pack, _ = evidence.build_breakout_board(_bundle(), {}, [])
    assert pack["available"] is False


def test_breakout_track_record_pack(base, monkeypatch):
    record = {"season": 2026,
              "weekly": {"available": True, "scoring_version": "weekly-v6",
                         "overall": {"calls": 14, "graded": 11, "hit": 6, "partial": 1,
                                     "miss": 4, "hit_rate": 0.5455}},
              "by_week": [{"week": 2, "calls": 11, "graded": 8, "hit": 4, "partial": 0,
                           "miss": 4, "hit_rate": 0.5}],
              "hits": [{"player_name": "Troy Franklin", "as_of_week": 2, "label": "Breakout Call"}],
              "misses": []}
    monkeypatch.setattr(evidence, "breakout_track_record_data", lambda season=None: record)
    pack, _ = evidence.build_breakout_track_record(_bundle(), {}, [])
    assert pack["overall"]["graded"] == 11
    assert pack["overall"]["hit_rate_pct"] == 54.5
    assert pack["by_week"][0]["hit_rate_pct"] == 50.0
    assert pack["biggest_hits"][0]["name"] == "Troy Franklin"
    assert pack["pending"] is False


def test_breakout_history_earliest_call_and_grade(base, monkeypatch):
    monkeypatch.setattr(evidence, "breakout_board_candidates", lambda season=None: dict(BOARD))
    monkeypatch.setattr(evidence, "breakout_call_history_for",
                        lambda pid, season=None: {"kind": "weekly", "season": 2026,
                                                  "as_of_week": 4, "label": "Week 5"})
    monkeypatch.setattr(evidence, "breakout_week_detail",
                        lambda pid, season, week: {"available": True, "grade": "hit",
                                                   "breakout_score": 82.3})
    index = {"rome": {"name": "Rome Odunze", "team": "CHI", "position": "WR"}}
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: dict(index))
    pack, _ = evidence.build_breakout_history(_bundle(), {}, ["rome"])
    assert pack["was_call"] is True
    assert pack["earliest_call"]["label"] == "Week 5"
    assert pack["earliest_call"]["breakout_score"] == 82.3
    assert pack["grade"] == "hit"
    assert pack["on_current_board"] is False


def test_breakout_history_never_called(base, monkeypatch):
    monkeypatch.setattr(evidence, "breakout_board_candidates", lambda season=None: dict(BOARD))
    monkeypatch.setattr(evidence, "breakout_call_history_for", lambda pid, season=None: None)
    pack, _ = evidence.build_breakout_history(_bundle(), {}, ["zz"])
    assert pack["was_call"] is False
    assert "earliest_call" not in pack


# -- pick ownership ---------------------------------------------------------------------

def test_pick_ownership_pack_names_current_owner(base):
    pack, _ = evidence.build_pick_ownership(_bundle(), {"picks": ["my 2027 first"]}, [])
    pick = pack["pick"]
    # My own 2027 1st: I am the original owner, the Pilots hold it now, and
    # my strong record makes it the late slot 4 pick.
    assert pick["pick_id"] == "2027_1_04"
    assert pick["label"] == "2027 1.04"
    assert pick["original_owner_team"] == "Caleb's Casting Couch"
    assert pick["current_owner_team"] == "Pittsburgh Pilots"
    assert pick["value"] == 1500.0 and pick["priced"] is True
    assert pack["owned_by_me"] is False


def test_pick_ownership_other_teams_pick_that_i_hold(base):
    pack, _ = evidence.build_pick_ownership(
        _bundle(), {"picks": ["2027 first"], "other_team_hint": "Gridiron Goblins"}, [])
    pick = pack["pick"]
    assert pick["pick_id"] == "2027_1_03"
    assert pick["original_owner_team"] == "Gridiron Goblins"
    assert pick["current_owner_team"] == "Caleb's Casting Couch"
    assert pick["value"] == 1900.0
    assert pack["owned_by_me"] is True


# -- trade eval with picks -----------------------------------------------------------------

def test_trade_eval_prices_and_attributes_picks(base):
    entities = {"give_player_ids": ["b2"], "receive_player_ids": ["a5"],
                "give_picks": ["my 2027 1st"]}
    pack, _ = evidence.build_trade_eval(_bundle(), entities, ["b2", "a5"])
    assert len(pack["give_picks"]) == 1
    pick = pack["give_picks"][0]
    # Attribution matters here: the viewer's OWN 2027 1st is already gone
    # (the Pilots hold it), so the only 2027 1st they can offer is the
    # Goblins' pick they hold: slot 3 by the Goblins' record, value 1900.
    # A naive "my pick, my slot" read would price their own slot 4 (1500)
    # instead and flip the verdict.
    assert pick["pick_id"] == "2027_1_03"
    assert pick["original_owner_team"] == "Gridiron Goblins"
    assert pick["current_owner_team"] == "Caleb's Casting Couch"
    assert pick["value"] == 1900.0
    # The priced pick counts in the side total: 3400 + 1900 vs 4680.
    assert pack["give"]["total_value"] == 5300.0
    assert pack["receive"]["total_value"] == 4680.0
    assert pack["fairness_gap"] == -620.0
    assert pack["trade_verdict"] == "DECLINE"
    assert pack["receive_picks"] == []
    # The Teams swap: fit data now carries the site's real roster grades.
    assert pack["roster_fit"]["grade"] is not None
    assert pack["roster_fit"]["grade_mode"] == "dynasty"
    assert pack["partner"]["team_name"] == "Pittsburgh Pilots"
    assert pack["partner"]["grade"] is not None


def test_trade_pack_picks_stage_in_calculator_link(base):
    entities = {"give_player_ids": ["b2"], "receive_player_ids": ["a5"],
                "give_picks": ["my 2027 1st"]}
    pack, _ = evidence.build_trade_eval(_bundle(), entities, ["b2", "a5"])
    # The calculator's URL contract (loadTradeFromURL in static/app.js and
    # PR #2611's link builder): a side's list is what that side receives,
    # so receive picks stage as ap and give picks as bp, by canonical id.
    from urllib.parse import urlencode
    from utils.draft import parse_pick_asset
    params = {}
    if pack["receive_picks"]:
        params["ap"] = ",".join(p["pick_id"] for p in pack["receive_picks"])
    if pack["give_picks"]:
        params["bp"] = ",".join(p["pick_id"] for p in pack["give_picks"])
    query = urlencode(params)
    assert "bp=2027_1_03" in query
    parsed = parse_pick_asset(pack["give_picks"][0]["pick_id"])
    assert parsed["season"] == 2027 and parsed["round"] == 1 and parsed["slot"] == 3
    # When the deep-link builder is on this branch (PR #2611), it must
    # stage the same ids from this pack unchanged.
    builder = getattr(compose, "_trade_calc_link", None)
    if builder is not None:
        link = builder(pack)
        assert link is not None and "bp=2027_1_03" in link["href"]


# -- compose: required receipts and act links for the new intents ----------------------------

def _draft(**over):
    data = {"verdict": "Call", "text": "See the numbers below.", "receipts": [],
            "caveat": None, "followups": [], "act_link": None}
    data.update(over)
    return data


def test_compose_track_record_injects_required_receipts(monkeypatch):
    import json
    from pathlib import Path
    questions = json.loads((Path(__file__).resolve().parent.parent
                            / "assistant_eval" / "questions.json").read_text())["questions"]
    pack = next(q["evidence_pack"] for q in questions if q["id"] == "bo2")
    monkeypatch.setattr(compose, "call_model_json", lambda *a, **k: _draft())
    answer = compose.compose_answer("breakout_track_record", "How good is BR at breakout calls?", pack)
    values = {r["value"] for r in answer["receipts"]}
    assert {"11", "54.5", "6", "4"} <= values
    assert answer["act_link"] == {"href": "/breakouts", "label": "View breakout board"}


def test_compose_teams_act_link_is_league_scoped(monkeypatch):
    import json
    from pathlib import Path
    questions = json.loads((Path(__file__).resolve().parent.parent
                            / "assistant_eval" / "questions.json").read_text())["questions"]
    pack = next(q["evidence_pack"] for q in questions if q["id"] == "tm1")
    monkeypatch.setattr(compose, "call_model_json", lambda *a, **k: _draft())
    answer = compose.compose_answer("teams", "Whose roster is strongest?", pack)
    assert answer["act_link"] == {"href": "/sleeper/2026/blackedraw/teams",
                                   "label": "View on Teams"}
    values = {r["value"] for r in answer["receipts"]}
    assert "91.4" in values and "86.2" in values


def test_compose_trade_verdict_stays_pinned_with_picks(monkeypatch):
    import json
    from pathlib import Path
    questions = json.loads((Path(__file__).resolve().parent.parent
                            / "assistant_eval" / "questions.json").read_text())["questions"]
    pack = next(q["evidence_pack"] for q in questions if q["id"] == "te4")
    monkeypatch.setattr(compose, "call_model_json",
                        lambda *a, **k: _draft(verdict="FAIR", text="Woody Marks for Zay Flowers."))
    answer = compose.compose_answer("trade_eval", "Should I trade my 2027 1st and Woody Marks for Zay Flowers?", pack)
    # Totals include the attributed pick (3430 vs 4680): inside the band is
    # wrong here, the pinned band verdict is ACCEPT and cannot be flipped.
    assert answer["verdict"] == "ACCEPT"
