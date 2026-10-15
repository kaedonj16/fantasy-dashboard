"""Live in-game + draft help tests.

Live answers read the same TTL-cached matchup path the portfolio card
uses (stubbed at the evidence module's live_matchups shim; the totals
and win-probability math is the site's own). Draft answers read the
league's Sleeper draft through the same primitives /api/draft/live
serves (stubbed at the evidence shims).
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence, router
from dashboard_services.assistant.router import _keyword_route


def _bundle(ctx=None, **over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": ctx if ctx is not None else {},
              "roster": {"roster_id": 3, "players": [], "starters": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


# -- router -----------------------------------------------------------------------

@pytest.mark.parametrize("message,intent", [
    ("Who is winning my game right now?", "live"),
    ("What do I need tonight to win?", "live"),
    ("How is Josh Allen doing today?", "live"),
    ("Who is the best available in my draft?", "draft"),
    ("Who is on the clock in my draft?", "draft"),
    ("What is my next pick in the draft?", "draft"),
    # Neighbors keep their intents.
    ("Will I win this week?", "matchup"),
    ("Break down my matchup", "matchup"),
    ("What are my playoff odds?", "playoffs"),
])
def test_keyword_live_and_draft(message, intent):
    assert _keyword_route(message)["intent"] == intent


def test_keyword_draft_slot_parsed():
    r = _keyword_route("Who should I target at 1.03 in my rookie draft?")
    assert r["intent"] == "draft"
    assert r["draft_slot"] == "1.03"


def test_model_path_passes_draft_slot(monkeypatch):
    def _fake(system, payload, schema, name):
        return {"intent": "draft", "player_names": [],
                "give_player_names": [], "receive_player_names": [],
                "picks": [], "give_picks": [], "receive_picks": [],
                "other_team_hint": None, "metric_hint": None,
                "draft_slot": "1.03"}
    monkeypatch.setattr(router, "call_model_json", _fake)
    out = router.classify("Who should I target at 1.03?", {}, None)
    assert out["intent"] == "draft"
    assert out["entities"]["draft_slot"] == "1.03"
    assert "live" in router.INTENTS and "draft" in router.INTENTS


def test_new_intents_have_builders():
    assert evidence.BUILDERS["live"] is evidence.build_live
    assert evidence.BUILDERS["draft"] is evidence.build_draft


# -- live --------------------------------------------------------------------------

def _team(rid, name, starters):
    return {"roster_id": rid, "name": name, "record": "5-2",
            "starters": starters}


def _starter(pid, name, pos, pts):
    return {"pid": pid, "name": name, "pos": pos, "pts": pts}


def _stub_live(monkeypatch, matchups, statuses, proj):
    monkeypatch.setattr(evidence, "live_matchups",
                        lambda *a: (matchups, statuses, proj))


def test_live_pack_from_cached_matchups(monkeypatch):
    mine = _team(3, "Couch", [_starter("a", "Allen", "QB", 30.0),
                              _starter("b", "Nabers", "WR", 0.0)])
    opp = _team(5, "Mustard", [_starter("c", "Hurts", "QB", 25.0),
                               _starter("d", "Andrews", "TE", 0.0)])
    statuses = {"a": "final", "b": "not_started", "c": "final",
                "d": "not_started"}
    proj = {"a": 28.0, "b": 14.0, "c": 25.0, "d": 8.0}
    _stub_live(monkeypatch, [{"left": mine, "right": opp}], statuses, proj)
    pack, tools = evidence.build_live(_bundle({"rosters": []}), {}, [])
    assert pack["available"] is True
    assert pack["state"] == "live"
    mu = pack["matchup"]
    assert mu["is_mine"] is True
    assert mu["team_a"]["current_points"] == 30.0
    assert mu["team_b"]["current_points"] == 25.0
    assert mu["team_a"]["win_prob_pct"] is not None
    need = pack["need"]
    assert need["lead"] == 5.0 and need["deficit"] == 0.0
    assert [r["name"] for r in need["my_remaining"]] == ["Nabers"]
    assert "matchups.compute_win_prob" in tools


def test_live_all_final_is_final_state(monkeypatch):
    mine = _team(3, "Couch", [_starter("a", "Allen", "QB", 30.0)])
    opp = _team(5, "Mustard", [_starter("c", "Hurts", "QB", 25.0)])
    _stub_live(monkeypatch, [{"left": mine, "right": opp}],
               {"a": "final", "c": "final"}, {"a": 28.0, "c": 25.0})
    pack, _tools = evidence.build_live(_bundle({"rosters": []}), {}, [])
    assert pack["state"] == "final"


def test_live_no_matchup_is_no_data(monkeypatch):
    _stub_live(monkeypatch, [], {}, {})
    pack, _tools = evidence.build_live(_bundle({"rosters": []}), {}, [])
    assert pack["available"] is False
    assert "_" not in pack["missing"]


def test_live_receipts_pin_both_scores():
    pack = {"type": "live",
            "matchup": {"is_mine": True,
                        "team_a": {"team_name": "Couch", "current_points": 84.5,
                                   "projected": 121.4, "win_prob_pct": 62.0},
                        "team_b": {"team_name": "Mustard", "current_points": 79.2,
                                   "projected": 118.9, "win_prob_pct": 38.0}}}
    rows = compose_mod._required_receipts("live", pack)
    labels = [r["label"] for r in rows]
    assert labels[0] == "Couch points so far"
    assert labels[1] == "Mustard points so far"
    assert len(rows) <= 4


# -- draft -------------------------------------------------------------------------

_SLOTS = {"1": "7", "2": "5", "3": "3", "4": "1", "5": "2", "6": "4",
          "7": "6", "8": "8", "9": "9", "10": "10", "11": "11", "12": "12"}
_NAMES = {"7": "Hurts So Good", "3": "Couch", "5": "Mustard"}


def _pick(pick_no, roster_id, pid, name, pos):
    rnd = (pick_no - 1) // 12 + 1
    return {"pick_no": pick_no, "round": rnd, "roster_id": roster_id,
            "player_id": pid,
            "metadata": {"first_name": name.split()[0],
                         "last_name": " ".join(name.split()[1:]),
                         "position": pos, "team": "BUF"}}


def _draft_picks():
    picks = []
    order = ["7", "5", "3", "1", "2", "4", "6", "8", "9", "10", "11", "12"]
    for i in range(24):
        rnd = i // 12 + 1
        slot = (i % 12) if rnd == 1 else 11 - (i % 12)
        rid = order[slot]
        picks.append(_pick(i + 1, rid, f"pid{i + 1}", f"Player {i + 1}", "WR"))
    picks[2] = _pick(3, "3", "chase", "Ja'Marr Chase", "WR")
    picks[21] = _pick(22, "3", "barkley", "Saquon Barkley", "RB")
    return picks


def _draft_ctx():
    return {"rosters": [{"roster_id": int(r), "players": []} for r in _SLOTS.values()],
            "roster_map": {int(k): v for k, v in _NAMES.items()},
            "league": {"roster_positions": ["QB", "RB", "RB", "WR", "WR",
                                            "TE", "FLEX", "SUPER_FLEX"]}}


_VALUES = [
    {"id": "bijan", "value": 7450}, {"id": "allen", "value": 7300},
    {"id": "chase", "value": 9900}, {"id": "barkley", "value": 6800},
    {"id": "bowers", "value": 5400},
]
_INDEX = {
    "bijan": {"full_name": "Bijan Robinson", "position": "RB", "team": "ATL"},
    "allen": {"full_name": "Josh Allen", "position": "QB", "team": "BUF"},
    "chase": {"full_name": "Ja'Marr Chase", "position": "WR", "team": "CIN"},
    "barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
    "bowers": {"full_name": "Brock Bowers", "position": "TE", "team": "LV"},
}
_ROOKIES = [
    {"name": "Jeremiah Smith", "position": "WR", "school": "Ohio State",
     "prospect_score": 96.2, "sleeper_id": None},
    {"name": "Julian Sayin", "position": "QB", "school": "Ohio State",
     "prospect_score": 94.0, "sleeper_id": None},
    {"name": "Jeremiyah Love", "position": "RB", "school": "Notre Dame",
     "prospect_score": 93.1, "sleeper_id": None},
    {"name": "Dante Moore", "position": "QB", "school": "Oregon",
     "prospect_score": 91.8, "sleeper_id": None},
    {"name": "Ahmad Hardy", "position": "RB", "school": "Missouri",
     "prospect_score": 91.0, "sleeper_id": None},
]


def _stub_draft(monkeypatch, *, status="drafting", picks=None):
    monkeypatch.setattr(evidence, "league_drafts",
                        lambda platform, league_id, season: [
                            {"draft_id": "D1", "status": status,
                             "season": 2026, "type": "snake"}])
    monkeypatch.setattr(evidence, "draft_detail", lambda draft_id: {
        "status": status, "season": 2026, "type": "snake",
        "settings": {"rounds": 15, "teams": 12, "reversal_round": 0},
        "slot_to_roster_id": dict(_SLOTS)})
    monkeypatch.setattr(evidence, "draft_picks_live",
                        lambda draft_id: list(picks if picks is not None
                                              else _draft_picks()))
    monkeypatch.setattr(evidence, "league_traded_picks", lambda league_id: [])
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda: list(_VALUES))
    monkeypatch.setattr(evidence, "_index", lambda: dict(_INDEX))
    monkeypatch.setattr(evidence, "active_rookie_class", lambda: 2027)
    monkeypatch.setattr(evidence, "rookie_rankings",
                        lambda year: list(_ROOKIES))
    return _bundle(_draft_ctx())


def test_draft_pack_live_state(monkeypatch):
    pack, tools = evidence.build_draft(_stub_draft(monkeypatch), {}, [])
    assert pack["available"] is True
    assert pack["status"] == "drafting"
    assert pack["picks_made"] == 24
    cur = pack["current_pick"]
    assert cur["pick_no"] == 25 and cur["round"] == 3
    assert cur["team_name"] == "Hurts So Good" and cur["is_mine"] is False
    assert pack["my_next_pick"]["pick_no"] == 27
    # Drafted players are gone; the best remaining is by site value.
    names = [r["name"] for r in pack["best_available"]]
    assert names[0] == "Bijan Robinson"
    assert "Ja'Marr Chase" not in names
    assert [p["name"] for p in pack["my_picks"]] == [
        "Ja'Marr Chase", "Saquon Barkley"]
    needs = {n["position"]: n for n in pack["my_needs"]}
    assert needs["QB"]["open_slots"] == 2  # QB slot + SUPER_FLEX
    assert needs["TE"]["open_slots"] == 1
    assert "api.get_draft_picks" in tools


def test_draft_traded_pick_moves_the_clock(monkeypatch):
    bundle = _stub_draft(monkeypatch)
    monkeypatch.setattr(evidence, "league_traded_picks", lambda league_id: [
        {"season": 2026, "round": 3, "roster_id": 7, "owner_id": 3,
         "previous_owner_id": 7}])
    pack, _tools = evidence.build_draft(bundle, {}, [])
    # Round 3 pick 1 was Hurts So Good's slot, traded to the viewer.
    assert pack["current_pick"]["roster_id"] == "3"
    assert pack["current_pick"]["is_mine"] is True
    assert pack["my_next_pick"]["pick_no"] == 25


def test_draft_prep_window_for_a_named_slot(monkeypatch):
    pack, _tools = evidence.build_draft(
        _stub_draft(monkeypatch, status="pre_draft", picks=[]),
        {"draft_slot": "1.03"}, [])
    prep = pack["prep"]
    assert prep["overall_pick"] == 3
    assert [c["rank"] for c in prep["candidates"]] == [1, 2, 3, 4, 5]
    at_pick = next(c for c in prep["candidates"] if c["rank"] == 3)
    assert at_pick["name"] == "Jeremiyah Love"
    assert at_pick["score"] == 93.1


def test_draft_complete_has_no_best_available(monkeypatch):
    pack, _tools = evidence.build_draft(
        _stub_draft(monkeypatch, status="complete"), {}, [])
    assert pack["status"] == "complete"
    assert pack["best_available"] == []
    assert pack["current_pick"] is None


def test_draft_no_draft_is_no_data(monkeypatch):
    monkeypatch.setattr(evidence, "league_drafts",
                        lambda platform, league_id, season: [])
    pack, _tools = evidence.build_draft(_bundle({}), {}, [])
    assert pack["available"] is False
    assert "_" not in pack["missing"]


def test_draft_receipts_pin_the_clock_and_best_value():
    pack = {"type": "draft",
            "current_pick": {"pick_no": 25},
            "my_next_pick": {"pick_no": 27},
            "best_available": [{"name": "Bijan Robinson", "value": 7450.0}],
            "picks_made": 24}
    rows = compose_mod._required_receipts("draft", pack)
    labels = [r["label"] for r in rows]
    assert labels == ["Current pick", "Your next pick", "Bijan Robinson value"]
