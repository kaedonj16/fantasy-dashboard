"""PR E: the last two Tier 1 assistant audit fixes.

1. GM directive coverage: build_gm_directive read brief["coverage"],
   but build_gm_brief returns its coverage map (the build_coverage
   dict: weeks / holes / requirements) under "qb_coverage", so the
   coverage-directive loop never fired and "tell me what to do"
   answers silently omitted coverage holes. The brief fixture below
   copies the real build_gm_brief return shape (see
   dashboard_services/gm_brief.py and tests/test_gm_brief_coverage.py).

2. Rookie draft board: build_draft built pack["best_available"] (and
   its by-position cut) from the full veteran value table even when
   draft_kind == "rookie", so compose cited a veteran as the best
   player available in a rookie draft. In a rookie draft the pool is
   the rookie class.
"""
import pytest

pytest.importorskip("pandas")

from dashboard_services import gm_brief
from dashboard_services.assistant import evidence


# -- GM directive coverage ------------------------------------------------

def _brief(qb_coverage):
    """build_gm_brief's return shape, coverage under "qb_coverage"."""
    return {
        "week": 9,
        "actions": [],
        "race_line": {"text": "Win and the 8 seed is yours to lose",
                      "path": {"win_and_in": False,
                               "note": "Butter Boys lose the tiebreak"}},
        "race_prompt": None,
        "qb_coverage": qb_coverage,
        "bye_panel": None,
        "claims": [],
        "league_culture": None,
        "return_horizons": [],
        "layers": {"rules": None, "roles": True, "partner_model": False},
    }


def _coverage(holes):
    """build_coverage's return shape; holes carry week/position/
    count/required/causes/fills_action."""
    return {"weeks": [], "holes": holes, "missing_weeks": [],
            "through_week": 14, "label": None, "roles_confirmed": True,
            "requirements": {"QB": 2, "TE": 1}}


def _directive_pack(monkeypatch, brief):
    monkeypatch.setattr(gm_brief, "build_gm_brief",
                        lambda ctx, rid, actions=None: brief)
    bundle = {"ctx": {"league": {}}, "roster": {"roster_id": "7"},
              "league_id": "L1", "platform": "fleaflicker",
              "season": 2026}
    pack, _tools = evidence.build_gm_directive(bundle)
    return pack


def test_coverage_directive_fires_from_qb_coverage(monkeypatch):
    brief = _brief(_coverage([
        {"week": 12, "position": "QB", "count": 0, "required": 2,
         "causes": ["bye: Josh Allen"], "fills_action": None},
        {"week": 13, "position": "TE", "count": 0, "required": 1,
         "causes": [], "fills_action": None},
    ]))
    pack = _directive_pack(monkeypatch, brief)
    cov = [d for d in pack["directives"] if d["kind"] == "coverage"]
    assert [d["call"] for d in cov] == [
        "Fix your QB hole in Week 12", "Fix your TE hole in Week 13"]
    assert cov[0]["why"] == "Only 0 of 2 startable QB that week"
    assert cov[1]["why"] == "Only 0 of 1 startable TE that week"


def test_no_coverage_directive_without_holes(monkeypatch):
    pack = _directive_pack(monkeypatch, _brief(_coverage([])))
    assert [d for d in pack["directives"] if d["kind"] == "coverage"] == []
    # No coverage map at all (build_coverage returned None) likewise.
    pack = _directive_pack(monkeypatch, _brief(None))
    assert [d for d in pack["directives"] if d["kind"] == "coverage"] == []


# -- rookie draft best available ------------------------------------------

_SLOTS = {"1": "7", "2": "5", "3": "3", "4": "1", "5": "2", "6": "4",
          "7": "6", "8": "8", "9": "9", "10": "10", "11": "11", "12": "12"}
_VALUES = [
    {"id": "bijan", "value": 7450}, {"id": "allen", "value": 7300},
    {"id": "chase", "value": 9900}, {"id": "barkley", "value": 6800},
    {"id": "bowers", "value": 5400},
]
_INDEX = {
    "bijan": {"full_name": "Bijan Robinson", "position": "RB",
              "team": "ATL"},
    "allen": {"full_name": "Josh Allen", "position": "QB", "team": "BUF"},
    "chase": {"full_name": "Ja'Marr Chase", "position": "WR",
              "team": "CIN"},
    "barkley": {"full_name": "Saquon Barkley", "position": "RB",
                "team": "PHI"},
    "bowers": {"full_name": "Brock Bowers", "position": "TE",
               "team": "LV"},
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
_VETERAN_NAMES = {r["full_name"] for r in _INDEX.values()}


def _bundle():
    ctx = {"rosters": [{"roster_id": int(r), "players": []}
                       for r in _SLOTS.values()],
           "roster_map": {7: "Hurts So Good", 3: "Couch", 5: "Mustard"},
           "league": {"roster_positions": ["QB", "RB", "RB", "WR", "WR",
                                           "TE", "FLEX", "SUPER_FLEX"]}}
    return {"platform": "sleeper", "league_id": "L1", "season": 2026,
            "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
            "ctx": ctx,
            "roster": {"roster_id": 3, "players": [], "starters": []},
            "roster_player_ids": set()}


def _stub_draft(monkeypatch, *, rounds, status="pre_draft", picks=None):
    monkeypatch.setattr(evidence, "league_drafts",
                        lambda platform, league_id, season: [
                            {"draft_id": "D1", "status": status,
                             "season": 2026, "type": "snake"}])
    monkeypatch.setattr(evidence, "draft_detail", lambda draft_id: {
        "status": status, "season": 2026, "type": "snake",
        "settings": {"rounds": rounds, "teams": 12, "reversal_round": 0},
        "slot_to_roster_id": dict(_SLOTS)})
    monkeypatch.setattr(evidence, "draft_picks_live",
                        lambda draft_id: list(picks or []))
    monkeypatch.setattr(evidence, "league_traded_picks", lambda league_id: [])
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda: list(_VALUES))
    monkeypatch.setattr(evidence, "_index", lambda: dict(_INDEX))
    monkeypatch.setattr(evidence, "active_rookie_class", lambda: 2027)
    monkeypatch.setattr(evidence, "rookie_rankings",
                        lambda year: list(_ROOKIES))
    return _bundle()


def test_rookie_draft_best_available_is_the_rookie_pool(monkeypatch):
    pack, _tools = evidence.build_draft(
        _stub_draft(monkeypatch, rounds=4), {}, [])
    assert pack["draft_kind"] == "rookie"
    names = [r["name"] for r in pack["best_available"]]
    assert names == [r["name"] for r in _ROOKIES]
    # The pack's best_available IS the rookie pool, entry for entry.
    assert pack["best_available"] == pack["best_available_rookies"]
    assert not (set(names) & _VETERAN_NAMES)
    by_pos = pack["best_available_by_position"]
    assert [r["name"] for r in by_pos["QB"]] == ["Julian Sayin",
                                                 "Dante Moore"]
    assert [r["name"] for r in by_pos["RB"]] == ["Jeremiyah Love",
                                                 "Ahmad Hardy"]
    assert [r["name"] for r in by_pos["WR"]] == ["Jeremiah Smith"]
    assert by_pos.get("TE", []) == []  # no TE in the class fixture


def test_rookie_draft_drafted_rookie_leaves_the_pool(monkeypatch):
    pick = {"pick_no": 1, "round": 1, "roster_id": "7",
            "player_id": "r1",
            "metadata": {"first_name": "Jeremiah", "last_name": "Smith",
                         "position": "WR", "team": ""}}
    pack, _tools = evidence.build_draft(
        _stub_draft(monkeypatch, rounds=4, status="drafting",
                    picks=[pick]), {}, [])
    assert pack["draft_kind"] == "rookie"
    names = [r["name"] for r in pack["best_available"]]
    assert "Jeremiah Smith" not in names
    assert names[0] == "Julian Sayin"


def test_full_draft_best_available_stays_veteran(monkeypatch):
    pack, _tools = evidence.build_draft(
        _stub_draft(monkeypatch, rounds=15), {}, [])
    assert pack["draft_kind"] == "full"
    best = pack["best_available"]
    assert best[0]["name"] == "Ja'Marr Chase"
    assert best[0]["value"] == 9900
    assert {r["name"] for r in best} <= _VETERAN_NAMES
    assert pack["best_available_rookies"] == []
    by_pos = pack["best_available_by_position"]
    assert by_pos["RB"][0] == {"name": "Bijan Robinson", "value": 7450}
