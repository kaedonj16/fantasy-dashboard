"""Playoff Path extension of the assistant playoffs pack.

Inside the scenarios window (scen["show"]) the pack gains the win/lose
fork, the root-for games, and the PF tiebreaker picture, all copied off
the raw rows of the same cached sim run build_playoffs already reads —
plus the router shapes and compose behaviors that surface them. Outside
the window the pack must stay byte-identical to the pre-Path shape: no
fork key on me, no root_for / tiebreakers keys at all.

Fixtures follow test_assistant_playoffs_seed: the real Blackedraw shape
(Pilots 7-1 PF 1259.88 on the 1 seed, Casting Couch 6-2 PF 1252.96 on
the 2 seed), with the viewer's sim row carrying the Path conditionals
the live sim packs since the Playoff Path build.
"""
import sys
import types

import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence
from dashboard_services.assistant.router import _keyword_route


def _bundle(ctx=None, **over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Caleb's Casting Couch",
              "week": 10,
              "ctx": ctx if ctx is not None else {},
              "roster": {"roster_id": 1, "players": [], "starters": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


_ODDS = [
    {"roster_id": 2, "wins": 7, "losses": 1, "ties": 0, "playoff_pct": 99.9,
     "bye_pct": 88.4, "first_seed_pct": 64.3, "avg_final_wins": 11.4,
     "is_complete": False},
    {"roster_id": 1, "wins": 6, "losses": 2, "ties": 0, "playoff_pct": 99.9,
     "bye_pct": 61.0, "first_seed_pct": 30.1, "avg_final_wins": 10.6,
     "is_complete": False,
     "impact_week": 10,
     "playoff_if_win": 99.9, "playoff_if_lose": 91.2,
     "bye_if_win": 82.0, "bye_if_lose": 38.0,
     "first_seed_if_win": 52.0, "first_seed_if_lose": 6.0,
     "swing": 8.7,
     "root_for": [
         {"a": 3, "b": 4, "playoff_if_a": 97.2, "playoff_if_b": 88.4,
          "bye_if_a": 66.0, "bye_if_b": 55.5},
         {"a": 5, "b": 6, "playoff_if_a": 99.9, "playoff_if_b": 99.0,
          "bye_if_a": None, "bye_if_b": None},
     ]},
    {"roster_id": 3, "wins": 5, "losses": 3, "ties": 0, "playoff_pct": 84.0,
     "bye_pct": 22.5, "first_seed_pct": 4.2, "avg_final_wins": 9.4,
     "is_complete": False},
    {"roster_id": 4, "wins": 4, "losses": 4, "ties": 0, "playoff_pct": 47.8,
     "bye_pct": 6.1, "first_seed_pct": 1.1, "avg_final_wins": 8.6,
     "is_complete": False},
    {"roster_id": 5, "wins": 3, "losses": 5, "ties": 0, "playoff_pct": 15.2,
     "bye_pct": 0.8, "first_seed_pct": 0.2, "avg_final_wins": 7.3,
     "is_complete": False},
    {"roster_id": 6, "wins": 2, "losses": 6, "ties": 0, "playoff_pct": 3.1,
     "bye_pct": 0.1, "first_seed_pct": 0.1, "avg_final_wins": 6.2,
     "is_complete": False},
]

_ROSTER_MAP = {1: "Caleb's Casting Couch", 2: "Pittsburgh Pilots",
               3: "Gamma", 4: "Delta", 5: "Epsilon", 6: "Zeta"}

_STATE_TEAMS = [
    {"roster_id": rid, "name": name, "wins": 0, "ties": 0, "pf": 0.0}
    for rid, name in _ROSTER_MAP.items()]

_ENTRY = {"status": "alive", "controls_destiny": True, "wins_to_clinch": 2,
          "clinch_if_win_next": False, "out_if_lose_next": False,
          "needs_help": False, "best_seed": 1, "worst_seed": 5,
          "next_game": {"week": 10, "opp": 2}}


def _ctx():
    pd = pytest.importorskip("pandas")
    records = [("Pittsburgh Pilots", 7, 1, 1259.88),
               ("Caleb's Casting Couch", 6, 2, 1252.96),
               ("Gamma", 5, 3, 1180.40), ("Delta", 4, 4, 1105.10),
               ("Epsilon", 3, 5, 1020.00), ("Zeta", 2, 6, 990.50)]
    team_stats = pd.DataFrame([
        {"owner": name, "Wins": w, "Losses": l, "Ties": 0, "PF": pf,
         "PA": 1000.0, "AVG": round(pf / (w + l), 1), "Streak": "W1"}
        for name, w, l, pf in records])
    return {"league_settings": {"playoff_teams": 6, "playoff_week_start": 15},
            "roster_map": dict(_ROSTER_MAP),
            "rosters": [], "team_stats": team_stats}


def _stub(monkeypatch, *, show):
    spo = types.ModuleType("data_building.simulate_playoff_odds")
    spo.build_sim_state = lambda ctx, platform="sleeper": {
        "teams": [dict(t) for t in _STATE_TEAMS],
        "matchups": {}, "playoff_teams": 6}
    spo._n_byes = lambda n: 2 if n >= 6 else 0
    spo.remaining_regular_season_weeks = (
        lambda odds, pws, current_week=0: 4)
    ps = types.ModuleType("data_building.playoff_scenarios")
    scen = ({"show": True, "teams": {1: dict(_ENTRY)}} if show
            else {"show": False, "teams": {}})
    ps.compute_scenarios = lambda teams, matchups, pt, **k: scen
    ps.scenario_summary = lambda e: "Alive"
    monkeypatch.setitem(sys.modules, "data_building.simulate_playoff_odds", spo)
    monkeypatch.setitem(sys.modules, "data_building.playoff_scenarios", ps)
    monkeypatch.setattr(evidence, "playoff_sim_cached",
                        lambda ctx, platform: [dict(r) for r in _ODDS])
    monkeypatch.setattr(evidence, "playoff_odds_series",
                        lambda league_id, season, roster_id: {"series": []})


# -- the pack, in window ---------------------------------------------------------

def test_path_detail_attached_in_window(monkeypatch):
    _stub(monkeypatch, show=True)
    pack, _tools = evidence.build_playoffs(_bundle(_ctx()), {}, [])
    assert pack["scenario"] is not None
    fork = pack["me"]["fork"]
    assert fork == {"impact_week": 10, "playoff_if_win": 99.9,
                    "playoff_if_lose": 91.2, "bye_if_win": 82.0,
                    "bye_if_lose": 38.0, "first_seed_if_win": 52.0,
                    "first_seed_if_lose": 6.0, "swing": 8.7}
    games = pack["root_for"]
    assert len(games) == 2
    top = games[0]
    assert top["a_name"] == "Gamma" and top["b_name"] == "Delta"
    assert top["root_for"] == "Gamma"
    assert top["playoff_if_rooted"] == 97.2
    assert top["playoff_if_other"] == 88.4
    assert top["delta"] == 8.8
    assert top["bye_if_rooted"] == 66.0 and top["bye_if_other"] == 55.5
    # Sorted by delta desc; a game with no bye split omits the bye keys.
    assert games[1]["root_for"] == "Epsilon" and games[1]["delta"] == 0.9
    assert "bye_if_rooted" not in games[1]
    tb = pack["tiebreakers"]
    # Wanted: the seeds around the viewer (1, 2, 3), the cut (6 in, 7
    # out — only 6 exists), and the bye line (seed 2). PF order desc.
    assert [(r["team_name"], r["seed"]) for r in tb["rows"]] == [
        ("Pittsburgh Pilots", 1), ("Caleb's Casting Couch", 2),
        ("Gamma", 3), ("Zeta", 6)]
    assert tb["rows"][0]["pf"] == 1259.88
    assert tb["my_pf_rank"] == 2


def test_pack_byte_shape_unchanged_out_of_window(monkeypatch):
    _stub(monkeypatch, show=False)
    pack, _tools = evidence.build_playoffs(_bundle(_ctx()), {}, [])
    assert pack["scenario"] is None
    assert "fork" not in pack["me"]
    assert "root_for" not in pack
    assert "tiebreakers" not in pack
    assert set(pack) == {
        "type", "available", "league", "week", "as_of", "field",
        "playoff_teams", "playoff_week_start", "current_week", "is_complete",
        "weeks_remaining", "cut_line", "projected_cut_wins", "me",
        "top_seed", "odds_trajectory", "odds_movers", "odds_soft",
        "margin", "scenario"}
    assert set(pack["me"]) == {
        "roster_id", "team_name", "wins", "losses", "ties", "record",
        "playoff_pct", "bye_pct", "first_seed_pct", "avg_final_wins",
        "seed", "wins_still_needed"}


def test_path_detail_fail_soft_without_standings(monkeypatch):
    # No standings snapshot: no seeds, no PF — tiebreakers stay None and
    # the fork (a sim product) still attaches.
    _stub(monkeypatch, show=True)
    ctx = {"league_settings": {"playoff_teams": 6, "playoff_week_start": 15},
           "roster_map": dict(_ROSTER_MAP), "rosters": []}
    pack, _tools = evidence.build_playoffs(_bundle(ctx), {}, [])
    assert pack["me"]["fork"]["playoff_if_lose"] == 91.2
    assert pack["tiebreakers"] is None
    assert pack["root_for"][0]["root_for"] == "Gamma"


# -- compose ----------------------------------------------------------------------

def _fork_pack():
    return {"type": "playoffs",
            "me": {"playoff_pct": 99.9, "seed": 2, "first_seed_pct": 30.1,
                   "bye_pct": 61.0, "wins_still_needed": 0.0,
                   "fork": {"impact_week": 10, "playoff_if_win": 99.9,
                            "playoff_if_lose": 91.2, "bye_if_win": 82.0,
                            "bye_if_lose": 38.0, "first_seed_if_win": 52.0,
                            "first_seed_if_lose": 6.0, "swing": 8.7}},
            "scenario": {"wins_to_clinch": 2},
            "weeks_remaining": 4, "odds_trajectory": None,
            "root_for": [{"a_name": "Gamma", "b_name": "Delta",
                          "root_for": "Gamma", "playoff_if_rooted": 97.2,
                          "playoff_if_other": 88.4, "delta": 8.8}]}


def test_receipts_fork_rows_lead_a_win_lose_question():
    rows = compose_mod._required_receipts(
        "playoffs", _fork_pack(), "What happens if I lose this week?")
    assert [r["label"] for r in rows] == [
        "Your playoff odds (%)", "Playoff odds if you win (%)",
        "Playoff odds if you lose (%)", "Current seed"]
    by_label = {r["label"]: r["value"] for r in rows}
    assert by_label["Playoff odds if you lose (%)"] == "91.2"


def test_receipts_fork_rows_displace_weeks_remaining():
    # A pack without seed data leaves a free receipt slot that the
    # weeks-remaining row used to take; in-window the fork takes it.
    pack = _fork_pack()
    pack["me"] = {"playoff_pct": 88.2, "bye_pct": 22.4,
                  "wins_still_needed": 1.1, "fork": pack["me"]["fork"]}
    rows = compose_mod._required_receipts(
        "playoffs", pack, "What do I need to make the playoffs?")
    labels = [r["label"] for r in rows]
    assert "Weeks remaining" not in labels
    assert labels == ["Your playoff odds (%)", "First-round bye odds (%)",
                      "Wins to clinch", "Playoff odds if you win (%)"]


def test_followup_root_for_chip_is_pack_gated():
    kept = compose_mod._clean_followups("playoffs", [], _fork_pack())
    assert kept[0]["ask"] == "Who should I root for this week?"
    pack = _fork_pack()
    pack["root_for"] = None
    kept = compose_mod._clean_followups("playoffs", [], pack)
    assert all(p["ask"] != "Who should I root for this week?" for p in kept)


# -- router ------------------------------------------------------------------------

@pytest.mark.parametrize("message,intent", [
    ("Who should I root for this week?", "playoffs"),
    ("Which other games should I be rooting for?", "playoffs"),
    ("What happens to my playoff odds if I lose this week?", "playoffs"),
    ("Can I still get the 1 seed?", "playoffs"),
    ("Can I still get a bye?", "playoffs"),
    ("Do I still have a shot at the first seed?", "playoffs"),
    # Controls: the seed table and settings shapes keep their owners.
    ("What seed am I right now?", "standings"),
    ("When do playoffs start?", "league_settings"),
    ("Who has the most points for?", "standings"),
])
def test_keyword_route_path_shapes(message, intent):
    assert _keyword_route(message)["intent"] == intent
