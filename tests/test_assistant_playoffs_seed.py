"""Playoffs seed race: current seeds and first-seed odds in the pack.

Regression for the live 1-seed answer (2026-11-06): the viewer was the
2 seed in Blackedraw (6-2, PF 1252.96; Pittsburgh Pilots 7-1, PF
1259.88 held the 1 seed) and asked about the 1 seed, but the playoffs
pack carried no seed data at all and dropped the sim's first_seed_pct,
so the only wins number in reach was the projected cut-line figure (0)
and the answer stapled it onto the 1 seed ("You need 0 wins still
needed to get the 1 seed") while inventing "holding the current lead"
for a team that was chasing.

The pack now stamps current seeds from build_standings_map (the
standings machinery's seeds, never the odds sort), carries
first_seed_pct through, and adds a top_seed block with games/pf back
from the viewer's side. Fixtures are shaped like the real case.
Everything external is monkeypatched at the evidence module's lazy
shims or via scoped sys.modules stubs, like test_assistant_league_context.
"""
import sys
import types

import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence


def _bundle(ctx=None, **over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Caleb's Casting Couch",
              "week": 9,
              "ctx": ctx if ctx is not None else {},
              "roster": {"roster_id": 1, "players": [], "starters": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


def _stub_sim_modules(monkeypatch, *, weeks=5):
    spo = types.ModuleType("data_building.simulate_playoff_odds")
    spo.build_sim_state = lambda ctx, platform="sleeper": None
    spo._n_byes = lambda n: 2 if n >= 6 else 0
    spo.remaining_regular_season_weeks = (
        lambda odds, pws, current_week=0: weeks)
    ps = types.ModuleType("data_building.playoff_scenarios")
    ps.compute_scenarios = lambda teams, matchups, pt, **k: {"show": False, "teams": {}}
    ps.scenario_summary = lambda e: "Alive"
    monkeypatch.setitem(sys.modules, "data_building.simulate_playoff_odds", spo)
    monkeypatch.setitem(sys.modules, "data_building.playoff_scenarios", ps)
    monkeypatch.setattr(
        evidence, "playoff_odds_series",
        lambda league_id, season, roster_id: {"series": []})


# The real Blackedraw shape at Week 9: Pilots 7-1 PF 1259.88 (1 seed),
# Casting Couch 6-2 PF 1252.96 (2 seed), four more teams behind.
_ODDS_CHASING = [
    {"roster_id": 2, "wins": 7, "losses": 1, "ties": 0, "playoff_pct": 99.9,
     "bye_pct": 88.4, "first_seed_pct": 64.3, "avg_final_wins": 11.4,
     "is_complete": False},
    {"roster_id": 1, "wins": 6, "losses": 2, "ties": 0, "playoff_pct": 99.9,
     "bye_pct": 61.0, "first_seed_pct": 30.1, "avg_final_wins": 10.6,
     "is_complete": False},
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


def _team_stats(rows):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(rows)


def _seeded_ctx(*, me_first=False):
    if me_first:
        records = [("Caleb's Casting Couch", 7, 1, 1259.88),
                   ("Pittsburgh Pilots", 6, 2, 1252.96),
                   ("Gamma", 5, 3, 1180.40), ("Delta", 4, 4, 1105.10),
                   ("Epsilon", 3, 5, 1020.00), ("Zeta", 2, 6, 990.50)]
    else:
        records = [("Pittsburgh Pilots", 7, 1, 1259.88),
                   ("Caleb's Casting Couch", 6, 2, 1252.96),
                   ("Gamma", 5, 3, 1180.40), ("Delta", 4, 4, 1105.10),
                   ("Epsilon", 3, 5, 1020.00), ("Zeta", 2, 6, 990.50)]
    team_stats = _team_stats([
        {"owner": name, "Wins": w, "Losses": l, "Ties": 0, "PF": pf,
         "PA": 1000.0, "AVG": round(pf / (w + l), 1), "Streak": "W1"}
        for name, w, l, pf in records])
    return {"league_settings": {"playoff_teams": 6, "playoff_week_start": 15},
            "roster_map": dict(_ROSTER_MAP),
            "rosters": [], "team_stats": team_stats}


def _odds_for(me_first):
    odds = [dict(r) for r in _ODDS_CHASING]
    if me_first:
        by_rid = {r["roster_id"]: r for r in odds}
        by_rid[1].update({"wins": 7, "losses": 1, "first_seed_pct": 64.3})
        by_rid[2].update({"wins": 6, "losses": 2, "first_seed_pct": 30.1})
    return odds


# -- the pack -----------------------------------------------------------------

def test_playoffs_pack_carries_seeds_first_seed_pct_and_top_seed(monkeypatch):
    _stub_sim_modules(monkeypatch)
    monkeypatch.setattr(evidence, "playoff_sim_cached",
                        lambda ctx, platform: _odds_for(me_first=False))
    pack, tools = evidence.build_playoffs(_bundle(_seeded_ctx()), {}, [])
    assert pack["available"] is True
    by_rid = {r["roster_id"]: r for r in pack["field"]}
    # Seeds come from the standings machinery, not the odds sort.
    assert by_rid["2"]["seed"] == 1
    assert by_rid["1"]["seed"] == 2
    assert by_rid["6"]["seed"] == 6
    assert all(r["seed"] is not None for r in pack["field"])
    # The sim's first_seed_pct is carried through, not dropped.
    assert by_rid["1"]["first_seed_pct"] == 30.1
    assert by_rid["2"]["first_seed_pct"] == 64.3
    assert pack["me"]["seed"] == 2
    assert pack["me"]["first_seed_pct"] == 30.1
    # The 1-seed race: Pilots hold it, I am 1.0 games and 6.92 PF back.
    top = pack["top_seed"]
    assert top["team_name"] == "Pittsburgh Pilots"
    assert top["wins"] == 7 and top["losses"] == 1
    assert top["record"] == "7-1"
    assert top["pf"] == 1259.88
    assert top["is_me"] is False
    assert top["games_back"] == 1.0
    assert top["pf_back"] == 6.92
    assert "service.build_standings_map" in tools


def test_playoffs_top_seed_when_viewer_holds_it(monkeypatch):
    _stub_sim_modules(monkeypatch)
    monkeypatch.setattr(evidence, "playoff_sim_cached",
                        lambda ctx, platform: _odds_for(me_first=True))
    pack, _tools = evidence.build_playoffs(
        _bundle(_seeded_ctx(me_first=True)), {}, [])
    assert pack["me"]["seed"] == 1
    top = pack["top_seed"]
    assert top["team_name"] == "Caleb's Casting Couch"
    assert top["is_me"] is True
    assert top["games_back"] == 0.0
    assert top["pf_back"] == 0.0


def test_playoffs_without_standings_snapshot_omits_seeds(monkeypatch):
    _stub_sim_modules(monkeypatch)
    monkeypatch.setattr(evidence, "playoff_sim_cached",
                        lambda ctx, platform: _odds_for(me_first=False))
    ctx = {"league_settings": {"playoff_teams": 6, "playoff_week_start": 15},
           "roster_map": dict(_ROSTER_MAP), "rosters": []}
    pack, _tools = evidence.build_playoffs(_bundle(ctx), {}, [])
    assert pack["available"] is True
    assert all(r["seed"] is None for r in pack["field"])
    assert pack["me"]["seed"] is None
    assert pack["top_seed"] is None
    # first_seed_pct still rides: it comes from the sim, not the standings.
    assert pack["me"]["first_seed_pct"] == 30.1


# -- compose receipts ------------------------------------------------------------

def test_playoffs_receipts_gain_seed_and_first_seed_odds(monkeypatch):
    _stub_sim_modules(monkeypatch)
    monkeypatch.setattr(evidence, "playoff_sim_cached",
                        lambda ctx, platform: _odds_for(me_first=False))
    pack, _tools = evidence.build_playoffs(_bundle(_seeded_ctx()), {}, [])
    rows = compose_mod._required_receipts("playoffs", pack)
    by_label = {r["label"]: r["value"] for r in rows}
    assert by_label["Current seed"] == "2"
    assert by_label["First-seed odds (%)"] == "30.1"
    assert by_label["Your playoff odds (%)"] == "99.9"


def test_playoffs_receipts_unchanged_without_seed_data():
    pack = {"type": "playoffs",
            "me": {"playoff_pct": 71.5, "wins_still_needed": 4.8},
            "scenario": None, "weeks_remaining": 7,
            "odds_trajectory": None}
    labels = [r["label"] for r in compose_mod._required_receipts("playoffs", pack)]
    assert "Current seed" not in labels
    assert "First-seed odds (%)" not in labels
    assert labels == ["Your playoff odds (%)",
                      "Wins still needed (projected)",
                      "Weeks remaining"]


# -- standings fallback seeds are marked ---------------------------------------

def _standings_ctx(team_stats):
    return {"roster_map": {1: "Alpha", 3: "Couch", 2: "Beta"},
            "rosters": [], "team_stats": team_stats, "league_settings": {}}


def _standings_bundle(ctx):
    bundle = _bundle(ctx)
    bundle["roster"] = {"roster_id": 3, "players": [], "starters": []}
    bundle["team_name"] = "Couch"
    return bundle


def test_standings_real_seeds_are_not_marked_estimated():
    team_stats = _team_stats([
        {"owner": "Alpha", "Wins": 6, "Losses": 1, "Ties": 0, "PF": 845.2,
         "PA": 700.3, "AVG": 120.7, "Streak": "W4"},
        {"owner": "Couch", "Wins": 5, "Losses": 2, "Ties": 0, "PF": 812.4,
         "PA": 745.0, "AVG": 116.1, "Streak": "W1"},
        {"owner": "Beta", "Wins": 4, "Losses": 3, "Ties": 0, "PF": 775.9,
         "PA": 779.4, "AVG": 110.8, "Streak": "L2"},
    ])
    pack, _tools = evidence.build_standings(
        _standings_bundle(_standings_ctx(team_stats)), {}, [])
    assert pack["seeds_estimated"] is False
    assert [r["team_name"] for r in pack["teams"]] == ["Alpha", "Couch", "Beta"]


def test_standings_fallback_seeds_are_marked_estimated():
    # "Stranger" matches no roster in roster_map, so build_standings_map
    # leaves that team out and the real seed map is incomplete: the
    # builder falls back to stamping seeds from the wins/PF sort, and
    # those stamps must be marked as estimates, not real seeds.
    team_stats = _team_stats([
        {"owner": "Alpha", "Wins": 6, "Losses": 1, "Ties": 0, "PF": 845.2,
         "PA": 700.3, "AVG": 120.7, "Streak": "W4"},
        {"owner": "Couch", "Wins": 5, "Losses": 2, "Ties": 0, "PF": 812.4,
         "PA": 745.0, "AVG": 116.1, "Streak": "W1"},
        {"owner": "Beta", "Wins": 4, "Losses": 3, "Ties": 0, "PF": 775.9,
         "PA": 779.4, "AVG": 110.8, "Streak": "L2"},
        {"owner": "Stranger", "Wins": 3, "Losses": 4, "Ties": 0, "PF": 700.0,
         "PA": 800.0, "AVG": 100.0, "Streak": "L1"},
    ])
    pack, _tools = evidence.build_standings(
        _standings_bundle(_standings_ctx(team_stats)), {}, [])
    assert pack["seeds_estimated"] is True
    assert [r["seed"] for r in pack["teams"]] == [1, 2, 3, 4]
    assert pack["teams"][0]["team_name"] == "Alpha"
