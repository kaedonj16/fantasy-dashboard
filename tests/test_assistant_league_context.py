"""P1 league-context types (workshop section 9): playoffs, power rankings,
standings, matchup, schedule.

Router disambiguation (keyword fallback), evidence builders wired to the
site's own engines (everything external monkeypatched at the evidence
module's lazy shims or, for the playoff sim/scenario modules, scoped
sys.modules stubs), and the compose citation contract + league act links.
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
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": ctx if ctx is not None else {},
              "roster": {"roster_id": 3, "players": [], "starters": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


# -- router disambiguation ------------------------------------------------------

@pytest.mark.parametrize("message,intent", [
    ("What are my playoff odds?", "playoffs"),
    ("Am I eliminated?", "playoffs"),
    ("What do I need to make the playoffs?", "playoffs"),
    ("Have I clinched a playoff spot?", "playoffs"),
    ("Where am I in the power rankings?", "power_rankings"),
    ("Who is the team to beat?", "power_rankings"),
    ("Power rankings this week?", "power_rankings"),
    ("What do the standings look like?", "standings"),
    ("Who has the most points for?", "standings"),
    ("Who leads my division?", "standings"),
    ("Break down my matchup this week", "matchup"),
    ("Who wins between Alpha and Beta?", "matchup"),
    ("What is my win probability this week?", "matchup"),
    ("Who has the easiest remaining schedule?", "schedule"),
    ("How hard is my playoff stretch?", "schedule"),
    ("How tough is my remaining schedule?", "schedule"),
    # Regression: the original intents keep their phrasings.
    ("Should I start Rome Odunze or Jordan Addison?", "start_sit"),
    ("Who should I add from waivers?", "waivers"),
    ("What should I do with my team this week?", "this_week"),
    ("Should I trade Odunze for Addison?", "trade_eval"),
])
def test_keyword_route_disambiguation(message, intent):
    assert _keyword_route(message)["intent"] == intent


def test_keyword_route_standings_is_not_power_rankings():
    # The two league tables must never cross-route.
    assert _keyword_route("Show me the standings")["intent"] == "standings"
    assert _keyword_route("Show me the power rankings")["intent"] == "power_rankings"
    assert _keyword_route("Who has the most points for in the standings?")["intent"] == "standings"


# -- playoffs ---------------------------------------------------------------------

def _stub_sim_modules(monkeypatch, *, state=None, scen=None, weeks=7):
    spo = types.ModuleType("data_building.simulate_playoff_odds")
    spo.build_sim_state = lambda ctx, platform="sleeper": state
    spo._n_byes = lambda n: 2 if n >= 6 else 0
    spo.remaining_regular_season_weeks = (
        lambda odds, pws, current_week=0: weeks)
    ps = types.ModuleType("data_building.playoff_scenarios")
    ps.compute_scenarios = lambda teams, matchups, pt, **k: scen or {"show": False, "teams": {}}
    ps.scenario_summary = lambda e: "Win and you're in" if e.get("clinch_if_win_next") else "Alive"
    monkeypatch.setitem(sys.modules, "data_building.simulate_playoff_odds", spo)
    monkeypatch.setitem(sys.modules, "data_building.playoff_scenarios", ps)


_ODDS = [
    {"roster_id": 1, "wins": 6, "losses": 1, "ties": 0, "playoff_pct": 93.6,
     "bye_pct": 41.2, "avg_final_wins": 11.2, "is_complete": False},
    {"roster_id": 3, "wins": 5, "losses": 2, "ties": 0, "playoff_pct": 71.5,
     "bye_pct": 18.2, "avg_final_wins": 9.8, "is_complete": False},
    {"roster_id": 2, "wins": 4, "losses": 3, "ties": 0, "playoff_pct": 47.8,
     "bye_pct": 7.1, "avg_final_wins": 8.9, "is_complete": False},
    {"roster_id": 4, "wins": 2, "losses": 5, "ties": 0, "playoff_pct": 12.3,
     "bye_pct": 0.5, "avg_final_wins": 7.0, "is_complete": False},
]


def _playoff_ctx():
    return {"league_settings": {"playoff_teams": 2, "playoff_week_start": 15},
            "roster_map": {1: "Alpha", 2: "Beta", 3: "Couch", 4: "Delta"},
            "rosters": []}


def test_playoffs_pack_from_cached_sim(monkeypatch):
    _stub_sim_modules(monkeypatch, weeks=7)
    monkeypatch.setattr(evidence, "playoff_sim_cached", lambda ctx, platform: list(_ODDS))
    pack, tools = evidence.build_playoffs(_bundle(_playoff_ctx()), {}, [])
    assert pack["available"] is True
    assert [r["team_name"] for r in pack["field"]] == ["Alpha", "Couch", "Beta", "Delta"]
    assert pack["me"]["team_name"] == "Couch"
    assert pack["me"]["playoff_pct"] == 71.5
    # Cut line is the last team in (2 playoff spots): Couch, 9.8 projected wins.
    assert pack["cut_line"]["last_in"]["team_name"] == "Couch"
    assert pack["projected_cut_wins"] == 9.8
    assert pack["me"]["wins_still_needed"] == 4.8
    assert pack["weeks_remaining"] == 7
    assert pack["odds_soft"] is False
    assert pack["scenario"] is None
    assert "simulate_playoff_odds.simulate_playoff_odds" in tools


def test_playoffs_early_season_flags_soft_odds(monkeypatch):
    _stub_sim_modules(monkeypatch, weeks=12)
    monkeypatch.setattr(evidence, "playoff_sim_cached", lambda ctx, platform: list(_ODDS))
    pack, _tools = evidence.build_playoffs(_bundle(_playoff_ctx(), week=2), {}, [])
    assert pack["odds_soft"] is True
    assert pack["margin"]["thin"] is True


def test_playoffs_scenario_line_when_window_open(monkeypatch):
    entry = {"status": "alive", "controls_destiny": True, "wins_to_clinch": 1,
             "clinch_if_win_next": True, "out_if_lose_next": False,
             "needs_help": False, "best_seed": 2, "worst_seed": 7,
             "next_game": {"week": 14, "opp": 1}}
    state = {"teams": [{"roster_id": 3, "name": "Couch", "wins": 8, "ties": 0, "pf": 900.0},
                       {"roster_id": 1, "name": "Alpha", "wins": 9, "ties": 0, "pf": 950.0}],
             "matchups": {}, "playoff_teams": 2}
    _stub_sim_modules(monkeypatch, state=state,
                      scen={"show": True, "teams": {3: entry}}, weeks=1)
    monkeypatch.setattr(evidence, "playoff_sim_cached", lambda ctx, platform: list(_ODDS))
    pack, tools = evidence.build_playoffs(_bundle(_playoff_ctx(), week=13), {}, [])
    assert pack["scenario"]["summary"] == "Win and you're in"
    assert pack["scenario"]["wins_to_clinch"] == 1
    assert pack["scenario"]["next_opponent"] == "Alpha"
    assert "playoff_scenarios.compute_scenarios" in tools


def test_playoffs_no_odds_is_no_data(monkeypatch):
    _stub_sim_modules(monkeypatch)
    monkeypatch.setattr(evidence, "playoff_sim_cached", lambda ctx, platform: [])
    pack, _tools = evidence.build_playoffs(_bundle(_playoff_ctx()), {}, [])
    assert pack["available"] is False


def _sim_must_not_run(ctx, platform):
    raise AssertionError("the current-season sim must not run for a declined season")


def test_playoffs_last_season_declines_without_running_the_sim(monkeypatch):
    # 2026-10-22 audit fold-in: the playoffs branch extracted the season
    # but build_playoffs never resolved it, so "last season" silently
    # served the CURRENT sim. The keyword router's -1 sentinel (and the
    # model path's corrected year) must decline at the builder instead.
    _stub_sim_modules(monkeypatch)
    monkeypatch.setattr(evidence, "playoff_sim_cached", _sim_must_not_run)
    pack, tools = evidence.build_playoffs(
        _bundle(_playoff_ctx()), {"season": -1}, [])
    assert pack["available"] is False
    assert pack["missing"] == ("playoff odds are only tracked for the current season, "
                               "so past-season odds are not available")
    assert pack["requested_season"] == 2025
    assert pack["historical"] is True
    assert tools == []
    ans = compose_mod.compose_answer(
        "playoffs", "What were my playoff odds last season?", pack)
    assert "Playoff odds are only tracked for the current season" in ans["text"]
    assert "71.5" not in ans["text"]


@pytest.mark.parametrize("season", [2024, 2027])
def test_playoffs_non_current_explicit_season_declines(monkeypatch, season):
    _stub_sim_modules(monkeypatch)
    monkeypatch.setattr(evidence, "playoff_sim_cached", _sim_must_not_run)
    pack, _tools = evidence.build_playoffs(
        _bundle(_playoff_ctx()), {"season": season}, [])
    assert pack["available"] is False
    assert pack["requested_season"] == season
    assert pack["historical"] is (season < 2026)


def test_playoffs_current_season_entity_still_builds(monkeypatch):
    _stub_sim_modules(monkeypatch, weeks=7)
    monkeypatch.setattr(evidence, "playoff_sim_cached", lambda ctx, platform: list(_ODDS))
    pack, _tools = evidence.build_playoffs(
        _bundle(_playoff_ctx()), {"season": 2026}, [])
    assert pack["available"] is True
    assert pack["me"]["playoff_pct"] == 71.5


# -- power rankings -----------------------------------------------------------------

def _pr_teams(rank_shift=0):
    base = [
        {"roster_id": 1, "team_name": "Alpha", "rank": 1, "power_score": 88.1,
         "wins": 6, "losses": 1, "avg": 120.7, "all_play_pct": 0.792,
         "playoff_pct": 93.6, "momentum_label": "Rising"},
        {"roster_id": 3, "team_name": "Couch", "rank": 2, "power_score": 78.4,
         "wins": 5, "losses": 2, "avg": 116.1, "all_play_pct": 0.679,
         "playoff_pct": 71.5, "momentum_label": "Rising"},
        {"roster_id": 2, "team_name": "Beta", "rank": 3, "power_score": 70.5,
         "wins": 4, "losses": 3, "avg": 110.8, "all_play_pct": 0.571,
         "playoff_pct": 47.8, "momentum_label": "Steady"},
    ]
    for t in base:
        t["rank"] += rank_shift
    return base


class _FakeSeries(list):
    def __eq__(self, other):
        return [x == other for x in self]

    def tolist(self):
        return list(self)


class _FakeDF:
    empty = False

    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, key):
        if isinstance(key, str):
            return _FakeSeries([r[key] for r in self._rows])
        return _FakeDF([r for r, keep in zip(self._rows, key) if keep])


def test_power_rankings_pack_and_movement(monkeypatch):
    monkeypatch.setattr(evidence, "power_rankings_context",
                        lambda ctx: {"teams": _pr_teams(1 if ctx.get("_capped") else 0),
                                     "season_phase": "in_season"})
    monkeypatch.setattr(evidence, "standings_as_of_week",
                        lambda ctx, week: {"_capped": True, "team_stats": None})
    ctx = {"roster_map": {1: "Alpha", 2: "Beta", 3: "Couch"},
           "df_weekly": _FakeDF([{"week": 5, "finalized": True},
                                 {"week": 6, "finalized": True}])}
    pack, tools = evidence.build_power_rankings(_bundle(ctx), {}, [])
    assert [r["team_name"] for r in pack["teams"]] == ["Alpha", "Couch", "Beta"]
    assert pack["leader"]["team_name"] == "Alpha"
    assert pack["me"]["rank"] == 2 and pack["me"]["power_score"] == 78.4
    # Last week Couch was rank 3 (shifted), so the movement is +1.
    assert pack["me"]["rank_change"] == 1
    assert "ai.context_builders.build_power_rankings_context" in tools


# -- standings -----------------------------------------------------------------------

def test_standings_pack_uses_finalize_machinery(monkeypatch):
    pd = pytest.importorskip("pandas")
    team_stats = pd.DataFrame([
        {"owner": "Alpha", "Wins": 6, "Losses": 1, "Ties": 0, "PF": 845.2,
         "PA": 700.3, "AVG": 120.7, "Streak": "W4"},
        {"owner": "Couch", "Wins": 5, "Losses": 2, "Ties": 0, "PF": 812.4,
         "PA": 745.0, "AVG": 116.1, "Streak": "W1"},
        {"owner": "Beta", "Wins": 4, "Losses": 3, "Ties": 0, "PF": 775.9,
         "PA": 779.4, "AVG": 110.8, "Streak": "L2"},
    ])
    ctx = {"roster_map": {1: "Alpha", 3: "Couch", 2: "Beta"},
           "rosters": [], "team_stats": team_stats, "league_settings": {}}
    pack, tools = evidence.build_standings(_bundle(ctx), {}, [])
    assert [r["team_name"] for r in pack["teams"]] == ["Alpha", "Couch", "Beta"]
    assert pack["teams"][0]["seed"] == 1
    assert pack["me"]["record"] == "5-2" and pack["me"]["pf"] == 812.4
    assert pack["points_leader"] == {"team_name": "Alpha", "pf": 845.2}
    assert pack["has_divisions"] is False
    assert "service.finalize_team_stats" in tools


# -- matchup ---------------------------------------------------------------------------

def _team(rid, name, pids):
    return {"roster_id": rid, "name": name, "record": "5-2",
            "starters": [{"pid": pid, "name": f"P{pid}", "pos": "RB", "pts": None}
                         for pid in pids],
            "pts_total": None, "proj_total": None}


def test_matchup_pack_win_prob_and_targeting(monkeypatch):
    matchups = [
        {"matchup_id": 1, "left": _team(3, "Couch", ["a1", "a2"]),
         "right": _team(1, "Alpha", ["b1", "b2"])},
        {"matchup_id": 2, "left": _team(2, "Beta", ["c1"]),
         "right": _team(4, "Delta", ["d1"])},
    ]
    proj = {"a1": 20.0, "a2": 10.0, "b1": 15.0, "b2": 5.0, "c1": 12.0, "d1": 9.0}
    monkeypatch.setattr(evidence, "live_matchups",
                        lambda *a: (matchups, {}, proj))
    pack, tools = evidence.build_matchup(_bundle({"roster_map": {}}), {}, [])
    m = pack["matchup"]
    assert m["is_mine"] is True
    assert m["team_a"]["team_name"] == "Couch"  # viewer's side oriented first
    assert m["team_a"]["projected"] == 30.0
    assert m["team_b"]["projected"] == 20.0
    assert m["team_a"]["win_prob_pct"] > 50.0
    assert abs(m["team_a"]["win_prob_pct"] + m["team_b"]["win_prob_pct"] - 100.0) < 0.2
    assert len(pack["board"]) == 2
    assert "matchups.compute_win_prob" in tools
    # Named pairing: Beta vs Delta routes to the other game.
    pack2, _ = evidence.build_matchup(
        _bundle({"roster_map": {}}), {"team_names": ["Beta", "Delta"]}, [])
    assert pack2["matchup"]["team_a"]["team_name"] == "Beta"
    assert pack2["matchup"]["is_mine"] is False


def test_matchup_no_games_is_no_data(monkeypatch):
    monkeypatch.setattr(evidence, "live_matchups", lambda *a: ([], {}, {}))
    pack, _tools = evidence.build_matchup(_bundle({"roster_map": {}}, week=18), {}, [])
    assert pack["available"] is False
    assert "no scheduled matchup" in pack["missing"]


# -- schedule ----------------------------------------------------------------------------

def _schedule_inputs():
    rosters = [{"roster_id": rid, "players": []} for rid in (1, 2, 3, 4)]
    points = {1: 130.0, 2: 110.0, 3: 100.0, 4: 80.0}
    mbw = {}
    for w in range(1, 18):
        rows = []
        for mid, (a, b) in enumerate([(1, 2), (3, 4)], start=1):
            rows.append({"roster_id": a, "matchup_id": mid,
                         "points": points[a] if w <= 7 else 0.0})
            rows.append({"roster_id": b, "matchup_id": mid,
                         "points": points[b] if w <= 7 else 0.0})
        mbw[w] = rows
    return rosters, mbw


def test_schedule_pack_easiest_hardest_and_stretch(monkeypatch):
    pytest.importorskip("pandas")
    monkeypatch.setattr(evidence, "projected_starter_avgs", lambda *a, **k: {})
    rosters, mbw = _schedule_inputs()
    ctx = {"rosters": rosters, "matchups_by_week": mbw,
           "roster_map": {1: "Alpha", 2: "Beta", 3: "Couch", 4: "Delta"},
           "league_settings": {"playoff_week_start": 15}}
    pack, tools = evidence.build_schedule(_bundle(ctx), {}, [])
    assert pack["available"] is True
    idx = [r["opp_index"] for r in pack["teams"]]
    assert idx == sorted(idx)
    # Couch (100/week) and Delta (80) face each other all season: Couch's
    # slate is the softest, Alpha's (vs 110 Beta) the hardest of the four.
    assert pack["me"]["team_name"] == "Couch"
    assert pack["easiest"]["opp_index"] <= pack["me"]["opp_index"] <= pack["hardest"]["opp_index"]
    stretch = pack["playoff_stretch"]
    assert stretch["weeks"] == [15, 17]
    assert [o["opponent"] for o in stretch["me"]["opponents"]] == ["Delta"] * 3
    assert "service.remaining_schedule_strength" in tools


# -- compose: citation contract + league act links ---------------------------------------

_FROZEN = None


def _frozen_question(qid):
    global _FROZEN
    if _FROZEN is None:
        import json
        from pathlib import Path
        doc = json.loads((Path(__file__).resolve().parent.parent
                          / "assistant_eval" / "questions.json").read_text())
        _FROZEN = {q["id"]: q for q in doc["questions"]}
    return _FROZEN[qid]


def _frozen_pack(qid):
    return _frozen_question(qid)["evidence_pack"]


@pytest.mark.parametrize("qid,href_part,label,key_numbers", [
    ("po1", "/sleeper/2026/blackedraw/dashboard", "View Playoff Odds", ["71.5", "7"]),
    ("pr1", "/sleeper/2026/blackedraw/standings", "View Power Rankings", ["78.4", "88.1"]),
    ("st1", "/sleeper/2026/blackedraw/standings", "View standings", ["812.4", "845.2"]),
    ("mu1", "/sleeper/2026/blackedraw/weekly", "View Matchups", ["46.2", "118.6", "121.3"]),
    ("sc1", "/sleeper/2026/blackedraw/schedule", "Open in Schedule Assistant", ["99.8", "91.2"]),
])
def test_compose_league_context_contract(monkeypatch, qid, href_part, label, key_numbers):
    monkeypatch.setattr(compose_mod, "call_model_json", lambda *a, **k: {
        "verdict": "Here is the picture", "text": "Details follow in the receipts.",
        "receipts": [], "caveat": None, "followups": [], "act_link": None})
    q = _frozen_question(qid)
    answer = compose_mod.compose_answer(q["type"], q["question"], q["evidence_pack"])
    assert answer["act_link"]["href"] == href_part
    # The label names the destination surface, never a generic "Open".
    assert answer["act_link"]["label"] == label
    cited = {str(r["value"]) for r in answer["receipts"]}
    for kn in key_numbers:
        assert kn in cited, f"{qid}: key number {kn} not in receipts {cited}"
    assert 2 <= len(answer["receipts"]) <= 4


def test_compose_matchup_no_data_links_weekly_page():
    answer = compose_mod.no_data_answer(_frozen_pack("mu3"))
    assert answer["verdict"] is None
    assert "No scheduled matchup" in answer["text"]
    assert answer["act_link"]["href"] == "/sleeper/2026/blackedraw/weekly"
    assert answer["act_link"]["label"] == "View Matchups"


# -- matchup required mentions (live eval round 3: mu4) ---------------------------

def test_matchup_required_mentions_name_both_teams():
    assert compose_mod._required_mentions("matchup", _frozen_pack("mu4")) == [
        ("Caleb's Casting Couch", "matchup_a"), ("Gibbs and Take", "matchup_b")]


def test_matchup_mention_injected_when_draft_skips_the_opponent():
    data = {"verdict": "Caleb's Casting Couch favored",
            "text": "Caleb's Casting Couch projects 156.7 against 96.7.",
            "caveat": None, "receipts": [], "followups": []}
    out = compose_mod._ensure_required_mentions(
        "matchup", _frozen_pack("mu4"), data, [])
    assert "Gibbs and Take" in out["text"]
    assert "The matchup is Caleb's Casting Couch against Gibbs and Take." \
        in out["text"]
