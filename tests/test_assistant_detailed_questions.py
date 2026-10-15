"""Detailed-question boards: multi-step analytical asks composed from
site computations (game-log finishes / floors / ceilings, the value
board with age and pool filters, schedule slates, within-team gaps,
positional-rank movers) plus the Teams pack's breadth counts and
category sweep. Everything external is monkeypatched at the evidence
module's lazy shims, the deep-cuts pattern.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence, tools
from dashboard_services.assistant.router import _keyword_route


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 5,
              "ctx": {"raw_scoring_settings": {"rec": 1.0},
                      "rosters": [], "roster_positions": []},
              "roster": {"roster_id": 1, "players": [], "starters": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


def _p(name, team, pos, age):
    return {"full_name": name, "team": team, "position": pos, "age": age}


INDEX = {
    "rb1": _p("Alpha Back", "NYJ", "RB", 24),
    "rb2": _p("Bravo Back", "MIA", "RB", 26),
    "rb3": _p("Charlie Back", "BUF", "RB", 23),
    "rb4": _p("Delta Back", "KC", "RB", 29),
    "rb5": _p("Echo Back", "DAL", "RB", 25),
    "wr1": _p("Foxtrot Wide", "CIN", "WR", 25),
    "wr2": _p("Golf Wide", "CIN", "WR", 27),
    "wr3": _p("Hotel Wide", "DAL", "WR", 22),
    "wr4": _p("India Wide", "DAL", "WR", 28),
    "qb1": _p("Juliet Passer", "NYJ", "QB", 27),
    "qb2": _p("Kilo Passer", "PIT", "QB", 30),
    "te1": _p("Lima Tight", "NYJ", "TE", 26),
    "te2": _p("Mike Tight", "PIT", "TE", 29),
}

GAME_LOGS = {  # week -> pid -> points (scored by the fake below)
    1: {"rb1": 20.0, "rb2": 25.0, "rb3": 10.0, "rb4": 15.0, "rb5": 30.0},
    2: {"rb1": 22.0, "rb2": 21.0, "rb3": 19.0, "rb4": 5.0, "rb5": 28.0},
    3: {"rb1": 18.0, "rb2": 16.0, "rb3": 17.0, "rb4": 14.0},
    4: {"rb1": 24.0, "rb2": 12.0, "rb3": 11.0, "rb4": 20.0},
}

VALUE_ROWS = {
    "rb1": {"value": 5000, "sf_value": 5200, "position": "RB", "age": 24},
    "rb2": {"value": 4500, "sf_value": 4600, "position": "RB", "age": 26},
    "rb3": {"value": 3000, "sf_value": 3100, "position": "RB", "age": 23},
    "rb4": {"value": 2500, "sf_value": 2600, "position": "RB", "age": 29},
    "wr1": {"value": 6000, "sf_value": 6100, "position": "WR", "age": 25},
    "wr2": {"value": 2000, "sf_value": 2100, "position": "WR", "age": 27},
    "wr3": {"value": 3500, "sf_value": 3600, "position": "WR", "age": 22},
}


@pytest.fixture()
def world(monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global", lambda: dict(INDEX))
    monkeypatch.setattr(
        evidence, "load_sleeper_week_stats",
        lambda season, week: {pid: {"pts": pts}
                              for pid, pts in GAME_LOGS.get(week, {}).items()})
    import utils.projections as proj
    monkeypatch.setattr(proj, "week_stat_points",
                        lambda line, scoring, pos: float(line.get("pts", 0.0)))
    monkeypatch.setattr(evidence, "_value_rows", lambda: dict(VALUE_ROWS))
    return monkeypatch


# -- game-log boards ---------------------------------------------------------------

def test_finishes_board_counts_weekly_top_finishes(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "finishes", "position": "RB", "finish_n": 2})
    assert pack["available"] is True
    board = pack["board"]
    # rb5 owns the two best single weeks (30, 28) and takes a top-2 spot
    # both weeks, but with only 2 games the minimum sample keeps him off
    # the finishes board entirely. Finishes: Alpha 3 (weeks 2-4), then
    # Bravo, Charlie, and Delta on 1 apiece, PPG breaking the tie.
    assert [r["name"] for r in board] == ["Alpha Back", "Bravo Back",
                                          "Charlie Back", "Delta Back"]
    assert [r["finishes"] for r in board] == [3, 1, 1, 1]
    assert board[0]["value"] == 3
    assert "top 2" in pack["qualification_text"]


def test_floor_board_needs_a_real_sample(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "floor", "position": "RB"})
    board = pack["board"]
    assert board[0]["name"] == "Alpha Back" and board[0]["floor"] == 18.0
    assert all(r["games"] >= 4 for r in board)  # rb5 (2 games) excluded


def test_ceiling_board_ranks_best_week(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "ceiling", "position": "RB"})
    board = pack["board"]
    assert board[0]["name"] == "Echo Back" and board[0]["ceiling"] == 30.0
    assert board[1]["name"] == "Bravo Back" and board[1]["ceiling"] == 25.0


# -- value board + filters -----------------------------------------------------------

def test_value_board_ranks_by_displayed_value(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "value", "position": "RB"})
    assert [r["name"] for r in pack["board"]] == [
        "Alpha Back", "Bravo Back", "Charlie Back", "Delta Back"]
    assert pack["board"][0]["value"] == 5000
    assert "1QB" in pack["qualification_text"]


def test_value_board_age_filter(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "value", "position": "RB", "max_age": 25})
    assert [r["name"] for r in pack["board"]] == ["Alpha Back", "Charlie Back"]
    assert "aged 25 or younger" in pack["qualification_text"]


def test_value_board_pool_filters(world):
    bundle = _bundle()
    bundle["ctx"]["rosters"] = [
        {"roster_id": 1, "players": ["rb1", "wr1"]},
        {"roster_id": 2, "players": ["rb2"]}]
    bundle["roster_player_ids"] = {"rb1", "wr1"}
    pack, _t = evidence.build_leaderboard(
        bundle, {"board_kind": "value", "pool": "waivers"})
    names = [r["name"] for r in pack["board"]]
    assert "Alpha Back" not in names and "Bravo Back" not in names
    assert "Foxtrot Wide" not in names and "Hotel Wide" in names
    pack, _t = evidence.build_leaderboard(
        bundle, {"board_kind": "value", "pool": "roster"})
    assert [r["name"] for r in pack["board"]] == ["Foxtrot Wide", "Alpha Back"]


# -- metric-shaped detailed boards ---------------------------------------------------

_GAMES_VOL = {"col": "games", "opts": [4, 8, 12, 16], "label": "Min Games"}
CONFIG = {
    "target_share": {"label": "Target Share", "desc": "Share of team targets.",
                     "positions": ["WR", "TE", "RB"], "min_vol": dict(_GAMES_VOL)},
    "opportunity_share": {"label": "Opportunity Share", "desc": "Share of team opportunities.",
                          "positions": ["RB", "WR", "TE"], "min_vol": dict(_GAMES_VOL)},
}
TS_GAP_ROWS = [
    {"player_id": "wr1", "name": "Foxtrot Wide", "team": "CIN", "position": "WR", "value": 0.31, "games": 6},
    {"player_id": "wr2", "name": "Golf Wide", "team": "CIN", "position": "WR", "value": 0.22, "games": 6},
    {"player_id": "wr3", "name": "Hotel Wide", "team": "DAL", "position": "WR", "value": 0.28, "games": 6},
    {"player_id": "wr4", "name": "India Wide", "team": "DAL", "position": "WR", "value": 0.25, "games": 6},
]
OPP_ROWS = [
    {"player_id": "rb1", "name": "Alpha Back", "team": "NYJ", "position": "RB", "value": 0.30, "games": 6},
    {"player_id": "rb2", "name": "Bravo Back", "team": "MIA", "position": "RB", "value": 0.28, "games": 6},
    {"player_id": "rb3", "name": "Charlie Back", "team": "BUF", "position": "RB", "value": 0.26, "games": 6},
]
SCHED = {  # week -> games (home, away)
    5: [("MIA", "NYJ"), ("KC", "BUF")],
    6: [("NYJ", "BUF"), ("MIA", "KC")],
    7: [("BUF", "NYJ"), ("MIA", "CIN")],
    8: [("NYJ", "DAL"), ("BUF", "MIA")],
}
RB_RANKS = {"MIA": 2, "BUF": 10, "KC": 20, "DAL": 28, "NYJ": 5, "CIN": 15}


class _Policy:
    games_min = 4
    provisional = False

    def minimum(self, col, full):
        return full


@pytest.fixture()
def metric_world(world):
    rows = {"target_share": TS_GAP_ROWS, "opportunity_share": OPP_ROWS}

    def fake_leaderboard(metric, position=None, limit=500, season=None,
                         min_vol=None, combine=False):
        out = list(rows.get(metric, []))
        if position:
            out = [r for r in out if r["position"] == position]
        return out

    world.setattr(evidence, "metric_leaderboard", fake_leaderboard)
    world.setattr(evidence, "_metric_config", lambda: dict(CONFIG))
    world.setattr(evidence, "qualification_policy_for",
                  lambda season, week_start=None, week_end=None: _Policy())
    world.setattr(evidence, "pro_metric_keys", lambda: {"wopr"})
    world.setattr(evidence, "premium_metric_keys", lambda: set())
    world.setattr(evidence, "premium_metrics_exposed", lambda: False)
    world.setattr(evidence, "matchup_rank_table",
                  lambda season, pos: (dict(RB_RANKS), RB_RANKS))
    world.setattr(evidence, "load_week_sched",
                  lambda season, week: [{"home": h, "away": a}
                                        for h, a in SCHED.get(week, [])])
    return world


def test_gap_board_ranks_intra_team_gap(metric_world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "gap", "metric_hint": "target share",
                    "position": "WR"})
    board = pack["board"]
    assert board[0]["name"] == "Foxtrot Wide"
    assert board[0]["gap"] == 0.09
    assert board[0]["second_name"] == "Golf Wide"
    assert board[0]["second_value"] == 0.22
    assert board[1]["gap"] == 0.03


def test_slate_board_reranks_by_schedule_ease(metric_world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "slate", "metric_hint": "opportunity share",
                    "position": "RB"})
    board = pack["board"]
    # Metric order is rb1 > rb2 > rb3; the slate re-ranks it: rb3's next
    # four opponents average rank 8.0, rb1 and rb2 tie at 12.5 with the
    # metric value breaking the tie.
    assert [r["name"] for r in board] == ["Charlie Back", "Alpha Back",
                                          "Bravo Back"]
    assert board[0]["slate_avg_rank"] == 8.0
    assert board[0]["slate_games"] == 4
    assert board[1]["slate_avg_rank"] == 12.5


# -- rank movers -----------------------------------------------------------------

def test_rank_movers_compares_positional_ranks(world):
    world.setattr(evidence, "latest_snapshot_date", lambda: "2026-10-15")
    old = {"rb1": 3000.0, "rb2": 4000.0, "rb3": 4800.0, "rb4": 2000.0,
           "wr1": 5500.0, "wr3": 3000.0, "wr2": 1500.0}
    world.setattr(evidence, "values_as_of",
                  lambda as_of, league_type=None, league_size=None: dict(old))
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "rank_movers"})
    board = pack["board"]
    assert [r["name"] for r in board] == ["Alpha Back"]
    assert board[0]["climb"] == 2
    assert board[0]["old_rank"] == 3 and board[0]["new_rank"] == 1
    assert pack["window_days"] == 30


def test_rank_movers_honest_no_data_without_history(world):
    world.setattr(evidence, "latest_snapshot_date", lambda: "2026-10-15")
    world.setattr(evidence, "values_as_of",
                  lambda as_of, league_type=None, league_size=None: {})
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "rank_movers"})
    assert pack["available"] is False
    assert pack["missing"] == "value history is not available 30 days back"


# -- teams breadth + sweep ---------------------------------------------------------

TEAM_VALUES = {
    "qb1": 4000, "rb1": 5000, "rb2": 4500, "wr1": 6000, "te1": 2500,
    "qb2": 3000, "rb3": 3000, "wr3": 3500, "wr2": 2000, "te2": 1500,
}
TEAM_POS = {"qb1": "QB", "rb1": "RB", "rb2": "RB", "wr1": "WR", "te1": "TE",
            "qb2": "QB", "rb3": "RB", "wr3": "WR", "wr2": "WR", "te2": "TE"}


@pytest.fixture()
def teams_world(world):
    by_id = {pid: {"value": v, "sf_value": v, "position": TEAM_POS[pid],
                   "redraft_value_sf": v, "redraft_value_1qb": v}
             for pid, v in TEAM_VALUES.items()}
    world.setattr(evidence, "league_format_values",
                  lambda ctx, force_scoring_type=None: dict(by_id))

    def fake_rank(vals_by_roster, slot_counts, positions=None):
        strengths, ranks = {}, {pos: {} for pos in (positions or [])}
        for pos in (positions or []):
            ordered = sorted(
                vals_by_roster.items(),
                key=lambda kv: -sum(sorted(kv[1].get(pos) or [],
                                           reverse=True)[:2]))
            for i, (rid, vals) in enumerate(ordered):
                strengths.setdefault(rid, {})[pos] = sum(
                    sorted(vals.get(pos) or [], reverse=True)[:2])
                ranks[pos][rid] = i + 1
        return strengths, ranks

    world.setattr(evidence, "rank_positions", fake_rank)
    world.setattr(evidence, "_roster_grades",
                  lambda *a, **k: {"1": {"score": 88.0, "grade": "A"},
                                   "2": {"score": 74.0, "grade": "C"}})
    return world


def _teams_bundle():
    bundle = _bundle()
    bundle["ctx"]["rosters"] = [
        {"roster_id": 1, "owner_id": "u1",
         "players": ["qb1", "rb1", "rb2", "wr1", "te1"],
         "metadata": {"team_name": "Caleb's Casting Couch"}},
        {"roster_id": 2, "owner_id": "u2",
         "players": ["qb2", "rb3", "wr3", "wr2", "te2"],
         "metadata": {"team_name": "Pittsburgh Pilots"}},
    ]
    bundle["ctx"]["roster_positions"] = ["QB", "RB", "RB", "WR", "WR", "TE",
                                         "FLEX"]
    bundle["ctx"]["users"] = [{"user_id": "u1", "display_name": "Kaedon"},
                              {"user_id": "u2", "display_name": "Jayden"}]
    bundle["roster"] = {"roster_id": 1,
                        "players": ["qb1", "rb1", "rb2", "wr1", "te1"]}
    return bundle


def test_teams_breadth_counts_top_players(teams_world):
    pack, _t = evidence.build_teams(_teams_bundle(), {})
    top = pack["most_top_players"]
    # Top-24 per position covers every fixture player, so breadth is a
    # roster-size count here; the Couch carries 5, the Pilots 5, and the
    # tie breaks on team name.
    assert top["count"] == 5
    assert pack["breadth"][0]["by_position"]["RB"] == 2


def test_teams_sweep_compares_every_category(teams_world):
    pack, _t = evidence.build_teams(
        _teams_bundle(), {"team_names": ["Pittsburgh Pilots"]})
    sweep = pack["sweep"]
    assert sweep["team_a"]["team_name"] == "Caleb's Casting Couch"
    assert sweep["team_b"]["team_name"] == "Pittsburgh Pilots"
    cats = {c["category"]: c for c in sweep["categories"]}
    assert cats["QB room"]["winner"] == "a"
    assert cats["Roster grade"]["a_value"] == 88.0
    assert cats["Total roster value"]["a_value"] == 22000
    assert sweep["a_wins"] == 6 and sweep["b_wins"] == 0


def test_sweep_compose_is_deterministic(teams_world):
    pack, _t = evidence.build_teams(
        _teams_bundle(), {"team_names": ["Pittsburgh Pilots"]})
    ans = compose_mod.compose_answer("teams", "compare us", pack)
    assert ans["verdict"].startswith("Caleb's Casting Couch wins")
    assert "6-0" in ans["verdict"]
    assert "QB room" in ans["text"]


# -- router keyword fallback -----------------------------------------------------------

def test_keyword_detailed_boards():
    r = _keyword_route("Which RB has the most top-12 weeks this season?")
    assert r["intent"] == "leaderboard" and r["board_kind"] == "finishes"
    assert r["finish_n"] == 12 and r["position"] == "RB"
    r = _keyword_route("Who has the highest floor at WR?")
    assert r["board_kind"] == "floor" and r["position"] == "WR"
    r = _keyword_route("Top RBs under 25 by value")
    assert r["board_kind"] == "value" and r["max_age"] == 24
    assert r["position"] == "RB"
    r = _keyword_route("Who beats his team's No. 2 by the most target share?")
    assert r["board_kind"] == "gap" and r["metric_hint"] == "target share"
    r = _keyword_route("Who climbed the most in positional rank this month?")
    assert r["board_kind"] == "rank_movers"


def test_keyword_waiver_pool_board_not_waiver_advice():
    r = _keyword_route("Top receivers on waivers by target share")
    assert r["intent"] == "leaderboard" and r["pool"] == "waivers"
    assert r["metric_hint"] == "target share"


def test_keyword_waiver_advice_unchanged():
    r = _keyword_route("Who should I add from waivers this week?")
    assert r["intent"] == "waivers"


def test_keyword_teams_breadth_and_sweep():
    r = _keyword_route("Which team has the most top-24 players?")
    assert r["intent"] == "teams" and r["top_n"] == 24
    r = _keyword_route("Compare my team vs the Pilots across every category, who wins more?")
    assert r["intent"] == "teams"


# -- open-layer tool ----------------------------------------------------------------

def test_deep_board_tool_finishes(world):
    entry = tools.run_tool("deep_board", {"kind": "finishes", "position": "RB"},
                           _bundle())
    assert entry["ok"] is True
    assert entry["board"][0]["name"] == "Alpha Back"
    assert entry["board"][0]["value"] == 4
    assert entry["receipts"]


def test_deep_board_tool_rejects_unknown_kind(world):
    entry = tools.run_tool("deep_board", {"kind": "nope"}, _bundle())
    assert entry["ok"] is False
