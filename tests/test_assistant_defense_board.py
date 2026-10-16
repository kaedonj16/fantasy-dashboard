"""Team-defense boards: "what is Denver's defense ranked" and its
siblings. Team defenses are Sleeper pseudo-players whose weekly lines
sit in the week-stat store keyed by team abbreviation; the board ranks
them by season fantasy points scored with the league's own settings
(the week_stat_points rule every game-log surface uses), and a named
defense rides as the pack focus with its model-value rank added only
when the value table actually carries DEF rows.

The fixtures are adapter-shaped: real Sleeper defense stat lines
(sack / ff / fum_rec / pts_allow bucket / the published pts_* totals)
fed through the REAL week_stat_points, never a stub of it. Only the
store reads (week stats, players index, value rows) are monkeypatched,
at the evidence module's lazy shims, the deep-cuts pattern.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence, tools
from dashboard_services.assistant.router import _keyword_route


SCORING = {"rec": 0.5, "sack": 1.0, "int": 2.0, "fum_rec": 2.0,
           "def_td": 6.0, "pts_allow_0": 10.0, "pts_allow_1_6": 7.0,
           "pts_allow_7_13": 4.0, "pts_allow_14_20": 1.0,
           "pts_allow_21_27": 0.0, "pts_allow_28_34": -1.0,
           "pts_allow_35p": -4.0, "safe": 2.0, "blk_kick": 2.0}


def _def_line(pts, sack=3.0, pts_allow=17.0):
    """One week of a Sleeper team-defense line, in the feed's shape."""
    bucket = ("pts_allow_0" if pts_allow == 0 else
              "pts_allow_1_6" if pts_allow <= 6 else
              "pts_allow_7_13" if pts_allow <= 13 else
              "pts_allow_14_20" if pts_allow <= 20 else
              "pts_allow_21_27" if pts_allow <= 27 else
              "pts_allow_28_34" if pts_allow <= 34 else "pts_allow_35p")
    return {"sack": sack, "ff": 1.0, "fum_rec": 1.0, "td": 0.0, "gp": 1.0,
            "pts_allow": pts_allow, bucket: 1.0,
            "pts_ppr": float(pts), "pts_half_ppr": float(pts),
            "pts_std": float(pts), "pos_rank_ppr": 999.0,
            "pos_rank_half_ppr": 999.0, "pos_rank_std": 999.0}


# Weekly fantasy points per defense, weeks 1-6 (a realistic DST range).
POINTS = {
    "DEN": [16, 9, 12, 7, 14, 10],    # 68, 11.3 PPG -> rank 2
    "HOU": [13, 14, 9, 12, 10, 13],   # 71, 11.8 PPG -> rank 1
    "KC": [4, 8, 6, 9, 5, 7],         # 39, 6.5 PPG  -> rank 9
    "BAL": [9, 6, 11, 8, 10, 6],      # 50 -> rank 5 (name breaks PPG tie)
    "MIN": [13, 10, 7, 9, 11, 8],     # 58 -> rank 3
    "PHI": [8, 12, 10, 6, 9, 9],      # 54 -> rank 4
    "BUF": [7, 5, 9, 11, 6, 8],       # 46 -> rank 7
    "CLE": [12, 7, 5, 8, 13, 5],      # 50 -> rank 6
    "NYJ": [3, 6, 4, 7, 5, 6],        # 31 -> rank 10
    "MIA": [5, 4, 8, 3, 6, 4],        # 30 -> rank 11
    "ARI": [6, 8, 5, 7, 4, 9],        # 39 -> rank 8 (name breaks PPG tie)
    "TEN": [2, 5, 6, 4, 3, 5],        # 25 -> rank 12
}

POINTS_2025 = {  # a finished season, weeks 1-3: DEN 30 (rank 2)
    "DEN": [10, 12, 8], "HOU": [11, 12, 10], "KC": [6, 6, 6],
}

_NAMES = {"DEN": "Denver Broncos", "HOU": "Houston Texans",
          "KC": "Kansas City Chiefs", "BAL": "Baltimore Ravens",
          "MIN": "Minnesota Vikings", "PHI": "Philadelphia Eagles",
          "BUF": "Buffalo Bills", "CLE": "Cleveland Browns",
          "NYJ": "New York Jets", "MIA": "Miami Dolphins",
          "ARI": "Arizona Cardinals", "TEN": "Tennessee Titans"}

INDEX = {abbr: {"full_name": name, "team": abbr, "position": "DEF"}
         for abbr, name in _NAMES.items()}
INDEX["4034"] = {"full_name": "Christian McCaffrey", "team": "SF",
                 "position": "RB"}

VALUE_ROWS = {
    "HOU": {"value": 1600, "sf_value": 1620, "position": "DEF"},
    "DEN": {"value": 1450, "sf_value": 1500, "position": "DEF"},
    "MIN": {"value": 1200, "sf_value": 1210, "position": "DEF"},
    "KC": {"value": 900, "sf_value": 950, "position": "DEF"},
    "4034": {"value": 9500, "sf_value": 9800, "position": "RB"},
}


def _weeks(points_by_team):
    weeks = {}
    for pid, pts_list in points_by_team.items():
        for i, pts in enumerate(pts_list, 1):
            weeks.setdefault(i, {})[pid] = _def_line(pts)
    # A skill player's line rides in every week file and must never
    # leak onto a defense board.
    for i in list(weeks):
        weeks[i]["4034"] = {"pts_ppr": 24.0, "pts_half_ppr": 24.0,
                            "pts_std": 24.0, "gp": 1.0}
    return weeks


WEEKS = _weeks(POINTS)
WEEKS_2025 = _weeks(POINTS_2025)


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": {"raw_scoring_settings": dict(SCORING),
                      "rosters": [], "roster_positions": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


@pytest.fixture()
def world(monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global",
                        lambda: dict(INDEX))

    def fake_stats(season, week):
        if int(season) == 2025:
            return dict(WEEKS_2025.get(int(week), {}))
        return dict(WEEKS.get(int(week), {}))

    monkeypatch.setattr(evidence, "load_sleeper_week_stats", fake_stats)
    monkeypatch.setattr(evidence, "_value_rows", lambda: dict(VALUE_ROWS))
    return monkeypatch


# -- routing -----------------------------------------------------------------

def test_keyword_route_exact_question():
    r = _keyword_route("what is denver's defense ranked")
    assert r["intent"] == "leaderboard"
    assert r["board_kind"] == "defenses"
    assert r["team"] == "DEN"


def test_keyword_route_siblings():
    r = _keyword_route("how good is the chiefs defense")
    assert (r["intent"], r["board_kind"], r["team"]) == (
        "leaderboard", "defenses", "KC")
    r = _keyword_route("what is the broncos defense ranked")
    assert (r["board_kind"], r["team"]) == ("defenses", "DEN")
    r = _keyword_route("what is denver's dst ranked")
    assert (r["board_kind"], r["team"]) == ("defenses", "DEN")
    r = _keyword_route("rank the defenses")
    assert (r["intent"], r["board_kind"], r["team"]) == (
        "leaderboard", "defenses", None)
    r = _keyword_route("top 5 defenses this season")
    assert (r["board_kind"], r["top_n"]) == ("defenses", 5)


def test_keyword_route_guards_keep_their_paths():
    r = _keyword_route("should I start the Broncos defense or the "
                       "Chiefs defense")
    assert r["intent"] == "start_sit"
    r = _keyword_route("which defenses give up the most to running backs")
    assert (r["intent"], r["board_kind"]) == ("leaderboard", "matchup")
    r = _keyword_route("should I stream a defense this week")
    assert r["intent"] == "waivers"
    r = _keyword_route("should I trade for the Broncos defense")
    assert r["intent"] == "trade_eval"


# -- builder -----------------------------------------------------------------

def test_defense_board_ranks_by_season_points(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "defenses", "team": "DEN"})
    assert pack["available"] is True
    assert pack["pool_size"] == 12
    board = pack["board"]
    assert [r["name"] for r in board[:3]] == [
        "Houston Texans", "Denver Broncos", "Minnesota Vikings"]
    assert board[0]["value"] == 71.0 and board[1]["value"] == 68.0
    assert board[1]["ppg"] == 11.3 and board[1]["games"] == 6
    assert all(r["position"] == "DEF" for r in board)
    assert "4034" not in {r["player_id"] for r in board}
    assert "weeks 1 to 6" in pack["qualification_text"]


def test_defense_focus_carries_both_bases(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "defenses", "team": "DEN"})
    focus = pack["focus"]
    assert focus["name"] == "Denver Broncos"
    assert focus["rank"] == 2 and focus["pool_size"] == 12
    assert focus["value"] == 68.0 and focus["ppg"] == 11.3
    # The value basis rides only because the table carries DEF rows,
    # ranked among defenses alone (not the RB carrying a bigger value).
    assert focus["value_rank"] == 2 and focus["value_pool"] == 4
    assert focus["model_value"] == 1450


def test_defense_focus_beyond_the_shown_board(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "defenses", "team": "KC", "top_n": 3})
    assert pack["board_count"] == 3
    assert pack["focus"]["rank"] == 9
    assert pack["focus"]["name"] == "Kansas City Chiefs"


def test_defense_focus_without_value_rows(world, monkeypatch):
    monkeypatch.setattr(evidence, "_value_rows", lambda: {})
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "defenses", "team": "DEN"})
    focus = pack["focus"]
    assert focus["rank"] == 2
    assert "value_rank" not in focus and "model_value" not in focus


def test_defense_board_form_has_no_focus(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "defenses"})
    assert pack["available"] is True
    assert "focus" not in pack


def test_defense_board_no_completed_weeks(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(week=1), {"board_kind": "defenses", "team": "DEN"})
    assert pack["available"] is False
    assert pack["missing"] == "no completed games yet this season"


def test_defense_board_historical_season(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "defenses", "team": "DEN", "season": -1})
    assert pack["available"] is True
    assert pack["historical"] is True and pack["target_season"] == 2025
    assert pack["focus"]["rank"] == 2
    assert pack["focus"]["value"] == 30.0
    assert pack["qualification_text"].endswith("Final 2025 season board.")


# -- compose -----------------------------------------------------------------

def test_composed_answer_names_the_rank_and_bases(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "defenses", "team": "DEN"})
    ans = compose_mod.compose_answer(
        "leaderboard", "what is denver's defense ranked", pack)
    assert ans["verdict"] == (
        "Denver Broncos defense is #2 of 12 by fantasy points")
    assert "ranks #2 of 12 by fantasy points this season" in ans["text"]
    assert "68 points in 6 games (11.3 per game)" in ans["text"]
    assert "By model trade value it ranks #2 of 4 defenses." in ans["text"]
    assert len(ans["receipts"]) == pack["board_count"]
    assert ans["receipts"][0]["label"] == "1. Houston Texans"
    assert ans["receipts"][1]["value"] == "68 points (11.3 PPG)"


# -- open layer ---------------------------------------------------------------

def test_open_layer_deep_board_defenses(world):
    out = tools.run_tool("deep_board", {"kind": "defenses", "team": "DEN"},
                         _bundle())
    assert out["ok"] is True
    assert out["board_kind"] == "defenses"
    assert out["focus"]["rank"] == 2
    assert out["board"][0]["name"] == "Houston Texans"
