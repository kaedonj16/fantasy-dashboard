"""Top-scorer regression: "Who was the top scorer in 2024?" once answered
with the CURRENT season's board (Jahmyr Gibbs, 153.7 through Week 6 of
2026) because the phrasing resolved nowhere in the keyword router and
fell to the open layer, whose top_board tool had no season argument and
read the bundle's current season. The correct 2024 answer is Lamar
Jackson, 434.4 PPR points (Gibbs 7th, 362.9), from the site's own
Sleeper week-stat source.

These tests pin the whole chain: the phrasing family routes to the
season-total PPR board on both router paths, the builder is called with
the named season and realistic adapter-shaped rows (weekly-aggregate
leaderboard rows with season totals in the 300s and 400s, the shape
get_weekly_range_leaderboard returns), and the open-layer tool honors a
season argument instead of silently serving the current board.
Everything external is monkeypatched at the evidence module's lazy
shims, matching the history-boards and open-layer suites.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import aliases, evidence, router, tools
from dashboard_services.assistant.router import _keyword_route


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": {"raw_scoring_settings": {"rec": 1.0},
                      "rosters": [], "roster_positions": []},
              "roster": {"players": [], "starters": []},
              "roster_player_ids": {"p5"}}
    bundle.update(over)
    return bundle


_GAMES_VOL = {"col": "games", "opts": [4, 8, 12, 16], "label": "Min Games"}
CONFIG = {
    "ppr_pts": {"label": "PPR Points",
                "desc": "Total fantasy points for the period using "
                        "full-PPR scoring.",
                "positions": ["QB", "RB", "WR", "TE"],
                "min_vol": dict(_GAMES_VOL)},
}


class _Policy:
    games_min = 4
    provisional = False

    def minimum(self, col, full):
        return full


def _wrow(pid, name, team, pos, value, games):
    """One row in the REAL weekly-aggregate leaderboard shape
    (data_building.advanced_metrics.get_weekly_range_leaderboard):
    a season total summed from player_weekly_metrics, not a rate."""
    return {"player_id": pid, "name": name, "team": team, "position": pos,
            "value": value, "games": games, "vol": games, "weeks": games,
            "rec": None, "tgt": None, "car": None}


# The real 2024 PPR leaderboard (Sleeper pts_ppr summed over Weeks 1-18
# of the site's own week-stat cache): a season leader sits in the 400s.
ROWS_2024 = [
    _wrow("4881", "Lamar Jackson", "BAL", "QB", 434.4, 17),
    _wrow("7564", "Ja'Marr Chase", "CIN", "WR", 403.0, 17),
    _wrow("4984", "Josh Allen", "BUF", "QB", 385.0, 16),
    _wrow("6770", "Joe Burrow", "CIN", "QB", 381.8, 17),
    _wrow("4892", "Baker Mayfield", "TB", "QB", 381.8, 17),
    _wrow("11566", "Jayden Daniels", "WAS", "QB", 364.8, 17),
    _wrow("9221", "Jahmyr Gibbs", "DET", "RB", 362.9, 17),
    _wrow("4866", "Saquon Barkley", "PHI", "RB", 355.3, 16),
]


@pytest.fixture()
def boards(monkeypatch):
    calls = []

    def fake_leaderboard(metric, position=None, limit=500, season=None,
                         min_vol=None, combine=False):
        calls.append({"metric": metric, "season": season})
        rows = ROWS_2024 if season == 2024 else []
        if position:
            rows = [r for r in rows if r["position"] == position]
        return list(rows)

    monkeypatch.setattr(evidence, "metric_leaderboard", fake_leaderboard)
    monkeypatch.setattr(evidence, "_metric_config", lambda: dict(CONFIG))
    monkeypatch.setattr(evidence, "qualification_policy_for",
                        lambda season, week_start=None, week_end=None: _Policy())
    monkeypatch.setattr(evidence, "pro_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metrics_exposed", lambda: False)
    return calls


# -- the phrasing family resolves to the PPR points board ----------------------

@pytest.mark.parametrize("phrase", [
    "top scorer", "top scorers", "highest scorer", "leading scorer",
    "scored the most points", "most points scored",
    "led the league in scoring", "fantasy points", "most points",
])
def test_scoring_phrases_resolve_to_ppr_points(phrase):
    resolved = aliases.resolve_phrase(phrase)
    assert resolved is not None
    assert resolved["kind"] == "metric"
    assert resolved["key"] == "ppr_pts"


# -- keyword router: scorer family + season, standings guards ------------------

@pytest.mark.parametrize("question,season", [
    ("Who was the top scorer in 2024?", 2024),
    ("Who scored the most points in 2024?", 2024),
    ("Who led the league in scoring in 2024?", 2024),
    ("Who had the most points in 2024?", 2024),
    ("Who had the most fantasy points in 2024?", 2024),
    ("Who was the top scorer last season?", -1),
    ("Who is the highest scorer?", None),
    ("Who is the top scorer?", None),
])
def test_keyword_scorer_family_routes_to_board(question, season):
    r = _keyword_route(question)
    assert r["intent"] == "leaderboard"
    assert r["board_kind"] == "board"
    assert r["season"] == season
    resolved = aliases.resolve_phrase(str(r["metric_hint"]).replace("_", " "))
    assert resolved is not None and resolved["key"] == "ppr_pts"


@pytest.mark.parametrize("question", [
    "Who has the most points for in my league?",
    "Who has the most points?",
    "Which team is the highest scoring team?",
    "Who is in first place in the standings?",
    # The pinned team points-for reading: a league frame, no season.
    "Who has scored the most points in the league?",
])
def test_keyword_standings_asks_stay_standings(question):
    r = _keyword_route(question)
    assert r["intent"] == "standings"


def test_keyword_scorer_with_season_beats_league_frame():
    r = _keyword_route("Who scored the most points in the league in 2024?")
    assert r["intent"] == "leaderboard"
    assert r["season"] == 2024


def test_model_path_scorer_question(monkeypatch):
    def _fake(system, payload, schema, name):
        return {"intent": "leaderboard", "player_names": [],
                "give_player_names": [], "receive_player_names": [],
                "picks": [], "give_picks": [], "receive_picks": [],
                "other_team_hint": None, "metric_hint": "fantasy points",
                "season": 2024}
    monkeypatch.setattr(router, "call_model_json", _fake)
    out = router.classify("Who was the top scorer in 2024?", {}, None)
    assert out["intent"] == "leaderboard"
    assert out["entities"]["season"] == 2024
    # The router prompt itself names the scorer family for the model.
    assert "top scorer" in router.SYSTEM


# -- the builder is fed the named season and serves its totals -----------------

def test_builder_serves_2024_board_for_top_scorer(boards):
    routed = _keyword_route("Who was the top scorer in 2024?")
    pack, _tools = evidence.build_leaderboard(_bundle(), routed)
    assert boards == [{"metric": "ppr_pts", "season": 2024}]
    assert pack["available"] is True
    assert pack["historical"] is True
    assert pack["target_season"] == 2024
    assert pack["metric_key"] == "ppr_pts"
    top = pack["board"][0]
    assert top["name"] == "Lamar Jackson"
    assert top["value"] == 434.4
    by_name = {r["name"]: r for r in pack["board"]}
    assert by_name["Jahmyr Gibbs"]["value"] == 362.9
    assert by_name["Jahmyr Gibbs"]["rank"] == 7
    assert "Final 2024 season board." in pack["qualification_text"]
    # A finished season's board credits nobody to the current roster.
    assert all(r["on_my_roster"] is False for r in pack["board"])


def test_builder_current_season_scorer_uses_bundle_season(boards):
    routed = _keyword_route("Who is the top scorer?")
    pack, _tools = evidence.build_leaderboard(_bundle(), routed)
    assert boards == [{"metric": "ppr_pts", "season": 2026}]
    assert pack["historical"] is False


# -- the open-layer tool honors a season argument -------------------------------

@pytest.fixture()
def season_board(monkeypatch):
    calls = []

    def fake_season_board(key, spec, season, position):
        calls.append({"key": key, "season": season})
        rows = ROWS_2024 if season in (2024, 2026) else []
        return list(rows), {}

    monkeypatch.setattr(evidence, "_season_board", fake_season_board)
    monkeypatch.setattr(evidence, "_metric_config", lambda: dict(CONFIG))
    monkeypatch.setattr(evidence, "pro_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metrics_exposed", lambda: False)
    return calls


def test_top_board_tool_reads_the_named_season(season_board):
    entry = tools.run_tool(
        "top_board", {"stat": "fantasy points", "season": "2024"}, _bundle())
    assert season_board == [{"key": "ppr_pts", "season": 2024}]
    assert entry["ok"] is True
    assert entry["board"][0]["name"] == "Lamar Jackson"
    assert entry["board"][0]["value"] == 434.4
    assert entry["basis"].startswith("Final 2024 season board")
    assert entry["receipts"][0]["value"] == "434.4"
    assert all(r["on_my_roster"] is False for r in entry["board"])


def test_top_board_tool_defaults_to_current_season(season_board):
    entry = tools.run_tool(
        "top_board", {"stat": "fantasy points"}, _bundle())
    assert season_board == [{"key": "ppr_pts", "season": 2026}]
    assert entry["basis"].startswith("Season board,")


def test_top_board_tool_declines_a_future_season(season_board):
    entry = tools.run_tool(
        "top_board", {"stat": "fantasy points", "season": "2027"}, _bundle())
    assert entry["ok"] is False
    assert season_board == []
