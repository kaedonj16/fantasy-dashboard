"""Breadth boards + the stat-board team-row fix.

Defect (live, 2026-10-15): "Top 10 receiving yards leaders" ranked
TEAM_LAR, TEAM_DET, ... — the week-stat cache's team aggregate rows
(TEAM_<abbr> keys, no player record) flowed through stat_board_rows
because nothing required a real player record with a real position.
Raw stat boards now rank only players the index knows at one of the
stat's positions, on every family the stat path serves.

Feature: "who is rated in the top 10 of the most receiving metrics?"
is a breadth question — count each player's top-N finishes across the
metric family's Advanced Metrics boards and rank by the count. Built
deterministically in the leaderboard family (board_kind "breadth"),
with the catalog category as the family definition and the single
boards' PRO gating applied per metric.
"""
import json

import pytest

pytest.importorskip("pandas")
pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence as ev
from dashboard_services.assistant import tools as tools_mod
from dashboard_services.assistant.router import _keyword_route


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 3,
              "ctx": None, "roster": None, "roster_player_ids": set(),
              "has_pro": True}
    bundle.update(over)
    return bundle


# -- Defect A: team rows in the week-stat cache --------------------------------

STAT_INDEX = {
    "1": {"name": "Alpha Receiver", "team": "CIN", "position": "WR"},
    "2": {"name": "Beta Receiver", "team": "MIN", "position": "WR"},
    "3": {"name": "Gamma Back", "team": "IND", "position": "RB"},
    "4": {"name": "Delta Passer", "team": "BUF", "position": "QB"},
    "LAR": {"name": "Los Angeles Rams", "team": "LAR", "position": "DEF"},
}

# Team aggregate rows (TEAM_LAR out-totals every real receiver), a DST
# record with receiving stats, and a pid with no index record at all,
# interleaved with real players the way the production cache mixes them.
STAT_WEEKS = {
    1: {"TEAM_LAR": {"rec_yd": 900, "rec": 60, "rec_tgt": 85, "rec_td": 5,
                     "pass_yd": 850},
        "TEAM_DET": {"rec_yd": 880, "rec": 58, "rec_tgt": 80, "rec_td": 4},
        "LAR": {"rec_yd": 500, "rec": 30},
        "ghost": {"rec_yd": 700, "rec": 40},
        "1": {"rec_yd": 100, "rec": 8, "rec_tgt": 12, "rec_td": 1},
        "2": {"rec_yd": 90, "rec": 7, "rec_tgt": 10, "rec_td": 0},
        "3": {"rec_yd": 40, "rec": 5, "rec_tgt": 6, "rec_td": 1,
              "rush_yd": 80, "rush_att": 15},
        "4": {"pass_yd": 300, "pass_td": 2, "pass_att": 40, "pass_cmp": 28}},
    2: {"TEAM_LAR": {"rec_yd": 950, "rec": 62, "rec_tgt": 88, "rec_td": 4,
                     "pass_yd": 870},
        "1": {"rec_yd": 120, "rec": 9, "rec_tgt": 13, "rec_td": 2},
        "2": {"rec_yd": 70, "rec": 6, "rec_tgt": 9, "rec_td": 1},
        "3": {"rec_yd": 30, "rec": 4, "rec_tgt": 5, "rec_td": 0,
              "rush_yd": 60, "rush_att": 12},
        "4": {"pass_yd": 280, "pass_td": 3, "pass_att": 38, "pass_cmp": 25}},
}


@pytest.fixture()
def stat_base(monkeypatch):
    monkeypatch.setattr(ev, "get_players_index_global",
                        lambda *a, **k: dict(STAT_INDEX))
    monkeypatch.setattr(ev, "load_sleeper_week_stats",
                        lambda season, week: dict(STAT_WEEKS.get(week) or {}))
    return monkeypatch


def test_stat_board_receiving_excludes_team_rows(stat_base):
    rows, completed = ev.stat_board_rows(_bundle(), "rec_yd")
    assert completed == 2
    names = [r["name"] for r in rows]
    assert names == ["Alpha Receiver", "Beta Receiver", "Gamma Back"]
    assert rows[0]["value"] == 220
    assert not any("TEAM" in n or n == "Los Angeles Rams" for n in names)


def test_stat_board_passing_excludes_team_rows(stat_base):
    rows, _completed = ev.stat_board_rows(_bundle(), "pass_yd")
    assert [(r["name"], r["value"]) for r in rows] == [("Delta Passer", 580)]


def test_stat_board_rushing_players_only(stat_base):
    rows, _completed = ev.stat_board_rows(_bundle(), "rush_yd")
    assert [(r["name"], r["value"]) for r in rows] == [("Gamma Back", 140)]


def test_leaderboard_stat_board_has_no_teams(stat_base):
    pack, _tools = ev.build_leaderboard(
        _bundle(), {"board_kind": "board", "metric_hint": "receiving yards",
                    "top_n": 10}, [])
    assert pack["available"] is True and pack["stat_board"] is True
    assert [r["name"] for r in pack["board"]] == [
        "Alpha Receiver", "Beta Receiver", "Gamma Back"]


def test_tool_top_board_has_no_teams(stat_base):
    entry = tools_mod.run_tool("top_board", {"stat": "receiving yards",
                                             "top_n": 5}, _bundle())
    assert entry["ok"] is True
    assert [r["name"] for r in entry["board"]] == [
        "Alpha Receiver", "Beta Receiver", "Gamma Back"]


# -- Feature B: the breadth builder ---------------------------------------------

_GAMES_VOL = {"col": "games", "opts": [4, 8, 12, 16], "label": "Min Games"}
CONFIG = {
    "target_share": {"label": "Target Share", "category": "Receiving",
                     "positions": ["WR", "TE", "RB"], "min_vol": dict(_GAMES_VOL)},
    "rec_yards_per_game": {"label": "Rec Yds/G", "category": "Receiving",
                           "positions": ["WR", "TE", "RB"],
                           "min_vol": dict(_GAMES_VOL)},
    "air_yards_share": {"label": "Air Yards Share", "category": "Receiving",
                        "positions": ["WR", "TE"], "min_vol": dict(_GAMES_VOL)},
    "wopr": {"label": "WOPR", "category": "Receiving",
             "positions": ["WR", "TE", "RB"], "min_vol": dict(_GAMES_VOL)},
    "target_quality_score": {"label": "Target Quality", "category": "Receiving",
                             "positions": ["WR", "TE", "RB"], "hidden": True,
                             "min_vol": dict(_GAMES_VOL)},
    "rush_yards_per_game": {"label": "Rush Yds/G", "category": "Rushing",
                            "positions": ["RB", "QB"],
                            "min_vol": dict(_GAMES_VOL)},
    "yards_per_carry": {"label": "Yards/Carry", "category": "Rushing",
                        "positions": ["RB", "QB"], "min_vol": dict(_GAMES_VOL)},
    "snap_share": {"label": "Snap Share", "category": "General",
                   "positions": ["QB", "RB", "WR", "TE"],
                   "min_vol": dict(_GAMES_VOL)},
}


def _prow(pid, name, pos, value):
    return {"player_id": pid, "name": name, "team": "T" + pid[1:],
            "position": pos, "value": value, "games": 6}


# Boards are position-scoped (the WR board ranks WRs only), so on the
# receiving boards: Alpha finishes 1/2/1 across target share, yards/game
# and air yards, plus 1st in WOPR (PRO): 4 finishes, rank sum 5. Beta
# (2/1/2) and Zeta (3/3/3) tie at 3 finishes; Beta's rank sum 5 beats
# Zeta's 9. Gamma the RB and Delta the TE never appear on WR boards.
# The hidden metric would give Beta another finish; it must never count.
ROWS = {
    "target_share": [_prow("p1", "Alpha Receiver", "WR", 0.31),
                     _prow("p2", "Beta Receiver", "WR", 0.29),
                     _prow("p6", "Zeta Receiver", "WR", 0.27),
                     _prow("p3", "Gamma Back", "RB", 0.20)],
    "rec_yards_per_game": [_prow("p2", "Beta Receiver", "WR", 95.0),
                            _prow("p1", "Alpha Receiver", "WR", 92.0),
                            _prow("p4", "Delta Tight End", "TE", 60.0),
                            _prow("p6", "Zeta Receiver", "WR", 55.0)],
    "air_yards_share": [_prow("p1", "Alpha Receiver", "WR", 0.40),
                        _prow("p2", "Beta Receiver", "WR", 0.35),
                        _prow("p4", "Delta Tight End", "TE", 0.30),
                        _prow("p6", "Zeta Receiver", "WR", 0.28)],
    "wopr": [_prow("p3", "Gamma Back", "RB", 0.55),
             _prow("p1", "Alpha Receiver", "WR", 0.50)],
    "target_quality_score": [_prow("p2", "Beta Receiver", "WR", 88.0),
                             _prow("p9", "Ninth Receiver", "WR", 80.0)],
    "rush_yards_per_game": [_prow("p3", "Gamma Back", "RB", 88.0),
                            _prow("p5", "Epsilon Back", "RB", 70.0)],
    "yards_per_carry": [_prow("p3", "Gamma Back", "RB", 5.1),
                        _prow("p5", "Epsilon Back", "RB", 4.8)],
}


class _Policy:
    games_min = 4
    provisional = False

    def minimum(self, col, full):
        return full


@pytest.fixture()
def breadth_boards(monkeypatch):
    def fake_leaderboard(metric, position=None, limit=500, season=None,
                         min_vol=None, combine=False):
        rows = ROWS.get(metric, [])
        if position:
            rows = [r for r in rows if r["position"] == position]
        return list(rows)

    monkeypatch.setattr(ev, "metric_leaderboard", fake_leaderboard)
    monkeypatch.setattr(ev, "_metric_config", lambda: dict(CONFIG))
    monkeypatch.setattr(ev, "qualification_policy_for",
                        lambda season, week_start=None, week_end=None: _Policy())
    monkeypatch.setattr(ev, "pro_metric_keys", lambda: {"wopr"})
    monkeypatch.setattr(ev, "premium_metric_keys", lambda: set())
    monkeypatch.setattr(ev, "premium_metrics_exposed", lambda: False)
    monkeypatch.setattr(ev, "get_players_index_global", lambda *a, **k: {})
    return monkeypatch


def test_breadth_ranks_by_count_then_rank_sum(breadth_boards):
    pack, _tools = ev.build_leaderboard(
        _bundle(), {"board_kind": "breadth", "family": "receiving",
                    "top_n": 10}, [])
    assert pack["available"] is True and pack["board_kind"] == "breadth"
    assert pack["label"] == "Receiving metrics"
    assert pack["position_filter"] == "WR"
    # Hidden metric excluded from the family entirely; WOPR counted for PRO.
    assert pack["metric_count"] == 4 and pack["gated_metric_count"] == 0
    board = pack["board"]
    names = [r["name"] for r in board]
    assert names == ["Alpha Receiver", "Beta Receiver", "Zeta Receiver"]
    assert board[0]["finish_count"] == 4 and board[0]["rank_sum"] == 5
    assert board[1]["finish_count"] == 3 and board[1]["rank_sum"] == 5
    assert board[2]["finish_count"] == 3 and board[2]["rank_sum"] == 9
    assert pack["pool_size"] == 3
    top_finishes = {(f["label"], f["rank"]) for f in board[0]["top_finishes"]}
    assert ("Target Share", 1) in top_finishes
    assert ("WOPR", 1) in top_finishes
    assert "Basis: top-10 finishes counted across the 4 receiving metrics" \
        in pack["qualification_text"]


def test_breadth_free_caller_excludes_pro_metrics(breadth_boards):
    pack, _tools = ev.build_leaderboard(
        _bundle(has_pro=False),
        {"board_kind": "breadth", "family": "receiving", "top_n": 10}, [])
    assert pack["available"] is True
    assert pack["metric_count"] == 3 and pack["gated_metric_count"] == 1
    assert pack["gated_metrics"] == ["WOPR"]  # the name may be listed...
    blob = json.dumps(pack)
    assert "0.55" not in blob and "0.5" not in blob  # ...never its values
    assert all(f["metric_key"] != "wopr"
               for r in pack["board"] for f in r["top_finishes"])
    top = pack["board"][0]
    assert top["name"] == "Alpha Receiver" and top["finish_count"] == 3
    assert "not counted on your plan" in pack["qualification_text"]


def test_breadth_rushing_family_defaults_to_rb(breadth_boards):
    pack, _tools = ev.build_leaderboard(
        _bundle(), {"board_kind": "breadth", "family": "rushing",
                    "top_n": 10}, [])
    assert pack["label"] == "Rushing metrics"
    assert pack["position_filter"] == "RB"
    assert pack["metric_count"] == 2
    assert pack["board"][0]["name"] == "Gamma Back"
    assert pack["board"][0]["finish_count"] == 2
    assert pack["board"][0]["rank_sum"] == 2


def test_breadth_tool_matches_builder(breadth_boards):
    entry = tools_mod.run_tool("breadth_board", {"family": "receiving"},
                               _bundle())
    assert entry["ok"] is True and entry["family"] == "receiving"
    assert entry["board"][0]["name"] == "Alpha Receiver"
    assert entry["board"][0]["finish_count"] == 4
    assert entry["receipts"][0]["value"] == "top 10 in 4 metrics"
    assert "Basis:" in entry["basis"]


def test_breadth_compose_is_deterministic(breadth_boards, monkeypatch):
    pack, _tools = ev.build_leaderboard(
        _bundle(), {"board_kind": "breadth", "family": "receiving",
                    "top_n": 10}, [])
    pack["type"] = "leaderboard"
    pack["league"] = {"platform": "sleeper", "league_id": "L1",
                      "season": 2026, "league_name": "Blackedraw",
                      "team_name": "Couch"}
    monkeypatch.setattr(
        compose_mod, "call_model_json",
        lambda system, payload, schema, name: {
            "verdict": "MODEL VERDICT",
            "text": "Breadth at the top of the receiving board is real.",
            "receipts": [], "caveat": None,
            "followups": ["Who leads in target share?"], "act_link": None})
    ans = compose_mod.compose_answer(
        "leaderboard", "who is rated in the top 10 of the most receiving "
        "metrics?", pack)
    assert ans["verdict"] == "Alpha Receiver has the most top-10 receiving finishes"
    assert len(ans["receipts"]) == 3
    assert ans["receipts"][0]["label"] == "1. Alpha Receiver"
    assert "Target Share #1 (0.31)" in ans["receipts"][0]["value"]
    assert "top-10 finishes counted across the 4 receiving metrics" in ans["text"]


# -- routing ---------------------------------------------------------------------

def test_keyword_breadth_kaedon_phrasing():
    r = _keyword_route("who is rated in the top 10 of the most receiving metrics?")
    assert r["intent"] == "leaderboard"
    assert r["board_kind"] == "breadth"
    assert r["family"] == "receiving"
    assert r["top_n"] == 10


def test_keyword_breadth_rushing():
    r = _keyword_route("which running back is top 10 in the most rushing metrics?")
    assert r["board_kind"] == "breadth" and r["family"] == "rushing"
    assert r["position"] == "RB"


def test_keyword_most_finishes_phrasing():
    r = _keyword_route("who has the most top 10 finishes in receiving metrics?")
    assert r["board_kind"] == "breadth" and r["family"] == "receiving"


def test_keyword_receiving_stats_stays_a_stat_board():
    # The earlier live question: "most receiving stats" is the
    # receiving-yards season-total board, NOT a breadth question.
    r = _keyword_route("who is top 5 in the most receiving stats")
    assert r["intent"] == "leaderboard"
    assert r["board_kind"] == "board"
    assert r["metric_hint"] == "receiving stats"


def test_keyword_single_metric_board_not_stolen():
    r = _keyword_route("Who are the top 10 in target share?")
    assert r["board_kind"] == "board" and r["metric_hint"] == "target share"


def test_router_schema_carries_breadth_and_family():
    from dashboard_services.assistant import router
    assert "breadth" in router.SCHEMA["properties"]["board_kind"]["enum"]
    assert set(router.SCHEMA["properties"]["family"]["enum"]) >= {
        "receiving", "rushing", "passing", "fantasy"}
