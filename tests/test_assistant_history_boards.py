"""History boards + playoff-odds trajectory tests.

Past-season boards (a season entity through both router paths, builders
reading that season's own data, honest declines where a source has no
historical form) and the playoffs pack's recorded-odds trajectory.
Everything external is monkeypatched at the evidence module's lazy
shims, matching the deep-cuts and league-context test suites.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence, router
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
    "target_share": {"label": "Target Share", "desc": "Share of team pass attempts.",
                     "positions": ["WR", "TE", "RB"], "min_vol": dict(_GAMES_VOL)},
}


class _Policy:
    games_min = 4
    provisional = False

    def minimum(self, col, full):
        return full


def _prow(pid, name, team, pos, value, games=17):
    return {"player_id": pid, "name": name, "team": team, "position": pos,
            "value": value, "games": games}


ROWS_2024 = [_prow(f"h{i}", f"Hist {i}", "T" + str(i), "WR",
                   round(0.32 - i * 0.01, 2)) for i in range(1, 13)]


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


# -- router: season entity on both paths -----------------------------------------

def test_keyword_board_carries_explicit_season():
    r = _keyword_route("Who were the top 10 in target share in 2024?")
    assert r["intent"] == "leaderboard"
    assert r["season"] == 2024


def test_keyword_last_season_is_the_sentinel():
    r = _keyword_route("Who had the most receiving yards last season?")
    assert r["intent"] == "leaderboard"
    assert r["season"] == -1


def test_keyword_class_year_is_not_a_season_board():
    r = _keyword_route("Who are the top rookies in the 2027 class?")
    assert r["intent"] == "rookie"
    assert r.get("season") in (None, -1) or r["intent"] == "rookie"


def test_keyword_current_season_board_has_no_season():
    r = _keyword_route("Who are the top 10 in target share?")
    assert r["season"] is None


def test_keyword_playoffs_trajectory_phrasing():
    r = _keyword_route("How have my playoff odds moved this season?")
    assert r["intent"] == "playoffs"


def test_model_path_passes_season_through(monkeypatch):
    def _fake(system, payload, schema, name):
        return {"intent": "leaderboard", "player_names": [],
                "give_player_names": [], "receive_player_names": [],
                "picks": [], "give_picks": [], "receive_picks": [],
                "other_team_hint": None, "metric_hint": "target share",
                "season": 2024}
    monkeypatch.setattr(router, "call_model_json", _fake)
    out = router.classify("Who were the top 10 in target share in 2024?", {}, None)
    assert out["intent"] == "leaderboard"
    assert out["entities"]["season"] == 2024
    assert "season" in router.SCHEMA["properties"]


def test_target_season_resolution():
    assert evidence._target_season({"season": 2026}, {"season": 2024}) == (2024, True)
    assert evidence._target_season({"season": 2026}, {"season": -1}) == (2025, True)
    assert evidence._target_season({"season": 2026}, {"season": 2026}) == (2026, False)
    assert evidence._target_season({"season": 2026}, {}) == (2026, False)
    assert evidence._target_season({"season": 2026}, {"season": None}) == (2026, False)


# -- builders: historical boards read the named season ----------------------------

def test_metric_board_builds_from_the_named_season(boards):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share", "season": 2024})
    assert pack["available"] is True
    assert pack["historical"] is True
    assert pack["target_season"] == 2024
    assert boards and all(c["season"] == 2024 for c in boards)
    assert pack["board"][0]["name"] == "Hist 1"
    # Nobody is "on my roster" in a finished season.
    assert all(r["on_my_roster"] is False for r in pack["board"])
    assert "2024" in pack["qualification_text"]


def test_current_board_unchanged_without_a_season(boards):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share"})
    assert pack["historical"] is False
    assert pack["target_season"] is None


def test_last_season_sentinel_resolves_in_the_builder(boards):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share", "season": -1})
    assert pack["historical"] is True
    assert pack["target_season"] == 2025


def test_value_board_declines_for_a_past_season(boards):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low", "season": 2024})
    assert pack["available"] is False
    assert pack["historical"] is True
    # Sentence-shaped marker: the no-data composer speaks it as-is.
    assert "_" not in pack["missing"]
    assert len(pack["missing"].split()) >= 4


def test_matchup_board_declines_for_a_past_season(boards):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "matchup", "season": 2024})
    assert pack["available"] is False
    assert "current season" in pack["missing"]


def test_future_season_declines(boards):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share", "season": 2027})
    assert pack["available"] is False
    assert "not been played" in pack["missing"]


def test_stat_board_reads_the_named_seasons_weeks(monkeypatch):
    seen = []

    def fake_stats(season, week):
        seen.append((season, week))
        if season == 2024 and 1 <= week <= 17:
            return {"p1": {"rec_yd": 100.0}, "p2": {"rec_yd": 50.0}}
        return {}

    monkeypatch.setattr(evidence, "load_sleeper_week_stats", fake_stats)
    monkeypatch.setattr(evidence, "_index",
                        lambda: {"p1": {"full_name": "Hist One", "position": "WR",
                                        "team": "BUF"},
                                 "p2": {"full_name": "Hist Two", "position": "WR",
                                        "team": "MIA"}})
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "receiving yards", "season": 2024})
    assert pack["available"] is True
    assert pack["stat_board"] is True
    assert pack["board"][0]["name"] == "Hist One"
    assert pack["board"][0]["value"] == 1700
    assert all(s == 2024 for s, _w in seen)
    assert "2024" in pack["qualification_text"]


def test_historical_finishes_board_reads_that_seasons_game_logs(monkeypatch):
    weeks = [{"p1": (30.0, "WR"), "p2": (10.0, "WR")}] * 10
    monkeypatch.setattr(evidence, "_weekly_scored",
                        lambda bundle: (weeks, bundle["week"] - 1))
    monkeypatch.setattr(evidence, "_index", lambda: {
        "p1": {"full_name": "Hist One", "position": "WR", "team": "BUF"},
        "p2": {"full_name": "Hist Two", "position": "WR", "team": "MIA"}})
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "finishes", "season": 2024})
    assert pack["available"] is True
    assert pack["board"][0]["name"] == "Hist One"
    assert pack["board"][0]["value"] == 10
    assert "2024" in pack["qualification_text"]


def test_historical_breadth_counts_that_seasons_finishes(monkeypatch):
    config = {"target_share": {"label": "Target Share", "desc": "x",
                               "category": "Receiving",
                               "positions": ["WR", "TE", "RB"],
                               "min_vol": dict(_GAMES_VOL)},
              "wopr": {"label": "WOPR", "desc": "x", "category": "Receiving",
                       "positions": ["WR", "TE", "RB"],
                       "min_vol": dict(_GAMES_VOL)}}
    seen = []

    def fake_leaderboard(metric, position=None, limit=500, season=None,
                         min_vol=None, combine=False):
        seen.append(season)
        return list(ROWS_2024)

    monkeypatch.setattr(evidence, "metric_leaderboard", fake_leaderboard)
    monkeypatch.setattr(evidence, "_metric_config", lambda: config)
    monkeypatch.setattr(evidence, "qualification_policy_for",
                        lambda season, week_start=None, week_end=None: _Policy())
    monkeypatch.setattr(evidence, "pro_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metrics_exposed", lambda: False)
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "breadth", "family": "receiving",
                    "season": 2024})
    assert pack["available"] is True
    assert seen and all(s == 2024 for s in seen)
    assert "2024" in pack["qualification_text"]


def test_historical_value_board_reads_the_as_of_snapshot(monkeypatch):
    import dashboard_services.player_value_history as pvh
    calls = []

    def fake_as_of(as_of, **kw):
        calls.append(as_of)
        return {"p1": 7000.0, "p2": 6500.0}

    monkeypatch.setattr(pvh, "get_values_as_of", fake_as_of)
    monkeypatch.setattr(evidence, "_index", lambda: {
        "p1": {"full_name": "Hist One", "position": "RB", "team": "BUF"},
        "p2": {"full_name": "Hist Two", "position": "RB", "team": "MIA"}})
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda: [])
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "value", "season": 2024})
    assert pack["available"] is True
    assert calls == ["2025-02-15"]
    assert pack["board"][0]["name"] == "Hist One"
    assert pack["board"][0]["value"] == 7000.0
    assert "end of the 2024 season" in pack["qualification_text"]


def test_historical_value_board_without_snapshots_declines(monkeypatch):
    import dashboard_services.player_value_history as pvh
    monkeypatch.setattr(pvh, "get_values_as_of", lambda as_of, **kw: {})
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "value", "season": 2024})
    assert pack["available"] is False
    assert "reach back" in pack["missing"]


@pytest.mark.parametrize("kind", ["slate", "rank_movers"])
def test_current_only_boards_decline_for_a_past_season(boards, kind):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": kind, "metric_hint": "target share",
                    "season": 2024})
    assert pack["available"] is False
    assert "current season" in pack["missing"]


def test_historical_decline_is_not_the_empty_value_board_answer():
    # compose_answer routes an available=False buy_low pack to the
    # closest-miss treatment unless the pack is historical; a past-season
    # ask must take the plain no-data shape instead.
    pack = {"type": "leaderboard", "available": False, "board_kind": "buy_low",
            "historical": True, "league": {"platform": "sleeper",
                                            "league_id": "L1", "season": 2026},
            "missing": "buy low and sell high boards only cover the current season",
            "nearest": []}
    ans = compose_mod.compose_answer("leaderboard",
                                     "Who were the biggest buy-low candidates in 2024?",
                                     pack)
    assert ans["verdict"] is None
    assert "current season" in ans["text"]


# -- playoffs trajectory ----------------------------------------------------------

_SERIES = {
    "1": [{"week": 2, "pct": 88.0}, {"week": 7, "pct": 93.6}],
    "3": [{"week": 2, "pct": 44.0}, {"week": 4, "pct": 55.0},
          {"week": 7, "pct": 71.5}],
    "2": [{"week": 2, "pct": 52.0}, {"week": 7, "pct": 47.8}],
    "4": [{"week": 2, "pct": 30.0}, {"week": 7, "pct": 12.3}],
}

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


def _stub_playoffs(monkeypatch):
    import sys
    import types
    spo = types.ModuleType("data_building.simulate_playoff_odds")
    spo.build_sim_state = lambda ctx, platform="sleeper": None
    spo._n_byes = lambda n: 2 if n >= 6 else 0
    spo.remaining_regular_season_weeks = lambda odds, pws, current_week=0: 7
    ps = types.ModuleType("data_building.playoff_scenarios")
    ps.compute_scenarios = lambda teams, matchups, pt, **k: {"show": False, "teams": {}}
    ps.scenario_summary = lambda e: "Alive"
    monkeypatch.setitem(sys.modules, "data_building.simulate_playoff_odds", spo)
    monkeypatch.setitem(sys.modules, "data_building.playoff_scenarios", ps)
    monkeypatch.setattr(evidence, "playoff_sim_cached",
                        lambda ctx, platform: list(_ODDS))
    monkeypatch.setattr(
        evidence, "playoff_odds_series",
        lambda league_id, season, roster_id: {
            "series": list(_SERIES.get(str(roster_id)) or []),
            "current": None, "delta": None})
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": _playoff_ctx(),
              "roster": {"roster_id": 3, "players": [], "starters": []},
              "roster_player_ids": set()}
    return bundle


def test_playoffs_pack_carries_the_trajectory(monkeypatch):
    pack, tools = evidence.build_playoffs(_stub_playoffs(monkeypatch), {}, [])
    traj = pack["odds_trajectory"]
    assert traj["team_name"] == "Couch"
    assert traj["start_pct"] == 44.0 and traj["start_week"] == 2
    assert traj["current_pct"] == 71.5 and traj["current_week"] == 7
    assert traj["delta"] == 27.5
    movers = pack["odds_movers"]
    assert movers["risers"][0]["team_name"] == "Couch"
    assert movers["risers"][0]["delta"] == 27.5
    assert movers["fallers"][0]["team_name"] == "Delta"
    assert "playoff_odds_history.get_series" in tools


def test_playoffs_without_recorded_series_has_no_trajectory(monkeypatch):
    bundle = _stub_playoffs(monkeypatch)
    monkeypatch.setattr(evidence, "playoff_odds_series",
                        lambda league_id, season, roster_id: {"series": []})
    pack, _tools = evidence.build_playoffs(bundle, {}, [])
    assert pack["odds_trajectory"] is None
    assert pack["odds_movers"] is None


def test_trajectory_start_odds_are_a_required_receipt():
    pack = {"type": "playoffs", "me": {"playoff_pct": 71.5,
                                       "wins_still_needed": 4.8},
            "scenario": None, "weeks_remaining": 7,
            "odds_trajectory": {"start_pct": 44.0, "start_week": 2}}
    rows = compose_mod._required_receipts("playoffs", pack)
    labels = [r["label"] for r in rows]
    assert "Your playoff odds (%)" in labels
    assert "Your playoff odds in Week 2 (%)" in labels
    start = next(r for r in rows if r["label"].startswith("Your playoff odds in"))
    assert start["value"] == "44"
    assert len(rows) <= 4
