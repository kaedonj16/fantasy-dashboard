"""Deep cuts tests: the leaderboard intent family.

Router keyword fallback, the evidence builder (boards computed from the
Advanced Metrics leaderboard layer with the AM route's qualification
minimums, PRO/PREMIUM gating parity, movers windows, matchup boards), and
the deterministic board composer (model phrases an intro only).
Everything external is monkeypatched at the evidence module's lazy shims.
"""
import json
import re

import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence
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
    "air_yards_share": {"label": "Air Yards Share", "desc": "Share of team air yards.",
                        "positions": ["WR", "TE"], "min_vol": dict(_GAMES_VOL)},
    "snap_share": {"label": "Snap Share", "desc": "Share of team snaps.",
                   "positions": ["RB", "WR", "TE"], "min_vol": dict(_GAMES_VOL)},
    "wopr": {"label": "WOPR", "desc": "Weighted opportunity rating.",
             "positions": ["WR", "TE", "RB"], "min_vol": dict(_GAMES_VOL)},
}


def _prow(pid, name, team, pos, value, games=6):
    return {"player_id": pid, "name": name, "team": team, "position": pos,
            "value": value, "games": games}


TS_ROWS = [_prow(f"p{i}", f"Player {i}", "T" + str(i), "WR", round(0.32 - i * 0.01, 2))
           for i in range(1, 13)]
TS_ROWS[4]["name"] = "Rome Odunze"
AYS_ROWS = [_prow("p2", "Player 2", "T2", "WR", 0.36),
            _prow("p1", "Player 1", "T1", "WR", 0.34),
            _prow("p9", "Player 9", "T9", "TE", 0.30),
            _prow("p5", "Rome Odunze", "T5", "WR", 0.25)]


class _Policy:
    games_min = 4
    provisional = False

    def minimum(self, col, full):
        return full


@pytest.fixture()
def boards(monkeypatch):
    calls = []

    def fake_leaderboard(metric, position=None, limit=500, season=None,
                         min_vol=None, combine=False):
        calls.append({"metric": metric, "position": position, "min_vol": min_vol})
        rows = {"target_share": TS_ROWS, "air_yards_share": AYS_ROWS,
                "wopr": TS_ROWS[:3]}.get(metric, [])
        if position:
            rows = [r for r in rows if r["position"] == position]
        return list(rows)

    monkeypatch.setattr(evidence, "metric_leaderboard", fake_leaderboard)
    monkeypatch.setattr(evidence, "_metric_config", lambda: dict(CONFIG))
    monkeypatch.setattr(evidence, "qualification_policy_for",
                        lambda season, week_start=None, week_end=None: _Policy())
    monkeypatch.setattr(evidence, "pro_metric_keys", lambda: {"wopr", "opportunity_trend"})
    monkeypatch.setattr(evidence, "premium_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metrics_exposed", lambda: False)
    return calls


# -- router keyword fallback --------------------------------------------------------

def test_keyword_top_n_is_a_board():
    r = _keyword_route("Who are the top 10 in target share?")
    assert r["intent"] == "leaderboard"
    assert r["board_kind"] == "board"
    assert r["top_n"] == 10
    assert r["metric_hint"] == "target share"


def test_keyword_player_metric_rank_stays_a_player_question():
    r = _keyword_route("What is Ja'Marr Chase's target share rank?")
    assert r["intent"] != "leaderboard"


def test_keyword_movers_and_matchup():
    r = _keyword_route("Who are the biggest risers in target share?")
    assert r["intent"] == "leaderboard" and r["board_kind"] == "movers"
    assert r["direction"] == "risers"
    r = _keyword_route("Which defenses give up the most to tight ends?")
    assert r["intent"] == "leaderboard" and r["board_kind"] == "matchup"
    assert r["position"] == "TE"


def test_keyword_intersection():
    r = _keyword_route("Who is in the top 10 in both target share and air yards share?")
    assert r["intent"] == "leaderboard" and r["board_kind"] == "intersection"
    assert r["metric_hint"] == "target share"
    assert r["second_metric_hint"] == "air yards share"


# -- season boards --------------------------------------------------------------------

def test_season_board_uses_am_pool_definition(boards):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share"})
    assert pack["available"] is True
    assert pack["pool_size"] == 12
    assert [r["rank"] for r in pack["board"]] == list(range(1, 11))
    assert pack["board"][0]["value"] == 0.31
    # The AM route's qualification minimum reached the leaderboard call.
    assert boards[0]["min_vol"] == 4
    assert "minimum 4 games" in pack["qualification_text"]
    assert pack["board"][4]["on_my_roster"] is True  # p5 is rostered


def test_position_filter_scopes_the_pool(boards):
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "air yards share", "position": "TE"})
    assert boards[0]["position"] == "TE"
    assert [r["name"] for r in pack["board"]] == ["Player 9"]
    assert pack["pool_size"] == 1


def test_window_falls_back_to_season_when_metric_has_no_weeks(boards, monkeypatch):
    monkeypatch.setattr(evidence, "weekly_metric_keys", lambda: {"target_share": {}})
    monkeypatch.setattr(evidence, "adv_weekly_supported", lambda m: False)
    monkeypatch.setattr(evidence, "situational_metric_keys", lambda: {})
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "air yards share", "window_weeks": 4})
    assert pack["available"] is True
    assert pack["window"]["applied"] is False
    assert pack["board"][0]["value"] == 0.36  # season board, not a range board


def test_window_applies_when_metric_is_range_capable(boards, monkeypatch):
    monkeypatch.setattr(evidence, "weekly_metric_keys", lambda: {"target_share": {}})
    monkeypatch.setattr(evidence, "adv_weekly_supported", lambda m: False)
    monkeypatch.setattr(evidence, "situational_metric_keys", lambda: {})

    def fake_range(metric, position=None, season=None, week_start=None,
                   week_end=None, min_vol=None, limit=500):
        assert (week_start, week_end) == (3, 6)  # last 4 of 6 completed weeks
        return [_prow("p1", "Player 1", "T1", "WR", 0.4)]

    monkeypatch.setattr(evidence, "weekly_range_leaderboard", fake_range)
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share", "window_weeks": 4})
    assert pack["window"]["applied"] is True
    assert pack["board"][0]["value"] == 0.4
    assert "Weeks 3 to 6 only" in pack["qualification_text"]


# -- intersections and dual ranks -------------------------------------------------------

def test_intersection_keeps_only_players_on_both_top_boards(boards):
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share",
                    "second_metric_hint": "air yards share",
                    "board_kind": "intersection", "top_n": 3})
    names = [r["name"] for r in pack["board"]]
    # TS top 3: p1, p2, p3. AYS top 3: p2, p1, p9. Common: p1, p2.
    assert names == ["Player 1", "Player 2"]
    first = pack["board"][0]
    assert first["value"] == 0.31
    assert first["secondary"] == {"value": 0.34, "rank": 2}


def test_dual_board_adds_second_metric_column(boards):
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share",
                    "second_metric_hint": "air yards share",
                    "board_kind": "dual", "top_n": 5})
    by_name = {r["name"]: r for r in pack["board"]}
    assert by_name["Rome Odunze"]["secondary"] == {"value": 0.25, "rank": 4}
    assert by_name["Player 3"]["secondary"] is None  # not on the AYS board


# -- movers ------------------------------------------------------------------------------

def _patch_movers(monkeypatch):
    monkeypatch.setattr(evidence, "weekly_metric_keys", lambda: {"target_share": {}})
    monkeypatch.setattr(evidence, "adv_weekly_supported", lambda m: False)
    monkeypatch.setattr(evidence, "situational_metric_keys", lambda: {})
    recent = [_prow("p1", "Player 1", "T1", "WR", 0.30),
              _prow("p2", "Player 2", "T2", "WR", 0.20)]
    base = [_prow("p1", "Player 1", "T1", "WR", 0.18),
            _prow("p2", "Player 2", "T2", "WR", 0.22)]

    def fake_range(metric, position=None, season=None, week_start=None,
                   week_end=None, min_vol=None, limit=500):
        return list(recent) if week_start == 3 else list(base)

    monkeypatch.setattr(evidence, "weekly_range_leaderboard", fake_range)


def test_movers_computes_deltas_against_baseline(boards, monkeypatch):
    _patch_movers(monkeypatch)
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share", "board_kind": "movers"})
    assert pack["available"] is True
    top = pack["board"][0]
    assert top["name"] == "Player 1"
    assert top["value"] == 0.30 and top["baseline_value"] == 0.18
    assert top["delta"] == 0.12
    assert pack["board"][1]["delta"] == -0.02
    assert "61" not in pack["qualification_text"]  # pool is the joined count
    assert "2 qualify" in pack["qualification_text"]


def test_movers_fallers_sort_ascending(boards, monkeypatch):
    _patch_movers(monkeypatch)
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "target share", "board_kind": "movers",
                    "direction": "fallers"})
    assert pack["board"][0]["name"] == "Player 2"
    assert pack["board"][0]["delta"] == -0.02


def test_movers_without_week_history_is_honest(boards, monkeypatch):
    monkeypatch.setattr(evidence, "weekly_metric_keys", lambda: {})
    monkeypatch.setattr(evidence, "adv_weekly_supported", lambda m: False)
    monkeypatch.setattr(evidence, "situational_metric_keys", lambda: {})
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "air yards share", "board_kind": "movers"})
    assert pack["available"] is False
    assert "week-by-week" in pack["missing"]
    assert "target_share" in pack["nearest"]


# -- matchup board -------------------------------------------------------------------------

def test_matchup_board_from_rank_table(boards, monkeypatch):
    monkeypatch.setattr(
        evidence, "matchup_rank_table",
        lambda season, pos: ({"JAX": 1, "ATL": 2, "CIN": 3}, 32,
                             {"JAX": {"fpts": 12.9}, "ATL": {"fpts": 12.4},
                              "CIN": {"fpts": 11.8}}, True))
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"board_kind": "matchup", "position": "TE"})
    assert pack["available"] is True
    assert [r["team"] for r in pack["board"]] == ["JAX", "ATL", "CIN"]
    assert pack["board"][0]["value"] == 12.9
    assert pack["pool_size"] == 32


# -- PRO / premium gating parity -------------------------------------------------------------

def test_pro_metric_gated_for_free_caller(boards):
    pack, _ = evidence.build_leaderboard(
        _bundle(has_pro=False), {"metric_hint": "wopr"})
    assert pack["available"] is False
    assert pack["gated"] is True
    assert pack["missing"] == "pro_only"
    assert "board" not in pack
    # No board numbers anywhere in the pack.
    assert "0.31" not in json.dumps(pack)


def test_pro_metric_board_for_pro_caller(boards):
    pack, _ = evidence.build_leaderboard(
        _bundle(has_pro=True), {"metric_hint": "wopr"})
    assert pack["available"] is True
    assert "gated" not in pack
    assert pack["board"][0]["value"] == 0.31


def test_free_metric_is_never_gated(boards):
    pack, _ = evidence.build_leaderboard(
        _bundle(has_pro=False), {"metric_hint": "target share"})
    assert pack["available"] is True and "gated" not in pack


def test_premium_unexposed_metric_is_unavailable(boards, monkeypatch):
    monkeypatch.setattr(evidence, "premium_metric_keys", lambda: {"target_share"})
    pack, _ = evidence.build_leaderboard(
        _bundle(has_pro=True), {"metric_hint": "target share"})
    assert pack["available"] is False
    assert pack["missing"] == "metric not available"
    assert "gated" not in pack


def test_routes_family_no_data(boards):
    pack, _ = evidence.build_leaderboard(
        _bundle(), {"metric_hint": "yards per route run"})
    assert pack["available"] is False
    assert pack["missing"] == "routes not tracked"
    assert "target_share" in pack["nearest"]


# -- player-in-pool framing ---------------------------------------------------------------------

def test_metric_pack_frames_player_in_qualified_pool(boards, monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: {})
    monkeypatch.setattr("data_building.advanced_metrics.get_metric_leaderboard",
                        lambda metric, **k: list(TS_ROWS))
    pack, _ = evidence.build_metric(
        _bundle(), {"metric_hint": "target share"}, player_ids=["p3"])
    assert pack["pool_size"] == 12
    assert pack["player_rank"]["rank"] == 3
    assert pack["player_rank"]["pool_size"] == 12
    assert pack["player_rank"]["percentile"] == round(100 * 9 / 11)


def test_player_pack_frames_target_share_pool(boards, monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global",
                        lambda *a, **k: {"p5": {"name": "Rome Odunze", "team": "CHI",
                                                "position": "WR"}})
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: [])
    monkeypatch.setattr(evidence, "_metrics_for",
                        lambda pid, season=None: ({"target_share": 0.27, "ppg": 15.0},
                                                  ["fake_metrics"]))
    monkeypatch.setattr("data_building.advanced_metrics.get_player_metric_ranks",
                        lambda pid, season=None: {})
    monkeypatch.setattr(evidence, "load_sleeper_week_stats", lambda *a, **k: {})
    monkeypatch.setattr(evidence, "load_week_sched", lambda *a, **k: [])
    pack, _ = evidence.build_player(_bundle(), {}, ["p5"])
    # Odunze is 5th on the target share board (position pool = full fake board).
    assert pack["pool_framing"]["rank"] == 5
    assert pack["pool_framing"]["pool_size"] == 12


# -- composer: the board is code, the model only frames -------------------------------------------

def _small_board_pack():
    return {"type": "leaderboard", "available": True, "board_kind": "board",
            "metric_key": "target_share", "label": "Target Share",
            "requested_n": 10, "pool_size": 12,
            "qualification_text": "Qualified pool: 12 players, minimum 4 games.",
            "board": [
                {"rank": 1, "player_id": "p1", "name": "Player 1", "team": "T1",
                 "position": "WR", "value": 0.31, "on_my_roster": False},
                {"rank": 2, "player_id": "p2", "name": "Player 2", "team": "T2",
                 "position": "WR", "value": 0.29, "on_my_roster": False},
                {"rank": 3, "player_id": "p5", "name": "Rome Odunze", "team": "T5",
                 "position": "WR", "value": 0.27, "on_my_roster": True}]}


def _canned(text, followups=None):
    def fake(system, payload, schema, name):
        return {"verdict": "MODEL VERDICT", "text": text, "receipts": [],
                "caveat": None, "followups": followups or [], "act_link": None}
    return fake


def test_compose_board_is_deterministic(monkeypatch):
    monkeypatch.setattr(compose_mod, "call_model_json",
                        _canned("Usage is concentrating at the top this season.",
                                ["Top target share for tight ends?"]))
    ans = compose_mod.compose_answer("leaderboard", "top target share",
                                     _small_board_pack())
    assert ans["verdict"] == "Player 1 leads Target Share"  # code, not model
    assert [r["label"] for r in ans["receipts"]] == ["1. Player 1", "2. Player 2",
                                                     "3. Rome Odunze"]
    assert [r["value"] for r in ans["receipts"]] == ["0.31", "0.29", "0.27"]
    assert "Usage is concentrating" in ans["text"]  # model intro, names nobody
    assert "2. Player 2 (T2, WR): 0.29" in ans["text"]
    assert "minimum 4 games" in ans["text"]
    assert ans["act_link"] == {"href": "/metrics?metric=target_share",
                               "label": "Open Advanced Metrics"}


def test_compose_board_link_is_league_scoped_with_position(monkeypatch):
    monkeypatch.setattr(compose_mod, "call_model_json",
                        _canned("Usage is concentrating at the top this season."))
    pack = _small_board_pack()
    pack["league"] = {"platform": "sleeper", "season": 2026,
                      "league_id": "1312067280816832512"}
    pack["position_filter"] = "WR"
    ans = compose_mod.compose_answer("leaderboard", "top WR target share", pack)
    assert ans["act_link"] == {
        "href": "/sleeper/2026/1312067280816832512/metrics?metric=target_share&pos=WR",
        "label": "Open Advanced Metrics"}


def test_compose_intro_naming_players_falls_back_to_template(monkeypatch):
    monkeypatch.setattr(compose_mod, "call_model_json",
                        _canned("Player 1 and Player 2 are pulling away from the field."))
    ans = compose_mod.compose_answer("leaderboard", "top target share",
                                     _small_board_pack())
    assert "Player 1 leads Target Share at 0.31 across 12 qualified players." in ans["text"]
    assert "pulling away" not in ans["text"]


def test_compose_model_failure_still_answers(monkeypatch):
    def boom(system, payload, schema, name):
        raise RuntimeError("model down")

    monkeypatch.setattr(compose_mod, "call_model_json", boom)
    ans = compose_mod.compose_answer("leaderboard", "top target share",
                                     _small_board_pack())
    assert ans["verdict"] == "Player 1 leads Target Share"
    assert len(ans["receipts"]) == 3


def test_compose_gated_pack_is_digit_free(monkeypatch):
    pack = {"type": "leaderboard", "available": False, "gated": True,
            "metric_key": "wopr", "label": "WOPR", "missing": "pro_only"}
    ans = compose_mod.compose_answer("leaderboard", "who leads in WOPR", pack)
    assert ans["verdict"] is None
    assert "PRO" in ans["text"]
    assert not re.search(r"\d", ans["text"])
    assert ans["receipts"] == []
