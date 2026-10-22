"""Compare ownership + number presentation tests.

Kaedon live, 2026-10-16: "compare garrett wilson and nico collins"
came back as a one-sided "Garrett Wilson update" citing
"17.279999999999998 points per game" and "0.9 snap share" next to
"30.2 target share". These tests pin the fixes: the compare shape is
owned deterministically (router -> open layer -> compare_players over
the full player universe), a partial profile still compares with its
gaps named, an unresolvable name is answered by name with the closest
match offered, and pack numbers ship rounded the way the site presents
them (PPG 1 decimal, shares as percents), with a grounding-gate net
that rewrites any surviving float artifact.
"""
import json

import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

import data_building.advanced_metrics as am
from dashboard_services.assistant import compare_table as ct_mod
from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import context as ctx_mod
from dashboard_services.assistant import evidence as ev
from dashboard_services.assistant import open_layer
from dashboard_services.assistant import resolve as resolve_mod
from dashboard_services.assistant import tools as tools_mod

INDEX = {
    "wilson": {"full_name": "Garrett Wilson", "team": "NYJ", "position": "WR"},
    "collins": {"full_name": "Nico Collins", "team": "HOU", "position": "WR"},
}
VALUES = [{"id": "wilson", "value": 526.3}, {"id": "collins", "value": 610.5}]
# Wilson's totals divide to exactly 17.279999999999998 in raw float.
METRICS = {
    "wilson": {"snap_share": 0.9, "target_share": 30.2, "expected_ppr": 88.3,
               "ppr_over_expected": 15.38, "games": 6,
               "touches_per_game": 9.8333333333},
    "collins": {"snap_share": 0.88, "target_share": 28.4, "expected_ppr": 92.0,
                "ppr_over_expected": 10.0, "games": 6,
                "touches_per_game": 8.5},
}


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch",
              "week": None, "roster_player_ids": set(), "has_pro": True}
    bundle.update(over)
    return bundle


@pytest.fixture()
def world(monkeypatch):
    metrics = {k: dict(v) for k, v in METRICS.items()}
    for mod in (ev, ctx_mod):
        monkeypatch.setattr(mod, "get_players_index_global",
                            lambda *a, **k: dict(INDEX))
        monkeypatch.setattr(mod, "get_model_value_table_cached",
                            lambda *a, **k: list(VALUES))
    monkeypatch.setattr(am, "get_player_metrics",
                        lambda pid, **k: dict(metrics.get(str(pid)) or {}))
    monkeypatch.setattr(am, "get_player_metrics_by_season",
                        lambda pid, season, **k: dict(metrics.get(str(pid)) or {}))
    return metrics


# -- metric presentation -----------------------------------------------------------

def test_metrics_for_rounds_derived_ppg_and_unifies_share_scale(world):
    m, _tools = ev._metrics_for("wilson", 2026)
    assert m["ppg"] == 17.3            # not 17.279999999999998
    assert m["snap_share"] == 90       # fraction in the store, percent out
    assert m["target_share"] == 30.2   # already a percent in the store
    assert m["touches_per_game"] == 9.8


def test_metrics_for_keeps_missing_metrics_missing(world):
    world["collins"] = {"games": 6}
    m, _tools = ev._metrics_for("collins", 2026)
    assert "ppg" not in m and "snap_share" not in m


def test_metrics_for_derives_touches_per_game_from_totals(world):
    # Real-shaped row: player_advanced_metrics stores the total_touches
    # season total and games; there is NO touches_per_game column, so
    # the per-game line must derive exactly like expected PPG does.
    world["collins"] = {"snap_share": 0.88, "target_share": 28.4,
                        "expected_ppr": 92.0, "ppr_over_expected": 10.0,
                        "games": 6, "total_touches": 51}
    m, _tools = ev._metrics_for("collins", 2026)
    assert m["touches_per_game"] == 8.5


def test_compare_tool_carries_derived_touches(world):
    world["collins"] = {"snap_share": 0.88, "target_share": 28.4,
                        "expected_ppr": 92.0, "ppr_over_expected": 10.0,
                        "games": 6, "total_touches": 51}
    out = tools_mod._tool_compare_players(
        _bundle(), {"player_a": "Garrett Wilson", "player_b": "Nico Collins"})
    assert out["players_detail"][1]["touches_per_game"] == 8.5
    table = ct_mod.build_compare_table(out["players_detail"])
    opp = next(g for g in table["groups"] if g["label"] == "OPPORTUNITY")
    assert opp["rows"][0] == {"key": "touches_per_game",
                              "label": "Touches per game",
                              "a": "9.8", "b": "8.5",
                              "winner": 0, "delta": "+1.3"}


# -- resolution offers --------------------------------------------------------------

def test_nearest_names_offers_the_real_near_miss(world):
    assert resolve_mod.nearest_names("nico collinzz", index=INDEX) == ["Nico Collins"]
    assert resolve_mod.nearest_names("zz nobody", index=INDEX) == []


# -- the compare tool -----------------------------------------------------------------

def test_compare_tool_builds_both_players(world):
    out = tools_mod._tool_compare_players(
        _bundle(), {"player_a": "Garrett Wilson", "player_b": "Nico Collins"})
    assert out["ok"]
    assert [p["name"] for p in out["players_detail"]] == [
        "Garrett Wilson", "Nico Collins"]
    assert out["players_detail"][0]["ppg"] == 17.3
    assert out["players_detail"][0]["snap_share"] == 90
    assert {r["label"] for r in out["receipts"]} == {
        "Garrett Wilson value", "Garrett Wilson points per game",
        "Nico Collins value", "Nico Collins points per game"}
    assert out["link"]["label"] == "View head-to-head in Compare"
    assert "p1=wilson" in out["link"]["href"]
    assert "p2=collins" in out["link"]["href"]


def test_compare_tool_unrostered_players_still_compare(world):
    # Neither player is on the asker's roster (or any roster in the
    # frozen world): resolution is universe-wide, so both compare.
    out = tools_mod._tool_compare_players(
        _bundle(roster_player_ids=set()),
        {"player_a": "garrett wilson", "player_b": "nico collins"})
    assert out["ok"] and len(out["players_detail"]) == 2


def test_compare_tool_partial_data_names_the_gap(world):
    world["collins"] = {"expected_ppr": 92.0, "ppr_over_expected": 10.0,
                         "games": 6}
    out = tools_mod._tool_compare_players(
        _bundle(), {"player_a": "Garrett Wilson", "player_b": "Nico Collins"})
    assert out["ok"]
    collins = out["players_detail"][1]
    assert collins["ppg"] == 17
    assert collins["missing_metrics"] == ["target share", "snap share"]


def test_compare_tool_unresolvable_name_is_named_with_nearest(world):
    out = tools_mod._tool_compare_players(
        _bundle(), {"player_a": "Garrett Wilson", "player_b": "Nico Collinzz"})
    assert not out["ok"]
    assert out["not_found"] == "Nico Collinzz"
    assert out["nearest"] == ["Nico Collins"]
    assert out["found"]["name"] == "Garrett Wilson"


# -- open layer: deterministic fetch + named decline -----------------------------------

def test_plan_and_fetch_compares_without_the_planner(world, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("planner model must not run for a compare")
    monkeypatch.setattr(open_layer, "call_model_json", _boom)
    pack, called = open_layer.plan_and_fetch(
        "compare garrett wilson and nico collins", _bundle())
    assert called == ["compare_players"]
    assert pack["available"] is True
    entry = pack["evidence"][0]
    assert entry["ok"] and len(entry["players_detail"]) == 2


def test_plan_and_fetch_not_found_pack_is_unavailable(world, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("planner model must not run for a compare")
    monkeypatch.setattr(open_layer, "call_model_json", _boom)
    pack, called = open_layer.plan_and_fetch(
        "compare garrett wilson and nico collinzz", _bundle())
    assert called == ["compare_players"]
    assert pack["available"] is False
    assert pack["evidence"][0]["not_found"] == "nico collinzz"


def test_compare_not_found_answer_names_player_and_offers_match(world, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("no model call for a deterministic decline")
    monkeypatch.setattr(open_layer, "call_model_json", _boom)
    pack, _called = open_layer.plan_and_fetch(
        "compare garrett wilson and nico collinzz", _bundle())
    answer = open_layer.compose_open_answer(
        "compare garrett wilson and nico collinzz", pack)
    assert answer["verdict"] is None
    assert "could not find nico collinzz" in answer["text"]
    assert "closest match is Nico Collins" in answer["text"]
    asks = [f["ask"] for f in answer["followups"]]
    assert asks == ["Compare Garrett Wilson and Nico Collins"]


# -- grounding gate: float-artifact net -----------------------------------------------

def _player_pack():
    return {"type": "player",
            "player": {"name": "Garrett Wilson", "value": 526.3},
            "role": {"ppg": 17.279999999999998, "snap_share": 90}}


def test_grounding_gate_rewrites_grounded_float_artifacts():
    answer = {"verdict": "Garrett Wilson update",
              "text": "Garrett Wilson averages 17.279999999999998 points "
                      "per game with a 90 snap share.",
              "receipts": [{"label": "Garrett Wilson points per game",
                            "value": "17.279999999999998"}],
              "caveat": None, "followups": [], "act_link": None,
              "actions": [], "mentions": []}
    out = compose_mod._enforce_grounding(
        answer, "player", _player_pack(), "tell me about garrett wilson")
    assert "17.279999999999998" not in out["text"]
    assert "17.3 points per game" in out["text"]
    assert out["receipts"][0]["value"] == "17.3"


# -- the side-by-side compare table ------------------------------------------------
# The approved mock's own pairing (Kaedon's live Gibbs vs Walker
# question, 2026-10-19): grouped rows, winner bold, triangle + gap.

GIBBS_WALKER = [
    {"player_id": "gibbs", "name": "Jahmyr Gibbs", "value": 1121.4,
     "ppg": 31.0, "recent_ppg": 30.0, "expected_ppg": 26.1,
     "target_share": 18.1, "snap_share": 80, "touches_per_game": None,
     "games": 5},
    {"player_id": "walker", "name": "Kenneth Walker III", "value": 752.3,
     "ppg": 27.6, "recent_ppg": 27.5, "expected_ppg": 23.9,
     "target_share": 15.0, "snap_share": 70, "touches_per_game": None,
     "games": 5},
]


def _row(table, group_label, key):
    group = next(g for g in table["groups"] if g["label"] == group_label)
    return next(r for r in group["rows"] if r["key"] == key)


def test_compare_table_matches_the_approved_mock_shape():
    table = ct_mod.build_compare_table(GIBBS_WALKER)
    assert [p["short"] for p in table["players"]] == ["Gibbs", "Walker III"]
    assert [g["label"] for g in table["groups"]] == [
        "PRODUCTION", "OPPORTUNITY", "VALUE"]
    assert _row(table, "PRODUCTION", "ppg") == {
        "key": "ppg", "label": "Points per game", "a": "31", "b": "27.6",
        "winner": 0, "delta": "+3.4"}
    assert _row(table, "PRODUCTION", "recent_ppg")["delta"] == "+2.5"
    assert _row(table, "PRODUCTION", "expected_ppg")["delta"] == "+2.2"
    assert _row(table, "OPPORTUNITY", "target_share") == {
        "key": "target_share", "label": "Target share", "a": "18.1%",
        "b": "15%", "winner": 0, "delta": "+3.1"}
    snap = _row(table, "OPPORTUNITY", "snap_share")
    assert snap["a"] == "80%" and snap["b"] == "70%" and snap["delta"] == "+10"
    assert _row(table, "VALUE", "value")["delta"] == "+369.1"
    # Shares print as percents, never as the raw fraction.
    assert "0.8" not in json.dumps(table)


def test_compare_table_tied_row_has_no_winner_or_delta():
    detail = [dict(GIBBS_WALKER[0]), dict(GIBBS_WALKER[1])]
    detail[1]["ppg"] = detail[0]["ppg"]
    row = _row(ct_mod.build_compare_table(detail), "PRODUCTION", "ppg")
    assert row["winner"] is None and row["delta"] is None
    assert row["a"] == row["b"] == "31"


def test_compare_table_omits_rows_and_groups_for_missing_metrics():
    detail = [dict(GIBBS_WALKER[0]), dict(GIBBS_WALKER[1])]
    detail[1].update({"target_share": None, "snap_share": None})
    table = ct_mod.build_compare_table(detail)
    # Walker has no opportunity metric left (touches is None on both),
    # so the whole group goes, not a row of placeholders.
    assert [g["label"] for g in table["groups"]] == ["PRODUCTION", "VALUE"]


def test_compare_table_winner_can_be_the_second_player():
    detail = [dict(GIBBS_WALKER[0]), dict(GIBBS_WALKER[1])]
    detail[1]["value"] = 1300.0
    row = _row(ct_mod.build_compare_table(detail), "VALUE", "value")
    assert row["winner"] == 1 and row["delta"] == "+178.6"


def test_compare_table_only_for_two_player_profiles():
    assert ct_mod.build_compare_table(GIBBS_WALKER[:1]) is None
    assert ct_mod.build_compare_table(GIBBS_WALKER + GIBBS_WALKER[:1]) is None
    assert ct_mod.build_compare_table([]) is None
    assert ct_mod.build_compare_table(
        [{"name": "A"}, {"name": "B"}]) is None  # no shared metric at all
    rank_entry = {"tool": "rank_players", "ok": True,
                  "players_detail": GIBBS_WALKER}
    assert ct_mod.compare_table_for_evidence([rank_entry]) is None
    dead_entry = {"tool": "compare_players", "ok": False,
                  "players_detail": GIBBS_WALKER}
    assert ct_mod.compare_table_for_evidence([dead_entry]) is None


def test_open_answer_carries_compare_table(monkeypatch):
    data = json.load(open("assistant_eval/questions.json"))
    q = next(x for x in data["questions"] if x["id"] == "ow22")
    draft = {"verdict": "Nico Collins over Garrett Wilson",
             "text": "Nico Collins leads on value, 610.5 to 526.3, while "
                     "Garrett Wilson edges points per game, 17.3 to 17.",
             "receipts": [{"label": "Nico Collins value", "value": "610.5"},
                          {"label": "Garrett Wilson value", "value": "526.3"}],
             "caveat": None, "followups": [], "actions": []}
    monkeypatch.setattr(compose_mod, "call_model_json",
                        lambda system, user, schema, name: dict(draft))
    answer = compose_mod.compose_answer("open", q["question"],
                                        q["evidence_pack"])
    table = answer["compare_table"]
    assert [p["name"] for p in table["players"]] == [
        "Garrett Wilson", "Nico Collins"]
    assert _row(table, "VALUE", "value")["winner"] == 1
    assert _row(table, "VALUE", "value")["delta"] == "+84.2"
    assert _row(table, "PRODUCTION", "ppg")["delta"] == "+0.3"
    assert _row(table, "OPPORTUNITY", "touches_per_game")["delta"] == "+1.3"
    keys = [r["key"] for g in table["groups"] for r in g["rows"]]
    assert "recent_ppg" not in keys  # None on both frozen profiles
    assert answer["act_link"]["label"] == "View head-to-head in Compare"


def test_grounding_gate_still_strips_invented_float_artifacts():
    answer = {"verdict": "Garrett Wilson update",
              "text": "He averages 99.123456789 points per game.",
              "receipts": [], "caveat": None, "followups": [],
              "act_link": None, "actions": [], "mentions": []}
    out = compose_mod._enforce_grounding(
        answer, "player", _player_pack(), "tell me about garrett wilson")
    assert "99.123456789" not in out["text"]


# -- Decimal-shaped rows (live regression, 2026-10-22) ----------------------------
#
# Real rows come back from psycopg with NUMERIC columns as Decimal.
# _metrics_for's presentation block guarded on isinstance(int/float), so
# a Decimal snap_share skipped BOTH the percent scaling and the rounding
# and compare tables printed the raw fraction as "0.8%" while every
# fixture above, faking plain floats, passed. These tests fake the real
# driver shape.

from decimal import Decimal


def test_metrics_for_decimal_row_scales_snap_share(world):
    world["wilson"] = {"snap_share": Decimal("0.82"),
                       "target_share": Decimal("18.1"),
                       "expected_ppr": Decimal("130.5"),
                       "ppr_over_expected": Decimal("20.5"),
                       "games": 5, "total_touches": 132}
    m, _tools = ev._metrics_for("wilson", 2026)
    assert m["snap_share"] == 82        # 0.82 fraction -> percent, not 0.8
    assert m["target_share"] == 18.1    # already a percent, unchanged
    assert not isinstance(m["snap_share"], Decimal)
    assert not isinstance(m["target_share"], Decimal)


def test_compare_table_decimal_row_shows_percent(world):
    world["wilson"] = {"snap_share": Decimal("0.82"),
                       "target_share": Decimal("18.1"),
                       "expected_ppr": Decimal("130.5"),
                       "ppr_over_expected": Decimal("20.5"),
                       "games": 5, "total_touches": 132}
    out = tools_mod._tool_compare_players(
        _bundle(), {"player_a": "Garrett Wilson", "player_b": "Nico Collins"})
    assert out["players_detail"][0]["snap_share"] == 82
    table = ct_mod.build_compare_table(out["players_detail"])
    opp = next(g for g in table["groups"] if g["label"] == "OPPORTUNITY")
    snap_row = next(r for r in opp["rows"] if r["key"] == "snap_share")
    assert snap_row["a"] == "82%"
    tgt_row = next(r for r in opp["rows"] if r["key"] == "target_share")
    assert tgt_row["a"] == "18.1%"
