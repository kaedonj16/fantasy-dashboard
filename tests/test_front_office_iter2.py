"""Iteration 2 pins for the Front Office report (items 3, 6, 7b):

- Shop moves respect the coverage horizon: "Shop Joe Milton III"
  was the top move in the week Milton was the only Week 11 QB cover
  (Lock and Sanders both bye 11). A non-load-bearing excess QB is
  preferred; when every excess QB is load-bearing the move stays
  but names the coverage cost.
- Injury rows use the actual designation: Questionable starters
  (Lock, Kincaid) are never labeled "out ...", and ``is_starter``
  comes from the real lineup, not the value-based roster role (the
  Reference summary counted "0 starters flagged" off that role).
- Positional grades read the shared positional_room_ranks helper.

Stub style mirrors tests/test_front_office_injury.py; per-test
stubs go through monkeypatch so nothing leaks across modules.
"""
import sys
import types

import pytest


_INSTALLED_STUBS = []


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    if name not in sys.modules:
        sys.modules[name] = m
        _INSTALLED_STUBS.append(name)


class _E(Exception):
    pass


_stub("dashboard_services.ai.client", AIRateLimitError=_E, AIUnavailableError=_E)
_stub(
    "dashboard_services.ai.context_builders",
    _ctx_is_sf=lambda *a: False,
    _record_pf_pa_for_roster=lambda *a: ("", 0.0, 0.0),
    build_model_value_lookup=lambda *a: {},
    build_team_gm_context=lambda *a: {},
    build_trade_suggestions_context=lambda *a: {},
    ctx_scoring_type=lambda *a: "ppr",
)
_stub(
    "dashboard_services.ai.prompts",
    build_front_office_prompt_payload=lambda *a: {},
    generate_front_office_report_result=lambda *a: {},
    normalize_trade_scoring_type=lambda x: x,
)
_stub(
    "dashboard_services.ai.renderer",
    _ai_error_notice=lambda r="": "",
    _ctx_with_playoff_odds=lambda x: x,
    _emit_ai_html=lambda x: x,
    ai_available=lambda: False,
)
_stub("dashboard_services.providers.espn_api", safe_float=lambda x, d=0.0: d)
_stub("utils.roster_strength", STARTER_THRESHOLD=0)

from dashboard_services.ai.front_office_report import (  # noqa: E402
    _build_injury_rows,
    _build_top_moves,
    _coverage_hole_weeks,
    _positional_grades,
)

for _stub_name in _INSTALLED_STUBS:
    sys.modules.pop(_stub_name, None)


def _stub_injury_return(monkeypatch, weeks_out_map=None, verdict_map=None):
    weeks_out_map = weeks_out_map or {}
    verdict_map = verdict_map or {}
    mod = types.ModuleType("dashboard_services.injury_return")
    mod.weeks_out_for_player = lambda pid: weeks_out_map.get(str(pid))
    mod.injury_roster_verdict = lambda **kw: verdict_map.get(
        str(kw.get("status") or ""), {"label": "Hold"})
    monkeypatch.setitem(sys.modules, "dashboard_services.injury_return", mod)


def _coverage(week_entries):
    """week_entries: {week: {pos: (startable_ids, count, required)}}"""
    weeks = []
    for wk, positions in sorted(week_entries.items()):
        weeks.append({
            "week": wk,
            "positions": {
                pos: {"startable": [{"id": pid} for pid in ids],
                      "count": count, "required": req}
                for pos, (ids, count, req) in positions.items()
            },
        })
    return {"weeks": weeks}


def _stub_coverage(monkeypatch, coverage):
    mod = types.ModuleType("dashboard_services.gm_brief")
    mod.build_coverage = lambda ctx, rid: (coverage, True)
    monkeypatch.setitem(sys.modules, "dashboard_services.gm_brief", mod)


# --- _coverage_hole_weeks -------------------------------------------------


def test_coverage_hole_weeks_margin_only():
    cov = _coverage({
        11: {"QB": (["milton"], 1, 1)},          # sole cover: margin
        12: {"QB": (["lock", "milton"], 2, 1)},  # surplus: not margin
    })
    assert _coverage_hole_weeks(cov, "milton") == [11]
    assert _coverage_hole_weeks(cov, "lock") == []
    assert _coverage_hole_weeks(None, "milton") == []
    assert _coverage_hole_weeks(cov, "") == []


# --- Shop moves (item 3) ----------------------------------------------------

_QB = {"excess": [{"id": "milton", "name": "Joe Milton III"},
                  {"id": "cousins", "name": "Kirk Cousins"}],
       "total": 4, "is_sf": True}


def _shop_moves(monkeypatch, coverage):
    _stub_coverage(monkeypatch, coverage)
    return _build_top_moves(
        trade_targets=[], drop_add_pairs=[], qb=dict(_QB),
        injury_rows=[], roster={"roster_id": 1}, ctx={}, grades=[],
        holes=[], qb_needy_count=0)


def test_shop_prefers_excess_qb_who_covers_nothing(monkeypatch):
    cov = _coverage({11: {"QB": (["milton"], 1, 1)}})
    moves = _shop_moves(monkeypatch, cov)
    shop = [m for m in moves if m["line"].startswith("Shop ")]
    assert len(shop) == 1
    assert shop[0]["line"] == "Shop Kirk Cousins"
    assert "Coverage cost" not in shop[0]["why"]


def test_shop_annotates_when_every_excess_qb_is_load_bearing(monkeypatch):
    cov = _coverage({
        11: {"QB": (["milton"], 1, 1)},
        12: {"QB": (["cousins"], 1, 1)},
    })
    moves = _shop_moves(monkeypatch, cov)
    shop = [m for m in moves if m["line"].startswith("Shop ")]
    assert len(shop) == 1
    assert shop[0]["line"] == "Shop Joe Milton III"
    assert ("Coverage cost: he is your only startable QB cover in "
            "Week 11") in shop[0]["why"]


def test_shop_falls_back_when_coverage_unavailable(monkeypatch):
    mod = types.ModuleType("dashboard_services.gm_brief")

    def _boom(ctx, rid):
        raise RuntimeError("no coverage")

    mod.build_coverage = _boom
    monkeypatch.setitem(sys.modules, "dashboard_services.gm_brief", mod)
    moves = _build_top_moves(
        trade_targets=[], drop_add_pairs=[], qb=dict(_QB),
        injury_rows=[], roster={"roster_id": 1}, ctx={}, grades=[],
        holes=[], qb_needy_count=0)
    shop = [m for m in moves if m["line"].startswith("Shop ")]
    assert len(shop) == 1
    assert shop[0]["line"] == "Shop Joe Milton III"
    assert "Coverage cost" not in shop[0]["why"]


# --- Injury language (item 6) ----------------------------------------------

def test_questionable_starters_are_game_time_decisions(monkeypatch):
    _stub_injury_return(
        monkeypatch,
        weeks_out_map={"lock": 0.4, "kincaid": 0.4, "outguy": 2.0},
        verdict_map={})
    ctx = {"current_week": 10, "players": {
        "lock": {"injury_status": "Questionable", "injury_body_part": "Oblique"},
        "kincaid": {"injury_status": "Questionable", "injury_body_part": "Ankle"},
        "outguy": {"injury_status": "Out", "injury_body_part": "Knee"},
    }}
    # Roster-row role is VALUE-based: both starters read "Depth" here,
    # which is exactly how the Reference summary missed them.
    rows = [
        {"id": "lock", "name": "Drew Lock", "position": "QB",
         "team": "SEA", "role": "Depth", "value": 120.0, "injury": "Q"},
        {"id": "kincaid", "name": "Dalton Kincaid", "position": "TE",
         "team": "BUF", "role": "Depth", "value": 90.0, "injury": "Q"},
        {"id": "outguy", "name": "Out Guy", "position": "RB",
         "team": "DAL", "role": "Starter", "value": 400.0, "injury": "O"},
    ]
    out = _build_injury_rows(ctx, rows, starter_ids={"lock", "kincaid"})
    by_id = {r["id"]: r for r in out}
    for pid in ("lock", "kincaid"):
        assert by_id[pid]["injury"] == "QUESTIONABLE"
        assert by_id[pid]["return_label"] == "game-time decision"
        assert not by_id[pid]["return_label"].startswith("out")
        assert by_id[pid]["is_starter"] is True
    # A real Out designation keeps its duration label.
    assert by_id["outguy"]["return_label"] == "out ~2 wks"
    assert by_id["outguy"]["is_starter"] is False


# --- Shared room ranks (item 7b wiring) -------------------------------------

def test_positional_grades_read_shared_room_ranks(monkeypatch):
    mod = types.ModuleType("dashboard_services.ai.context_builders")
    mod.positional_room_ranks = lambda ctx: {
        "ranks": {"QB": {"1": 11}, "RB": {"1": 3},
                  "WR": {"1": 5}, "TE": {"1": 8}},
        "strengths": {"1": {"QB": 41.0}},
        "of": 12,
    }
    monkeypatch.setitem(
        sys.modules, "dashboard_services.ai.context_builders", mod)
    grades = _positional_grades({"rosters": [{}] * 12}, "1", {})
    by_pos = {g["pos"]: g for g in grades}
    assert by_pos["QB"]["rank"] == 11
    assert by_pos["QB"]["of"] == 12
    assert by_pos["RB"]["rank"] == 3
