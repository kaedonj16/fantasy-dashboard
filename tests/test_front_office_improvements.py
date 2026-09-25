"""Tests for the Front Office v2 improvement batch (Sept 2026):

1. "Why they'd say yes": trade targets carry partner_record, partner_needs,
   and a computed why_they_say_yes line.
2. Trade deadline countdown from league settings.
3. Waiver-dupe check: a trade target gets flagged when a free agent at the
   same position is worth at least half as much.
4. Drop/add pairing: each waiver add paired with its cleanest drop.
5. Bye/injury urgency: starters on bye in the next two weeks or carrying a
   serious injury designation bubble their position's targets to the front.

Pure-function extraction (the report module pulls in the OpenAI client via
dashboard_services.ai.client, which is unavailable in the base test env; CI
covers the full import).
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _extract(mod_path: str, *names: str) -> dict:
    src = (ROOT / mod_path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = set(names)
    chunks = [
        ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert len(chunks) == len(names), f"missing fns in {mod_path}: {names}"
    return chunks


def _load_report_fns():
    chunks = _extract(
        "dashboard_services/ai/front_office_report.py",
        "_trade_deadline_info",
        "_urgent_needs",
        "_apply_urgency",
        "_annotate_waiver_alternatives",
        "_drop_add_pairs",
    )
    ns: dict = {
        "safe_float": lambda v: float(v or 0),
        "_safe_str": lambda v: str(v or ""),
        "__name__": "front_office_report_test",
    }
    import logging

    ns["logger"] = logging.getLogger("test")
    for chunk in chunks:
        exec(compile(chunk, "front_office_report.py", "exec"), ns)
    return ns


def _load_payload_fn():
    chunks = _extract(
        "dashboard_services/ai/prompts.py",
        "_for_trade_target_label",
        "build_front_office_prompt_payload",
    )
    ns: dict = {"__name__": "prompts_test"}
    for chunk in chunks:
        exec(compile(chunk, "prompts.py", "exec"), ns)
    return ns["build_front_office_prompt_payload"]


_FNS = _load_report_fns()
_trade_deadline_info = _FNS["_trade_deadline_info"]
_urgent_needs = _FNS["_urgent_needs"]
_apply_urgency = _FNS["_apply_urgency"]
_annotate_waiver_alternatives = _FNS["_annotate_waiver_alternatives"]
_drop_add_pairs = _FNS["_drop_add_pairs"]
build_front_office_prompt_payload = _load_payload_fn()


# ── #2 trade deadline ──────────────────────────────────────────────────────

def test_deadline_week_number_counts_down():
    info = _trade_deadline_info(
        {"league_settings": {"trade_deadline": 10}, "current_week": 4}
    )
    assert info == {"deadline_week": 10, "weeks_remaining": 6}


def test_deadline_passed_returns_none():
    assert (
        _trade_deadline_info(
            {"league_settings": {"trade_deadline": 3}, "current_week": 4}
        )
        is None
    )


def test_deadline_missing_returns_none():
    assert _trade_deadline_info({}) is None
    assert _trade_deadline_info({"league_settings": {"trade_deadline": 0}}) is None


def test_deadline_nested_league_settings():
    info = _trade_deadline_info(
        {"league": {"settings": {"trade_deadline": 11}}, "current_week": 9}
    )
    assert info["weeks_remaining"] == 2


# ── #3 waiver alternatives ──────────────────────────────────────────────────

def _trade(pos="WR", val=700.0):
    return [{"gets": [{"id": "t1", "name": "Star", "position": pos, "value": val}], "gives": []}]


def test_waiver_alternative_flagged_at_half_value():
    targets = _trade(val=700.0)
    waivers = [{"id": "w1", "name": "Wire", "position": "WR", "value": 350.0}]
    _annotate_waiver_alternatives(targets, waivers)
    assert targets[0]["waiver_alternative"] == {"name": "Wire", "value": 350.0}


def test_waiver_alternative_ignored_below_half_value():
    targets = _trade(val=700.0)
    waivers = [{"id": "w1", "name": "Wire", "position": "WR", "value": 349.0}]
    _annotate_waiver_alternatives(targets, waivers)
    assert "waiver_alternative" not in targets[0]


def test_waiver_alternative_requires_same_position():
    targets = _trade(pos="WR", val=700.0)
    waivers = [{"id": "w1", "name": "Wire RB", "position": "RB", "value": 690.0}]
    _annotate_waiver_alternatives(targets, waivers)
    assert "waiver_alternative" not in targets[0]


# ── #5 urgency ─────────────────────────────────────────────────────────────

# ── #5 bye/injury urgency ───────────────────────────────────────────────────

import pytest


@pytest.fixture
def _bye_and_injury_stubs(monkeypatch):
    # _urgent_needs lazily imports utils.utils / utils.lineup_issues; stub
    # them for the duration of the test only so other test modules still
    # resolve the real utils package.
    uu = types.ModuleType("utils.utils")
    uu.path_teams_index = lambda: "/nonexistent"
    uu.read_json_cached = lambda p: {"DET": {"byeWeek": 5}, "KC": {"byeWeek": 9}}
    li = types.ModuleType("utils.lineup_issues")
    li.SERIOUS_INJURY_STATUSES = {"OUT", "DOUBTFUL", "IR", "PUP", "SUS", "SUSP", "NA", "NFI"}
    u = types.ModuleType("utils")
    u.utils = uu
    u.lineup_issues = li
    monkeypatch.setitem(sys.modules, "utils", u)
    monkeypatch.setitem(sys.modules, "utils.utils", uu)
    monkeypatch.setitem(sys.modules, "utils.lineup_issues", li)


def _rows():
    return [
        {"id": "1", "name": "Gibbs", "position": "RB", "team": "DET", "injury": ""},
        {"id": "2", "name": "Rice", "position": "WR", "team": "KC", "injury": "OUT"},
        {"id": "3", "name": "Allen", "position": "QB", "team": "BUF", "injury": ""},
        {"id": "4", "name": "Bench", "position": "WR", "team": "DET", "injury": ""},
    ]


def test_urgent_needs_bye_and_injury(_bye_and_injury_stubs):
    roster = {"starters": ["1", "2", "3"]}
    urgent = _urgent_needs({}, roster, _rows(), 4)
    by_pos = {u["position"]: u for u in urgent}
    # DET bye is week 5; current week 4 -> bye in week+1.
    assert by_pos["RB"]["reason"] == "bye"
    assert "week 5" in by_pos["RB"]["detail"]
    # Rice is OUT.
    assert by_pos["WR"]["reason"] == "injury"
    # Allen (BUF, no bye data in the stub) is not urgent.
    assert "QB" not in by_pos


def test_urgent_needs_skips_bench_and_questionable(_bye_and_injury_stubs):
    rows = [
        {"id": "1", "name": "Gibbs", "position": "RB", "team": "DET", "injury": ""},
        {"id": "2", "name": "Lamb", "position": "WR", "team": "DAL", "injury": "QUESTIONABLE"},
    ]
    # Only Gibbs starts; Lamb is questionable (not a serious designation)
    # and doesn't start anyway.
    urgent = _urgent_needs({}, {"starters": ["1"]}, rows, 4)
    assert [u["position"] for u in urgent] == ["RB"]


def test_apply_urgency_bubbles_urgent_targets_first():
    trades = [{"gets": [{"position": "RB"}]}, {"gets": [{"position": "WR"}]}]
    waivers = [
        {"position": "TE", "name": "te"},
        {"position": "WR", "name": "wr"},
    ]
    _apply_urgency(trades, waivers, [{"position": "WR", "detail": "Rice is OUT"}])
    assert trades[0]["gets"][0]["position"] == "WR"
    assert trades[0]["urgent"] is True
    assert trades[0]["urgent_reason"] == "Rice is OUT"
    assert waivers[0]["name"] == "wr"
    assert waivers[0]["urgent"] is True


def test_apply_urgency_noop_without_urgent_needs():
    trades = [{"gets": [{"position": "RB"}]}]
    _apply_urgency(trades, [], [])
    assert "urgent" not in trades[0]


# ── #4 drop/add pairs ───────────────────────────────────────────────────────

def test_drop_add_pairs_prefers_same_position():
    cuts = [
        {"id": "c1", "name": "Cut WR", "position": "WR", "value": 100.0},
        {"id": "c2", "name": "Cut RB", "position": "RB", "value": 50.0},
    ]
    waivers = [{"id": "w1", "name": "Add WR", "position": "WR", "value": 300.0}]
    pairs = _drop_add_pairs(cuts, waivers)
    assert len(pairs) == 1
    assert pairs[0]["drop"]["name"] == "Cut WR"
    assert pairs[0]["add"]["name"] == "Add WR"


def test_drop_add_pairs_never_reuse_a_cut():
    cuts = [{"id": "c1", "name": "Only Cut", "position": "RB", "value": 50.0}]
    waivers = [
        {"id": "w1", "name": "Add WR", "position": "WR", "value": 300.0},
        {"id": "w2", "name": "Add TE", "position": "TE", "value": 250.0},
    ]
    pairs = _drop_add_pairs(cuts, waivers)
    assert len(pairs) == 1
    assert pairs[0]["drop"]["name"] == "Only Cut"


# ── payload carries the new signals ─────────────────────────────────────────

def test_payload_includes_new_signals():
    data = {
        "team_name": "T", "record": "2-1", "week": 4, "season_phase": "regular",
        "scoring_type": "dynasty", "direction": "contend", "grades": [],
        "roster_rows": [], "risers_7d": [], "fallers_7d": [],
        "last_week": None, "this_week": None,
        "trade_targets": [
            {
                "partner": "Rival", "partner_record": "1-2",
                "partner_needs": ["RB"],
                "why_they_say_yes": "Fills their RB need.",
                "gets": [{"id": "t1", "name": "Star", "position": "WR", "value": 700.0}],
                "gives": [{"name": "Depth", "position": "RB", "value": 500.0}],
                "waiver_alternative": {"name": "Wire", "value": 400.0},
                "urgent": True, "urgent_reason": "Rice is OUT",
            }
        ],
        "waiver_targets": [
            {"id": "w1", "name": "Wire", "position": "WR", "team": "KC",
             "value": 400.0, "pos_rank_label": "", "urgent": True,
             "urgent_reason": "Rice is OUT"},
        ],
        "cut_candidates": [],
        "drop_add_pairs": [{"drop": {"name": "Cut", "position": "WR", "value": 50.0},
                            "add": {"name": "Wire", "position": "WR", "value": 400.0}}],
        "urgent_needs": [{"position": "WR", "detail": "Rice is OUT"}],
        "trade_deadline": {"deadline_week": 10, "weeks_remaining": 6},
    }
    payload = build_front_office_prompt_payload(data)
    t = payload["trade_targets"][0]
    assert t["partner_record"] == "1-2"
    assert t["why_they_say_yes"] == "Fills their RB need."
    assert t["waiver_alternative"] == {"name": "Wire", "value": 400.0}
    assert t["urgent_need"] == "Rice is OUT"
    assert payload["waiver_targets"][0]["urgent_need"] == "Rice is OUT"
    assert payload["trade_deadline"] == {"deadline_week": 10, "weeks_remaining": 6}
    assert payload["urgent_needs"] == [{"position": "WR", "detail": "Rice is OUT"}]
    assert payload["drop_add_pairs"] == [{"drop": "Cut", "add": "Wire"}]
