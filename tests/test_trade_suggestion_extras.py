"""Parity 3: trade-suggestion extras (trade_suggestion_extras.py).

Coverage justification, cross-list asset conflicts, counter ladders
and race/deadline weighting as one post-pass over the engine's
finished list. Dependencies (brief coverage, keeper locks, rules
record) are stubbed at their module seams.
"""
from datetime import date, timedelta

import pytest

from dashboard_services import gm_brief, trade_suggestion_extras as extras
from utils import league_rules


@pytest.fixture()
def seams(monkeypatch):
    monkeypatch.setattr(
        gm_brief, "build_coverage",
        lambda ctx, rid, rules=None, actions=None: (
            {"holes": [{"position": "QB", "week": 11},
                       {"position": "QB", "week": 13},
                       {"position": "RB", "week": 10}]}, True))
    monkeypatch.setattr(league_rules, "keeper_locked_ids",
                        lambda ctx, roster_ids=None: frozenset({"core1"}))
    monkeypatch.setattr(
        gm_brief, "get_rules",
        lambda ctx: {"source": "feed", "trade_deadline": None})


def _partner(name, target, sends, score=5.0, contention=None):
    return {
        "team_name": name,
        "targets_they_have": [target],
        "targets_viewer_sends": sends,
        "value_you_get": float(target["value"]),
        "value_you_give": float(sum(s["value"] for s in sends)),
        "suggestion_score": score,
        "partner_contention": contention,
    }


def _viewer():
    return {"roster_id": "1", "starters": ["s1", "s2"],
            "players": ["s1", "s2", "core1", "sw1", "sw2", "sw3",
                        "spent1"]}


VALUES = {
    "core1": {"name": "Cornerstone", "position": "WR", "value": 900.0},
    "sw1": {"name": "Sweet One", "position": "RB", "value": 100.0},
    "sw2": {"name": "Sweet Two", "position": "WR", "value": 200.0},
    "sw3": {"name": "Sweet Three", "position": "TE", "value": 700.0},
    "spent1": {"name": "Spent Asset", "position": "RB", "value": 300.0},
}


def test_coverage_conflicts_and_ladder(seams):
    target = {"id": "tqb", "name": "Target QB", "position": "QB",
              "value": 1000.0}
    sends = [{"id": "spent1", "name": "Spent Asset", "value": 800.0}]
    p1 = _partner("Alpha", target, sends)
    p2 = _partner("Beta", {"id": "twr", "name": "Target WR",
                           "position": "WR", "value": 900.0},
                  [{"id": "spent1", "name": "Spent Asset",
                    "value": 850.0}])
    out = extras.annotate_suggestions({}, [p1, p2], _viewer(), VALUES)
    # Coverage: the QB target addresses the Week 11 (and 13) hole.
    assert out[0]["coverage"]["hole_weeks"] == [11, 13]
    assert "Week 11" in out[0]["coverage"]["text"]
    assert "coverage" not in out[1]      # no WR hole in the map
    # Conflicts: spent1 is promised to both partners.
    assert out[0]["asset_conflicts"] == {"spent1": ["Beta"]}
    assert out[1]["asset_conflicts"] == {"spent1": ["Alpha"]}
    # Ladder: ceiling = 1000*1.40 - 800 = 600; the two cheap bench
    # pieces fit, the 700 piece would break the band.
    ladder = out[0]["counter_ladder"]
    assert ladder["ceiling_value"] == 600.0
    assert [s["id"] for s in ladder["sweeteners"]] == ["sw1", "sw2"]
    rules = {r["rule"] for r in ladder["refuse_outright"]}
    assert rules == {"target_must_remain",
                     "never_send_designated_keeper"}


def test_race_weight_needs_a_known_deadline(seams, monkeypatch):
    target = {"id": "tqb", "name": "Target QB", "position": "QB",
              "value": 1000.0}
    sends = [{"id": "spent1", "name": "Spent Asset", "value": 800.0}]
    p1 = _partner("Alpha", target, sends, score=5.0,
                  contention="contender")
    out = extras.annotate_suggestions({}, [p1], _viewer(), VALUES)
    assert "race_weight" not in out[0]          # no deadline known
    assert out[0]["suggestion_score"] == 5.0
    soon = (date.today() + timedelta(days=10)).isoformat()
    monkeypatch.setattr(
        gm_brief, "get_rules",
        lambda ctx: {"source": "feed", "trade_deadline": soon})
    p2 = _partner("Alpha", target, sends, score=5.0,
                  contention="contender")
    out = extras.annotate_suggestions({}, [p2], _viewer(), VALUES)
    assert out[0]["race_weight"]["points"] == 0.4
    assert out[0]["race_weight"]["weeks_to_deadline"] == 1
    assert out[0]["suggestion_score"] == 5.4


def test_race_weight_fails_closed_on_unverified_rules(monkeypatch):
    monkeypatch.setattr(
        gm_brief, "build_coverage",
        lambda ctx, rid, rules=None, actions=None: ({"holes": []}, False))
    monkeypatch.setattr(league_rules, "keeper_locked_ids",
                        lambda ctx, roster_ids=None: frozenset())
    soon = (date.today() + timedelta(days=5)).isoformat()
    monkeypatch.setattr(
        gm_brief, "get_rules",
        lambda ctx: {"source": "default_unverified",
                     "trade_deadline": soon})
    p = _partner("Alpha",
                 {"id": "t", "name": "T", "position": "QB",
                  "value": 100.0},
                 [{"id": "s", "name": "S", "value": 90.0}],
                 contention="contender")
    out = extras.annotate_suggestions({}, [p], _viewer(), VALUES)
    assert "race_weight" not in out[0]
