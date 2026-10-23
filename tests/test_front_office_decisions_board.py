"""Tests for the Front Office decision board (v3, Kaedon 2026-10-23).

The modal leads with a sequenced 'Do this first' board: moves in
deadline order (IR, waiver, trade, lineup) with dependency tags,
deterministic After lines, and their evidence rendered in place. The
standalone trade / waiver / injury sections and the bottom GM alert
fold into the board; grades and roster drop to a Reference section.
Fixtures without decision_moves keep the legacy layout.

Uses the same stub harness as tests/test_front_office_rework.py.
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
_stub("utils.roster_strength", STARTER_THRESHOLD={"QB": 500, "RB": 350, "WR": 350, "TE": 200})

import dashboard_services.ai.front_office_report as fmod

for _stub_name in _INSTALLED_STUBS:
    sys.modules.pop(_stub_name, None)


@pytest.fixture(autouse=True)
def _real_safe_float(monkeypatch):
    monkeypatch.setattr(
        fmod, "safe_float", lambda x, default=0.0: default if x is None else float(x)
    )
    # Whichever FO harness imports front_office_report first binds its
    # STARTER_THRESHOLD (test_front_office_report_render stubs it as the
    # int 0). The builder does real threshold math, so pin the dict for
    # these tests and let monkeypatch restore it afterwards.
    monkeypatch.setattr(
        fmod, "STARTER_THRESHOLD",
        {"QB": 500, "RB": 350, "WR": 350, "TE": 200},
    )


def _row(pid, name, pos, value, injury="", role=None, team="BUF"):
    if role is None:
        role = "Starter" if value >= {"QB": 500, "RB": 350, "WR": 350, "TE": 200}.get(pos, 350) else "Depth"
    return {"id": pid, "name": name, "position": pos, "team": team,
            "value": value, "injury": injury, "role": role,
            "age": 25, "pos_rank_label": "", "trend_7d": None}


def _inj(pid, name, pos, injury, body="", ret="", role="Starter", action="Hold"):
    return {"id": pid, "name": name, "position": pos, "team": "BUF", "role": role,
            "injury": injury, "body": body, "weeks_out": None,
            "return_label": ret, "action": action}


def _scenario():
    roster_rows = [
        _row("q1", "QB One", "QB", 900, "QUESTIONABLE"),
        _row("q2", "QB Two", "QB", 500),
        _row("r1", "RB One", "RB", 800),
        _row("r3", "RB Three", "RB", 400, "IR"),
        _row("r5", "Worst RB", "RB", 200),
        _row("w2", "WR Two", "WR", 700, "OUT"),
        _row("w3", "WR Three", "WR", 500),
        _row("t1", "TE One", "TE", 300, "QUESTIONABLE"),
    ]
    injury_rows = [
        _inj("r3", "RB Three", "RB", "IR", "Knee", "out ~4 wks", action="Move to IR"),
        _inj("w2", "WR Two", "WR", "OUT", "Hamstring", "out ~2 wks", action="Hold"),
        _inj("q1", "QB One", "QB", "QUESTIONABLE", "Ankle", "", action="Hold"),
        _inj("t1", "TE One", "TE", "QUESTIONABLE", "Shoulder", "", action="Hold"),
    ]
    roster = {"starters": ["q1", "r1", "w2", "t1"], "reserve": [],
              "players": [r["id"] for r in roster_rows]}
    ctx = {"roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "IR"]}
    grades = [
        {"pos": "QB", "grade": "B", "rank": 4, "of": 10},
        {"pos": "RB", "grade": "D", "rank": 9, "of": 10},
        {"pos": "WR", "grade": "A", "rank": 2, "of": 10},
        {"pos": "TE", "grade": "C", "rank": 6, "of": 10},
    ]
    trade_targets = [
        {
            "gets": [{"id": "g1", "name": "Their RB2", "position": "RB", "value": 1240}],
            "gives": [{"name": "QB Three", "position": "QB", "value": 720},
                      {"name": "WR Five", "position": "WR", "value": 460}],
            "partner": "Partner A", "partner_record": "4-2",
            "partner_direction": "rebuilding",
            "fairness_label": "Fair deal, within 5%",
            "acceptance_label": "Likely to accept",
            "why_they_say_yes": "He is surplus for them.",
            "depth_checked": True, "depth_warnings": [],
            "analyzer_url": "/trade?x=1",
        },
        {
            "gets": [{"id": "g2", "name": "Plan B RB", "position": "RB", "value": 860}],
            "gives": [{"name": "QB Three", "position": "QB", "value": 720}],
            "partner": "Partner A",
            "fairness_label": "Fair deal, within 10%",
            "acceptance_label": "Possible",
            "why_they_say_yes": "Depth for them.",
            "depth_checked": True, "depth_warnings": [],
            "analyzer_url": "/trade?x=2",
        },
    ]
    pairs = [{"drop": {"name": "Worst RB", "position": "RB", "value": 200},
              "add": {"name": "Add RB", "position": "RB", "value": 640}}]
    waiver_targets = [{"id": "a1", "name": "Add RB", "position": "RB", "team": "FA",
                       "value": 640, "pos_rank_label": "RB24", "injury": ""}]
    cut_candidates = [{"id": "r5", "name": "Worst RB", "position": "RB", "value": 200}]
    return dict(roster_rows=roster_rows, injury_rows=injury_rows, roster=roster,
                ctx=ctx, grades=grades, trade_targets=trade_targets,
                drop_add_pairs=pairs, waiver_targets=waiver_targets,
                cut_candidates=cut_candidates)


def _build(sc, **over):
    kw = dict(
        trade_targets=sc["trade_targets"], potential_targets=[],
        drop_add_pairs=sc["drop_add_pairs"], waiver_targets=sc["waiver_targets"],
        cut_candidates=sc["cut_candidates"], injury_rows=sc["injury_rows"],
        roster=sc["roster"], roster_rows=sc["roster_rows"], ctx=sc["ctx"],
        grades=sc["grades"], holes=[],
    )
    kw.update(over)
    return fmod._build_decision_moves(**kw)


# ── Builder: sequence, dependencies, after lines ──────────────────────────

def test_decision_moves_sequence_deps_and_after_lines():
    sc = _scenario()
    moves = _build(sc)
    assert [m["kind"] for m in moves] == ["ir", "waiver", "trade", "lineup"]
    assert [m["rank"] for m in moves] == [1, 2, 3, 4]
    assert [m["timing"] for m in moves] == [
        "Do now", "Before waivers", "This week", "Before Sunday"]
    ir, waiver, trade, lineup = moves
    assert ir["line"] == "Move RB Three to the IR slot"
    assert ir["dep"] == "Unlocks Move 2"
    assert waiver["dep"] == "Needs Move 1"
    assert "3 startable RBs, up from 2" in waiver["after"]
    assert "No drop needed with Move 1 done" in waiver["after"]
    assert "Worst RB" in waiver["after"]
    assert waiver["add"]["pos_rank_label"] == "RB24"  # enriched from waiver row
    assert trade["line"] == "Offer QB Three + WR Five for Their RB2"
    assert trade["plan_b_index"] == 1
    assert "No new hole created" in trade["after"]
    assert lineup["line"] == "Bench WR Two, start WR Three"
    assert set(lineup["detail_injury_ids"]) == {"w2", "q1", "t1"}


def test_decision_moves_without_ir_slot_waiver_names_its_drop():
    sc = _scenario()
    sc["ctx"] = {"roster_positions": ["QB", "RB", "WR", "TE", "BN"]}
    moves = _build(sc)
    assert [m["kind"] for m in moves] == ["waiver", "trade", "lineup"]
    waiver = moves[0]
    assert waiver["dep"] is None
    assert "You drop Worst RB." in waiver["after"]


def test_lineup_move_requires_a_healthy_fill_in():
    sc = _scenario()
    sc["roster_rows"] = [r for r in sc["roster_rows"] if r["id"] != "w3"]
    moves = _build(sc)
    assert "lineup" not in [m["kind"] for m in moves]


def test_questionable_starter_alone_makes_no_lineup_move():
    sc = _scenario()
    sc["injury_rows"] = [r for r in sc["injury_rows"] if r["injury"] == "QUESTIONABLE"]
    sc["roster"]["reserve"] = []
    moves = _build(sc, drop_add_pairs=[], waiver_targets=[], trade_targets=[])
    assert moves == []


def test_potential_targets_become_the_trade_move_when_nothing_priced():
    sc = _scenario()
    pot = [{"id": "p1", "name": "Gettable RB", "position": "RB", "value": 860,
            "partner": "Partner A", "gap_pct": 85, "attainability": "Realistic",
            "injury": "", "depth_rank": 3, "flex_slots": 0,
            "is_surplus": True, "sole_starter": False, "starting_point": {}}]
    moves = _build(sc, trade_targets=[], potential_targets=pot,
                   drop_add_pairs=[], waiver_targets=[])
    trade = [m for m in moves if m["kind"] == "potential"]
    assert len(trade) == 1
    assert trade[0]["line"] == "Target Gettable RB (RB)"
    assert "85%" in trade[0]["why"]


# ── Renderer: board layout vs legacy fallback ─────────────────────────────

def _render_data(sc, moves):
    return {
        "team_name": "Caleb's Casting Couch", "week": 7, "record": "4-2",
        "playoff_pct": 68.0, "grades": sc["grades"],
        "roster_rows": sc["roster_rows"], "injury_rows": sc["injury_rows"],
        "trade_targets": sc["trade_targets"], "potential_trade_targets": [],
        "waiver_targets": sc["waiver_targets"],
        "cut_candidates": sc["cut_candidates"],
        "drop_add_pairs": sc["drop_add_pairs"],
        "decision_moves": moves,
        "qb_note": {"covered": True, "lead": "QB: covered, no move.",
                    "text": "QB One is QUESTIONABLE, a game-time call. "
                            "You hold 1 healthy QB behind him, so no QB add is suggested."},
        "opponent_injuries": {"entries": [{"name": "Their WR1"}],
                              "line": "Their WR1 is OUT this week."},
        "last_week": {"week": 6, "result": "W", "pf": 128.4, "pa": 112.7,
                      "opponent": "Partner B"},
        "risers_7d": [], "fallers_7d": [],
        "trade_deadline": {"weeks_remaining": 4},
    }


_AI = {"verdict": "HOLD", "headline": "Covered at QB. The leak is at RB depth.",
       "posture": "Fix RB depth this week.", "top_move": "",
       "gm_alert": "Trade deadline is 4 weeks out.",
       "trade_notes": {}, "waiver_notes": {}}


def test_board_layout_folds_sections_and_orders_decisions_first():
    sc = _scenario()
    out = fmod.render_front_office_report_html(_render_data(sc, _build(sc)), _AI)
    board = out.index("Do this first")
    assert board < out.index("Since last week") < out.index("Reference")
    # Evidence lives inside the board: swap card and Plan B precede the jump nav.
    assert out.index("for-swap-card") < out.index("for-jumpbar")
    assert "Plan B: Plan B RB" in out
    assert "Fallback drop: Worst RB" in out
    assert "Add RB" in out and "RB24" in out
    # Hold / Watch / Alert rows ride in the board, not at the bottom.
    holdbox = out.index("for-holdbox")
    assert board < holdbox < out.index("for-jumpbar")
    assert "QB: covered, no move." in out
    assert "TE One (TE) questionable." in out
    assert "Trade deadline is 4 weeks out." in out
    # Standalone sections are gone in board layout.
    assert "for-sec-title'>Trade targets" not in out
    assert "for-sec-title'>Waivers and cuts" not in out
    assert "for-sec-title'>GM alert" not in out
    assert "for-sec-title'>Positional grades" not in out
    # Reference carries the collapsed roster and injury bars.
    assert "Show table" in out and "Show all" in out
    assert "Nothing to decide" in out
    # Opponent line lives in the lineup evidence, inside the board.
    assert out.index("Opponent missing") < out.index("for-jumpbar")


def test_legacy_layout_when_no_decision_moves():
    sc = _scenario()
    data = _render_data(sc, [])
    out = fmod.render_front_office_report_html(data, _AI)
    assert "Do this first" not in out
    assert "Trade targets" in out
    assert "GM alert" in out
