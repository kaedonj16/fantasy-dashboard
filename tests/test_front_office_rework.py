"""Tests for the Front Office Report rework (Kaedon, 2026-10-14).

Pins the fixes behind her report: 3 QBs rostered in a 1QB league with a
QUESTIONABLE starter produced a suggestion to add a 4th QB who was on IR.
Covers the deterministic Top Moves and their hard gates (QB coverage,
injury, drop pairing), the two-column injury grid, the Swap trade card,
gettability-ranked potential targets, and the trade engine's injury
filter + ranking fields.

FO helpers are tested with the same stub harness as
tests/test_front_office_injury.py (stubs popped after import so they
never leak into other modules); the engine tests import the real
context_builders, like tests/test_trade_suggestion_fairness.py.
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
    # The stubbed espn safe_float returns the default for everything; the
    # helpers under test do real value math.
    monkeypatch.setattr(
        fmod, "safe_float", lambda x, default=0.0: default if x is None else float(x)
    )


def _row(pid, name, pos, value, injury=""):
    return {"id": pid, "name": name, "position": pos, "value": value, "injury": injury}


_CTX_1QB = {"roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "BN", "BN"]}


# ── QB gates ────────────────────────────────────────────────────────────────

def test_qb_context_covered_when_starter_questionable_with_healthy_backups():
    rows = [
        _row("q1", "Starter QB", "QB", 900, "Q"),
        _row("q2", "Backup QB", "QB", 500),
        _row("q3", "Third QB", "QB", 300),
    ]
    qb = fmod._qb_context(_CTX_1QB, {"starters": ["q1"]}, rows)
    assert qb["total"] == 3
    assert qb["healthy"] == 2
    assert qb["covered"] is True
    assert qb["add_allowed"] is False
    assert qb["starter_questionable_name"] == "Starter QB"
    # Keep starter + one backup in 1QB; the worst healthy QB is the shop piece.
    assert [r["name"] for r in qb["excess"]] == ["Third QB"]
    note = fmod._qb_note(qb)
    assert note["covered"] is True
    assert note["lead"] == "QB: covered, no move."


def test_qb_context_real_hole_allows_adds():
    rows = [_row("q1", "Hurt QB", "QB", 900, "IR")]
    qb = fmod._qb_context(_CTX_1QB, {"starters": ["q1"]}, rows)
    assert qb["healthy"] == 0
    assert qb["add_allowed"] is True
    assert qb["covered"] is False
    assert fmod._qb_note(qb)["lead"] == "QB: real hole."


def test_hole_positions_use_healthy_startable_counts():
    rows = [
        _row("q1", "QB1", "QB", 600),
        _row("r1", "RB1", "RB", 600),
        _row("r2", "RB2", "RB", 600, "IR"),  # startable but cannot play
        _row("w1", "WR1", "WR", 600),
        _row("w2", "WR2", "WR", 600),
    ]
    holes = fmod._hole_positions(_CTX_1QB, rows)
    # RB: only one healthy startable for two slots. TE: nobody rostered.
    assert set(holes) == {"RB", "TE"}


def test_ai_top_move_guard_blocks_covered_qb_and_stash_adds():
    data = {
        "qb_context": {"add_allowed": False},
        "waiver_targets": [{"name": "Hurt Stash", "stash_only": True}],
    }
    assert fmod._ai_top_move_ok("Add a quarterback off waivers.", data) is False
    assert fmod._ai_top_move_ok("Add Hurt Stash as an IR stash.", data) is False
    assert fmod._ai_top_move_ok("Drop your RB5, add the top free-agent RB.", data) is True
    assert fmod._ai_top_move_ok("", data) is False


def test_waiver_bar_filter_applies_kaedons_drop_bar():
    roster_rows = [
        _row("r1", "RB1", "RB", 800),
        _row("r2", "RB2", "RB", 400),
        _row("r3", "Worst RB", "RB", 200),
    ]
    targets = [
        {"name": "Marginal RB", "position": "RB", "value": 205},   # < 200 x 1.05
        {"name": "Real RB", "position": "RB", "value": 300},       # beats the bar
        {"name": "Hole TE", "position": "TE", "value": 50},        # fills a hole
        {"name": "IR Stash RB", "position": "RB", "value": 100, "stash_only": True},
    ]
    out = fmod._waiver_bar_filter(targets, roster_rows, holes=["TE"])
    assert [w["name"] for w in out] == ["Real RB", "Hole TE", "IR Stash RB"]


# ── Deterministic Top Moves ─────────────────────────────────────────────────

def test_build_top_moves_ranked_capped_and_gated():
    moves = fmod._build_top_moves(
        trade_targets=[{
            "gets": [{"name": "Target RB", "position": "RB"}],
            "gives": [{"name": "My WR"}],
            "fairness": 0.95,
            "acceptance_rate": 70,
            "acceptance_label": "Likely to accept",
            "why_they_say_yes": "He never starts for them.",
        }],
        drop_add_pairs=[{
            "drop": {"name": "Drop RB", "value": 100},
            "add": {"name": "Add RB", "value": 300, "position": "RB"},
        }],
        qb={"total": 3, "is_sf": False, "excess": [_row("q3", "Third QB", "QB", 300)]},
        injury_rows=[{"id": "h1", "name": "Hurt RB", "injury": "IR"}],
        roster={"reserve": []},
        ctx={"roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "IR"]},
        grades=[{"pos": "RB", "rank": 9, "of": 12}],
        holes=["RB"],
        qb_needy_count=2,
    )
    assert [m["line"] for m in moves] == [
        "Trade for Target RB (RB)",
        "Drop Drop RB, add Add RB",
        "Shop Third QB",
    ]
    assert moves[0]["type"] == "Trade"
    assert "Fills your RB hole" in moves[1]["why"]
    assert "2 teams in this league are thinner at QB" in moves[2]["why"]


def test_top_moves_html_renders_ranked_list_and_qb_note():
    data = {
        "top_moves": [
            {"type": "Waiver", "line": "Drop Drop RB, add Add RB", "why": "Beats him on value."},
        ],
        "qb_note": {"covered": True, "lead": "QB: covered, no move.", "text": "You hold 2 healthy QBs."},
    }
    out = fmod._top_moves_html(data, {})
    assert "Top moves" in out
    assert "for-move-card" in out
    assert "QB: covered, no move." in out


# ── Rendering: injury grid + Swap card + potential rows ────────────────────

def test_injury_section_uses_two_column_grid():
    rows = [{
        "name": "Hurt RB", "position": "RB", "team": "SF", "injury": "IR",
        "body": "Knee", "return_label": "out ~4 wks", "action": "IR slot",
    }]
    out = fmod._injury_section_html(rows)
    assert "for-inj-grid" in out
    assert "Hurt RB" in out


def test_swap_card_carries_fairness_acceptance_and_depth():
    targets = [{
        "gets": [{"id": "1", "name": "Target RB", "position": "RB", "age": 25,
                  "value": 1240, "tier": ""}],
        "gives": [{"id": "2", "name": "My QB3", "position": "QB", "value": 720},
                  {"id": "3", "name": "My WR5", "position": "WR", "value": 460}],
        "partner": "Partner Team A",
        "partner_record": "4-2",
        "partner_direction": "rebuilding",
        "fairness_label": "Fair deal, within 5%",
        "acceptance_label": "Likely to accept",
        "why_they_say_yes": "He never starts for them.",
        "depth_checked": True,
        "depth_warnings": [],
        "analyzer_url": "/trade?a=1",
    }]
    out = fmod._trade_targets_html(targets, {})
    assert "for-swap-card" in out and "for-swap-get" in out and "for-swap-give" in out
    assert "You get" in out and "You give" in out
    assert "Fair deal, within 5%" in out
    assert "Likely to accept" in out
    assert "no new hole created" in out
    assert "Partner Team A" in out


def test_potential_targets_rank_surplus_first_with_gap():
    lookup = {
        "vrb1": {"name": "My RB1", "position": "RB", "value": 900},
        "vrb2": {"name": "My RB2", "position": "RB", "value": 800},
        "vrb3": {"name": "My RB3", "position": "RB", "value": 700},
        "p1": {"name": "Their RB1", "position": "RB", "value": 1000},
        "p2": {"name": "Their RB2", "position": "RB", "value": 900},
        "p3": {"name": "Their RB3", "position": "RB", "value": 500},
    }
    ctx = {
        "rosters": [
            {"roster_id": 1, "players": ["vrb1", "vrb2", "vrb3"]},
            {"roster_id": 2, "players": ["p1", "p2", "p3"]},
        ],
        "roster_map": {"2": "Partner Team A"},
    }
    out = fmod._potential_trade_targets(ctx, 1, lookup, ["RB"])
    # Their RB3 is genuine surplus (behind two starters) and ranks first
    # despite the lowest value; the per-position cap keeps two.
    assert [p["name"] for p in out] == ["Their RB3", "Their RB2"]
    top = out[0]
    assert top["is_surplus"] is True
    assert top["attainability"] == "Realistic"
    # Viewer surplus package is My RB3 (700) vs his 500: gap = 500/700.
    assert top["gap_pct"] == 71
    assert top["starting_point"]["name"] == "My RB3"


# ── Trade engine: injury filter + ranking fields ────────────────────────────

def test_engine_never_targets_seriously_injured_players(monkeypatch):
    import dashboard_services.ai.context_builders as cb

    def _mv(pid, name, pos, val):
        return {"player_id": pid, "name": name, "position": pos, "value": val}

    mvt = [
        _mv("v_wr1", "Viewer WR1", "WR", 3600),
        _mv("v_wr2", "Viewer WR2", "WR", 3100),
        _mv("v_wr3", "Viewer WR3", "WR", 2400),
        _mv("v_rb1", "Viewer RB1", "RB", 700),
        _mv("v_qb1", "Viewer QB1", "QB", 1500),
        _mv("v_te1", "Viewer TE1", "TE", 1000),
        _mv("r2_qb1", "T2 QB1", "QB", 1100),
        _mv("r2_te1", "T2 TE1", "TE", 1000),
        # Team 2's RB1 (2400) would be the fair deal at exactly the
        # viewer's spare-WR value, but he is on IR.
        _mv("r2_rb1", "T2 RB1", "RB", 2400),
        _mv("r2_rb2", "T2 RB2", "RB", 2700),
        _mv("r2_rb3", "T2 RB3", "RB", 900),
        _mv("r2_wr1", "T2 WR1", "WR", 520),
    ]
    rosters = [
        {"roster_id": "1", "players": ["v_wr1", "v_wr2", "v_wr3", "v_rb1", "v_qb1", "v_te1"]},
        {"roster_id": "2", "players": ["r2_qb1", "r2_te1", "r2_rb1", "r2_rb2", "r2_rb3", "r2_wr1"]},
    ]
    for i in range(3, 11):
        pids = [f"r{i}_qb1", f"r{i}_te1", f"r{i}_rb1", f"r{i}_rb2", f"r{i}_wr1", f"r{i}_wr2"]
        mvt += [
            _mv(f"r{i}_qb1", f"T{i} QB1", "QB", 1100),
            _mv(f"r{i}_te1", f"T{i} TE1", "TE", 1000),
            _mv(f"r{i}_rb1", f"T{i} RB1", "RB", 1500),
            _mv(f"r{i}_rb2", f"T{i} RB2", "RB", 1200),
            _mv(f"r{i}_wr1", f"T{i} WR1", "WR", 1500),
            _mv(f"r{i}_wr2", f"T{i} WR2", "WR", 1300),
        ]
        rosters.append({"roster_id": str(i), "players": pids})
    ctx = {
        "rosters": rosters,
        "model_value_table": mvt,
        "roster_map": {r["roster_id"]: f"Team {r['roster_id']}" for r in rosters},
        "picks_by_roster": {},
        "standings_map": {r["roster_id"]: {"wins": 5, "losses": 5} for r in rosters},
        "rookie_rankings": [],
        "league_type": "1qb",
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE"],
        "players": {"r2_rb1": {"injury_status": "IR"}},
    }
    monkeypatch.setattr(
        cb, "build_team_gm_context",
        lambda c, rid: {"team_name": "Viewer", "direction": "balanced"},
    )
    res = cb.build_trade_suggestions_context(ctx, "1")
    assert res is not None
    all_targets = [
        t["name"] for p in res["top_partners"] for t in (p.get("targets_they_have") or [])
    ]
    assert "T2 RB1" not in all_targets
    t2 = next(p for p in res["top_partners"] if p["team_name"] == "Team 2")
    assert [t["name"] for t in t2["targets_they_have"]] == ["T2 RB2"]
    # Ranking now carries the signals it ranks by.
    assert "gettability" in t2
    assert "alt_deal" in t2
    assert "depth_checked" in t2
