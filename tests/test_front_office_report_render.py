"""Regression tests for the Front Office Report visual redesign.

The full modal report was rebuilt with a hero block (verdict stamp + team
name + stat chips), move rows with a "spots in trade-value rank" explainer,
grade-tinted cards, picklist rows for waivers/cuts, and a quieter
AI-unavailable notice. These tests pin the new markup contract.

The module's heavy deps (openai, espn_api) are absent from the pure test
env, so they are stubbed; the render helpers under test are pure.
"""
import sys
import types

import pytest


# Names this module installed into sys.modules (only when the real module was
# not already imported). Undone after the front_office_report import below so
# later-collected test modules resolve the real dashboard_services.ai.*
# modules instead of these bare stubs.
_INSTALLED_STUBS = []


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    # Track only the entries we actually install: if the real module is
    # already imported (e.g. the full-stack CI job), setdefault leaves it
    # alone and there is nothing to undo below.
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
_stub("utils.lineup_slots", canonicalize_slots=lambda x: x, count_lineup_slots=lambda x: 0)
_stub("utils.roster_strength", STARTER_THRESHOLD=0)

from dashboard_services.ai.front_office_report import (  # noqa: E402
    render_front_office_report_html,
)

# Remove the stub modules installed above. They exist only so this module can
# be imported without the heavy optional deps (openai, espn_api); the names
# front_office_report needed are already bound into its own namespace via
# from-imports. Leaving the stubs in sys.modules would poison later-collected
# test modules: e.g. `from dashboard_services.ai.prompts import GM_MEMO_SYSTEM`
# would resolve to the bare stub (no __file__, missing attrs) instead of the
# real module and fail collection.
for _stub_name in _INSTALLED_STUBS:
    sys.modules.pop(_stub_name, None)
# NOTE: _INSTALLED_STUBS is intentionally not deleted; a module-level `del`
# makes ruff F821 flag the earlier use inside _stub().


def _sample_data(**over):
    data = {
        "team_name": "Caleb's Casting Couch",
        "week": 3,
        "record": "2-0",
        "playoff_pct": 99.4,
        "last_week": {"week": 2, "result": "W", "pf": 157.6, "pa": 105.3, "opponent": "Veiny Oilers"},
        "risers_7d": [{"id": "101", "name": "Michael Wilson", "position": "WR", "trend_7d": 29}],
        "fallers_7d": [{"id": "202", "name": "Jaylen Warren", "position": "RB", "trend_7d": -58}],
        "grades": [
            {"pos": "QB", "grade": "D", "rank": 8, "of": 10},
            {"pos": "RB", "grade": "B", "rank": 4, "of": 10},
            {"pos": "WR", "grade": "A", "rank": 2, "of": 10},
            {"pos": "TE", "grade": "A", "rank": 2, "of": 10},
        ],
        "roster_rows": [
            {
                "name": "CeeDee Lamb",
                "position": "WR",
                "team": "DAL",
                "age": 27.4,
                "value": 782.6,
                "pos_rank_label": "WR5",
                "role": "Starter",
                "trend_7d": 4,
                "injury": "",
            }
        ],
        "trade_targets": [
            {
                "gets": [{"id": "1", "name": "Puka Nacua", "position": "WR", "age": 24, "value": 900.1}],
                "gives": [{"name": "Jaylen Warren", "position": "RB"}],
                "partner": "Veiny Oilers",
                "analyzer_url": "/trade?x=1",
            }
        ],
        "waiver_targets": [
            {"id": "9", "name": "WanDale Robinson", "position": "WR", "team": "NYG", "pos_rank_label": "WR48"}
        ],
        "cut_candidates": [{"name": "Chris Rodriguez Jr.", "position": "RB", "value": 12.5}],
    }
    data.update(over)
    return data


def _sample_ai(**over):
    ai = {
        "verdict": "CONTENDER",
        "headline": "Profiles as a balanced team.",
        "posture": "Buy now.",
        "gm_alert": "Watch the QB room.",
        "trade_notes": {"1": "Elite target."},
        "waiver_notes": {"9": "Pace rising."},
    }
    ai.update(over)
    return ai


def _render(data=None, ai=None):
    return render_front_office_report_html(data or _sample_data(), ai or _sample_ai())


def test_hero_renders_chips_not_dot_meta():
    out = _render()
    assert "for-hero" in out
    assert "Week 3" in out and "2-0" in out and "99% playoff odds" in out
    assert "for-report-meta" not in out
    assert "CONTENDER" in out  # verdict stamp still present


def test_move_rows_label_trade_value_spots():
    out = _render()
    assert "for-move-delta for-up" in out
    assert "for-move-delta for-down" in out
    assert "Michael Wilson" in out and "Jaylen Warren" in out
    assert "Spots gained or lost in dynasty trade-value rank" in out


def test_moves_split_into_risers_left_fallers_right():
    out = _render()
    assert "for-moves-cols" in out
    assert out.index("Risers") < out.index("Michael Wilson")
    assert out.index("Fallers") < out.index("Jaylen Warren")
    # Risers column comes before the fallers column.
    assert out.index("Michael Wilson") < out.index("Fallers")
    assert out.index("Jaylen Warren") < out.index("Spots gained or lost")


def test_move_rows_are_clickable_player_names():
    out = _render()
    assert "player-clickable" in out
    assert "data-player-id='101'" in out
    assert "data-player-id='202'" in out
    assert "data-player-name='Michael Wilson'" in out


def test_score_row_renders_win_badge():
    out = _render()
    assert "for-score-w" in out
    assert "157.6-105.3" in out


def test_grades_carry_tint_class_on_card():
    out = _render()
    assert "for-grade-card for-grade-d" in out
    assert "for-grade-card for-grade-b" in out
    assert "for-grade-card for-grade-a" in out
    # letter element itself no longer carries the color class
    assert "for-grade-letter for-grade-" not in out


def test_roster_header_says_trend():
    out = _render()
    assert ">Trend</th>" in out
    assert ">7d</th>" not in out


def test_trade_target_card_structure():
    out = _render()
    assert "for-target-from" in out and "From Veiny Oilers" in out
    assert "You give" in out
    assert "Analyze this trade" in out


def test_waivers_cuts_use_picklist():
    out = _render()
    assert "for-picklist" in out
    assert "for-pick-badge for-add" in out
    assert "for-pick-badge for-cut" in out
    assert "WanDale Robinson" in out and "Chris Rodriguez Jr." in out


def test_no_em_dashes_in_report_copy():
    out = _render()
    assert "\u2014" not in out


def test_html_escaping_preserved():
    data = _sample_data(team_name="<script>alert(1)</script>")
    out = _render(data)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_empty_sections_render_minimal_report():
    out = render_front_office_report_html({"team_name": "X"}, {})
    assert "for-hero" in out
    assert "Since last week" not in out
    assert "Trade targets" not in out


def test_css_has_new_report_selectors():
    css = open("static/dashboard.css", encoding="utf-8").read()
    for sel in (
        ".for-hero", ".for-chip", ".for-chip-hot", ".for-moves", ".for-move-delta",
        ".for-score-row", ".for-grade-card", ".for-target-from", ".for-picklist",
        ".for-pick-badge", ".for-modal-body .ai-error-notice",
    ):
        assert sel in css, f"missing CSS selector {sel}"


def test_positional_grades_match_teams_page_ranking(monkeypatch):
    import dashboard_services.ai.front_office_report as fmod
    from utils.roster_strength import rank_rosters_by_position
    from utils.utils import count_roster_positions

    # The pure-test env stubs safe_float to always return the default; the
    # grades math needs the real conversion.
    monkeypatch.setattr(
        fmod, "safe_float", lambda x, default=0.0: default if x is None else float(x)
    )
    _positional_grades = fmod._positional_grades

    lookup = {
        "p1": {"position": "WR", "value": 900},
        "p2": {"position": "WR", "value": 100},
        "p3": {"position": "WR", "value": 800},
    }
    rosters = [
        {"roster_id": 1, "players": ["p1", "p2"]},
        {"roster_id": 2, "players": ["p3"]},
    ]
    ctx = {"rosters": rosters, "roster_positions": ["WR", "WR", "BN", "BN"]}
    grades = _positional_grades(ctx, 1, lookup)
    by_pos = {g["pos"]: g for g in grades}

    team_pos_values = {
        "1": {"QB": [], "RB": [], "WR": [900.0, 100.0], "TE": []},
        "2": {"QB": [], "RB": [], "WR": [800.0], "TE": []},
    }
    _, expected_ranks = rank_rosters_by_position(
        team_pos_values,
        count_roster_positions(ctx["roster_positions"]),
        positions=["QB", "RB", "WR", "TE"],
    )
    assert by_pos["WR"]["rank"] == expected_ranks["WR"]["1"]
    assert by_pos["WR"]["of"] == 2


def test_second_of_ten_grades_an_a():
    from dashboard_services.ai.front_office_report import _grade_for_percentile
    assert _grade_for_percentile(100.0 * (10 - 2) / 9) == "A"
    assert _grade_for_percentile(100.0 * (10 - 8) / 9) == "D"
