"""Regression: None placeholders for empty lineup slots must not crash
numeric consumers of matchup team blocks.

#2077 keeps empty slots as None placeholders in team["starters"] so the
matchup card's rows stay aligned with roster_positions order. Every consumer
that does math over starters must skip them instead of calling .get on None.
Production 500'd on page_dashboard (compute_team_projections_for_weeks ->
team_live_totals) the first time a real lineup carried an empty slot.
"""
import pytest

pd = pytest.importorskip("pandas")  # dashboard_services.ai.weekly_recap imports it at module top
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import dashboard_services.matchups as matchups
from dashboard_services import recap_calculations as rc
from dashboard_services.ai import weekly_recap as wr
from dashboard_services.pages import recap_page


def _p(pid, name=None, pos="RB", pts=10.0):
    return {"pid": pid, "name": name or pid, "pos": pos, "nfl": "DAL", "pts": pts}


def _team(starters, rid="1"):
    return {
        "name": f"Team {rid}",
        "roster_id": rid,
        "owner_id": "u1",
        "starters": starters,
        "bench": [],
        "pts_total": 20.0,
        "proj_total": 100.0,
    }


def _blank_slot_team(rid="1"):
    # Blank RB slot at index 1 (the ESPN shape #2077 normalized).
    return _team([_p("qb1", pos="QB", pts=18.0), None, _p("wr1", pos="WR", pts=12.0)], rid)


def _historical_team(rid="1"):
    t = _blank_slot_team(rid)
    t["lineup_is_historical"] = True
    return t


# ---------- dashboard_services.matchups ----------

def test_team_live_totals_skips_none_placeholder():
    actual, live = matchups.team_live_totals(
        _blank_slot_team(), {}, {"qb1": 20.0, "wr1": 15.0},
    )
    assert actual == pytest.approx(30.0)
    assert live == pytest.approx(35.0)


def test_compute_team_projections_for_weeks_skips_none():
    # The exact production traceback: page_dashboard -> ensure_weekly_bits ->
    # compute_team_projections_for_weeks -> team_live_totals.
    out = matchups.compute_team_projections_for_weeks(
        {4: [{"left": _blank_slot_team("1"), "right": _team([_p("qb2", pos="QB", pts=5.0)], "2")}]},
        {4: {"statuses": {}}},
        {4: {"projections": {"qb1": 20.0, "wr1": 15.0, "qb2": 18.0}}},
        {"1": "A", "2": "B"},
    )
    assert out[(4, "1")] == pytest.approx(35.0)
    assert out[(4, "2")] == pytest.approx(18.0)


def test_compute_win_prob_skips_none_placeholder():
    prob = matchups.compute_win_prob(
        _blank_slot_team("1"), _team([_p("qb2", pos="QB", pts=5.0)], "2"),
        {}, {"qb1": 20.0, "wr1": 15.0, "qb2": 18.0},
    )
    assert 0.0 <= prob <= 1.0


def test_matchup_games_in_progress_skips_none_placeholder():
    m = {"left": _blank_slot_team("1"), "right": _blank_slot_team("2")}
    assert matchups._matchup_games_in_progress(m, {}) is False


def test_render_matchup_slide_proj_mode_with_blank_slot(monkeypatch):
    monkeypatch.setattr(matchups, "load_teams_index", lambda: {})
    monkeypatch.setattr(matchups, "build_offense_rankings", lambda *_a, **_k: {})
    monkeypatch.setattr(matchups, "load_week_stats", lambda *_a, **_k: {})
    monkeypatch.setattr(matchups, "load_week_schedule", lambda *_a, **_k: {})
    monkeypatch.setattr(matchups, "build_team_schedule_lookup", lambda *_a, **_k: {})
    monkeypatch.setattr(matchups, "_allow_live_game_indicators", lambda *_a, **_k: False)
    monkeypatch.setattr("utils.data_cache.load_week_projection", lambda *_a, **_k: {})
    m = {"left": _blank_slot_team("1"), "right": _blank_slot_team("2")}
    # w > proj_week turns proj mode on: _score_html -> team_live_totals,
    # compute_win_prob, _matchup_games_in_progress, pid lists all run.
    html = matchups.render_matchup_slide(
        "2026", m, w=5, proj_week=4,
        status_by_pid={},
        projections={5: {"projections": {"qb1": 20.0, "wr1": 15.0}}},
        players={}, teams={},
        team_game_lookup={},
        roster_positions=["QB", "RB", "WR", "BN"],
    )
    assert "qb1" in html or "Team 1" in html


# ---------- weekly_recap helpers ----------

def test_weekly_recap_helpers_skip_none_placeholder():
    starters = [_p("qb1", pos="QB", pts=18.0), None, _p("wr1", pos="WR", pts=12.0)]
    proj_by_pid = {"qb1": 20.0, "wr1": 15.0}
    assert wr._team_proj_total(starters, proj_by_pid) == pytest.approx(35.0)
    prob = wr._proj_win_prob(starters, [_p("qb2", pos="QB")], proj_by_pid)
    assert 0.0 <= prob <= 1.0
    out, maybe, byes, risk = wr._starter_flags(
        _team(starters), {}, proj_by_pid, {}, set(),
    )
    assert isinstance(out, list) and isinstance(risk, float)


# ---------- recap_page / recap_calculations ----------

def test_top_performers_by_roster_skips_none_placeholder():
    out = recap_page._top_performers_by_roster(
        [{"left": _historical_team("1"), "right": _historical_team("2")}],
    )
    assert out["1"][0]["pid"] == "qb1"
    assert out["1"][0]["pts"] == pytest.approx(18.0)


def test_build_lineup_analysis_skips_none_placeholder():
    result = rc.build_lineup_analysis(
        {4: [{"left": _historical_team("1"), "right": _historical_team("2")}]},
        4,
        roster_positions=["QB", "RB", "WR", "BN"],
    )
    assert isinstance(result, dict)
