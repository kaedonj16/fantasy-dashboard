"""Bye-week players must not be marked FINAL.

A gameless team on its bye week used to make build_status_by_pid stamp its
players STATUS_FINAL (only defenses got the "BYE" treatment). That locked 0
flowed into the matchup score header: the header's any-started check flipped
on, so a pre-game team showed a big final-style 0.0 with a small "proj N"
underneath instead of the clean proj-only number, and win probability either
vanished (bye starter without numeric points makes the stats pass bail) or
treated the bye starters' projections as already-decided zeros.

Kaedon's Week 6 case: "Never (Jamar) Chase a Ho!" starts Chase and Higgins
(CIN bye) and showed 0.0 / proj 98.4 with a 6% win bar on Tuesday.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _statuses():
    from utils.game_status import build_status_by_pid
    return build_status_by_pid


# CIN is on bye in week 6; KC and BUF play each other.
_TEAMS = {
    "KC": {"byeWeek": 10},
    "BUF": {"byeWeek": 7},
    "CIN": {"byeWeek": 6},
}
_GAMES = {
    "KC": {"status": "pre", "game": {}},
    "BUF": {"status": "pre", "game": {}},
}
_PLAYERS = {
    "qb_kc": {"team": "KC"},
    "rb_buf": {"team": "BUF"},
    "wr_cin": {"team": "CIN"},
    "te_cin": {"team": "CIN"},
}


def test_bye_week_skill_players_get_bye_status():
    statuses = _statuses()(_PLAYERS, _GAMES, _TEAMS, 6)
    assert statuses["wr_cin"] == "BYE"
    assert statuses["te_cin"] == "BYE"
    assert statuses["qb_kc"] == "not_started"
    assert statuses["rb_buf"] == "not_started"


def test_gameless_team_off_bye_week_stays_final():
    # Same shape in week 5 (CIN's bye is week 6): a missing game is a data
    # gap, not a bye, and keeps the historical FINAL verdict.
    statuses = _statuses()(_PLAYERS, _GAMES, _TEAMS, 5)
    assert statuses["wr_cin"] == "final"


def _bye_heavy_matchup():
    """Right team starts two CIN players (bye) plus one KC player."""
    left = {
        "roster_id": "1", "name": "Team A", "avatar": "", "record": "4-1",
        "pts_total": 0.0, "proj_total": 40.0,
        "starters": [
            {"pid": "qb_kc", "name": "P. Mahomes", "pos": "QB", "nfl": "KC", "pts": 0.0},
            {"pid": "rb_buf", "name": "J. Cook", "pos": "RB", "nfl": "BUF", "pts": 0.0},
        ],
    }
    right = {
        "roster_id": "2", "name": "Team B", "avatar": "", "record": "3-2",
        "pts_total": 0.0, "proj_total": 45.6,
        "starters": [
            {"pid": "wr_kc", "name": "R. Rice", "pos": "WR", "nfl": "KC", "pts": 0.0},
            {"pid": "wr_cin", "name": "J. Chase", "pos": "WR", "nfl": "CIN", "pts": 0.0},
            {"pid": "te_cin", "name": "T. Higgins", "pos": "WR", "nfl": "CIN", "pts": 0.0},
        ],
    }
    players = dict(_PLAYERS)
    players["wr_kc"] = {"team": "KC"}
    proj = {"qb_kc": 25.0, "rb_buf": 15.0, "wr_kc": 12.0, "wr_cin": 18.6, "te_cin": 15.0}
    statuses = _statuses()(players, _GAMES, _TEAMS, 6)
    return left, right, proj, statuses


def _matchups_mod():
    pytest.importorskip("requests")
    from dashboard_services import matchups as mmod
    return mmod


def test_team_live_totals_counts_bye_projections_as_pending():
    m = _matchups_mod()
    _, right, proj, statuses = _bye_heavy_matchup()
    actual, live_proj = m.team_live_totals(right, statuses, proj)
    assert actual == 0.0
    # All three starters, bye players included, count at projection: the
    # total is the full pregame number, not the non-bye slice.
    assert live_proj == pytest.approx(45.6)


def test_win_prob_counts_bye_players_as_pending():
    m = _matchups_mod()
    left, right, proj, statuses = _bye_heavy_matchup()
    p = m.compute_win_prob(left, right, statuses, proj)
    assert p is not None
    # Right projects higher (45.6 vs 40.0) with everyone pending, so the
    # left win probability must sit below 50%: the bye starters' projections
    # are undecided points, not locked zeros.
    assert p < 0.5


def test_slide_header_is_proj_only_for_bye_heavy_team(monkeypatch):
    m = _matchups_mod()
    monkeypatch.setattr("dashboard_services.api.get_nfl_state", lambda: {})
    monkeypatch.setattr(m, "load_week_stats", lambda *a, **k: {})
    monkeypatch.setattr(m, "load_week_schedule", lambda *a, **k: [])
    monkeypatch.setattr(m, "load_teams_index", lambda: {})
    monkeypatch.setattr(m, "build_offense_rankings", lambda *a: {})
    monkeypatch.setattr(m, "get_nfl_scores_for_date", lambda *a: None)
    monkeypatch.setattr("utils.data_cache.load_week_projection", lambda *a, **k: {})

    left, right, proj, statuses = _bye_heavy_matchup()
    html = m.render_matchup_slide(
        "2026", {"left": left, "right": right, "h2h": {}}, 6, 5,
        status_by_pid=statuses,
        projections={6: {"projections": proj}},
        players={}, teams={},
        team_game_lookup={},
    )
    # Neither header may render the stacked live/final form (big actual with
    # a small "proj N" line): before kickoff both are proj-only numbers.
    assert "m-espn-num'>0.0<" not in html
    assert html.count("m-proj-only") >= 2
    assert ">45.6<" in html
    # The win bar exists and favors the higher-projected (bye-heavy) team.
    bar = re.search(r'Win probability: Team A (\d+) percent, Team B (\d+) percent', html)
    assert bar, "win probability bar missing"
    assert int(bar.group(2)) > 50
