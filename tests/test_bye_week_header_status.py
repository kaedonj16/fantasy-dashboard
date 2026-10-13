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

Follow-up: #2556 gated the BYE verdict on the teams-index byeWeek metadata.
That metadata is stale in production (it still carries prior-season bye
weeks), so the gate fired only for the teams whose bye did not move (HOU
and MIN, week 6 both years) and kept failing for CIN and DET. The verdict
now keys off the loaded week's schedule itself: a team absent from a loaded
schedule is on its bye, whatever the metadata says.
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
# The production shape that defeated #2556: the teams index still carries
# the PRIOR season's bye weeks (CIN was week 10 last year), so a metadata
# gate misses the bye entirely.
_TEAMS_STALE = {
    "KC": {"byeWeek": 10},
    "BUF": {"byeWeek": 7},
    "CIN": {"byeWeek": 10},
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


def test_bye_verdict_ignores_stale_bye_week_metadata():
    # Regression for the live miss after #2556 merged and deployed: with
    # last season's byeWeek values in the index, CIN's bye players must
    # still come out BYE, because the week 6 schedule has no CIN game.
    statuses = _statuses()(_PLAYERS, _GAMES, _TEAMS_STALE, 6)
    assert statuses["wr_cin"] == "BYE"
    assert statuses["te_cin"] == "BYE"
    # The DEF verdict (pid = team code) follows the same schedule rule.
    assert statuses["CIN"] == "BYE"


def test_bye_verdict_needs_no_metadata_at_all():
    teams = {"KC": {}, "BUF": {}, "CIN": {}}
    statuses = _statuses()(_PLAYERS, _GAMES, teams, 6)
    assert statuses["wr_cin"] == "BYE"
    assert statuses["CIN"] == "BYE"


def test_empty_schedule_stays_not_started():
    # No schedule loaded at all is a data outage, not a league-wide bye:
    # keep the historical NOT_STARTED fallback.
    statuses = _statuses()(_PLAYERS, {}, _TEAMS_STALE, 6)
    assert statuses["wr_cin"] == "not_started"
    assert statuses["CIN"] == "not_started"


def _bye_heavy_matchup(teams=None):
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
    statuses = _statuses()(players, _GAMES, teams or _TEAMS, 6)
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


def test_slide_header_proj_only_with_stale_bye_metadata(monkeypatch):
    """Kaedon's live case after #2556 deployed: stale teams-index byeWeeks.

    The left team (no bye starters) rendered proj-only while the right
    team, starting Chase and Higgins, rendered the stacked 0.0 / proj N
    form because #2556's metadata gate never fired for CIN. With the
    schedule-keyed verdict both headers are proj-only.
    """
    m = _matchups_mod()
    monkeypatch.setattr("dashboard_services.api.get_nfl_state", lambda: {})
    monkeypatch.setattr(m, "load_week_stats", lambda *a, **k: {})
    monkeypatch.setattr(m, "load_week_schedule", lambda *a, **k: [])
    monkeypatch.setattr(m, "load_teams_index", lambda: {})
    monkeypatch.setattr(m, "build_offense_rankings", lambda *a: {})
    monkeypatch.setattr(m, "get_nfl_scores_for_date", lambda *a: None)
    monkeypatch.setattr("utils.data_cache.load_week_projection", lambda *a, **k: {})

    left, right, proj, statuses = _bye_heavy_matchup(teams=_TEAMS_STALE)
    assert statuses["wr_cin"] == "BYE"
    html = m.render_matchup_slide(
        "2026", {"left": left, "right": right, "h2h": {}}, 6, 5,
        status_by_pid=statuses,
        projections={6: {"projections": proj}},
        players={}, teams={},
        team_game_lookup={},
    )
    assert "m-espn-num'>0.0<" not in html
    assert html.count("m-proj-only") >= 2
    assert ">45.6<" in html
