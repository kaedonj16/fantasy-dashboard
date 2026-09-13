"""Live matchup projections: a player's projection updates as their game plays.

Before kickoff a starter shows the pregame projection. Once their game is in
progress the matchup page shows a *live projected finish* -- the points already
banked plus the slice of the pregame projection still to come -- instead of a
frozen pregame number. These contracts guard that behaviour and the shared
helpers (game_fraction_remaining / live_projected_final) it is built on.
"""
from __future__ import annotations

import pytest


def _matchups():
    pytest.importorskip("flask")
    pytest.importorskip("requests")
    from dashboard_services import matchups as mmod
    return mmod


def test_game_fraction_remaining_by_state():
    m = _matchups()
    f = m.game_fraction_remaining
    # Scheduled -> whole game ahead; final -> none left.
    assert f({"gameStatusCode": "0"}) == 1.0
    assert f({"gameStatusCode": "2"}) == 0.0
    # Start of Q1 (15:00 on the clock) is effectively the full game.
    assert f({"gameStatusCode": "1", "lineScore": {"period": "1"}, "gameClock": "15:00"}) == pytest.approx(1.0)
    # End of the first half (Q2, 0:00) -> exactly half of regulation remains.
    assert f({"gameStatusCode": "1", "lineScore": {"period": "2"}, "gameClock": "0:00"}) == pytest.approx(0.5)
    # Mid third quarter.
    assert f({"gameStatusCode": "1", "lineScore": {"period": "3"}, "gameClock": "7:30"}) == pytest.approx(0.375)
    # Late fourth quarter -> almost over.
    assert f({"gameStatusCode": "1", "lineScore": {"period": "4"}, "gameClock": "2:00"}) == pytest.approx(2.0 / 60.0)
    # Overtime: regulation spent, treat as essentially over.
    assert f({"gameStatusCode": "1", "lineScore": {"period": "OT"}, "gameClock": "8:00"}) == pytest.approx(0.02)
    # ESPN-style "Q2" period label parses the same as "2".
    assert f({"gameStatusCode": "1", "lineScore": {"period": "Q2"}, "gameClock": "0:00"}) == pytest.approx(0.5)


def test_game_fraction_remaining_unknown_returns_none():
    m = _matchups()
    f = m.game_fraction_remaining
    # No game, or a live game with no readable quarter -> undeterminable.
    assert f(None) is None
    assert f({"gameStatusCode": "1"}) is None
    assert f({}) is None


def test_live_projected_final_blends_actual_and_remaining_projection():
    m = _matchups()
    half = {"gameStatusCode": "1", "lineScore": {"period": "2"}, "gameClock": "0:00"}
    # 4 banked + 16 pregame * 0.5 remaining = 12.0 projected finish.
    assert m.live_projected_final(4.0, 16.0, half) == pytest.approx(12.0)
    # A player already past their projection still climbs above it late.
    late = {"gameStatusCode": "1", "lineScore": {"period": "4"}, "gameClock": "7:30"}
    assert m.live_projected_final(18.0, 16.0, late) == pytest.approx(18.0 + 16.0 * 0.125)
    # Unknown progress falls back to the pregame projection (prior behaviour).
    assert m.live_projected_final(4.0, 16.0, None) == pytest.approx(16.0)


def test_team_live_totals_projects_in_progress_finish_with_frac_lookup():
    m = _matchups()
    team = {"starters": [{"pid": "a", "nfl": "KC", "pts": 4.0, "pos": "QB"}]}
    proj = {"a": 16.0}
    tgl = {"KC": {"gameStatusCode": "1", "lineScore": {"period": "2"}, "gameClock": "0:00"}}
    frac = m.make_frac_lookup(tgl)

    actual, live = m.team_live_totals(team, {"a": m.STATUS_IN_PROGRESS}, proj, frac_lookup=frac)
    assert actual == pytest.approx(4.0)
    # Live projected finish, not frozen at the 4.0 already scored.
    assert live == pytest.approx(12.0)

    # Without the lookup, the old behaviour holds: in-progress freezes at actual.
    _, live_frozen = m.team_live_totals(team, {"a": m.STATUS_IN_PROGRESS}, proj)
    assert live_frozen == pytest.approx(4.0)


def test_team_live_totals_not_started_uses_pregame_projection():
    m = _matchups()
    team = {"starters": [{"pid": "a", "nfl": "KC", "pts": 0.0, "pos": "QB"}]}
    proj = {"a": 16.0}
    tgl = {"KC": {"gameStatusCode": "0"}}
    frac = m.make_frac_lookup(tgl)
    _, live = m.team_live_totals(team, {"a": m.STATUS_NOT_STARTED}, proj, frac_lookup=frac)
    assert live == pytest.approx(16.0)


def test_win_prob_uses_live_remaining_projection():
    m = _matchups()
    # Left QB is crushing their projection at halftime; right QB is on pace.
    left = {"starters": [{"pid": "l", "nfl": "KC", "pts": 20.0, "pos": "QB"}]}
    right = {"starters": [{"pid": "r", "nfl": "SF", "pts": 8.0, "pos": "QB"}]}
    proj = {"l": 18.0, "r": 18.0}
    status = {"l": m.STATUS_IN_PROGRESS, "r": m.STATUS_IN_PROGRESS}
    tgl = {
        "KC": {"gameStatusCode": "1", "lineScore": {"period": "2"}, "gameClock": "0:00"},
        "SF": {"gameStatusCode": "1", "lineScore": {"period": "2"}, "gameClock": "0:00"},
    }
    frac = m.make_frac_lookup(tgl)

    live_p = m.compute_win_prob(left, right, status, proj, frac_lookup=frac)
    # Live model: left projects ~29 (20 + 9), right ~17 (8 + 9) -> left favoured.
    assert live_p > 0.5

    # The frozen model banks both at their current points (20 vs 8) with no
    # pending variance, so it snaps to a near-certain result. The live model,
    # carrying remaining projection as variance, should be less extreme.
    frozen_p = m.compute_win_prob(left, right, status, proj)
    assert live_p < frozen_p


def _render_live_slide(monkeypatch, *, status, actual, proj, game):
    m = _matchups()
    monkeypatch.setattr("dashboard_services.api.get_nfl_state", lambda: {})
    monkeypatch.setattr(m, "load_week_stats", lambda *a, **k: {})
    monkeypatch.setattr(m, "load_week_schedule", lambda *a, **k: [])
    monkeypatch.setattr(m, "load_teams_index", lambda: {})
    monkeypatch.setattr(m, "build_offense_rankings", lambda *a: {})
    monkeypatch.setattr(m, "get_nfl_scores_for_date", lambda *a: None)
    monkeypatch.setattr("utils.utils.load_week_projection", lambda *a, **k: {})

    matchup = {
        "left": {
            "roster_id": "1", "name": "Team A", "avatar": "", "record": "1-0",
            "pts_total": actual, "proj_total": proj,
            "starters": [{"pid": "p1", "name": "Patrick Mahomes", "pos": "QB", "pts": actual, "nfl": "KC"}],
        },
        "right": {
            "roster_id": "2", "name": "Team B", "avatar": "", "record": "0-1",
            "pts_total": 0.0, "proj_total": 0.0, "starters": [],
        },
        "h2h": {},
    }
    return m.render_matchup_slide(
        "2025", matchup, w=3, proj_week=2,
        status_by_pid={"p1": status},
        projections={3: {"projections": {"p1": proj}}},
        players={}, teams={},
        team_game_lookup={"KC": game} if game else {},
    )


def test_slide_shows_live_projected_finish_for_in_progress_player(monkeypatch):
    m = _matchups()
    half = {"gameStatusCode": "1", "lineScore": {"period": "2"}, "gameClock": "0:00"}
    # actual 4.0, pregame proj 16.0, halftime -> live finish 12.0.
    html = _render_live_slide(
        monkeypatch, status=m.STATUS_IN_PROGRESS, actual=4.0, proj=16.0, game=half,
    )
    # The live projected finish is rendered; the frozen pregame 16.0 is not the
    # projection shown for the in-progress starter.
    assert "12.0" in html


def test_slide_keeps_pregame_projection_before_kickoff(monkeypatch):
    m = _matchups()
    scheduled = {"gameStatusCode": "0"}
    html = _render_live_slide(
        monkeypatch, status=m.STATUS_NOT_STARTED, actual=0.0, proj=16.0, game=scheduled,
    )
    # Not started -> the pregame projection is shown as-is.
    assert "16.0" in html
