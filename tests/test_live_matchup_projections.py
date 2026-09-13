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


def test_live_projected_final_clock_only_without_position():
    m = _matchups()
    half = {"gameStatusCode": "1", "lineScore": {"period": "2"}, "gameClock": "0:00"}
    # With no position the pace blend is off, so it's the pure clock estimate:
    # 4 banked + 16 pregame * 0.5 remaining = 12.0.
    assert m.live_projected_final(4.0, 16.0, half) == pytest.approx(12.0)
    late = {"gameStatusCode": "1", "lineScore": {"period": "4"}, "gameClock": "7:30"}
    assert m.live_projected_final(18.0, 16.0, late) == pytest.approx(18.0 + 16.0 * 0.125)
    # Unknown progress falls back to the pregame projection (prior behaviour).
    assert m.live_projected_final(4.0, 16.0, None, pos="QB") == pytest.approx(16.0)


def test_pace_blend_reacts_to_over_and_under_performance():
    m = _matchups()
    # At halftime a skill player's live finish blends pregame rate with the pace
    # they've actually set. An on-pace player lands right on their projection...
    assert m.live_final_from_frac(8.0, 16.0, 0.5, "QB") == pytest.approx(16.0)
    # ...a cold start is marked down below the pregame number...
    cold = m.live_final_from_frac(4.0, 16.0, 0.5, "QB")
    assert 4.0 < cold < 16.0
    # ...and a hot start is marked up above it.
    hot = m.live_final_from_frac(12.0, 16.0, 0.5, "QB")
    assert hot > 16.0
    # The pure clock model (what a no-position / K/DEF call uses) would sit at a
    # flat 12.0 for the cold case regardless of the slow start; the pace blend
    # is strictly more pessimistic there.
    assert cold < m.live_final_from_frac(4.0, 16.0, 0.5, "")


def test_pace_blend_is_damped_early_and_position_aware():
    m = _matchups()
    # One early score barely moves the number: at ~5% elapsed the pace term is
    # weighted near zero, so a hot Q1 stays close to the clock estimate.
    early = m.live_final_from_frac(7.0, 12.0, 0.95, "WR")
    clock_early = 7.0 + 12.0 * 0.95
    assert early == pytest.approx(clock_early, abs=1.5)
    # Kicker / defense scoring is too lumpy to extrapolate: an early made FG or a
    # defensive TD uses the pure clock estimate, not pace.
    assert m.live_final_from_frac(3.0, 8.0, 0.5, "K") == pytest.approx(3.0 + 8.0 * 0.5)
    assert m.live_final_from_frac(6.0, 7.0, 0.5, "DEF") == pytest.approx(6.0 + 7.0 * 0.5)
    # An unreadable fraction returns the pregame projection unchanged.
    assert m.live_final_from_frac(4.0, 16.0, None, "QB") == pytest.approx(16.0)


def test_team_live_totals_projects_in_progress_finish_with_frac_lookup():
    m = _matchups()
    team = {"starters": [{"pid": "a", "nfl": "KC", "pts": 4.0, "pos": "QB"}]}
    proj = {"a": 16.0}
    tgl = {"KC": {"gameStatusCode": "1", "lineScore": {"period": "2"}, "gameClock": "0:00"}}
    frac = m.make_frac_lookup(tgl)

    actual, live = m.team_live_totals(team, {"a": m.STATUS_IN_PROGRESS}, proj, frac_lookup=frac)
    assert actual == pytest.approx(4.0)
    # Live projected finish (pace-blended for a QB, cold start at halftime), not
    # frozen at the 4.0 already scored and below the flat pregame 16.0.
    assert live == pytest.approx(11.0)

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


def test_win_prob_floors_team_variance_to_realistic_cv():
    """A full pregame lineup's spread must reflect team-level dispersion
    (CV ~0.24), not the tight independent per-player sum that ran the win bar
    to 1%/99%. Summing nine independent starter variances dilutes the team
    total by ~1/sqrt(9); the floor restores a realistic coefficient of
    variation."""
    from math import erf, sqrt

    m = _matchups()
    # Nine even starters per side; left projects 165, right 126 (a 39-pt margin).
    left = {"starters": [{"pid": f"l{i}", "pts": 0.0, "pos": "WR"} for i in range(9)]}
    right = {"starters": [{"pid": f"r{i}", "pts": 0.0, "pos": "WR"} for i in range(9)]}
    proj = {f"l{i}": 165.0 / 9 for i in range(9)}
    proj.update({f"r{i}": 126.0 / 9 for i in range(9)})
    status = {pid: m.STATUS_NOT_STARTED for pid in proj}

    p = m.compute_win_prob(left, right, status, proj)

    # Analytic value with the CV floor active on both sides (it dominates the
    # independent sum for a full lineup).
    cv = 0.24
    var = (cv * 165.0) ** 2 + (cv * 126.0) ** 2
    z = (165.0 - 126.0) / (sqrt(var) * sqrt(2))
    expected = 0.5 * (1 + erf(z))
    assert p == pytest.approx(expected, abs=1e-6)
    # Well short of the old near-certain read (~92% before the floor).
    assert p < 0.85


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
    # QB, actual 4.0 at halftime on a 16.0 pregame -> pace-blended finish 11.0.
    html = _render_live_slide(
        monkeypatch, status=m.STATUS_IN_PROGRESS, actual=4.0, proj=16.0, game=half,
    )
    # The live projected finish is rendered; the frozen pregame 16.0 is not the
    # projection shown for the in-progress starter.
    assert "11.0" in html
    assert ">16.0<" not in html


def test_slide_keeps_pregame_projection_before_kickoff(monkeypatch):
    m = _matchups()
    scheduled = {"gameStatusCode": "0"}
    html = _render_live_slide(
        monkeypatch, status=m.STATUS_NOT_STARTED, actual=0.0, proj=16.0, game=scheduled,
    )
    # Not started -> the pregame projection is shown as-is.
    assert "16.0" in html
