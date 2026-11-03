"""Mid-season Playoff Odds must never fall back to a preseason projection.

Regression for Fleaflicker league 92916 (verified live, Week 9): identical
/api/playoff-odds calls alternated between the correct in-season sim and a
bogus preseason projection (every team 0-0, 14 weeks remaining). One worker
held a league ctx whose team_stats was seeded 0-0 (built while the weekly
fetch came back empty), and simulate_playoff_odds keyed its preseason path
on team_stats alone, so it fabricated a full 0-0 season and cached it.

The guard: when team_stats shows no games but the same ctx carries hard
evidence the season already started (finalized scored df_weekly rows, or
the NFL-state week for a non-future season), the sim returns [] instead of
running the projected path. A genuine preseason ctx still projects.
"""

import pytest

# simulate_playoff_odds imports numpy at module load and the fixtures below
# build pandas frames. The fast "lint" CI shard lacks both, so guard first --
# otherwise importing here raises at COLLECTION and aborts the whole run.
pytest.importorskip("numpy")
pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

import data_building.simulate_playoff_odds as spo  # noqa: E402


def _gameless_team_stats(n=4):
    return pd.DataFrame({
        "owner": [f"Team {i}" for i in range(n)],
        "Wins": [0] * n, "Losses": [0] * n, "Ties": [0] * n,
        "PF": [0.0] * n, "PA": [0.0] * n, "G": [0] * n,
    })


def _finalized_weekly():
    # Real df_weekly shape from build_matchups_by_week: one finalized,
    # scored week (week 1) plus the in-progress week flagged not finalized.
    return pd.DataFrame({
        "week": [1, 1, 2, 2],
        "matchup_id": [1, 1, 1, 1],
        "roster_id": ["1", "2", "1", "2"],
        "owner": ["Team 0", "Team 1", "Team 0", "Team 1"],
        "points": [110.5, 90.0, 40.0, 35.0],
        "finalized": [True, True, False, False],
    })


def _ctx(**over):
    ctx = {
        "league_id": "92916",
        "season": 2026,
        "current_week": 0,
        "league_settings": {"playoff_week_start": 15, "playoff_teams": 6},
        "team_stats": _gameless_team_stats(),
        "roster_map": {},
        "rosters": [],
        "roster_positions": [],
    }
    ctx.update(over)
    return ctx


@pytest.fixture
def no_projection(monkeypatch):
    """Fail loudly if the preseason projection machinery is entered."""
    def _boom(*a, **k):
        raise AssertionError("projected path must not run for a started season")
    monkeypatch.setattr(spo, "build_ppg_map", _boom)
    monkeypatch.setattr(spo, "_run_mc", _boom)


def test_gameless_stats_with_finalized_weekly_returns_empty(no_projection):
    ctx = _ctx(current_week=9, current_season=2026, df_weekly=_finalized_weekly())
    assert spo.simulate_playoff_odds(ctx, platform="fleaflicker") == []


def test_gameless_stats_with_advanced_week_and_empty_weekly_returns_empty(no_projection):
    # The exact live shape: df_weekly came back empty at ctx build time, so
    # there are no finalized rows -- only the NFL-state week proves the
    # season started (season == current_season, week 9).
    ctx = _ctx(current_week=9, current_season=2026, df_weekly=pd.DataFrame())
    assert spo.simulate_playoff_odds(ctx, platform="fleaflicker") == []


def test_gameless_stats_future_season_still_projects(monkeypatch):
    # A future season's ctx built during the current season legitimately
    # has current_week >= 2 (the live NFL week) and no games: the week
    # signal alone must not block its preseason projection.
    called = {}

    def fake_run_mc(teams, matchups, profiles, playoff_teams, n_sims, seed, **kw):
        called["teams"] = teams
        return [{"roster_id": t["roster_id"], "playoff_pct": 25.0} for t in teams]

    monkeypatch.setattr(spo, "build_ppg_map", lambda ctx: ({}, {}))
    monkeypatch.setattr(
        spo, "_estimate_from_rosters",
        lambda ctx, ppg_map=None, pos_map=None: [
            {"roster_id": i, "wins": 0, "losses": 0, "ties": 0} for i in range(1, 5)
        ],
    )
    monkeypatch.setattr(spo, "_resolve_schedule", lambda *a, **k: {})
    monkeypatch.setattr(spo, "_build_ctx_week_profiles", lambda *a, **k: {})
    monkeypatch.setattr(spo, "_run_mc", fake_run_mc)

    ctx = _ctx(season=2027, current_week=9, current_season=2026,
               df_weekly=pd.DataFrame())
    out = spo.simulate_playoff_odds(ctx, platform="fleaflicker")
    assert called.get("teams"), "projected path should still run preseason"
    assert out and all(r["is_projected"] for r in out)


def test_genuine_preseason_ctx_still_projects(monkeypatch):
    called = {}

    def fake_run_mc(teams, matchups, profiles, playoff_teams, n_sims, seed, **kw):
        called["teams"] = teams
        return [{"roster_id": t["roster_id"], "playoff_pct": 25.0} for t in teams]

    monkeypatch.setattr(spo, "build_ppg_map", lambda ctx: ({}, {}))
    monkeypatch.setattr(
        spo, "_estimate_from_rosters",
        lambda ctx, ppg_map=None, pos_map=None: [
            {"roster_id": i, "wins": 0, "losses": 0, "ties": 0} for i in range(1, 5)
        ],
    )
    monkeypatch.setattr(spo, "_resolve_schedule", lambda *a, **k: {})
    monkeypatch.setattr(spo, "_build_ctx_week_profiles", lambda *a, **k: {})
    monkeypatch.setattr(spo, "_run_mc", fake_run_mc)

    ctx = _ctx(current_week=0, df_weekly=pd.DataFrame())
    out = spo.simulate_playoff_odds(ctx, platform="fleaflicker")
    assert called.get("teams"), "projected path should still run preseason"
    assert out and all(r["is_projected"] for r in out)


def test_season_started_evidence_signals():
    assert spo.season_started_evidence(
        _ctx(current_week=9, df_weekly=_finalized_weekly()))
    assert not spo.season_started_evidence(
        _ctx(current_week=1, df_weekly=_finalized_weekly().iloc[:2].assign(
            finalized=[False, False], week=[1, 1])))
    assert not spo.season_started_evidence(_ctx())
    assert not spo.season_started_evidence({})
    assert not spo.season_started_evidence(None)
