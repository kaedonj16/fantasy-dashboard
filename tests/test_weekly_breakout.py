"""Regression tests for the in-season weekly breakout engine.

These are pure (no DB, no Flask): the scorer takes plain weekly-row dicts and the
context resolver takes an injected schedule reader. They lock down the behaviors
the rewrite exists to guarantee - correct in-season path selection, non
overlapping trend windows, strict as-of cutoffs, opportunity-before-production,
provisional rookie handling, share-unit consistency, and preserve-on-failure.
"""
from __future__ import annotations

import sys
import types
from datetime import date

import pytest

wb = pytest.importorskip("data_building.breakout_engine.weekly_breakout")
wr = pytest.importorskip("data_building.breakout_engine.weekly_runner")


def wk(week, snap=None, ts=None, tgt=None, car=None, pa=None, ppr=0.0, snaps=1):
    """One weekly_metrics-shaped row. Shares are 0-100; None means unknown."""
    return {
        "week": week, "snap_pct": snap, "target_share": ts, "targets": tgt,
        "carries": car, "pass_att": pa, "ppr_pts": ppr, "snaps": snaps,
    }


WR = {"player_id": "1", "player_name": "Test WR", "team": "AAA", "position": "WR"}
RB = {"player_id": "2", "player_name": "Test RB", "team": "BBB", "position": "RB"}


# ---------------------------------------------------------------------------
# trend windows: non-overlapping, momentum through three games
# ---------------------------------------------------------------------------

def test_split_windows_never_overlap():
    for n in range(1, 9):
        rows = [wk(i) for i in range(1, n + 1)]
        recent, baseline = wb.split_windows(rows)
        r_weeks = {r["week"] for r in recent}
        b_weeks = {r["week"] for r in baseline}
        assert not (r_weeks & b_weeks), f"overlap at n={n}"


def test_three_games_produce_momentum_not_zero():
    # The old recent-3-vs-season comparison reported zero momentum through three
    # games (same games on both sides). Here recent and baseline are disjoint.
    rows = [wk(1, 30), wk(2, 30), wk(3, 60)]
    recent, baseline = wb.split_windows(rows)
    assert [r["week"] for r in recent] == [3]
    assert [r["week"] for r in baseline] == [1, 2]


def test_four_plus_games_uses_two_recent_up_to_four_baseline():
    rows = [wk(i) for i in range(1, 8)]  # 7 games
    recent, baseline = wb.split_windows(rows)
    assert [r["week"] for r in recent] == [6, 7]
    assert [r["week"] for r in baseline] == [2, 3, 4, 5]  # preceding 4, non-overlapping


# ---------------------------------------------------------------------------
# strict as-of cutoff
# ---------------------------------------------------------------------------

def test_cutoff_ignores_future_weeks():
    rows = [wk(1, 20, tgt=2), wk(2, 25, tgt=3), wk(3, 60, tgt=8), wk(4, 70, tgt=9)]
    res = wb.score_player(WR, rows, cutoff_week=2)
    assert res["evaluated_weeks"] == [1, 2]
    assert 3 not in res["recent_weeks"] and 4 not in res["recent_weeks"]


def test_cutoff_is_strict_for_historical_scoring():
    # A future breakout must not leak backwards into an earlier as-of read.
    rows = [wk(1, 55, tgt=6), wk(2, 56, tgt=6), wk(3, 90, tgt=14)]
    early = wb.score_player(WR, rows, cutoff_week=2)
    assert early["evaluated_weeks"] == [1, 2]
    assert early["breakout_score"] < 20  # no jump visible yet


# ---------------------------------------------------------------------------
# opportunity before production
# ---------------------------------------------------------------------------

def test_rising_usage_scores_even_without_fantasy_spike():
    # Snaps and targets climb, but fantasy points stay flat/low (no scores yet).
    rows = [wk(1, 25, 8, 2, ppr=3), wk(2, 30, 9, 3, ppr=4),
            wk(3, 55, 18, 6, ppr=5), wk(4, 68, 24, 8, ppr=6)]
    res = wb.score_player(WR, rows, cutoff_week=4)
    assert res["breakout_score"] >= wb.EMERGING_MIN_SCORE
    assert res["fantasy"]["spike_without_role"] is False


def test_td_spike_without_role_growth_is_not_a_breakout():
    # Flat usage, one big fantasy game (TDs/long play). Score stays low and the
    # spike is flagged as a risk rather than establishing a breakout.
    rows = [wk(1, 55, 18, 6, ppr=8), wk(2, 54, 17, 6, ppr=7),
            wk(3, 56, 18, 6, ppr=7), wk(4, 55, 17, 5, ppr=28)]
    res = wb.score_player(WR, rows, cutoff_week=4)
    assert res["breakout_score"] < wb.WATCHLIST_MIN_SCORE
    assert res["fantasy"]["spike_without_role"] is True
    assert res["classification"] == "watchlist"


def test_role_growth_and_workload_both_count():
    # A 45%->65% snap move (bigger role, bigger jump) must outrank 5%->15%.
    small = wb._share_growth(5.0, 15.0)["points"]
    big = wb._share_growth(45.0, 65.0)["points"]
    assert big > small
    # And a trivial wiggle on an established starter earns almost nothing.
    assert wb._share_growth(54.0, 55.0)["points"] < 5.0


def test_correlated_signals_not_triple_counted():
    # snaps + target share + targets all rise together; the composite must not be
    # the naive sum of three ~full signals (which would blow past 100 trivially
    # and reward one change three times).
    rows = [wk(1, 30, 10, 3), wk(2, 32, 11, 3), wk(3, 62, 22, 7), wk(4, 66, 24, 8)]
    res = wb.score_player(WR, rows, cutoff_week=4)
    pts = [s["points"] for s in res["signals"].values() if s.get("available")]
    assert res["breakout_score"] <= 100.0
    assert res["breakout_score"] < sum(pts)  # deduplicated, not summed


# ---------------------------------------------------------------------------
# early season / rookies / prior baseline
# ---------------------------------------------------------------------------

def test_rookie_week1_is_provisional_without_prior_history():
    res = wb.score_player(RB, [wk(1, 70, None, 4, car=18, ppr=15)], cutoff_week=1)
    assert res["provisional"] is True
    assert res["baseline_source"] == "none"
    assert res["confidence"] <= 35.0            # provisional cap
    assert res["classification"] in ("watchlist", "temporary_opportunity")


def test_prior_season_baseline_used_when_thin_current_data():
    prior = {"snap_pct": 20.0, "target_share": 6.0, "targets_pg": 2.0,
             "carries_pg": 0.0, "pass_att_pg": 0.0}
    res = wb.score_player(WR, [wk(3, 68, 22, 8)], prior_baseline=prior, cutoff_week=3)
    assert res["baseline_source"] == "prior_season"
    # growth is measured off the prior baseline, not zero
    assert res["signals"]["snap_share"]["baseline"] == pytest.approx(20.0)
    assert any("last season" in r.lower() for r in res["risks"])


def test_week1_candidate_without_prior_history_still_scores():
    # A player who debuts week 1 with a real role should be recognized (as a
    # provisional watchlist/opportunity), not silently dropped.
    res = wb.score_player(WR, [wk(1, 65, 20, 7, ppr=6)], cutoff_week=1)
    assert res["breakout_score"] > 0
    assert res["evaluated_weeks"] == [1]


# ---------------------------------------------------------------------------
# classification / confidence separation
# ---------------------------------------------------------------------------

def test_temporary_opportunity_from_injury_context():
    rows = [wk(6, 28, None, 3, car=4), wk(7, 66, None, 5, car=17, ppr=14)]
    res = wb.score_player(RB, rows, cutoff_week=7,
                          injury_context={"vacated": True, "source": "Starter X (IR)"})
    assert res["classification"] == "temporary_opportunity"
    assert any("Starter X" in r for r in res["reasons"])
    assert any("return" in r.lower() for r in res["risks"])


def test_emerging_breakout_requires_sustained_multi_game_growth():
    rows = [wk(1, 25, 8, 2), wk(2, 28, 9, 3), wk(3, 55, 18, 6),
            wk(4, 64, 22, 8), wk(5, 68, 24, 8)]
    res = wb.score_player(WR, rows, cutoff_week=5)
    assert res["classification"] == "emerging_breakout"
    assert res["provisional"] is False
    assert res["sample"]["baseline_games"] >= wb.EMERGING_MIN_BASELINE_GAMES


def test_confidence_is_separate_from_score():
    # Same large role change, but a 2-game sample must carry lower confidence than
    # a 6-game one, while the score itself stays comparably high.
    thin = wb.score_player(WR, [wk(1, 30, 10, 3), wk(2, 66, 24, 8)], cutoff_week=2)
    thick = wb.score_player(WR, [wk(1, 30, 10, 3), wk(2, 31, 10, 3), wk(3, 32, 11, 3),
                                 wk(4, 62, 22, 7), wk(5, 65, 23, 8), wk(6, 66, 24, 8)],
                            cutoff_week=6)
    assert thin["confidence"] < thick["confidence"]
    assert thin["breakout_score"] > 0 and thick["breakout_score"] > 0


def test_low_prior_usage_backup_not_eliminated():
    # A backup with a near-zero prior season who has earned a substantial current
    # role must still score - low historical usage alone cannot disqualify him.
    prior = {"snap_pct": 4.0, "target_share": 1.0, "targets_pg": 0.3,
             "carries_pg": 0.0, "pass_att_pg": 0.0}
    rows = [wk(4, 40, 12, 4), wk(5, 70, 25, 9)]
    res = wb.score_player(WR, rows, prior_baseline=prior, cutoff_week=5)
    assert res["breakout_score"] >= wb.WATCHLIST_MIN_SCORE


# ---------------------------------------------------------------------------
# game-status classification, byes, missing data, share units
# ---------------------------------------------------------------------------

def test_week_status_bye_vs_inactive_vs_missing():
    team_played = {1, 2, 4}  # week 3 was the bye
    assert wb.classify_week_status(None, 3, team_played) == wb.STATUS_BYE
    assert wb.classify_week_status(None, 2, team_played) == wb.STATUS_INACTIVE
    assert wb.classify_week_status(None, 5, None) == wb.STATUS_MISSING
    assert wb.classify_week_status(wk(1, 60, 20, 6, snaps=40), 1, team_played) == wb.STATUS_ACTIVE
    assert wb.classify_week_status(wk(1, snaps=0), 1, team_played) == wb.STATUS_ACTIVE_NO_USAGE


def test_missing_share_stays_unknown_not_zero():
    # target_share None in some weeks must be excluded from the mean, not averaged
    # in as 0 (which would fake a role collapse).
    val, count = wb._mean_present([wk(1, ts=20), wk(2, ts=None), wk(3, ts=24)], "target_share")
    assert count == 2
    assert val == pytest.approx(22.0)


def test_shares_are_percent_scale():
    # A player at ~68% recent snaps must read as 68, never 0.68.
    res = wb.score_player(WR, [wk(1, 30, 10, 3), wk(2, 32, 11, 3), wk(3, 68, 24, 8)],
                          cutoff_week=3)
    assert res["signals"]["snap_share"]["recent"] > 1.0


# ---------------------------------------------------------------------------
# context resolution / in-season path selection
# ---------------------------------------------------------------------------

def _sched(dates_by_week):
    def reader(season, week):
        ds = dates_by_week.get(week)
        return [{"seasonType": "Regular Season", "gameDate": d} for d in (ds or [])]
    return reader


def test_resolver_picks_weekly_in_season():
    reader = _sched({1: ["20260910"], 2: ["20260917"], 3: ["20260924"]})
    ctx = wr.resolve_scoring_context({"season": 2026}, as_of_date=date(2026, 9, 30),
                                     week_games=reader)
    assert ctx.mode == wr.MODE_WEEKLY
    assert ctx.cutoff_week == 3


def test_resolver_picks_offseason_before_games():
    reader = _sched({1: ["20260910"]})
    ctx = wr.resolve_scoring_context({"season": 2026}, as_of_date=date(2026, 8, 20),
                                     week_games=reader)
    assert ctx.mode == wr.MODE_OFFSEASON
    assert ctx.cutoff_week is None


def test_resolver_handles_january_regular_season_week():
    # Week 18 played Jan 4 must be recognized as in-season, which the old
    # month-in-(9..12) rule dropped.
    reader = _sched({17: ["20251228"], 18: ["20260104"]})
    ctx = wr.resolve_scoring_context({"season": 2025}, as_of_date=date(2026, 1, 6),
                                     week_games=reader)
    assert ctx.mode == wr.MODE_WEEKLY
    assert ctx.cutoff_week == 18


def test_resolver_excludes_week_still_in_progress():
    reader = _sched({1: ["20260910"], 2: ["20260917"],
                     3: ["20260924", "20260925"]})  # wk3 has a game on/after as_of
    ctx = wr.resolve_scoring_context({"season": 2026}, as_of_date=date(2026, 9, 24),
                                     week_games=reader)
    assert ctx.cutoff_week == 2


# ---------------------------------------------------------------------------
# injury context helper (pure)
# ---------------------------------------------------------------------------

def test_injury_context_map_flags_player_behind_injured_starter():
    feed = {
        "starter": {"position": "RB", "team": "AAA", "depth_chart_order": 1,
                    "full_name": "Starter", "injury_status": "IR"},
        "backup": {"position": "RB", "team": "AAA", "depth_chart_order": 2,
                   "full_name": "Backup", "injury_status": ""},
        "other": {"position": "RB", "team": "BBB", "depth_chart_order": 1,
                  "full_name": "Other", "injury_status": ""},
    }
    ctx = wr._injury_context_map(feed)
    assert ctx.get("backup", {}).get("vacated") is True
    assert "Starter" in ctx["backup"]["source"]
    assert "starter" not in ctx  # the injured starter himself has no opening
    assert "other" not in ctx    # healthy chart elsewhere


# ---------------------------------------------------------------------------
# preserve-on-failure orchestration (runner with stubbed DB layer)
# ---------------------------------------------------------------------------

def _install_stubs(monkeypatch, *, refresh_raises, series):
    """Stub the DB/data modules the runner imports lazily so we can drive it
    without Postgres or bs4."""
    # data_building.weekly_metrics (fails to import for real here: needs bs4).
    wm = types.ModuleType("data_building.weekly_metrics")

    def _build(season, weeks=None):
        if refresh_raises:
            raise RuntimeError("simulated refresh failure")
        return len(weeks or [])

    wm.build_weekly_metrics = _build
    wm.get_player_weekly_series = lambda pid, season: series.get(str(pid), [])
    monkeypatch.setitem(sys.modules, "data_building.weekly_metrics", wm)

    # utils.utils.load_players_index
    uu = types.ModuleType("utils.utils")
    uu.load_players_index = lambda: {
        "1": {"pos": "WR", "team": "AAA", "name": "Test WR"},
    }
    monkeypatch.setitem(sys.modules, "utils.utils", uu)

    # weekly_store spies (real module, monkeypatched functions)
    from data_building.breakout_engine import weekly_store
    calls = {"save": [], "record": []}
    monkeypatch.setattr(weekly_store, "save_weekly_scores",
                        lambda season, week, results, as_of_date=None: (
                            calls["save"].append((season, week, len(results))) or len(results)))
    monkeypatch.setattr(weekly_store, "record_run",
                        lambda *a, **k: calls["record"].append(k.get("status")))
    return calls


def test_failed_refresh_preserves_previous_snapshot(monkeypatch):
    calls = _install_stubs(monkeypatch, refresh_raises=True, series={})
    ctx = wr.ScoringContext(season=2026, mode=wr.MODE_WEEKLY, as_of_date=date(2026, 10, 1),
                            cutoff_week=4, completed_weeks=[1, 2, 3, 4])
    summary = wr.run_weekly_breakout(ctx, refresh=True)
    assert summary["status"] == "skipped"
    assert calls["save"] == []               # nothing written -> last snapshot preserved
    assert "skipped" in calls["record"]


def test_successful_run_saves_scores(monkeypatch):
    series = {"1": [wk(1, 25, 8, 2), wk(2, 28, 9, 3), wk(3, 58, 20, 7), wk(4, 68, 24, 8)]}
    calls = _install_stubs(monkeypatch, refresh_raises=False, series=series)
    ctx = wr.ScoringContext(season=2026, mode=wr.MODE_WEEKLY, as_of_date=date(2026, 10, 1),
                            cutoff_week=4, completed_weeks=[1, 2, 3, 4])
    summary = wr.run_weekly_breakout(ctx, refresh=True)
    assert summary["status"] == "success"
    assert calls["save"] and calls["save"][0][2] >= 1
    assert "success" in calls["record"]


def test_offseason_context_does_not_run_weekly(monkeypatch):
    calls = _install_stubs(monkeypatch, refresh_raises=False, series={})
    ctx = wr.ScoringContext(season=2026, mode=wr.MODE_OFFSEASON, as_of_date=date(2026, 8, 1))
    summary = wr.run_weekly_breakout(ctx, refresh=True)
    assert summary["status"] == "skipped"
    assert calls["save"] == []


# ---------------------------------------------------------------------------
# backtest harness (pure)
# ---------------------------------------------------------------------------

bt = pytest.importorskip("data_building.breakout_engine.backtest_weekly_breakout")


def _bwk(w, snap=None, ts=None, tgt=None, ppr=0.0):
    return {"week": w, "snap_pct": snap, "target_share": ts, "targets": tgt,
            "carries": None, "pass_att": None, "ppr_pts": ppr, "snaps": (snap or 0)}


def test_backtest_rewards_sustained_riser_not_td_chaser():
    # A genuine usage riser that stays elevated vs a flat player with one big TD
    # week. The model should flag the riser; box-score chasing flags the chaser
    # and misses (it does not sustain).
    riser = [_bwk(1, 25, 8, 2, 3), _bwk(2, 28, 9, 3, 4), _bwk(3, 55, 18, 6, 6),
             _bwk(4, 64, 22, 8, 8), _bwk(5, 68, 24, 8, 10), _bwk(6, 70, 25, 9, 13),
             _bwk(7, 69, 24, 8, 12), _bwk(8, 71, 26, 9, 14)]
    chaser = [_bwk(1, 50, 15, 5, 6), _bwk(2, 50, 15, 5, 6), _bwk(3, 50, 15, 5, 7),
              _bwk(4, 50, 15, 5, 6), _bwk(5, 50, 15, 5, 30), _bwk(6, 50, 15, 5, 6),
              _bwk(7, 50, 15, 5, 7), _bwk(8, 50, 15, 5, 6)]
    series = {"r": riser, "c": chaser}
    meta = {"r": {"position": "WR"}, "c": {"position": "WR"}}
    rep = bt.run_backtest(series, meta, eval_weeks=[5], top_n=1, horizon=3)
    assert rep["methods"]["model"]["precision"] == 1.0
    assert rep["methods"]["recent_points"]["precision"] == 0.0


def test_backtest_flat_player_is_not_a_true_riser():
    flat = [_bwk(w, 50, 15, 5, 6) for w in range(1, 9)]
    assert bt._did_sustain(flat, "WR", 5, 3) is False


def test_backtest_none_when_no_lookahead():
    riser = [_bwk(w, 20 + 8 * w, 5 + 2 * w, w) for w in range(1, 6)]
    assert bt._did_sustain(riser, "WR", 5, 3) is None  # nothing after week 5


def test_backtest_usefulness_metric_tracks_ppg():
    # Startable PPR over the lookahead counts as useful; below-threshold does not.
    useful = [_bwk(w, 60, 20, 6, ppr=14) for w in range(1, 9)]
    thin = [_bwk(w, 60, 20, 6, ppr=4) for w in range(1, 9)]
    assert bt._became_useful(useful, "WR", 5, 3) is True
    assert bt._became_useful(thin, "WR", 5, 3) is False
    assert bt._became_useful(useful, "WR", 8, 3) is None  # no lookahead


def test_backtest_reports_both_precisions():
    riser = [_bwk(1, 25, 8, 2, 3), _bwk(2, 28, 9, 3, 4), _bwk(3, 55, 18, 6, 6),
             _bwk(4, 64, 22, 8, 8), _bwk(5, 68, 24, 8, 12), _bwk(6, 70, 25, 9, 14),
             _bwk(7, 69, 24, 8, 13), _bwk(8, 71, 26, 9, 15)]
    rep = bt.run_backtest({"r": riser}, {"r": {"position": "WR"}},
                          eval_weeks=[5], top_n=1, horizon=3)
    assert "precision" in rep["methods"]["model"]
    assert "precision_useful" in rep["methods"]["model"]
