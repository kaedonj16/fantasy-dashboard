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
    assert res["breakout_score"] >= 40
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


def test_routes_and_high_value_usage_are_explainable_when_available():
    rows = [wk(1, 25, 8, 2), wk(2, 28, 9, 2), wk(3, 70, 24, 8)]
    rows[0].update(routes=10, red_zone_opportunities=0)
    rows[1].update(routes=11, red_zone_opportunities=0)
    rows[2].update(routes=31, red_zone_opportunities=3)
    res = wb.score_player(WR, rows, cutoff_week=3)
    assert res["signals"]["routes_pg"]["delta"] > 15
    assert res["components"]["high_value_touches"] > 0
    assert res["components"]["unexpected_usage"] > 0


def test_garbage_time_and_returning_starter_lower_sustainability():
    rows = [wk(1, 20, tgt=2), wk(2, 22, tgt=2), wk(3, 68, tgt=8)]
    normal = wb.score_player(WR, rows, cutoff_week=3,
                             injury_context={"vacated": True, "multi_week": True})
    rows[-1]["garbage_time"] = True
    risky = wb.score_player(WR, rows, cutoff_week=3,
                            injury_context={"vacated": True, "starter_returning": True})
    assert risky["components"]["sustainability"] < normal["components"]["sustainability"]


def test_current_backtest_emits_calibration_records_without_future_leakage():
    from data_building.breakout_engine.backtest_weekly_breakout import (
        build_evaluation_records, calibration_report,
    )
    rows = [wk(1, 25, 8, 2, ppr=3), wk(2, 30, 9, 3, ppr=4),
            wk(3, 65, 22, 8, ppr=7), wk(4, 68, 24, 9, ppr=12)]
    records = build_evaluation_records({"1": rows}, {"1": WR}, season=2025,
                                       eval_weeks=[2], horizons=(1, 3))
    assert len(records) == 1
    assert records[0]["week"] == 2
    assert records[0]["inputs"]["targets"] == 3
    report = calibration_report(records, horizon=1)
    assert report["sample_size"] == 1
    assert sum(v["sample_size"] for v in report["buckets"].values()) == 1


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
    # Provisional confidence is a smooth 0.6x discount of the underlying
    # evidence (not the old hard 35.0 cliff): still low for one game, but it
    # varies player to player instead of pinning at exactly 35.
    assert res["confidence"] < 60.0
    assert res["classification"] in ("watchlist", "temporary_opportunity")
    assert res["signals"]["snap_share"]["baseline"] is None
    assert res["signals"]["snap_share"]["delta"] is None
    assert res["breakout_score"] < 100


def test_provisional_confidence_discount_is_proportional():
    # Unit-level: the provisional discount scales the evidence-based
    # confidence instead of clipping it at a fixed ceiling.
    rows = [wk(1, 70, None, 4, car=18, ppr=15)]
    signals, _ = wb._position_signals("RB", rows, [], initial_role=True)
    conf_prov, _ = wb._compute_confidence(signals, rows, [], "RB", 0, True)
    conf_full, _ = wb._compute_confidence(signals, rows, [], "RB", 0, False)
    assert conf_prov == pytest.approx(conf_full * 0.6, abs=0.15)


def test_prior_season_baseline_used_when_thin_current_data():
    prior = {"snap_pct": 20.0, "target_share": 6.0, "targets_pg": 2.0,
             "carries_pg": 0.0, "pass_att_pg": 0.0}
    res = wb.score_player(WR, [wk(3, 68, 22, 8)], prior_baseline=prior, cutoff_week=3)
    assert res["baseline_source"] == "prior_season"
    # growth is measured off the prior baseline, not zero
    assert res["signals"]["snap_share"]["baseline"] == pytest.approx(20.0)
    assert any("last season" in r.lower() for r in res["risks"])


def test_flat_week_one_usage_against_prior_is_not_breakout():
    prior = {"snap_pct": 64.0, "target_share": 21.0, "targets_pg": 7.0,
             "carries_pg": 0.0, "pass_att_pg": 0.0}
    res = wb.score_player(WR, [wk(1, 64, 21, 7)], prior_baseline=prior, cutoff_week=1)
    assert res["baseline_source"] == "prior_season"
    assert res["breakout_score"] == 0


def test_large_multi_signal_growth_outranks_mild_without_saturation():
    mild = [wk(1, 40, 12, 4), wk(2, 42, 13, 4), wk(3, 54, 17, 6)]
    large = [wk(1, 20, 6, 2), wk(2, 22, 7, 2), wk(3, 72, 27, 10)]
    mild_score = wb.score_player(WR, mild, cutoff_week=3)["breakout_score"]
    large_score = wb.score_player(WR, large, cutoff_week=3)["breakout_score"]
    assert large_score > mild_score
    assert large_score < 100


def test_prior_cache_accepts_player_id_and_normalizes_percent_units(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "player_history"
    cache.mkdir(parents=True)
    (cache / "usage_rows_2025.json").write_text(
        '[{"player_id":"abc","usage":{"games":10,"avg_off_snap_pct":0.64,'
        '"target_share":0.21,"avg_targets":7}}]')
    monkeypatch.chdir(tmp_path)
    prior = wr._prior_baseline_map(2026)
    assert prior["abc"]["snap_pct"] == pytest.approx(64.0)
    assert prior["abc"]["target_share"] == pytest.approx(21.0)


def test_prior_cache_accepts_id_and_does_not_double_scale_percent(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "player_history"
    cache.mkdir(parents=True)
    (cache / "usage_rows_2025.json").write_text(
        '[{"id":"xyz","usage":{"games":10,"avg_off_snap_pct":64,'
        '"target_share":21,"avg_targets":7}}]')
    monkeypatch.chdir(tmp_path)
    prior = wr._prior_baseline_map(2026)
    assert prior["xyz"]["snap_pct"] == pytest.approx(64.0)
    assert prior["xyz"]["target_share"] == pytest.approx(21.0)


def test_prior_cache_rejects_provider_default_zero_share(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "player_history"
    cache.mkdir(parents=True)
    (cache / "usage_rows_2025.json").write_text(
        '[{"id":"qb","usage":{"games":17,"avg_off_snap_pct":0,'
        '"avg_off_snaps":65,"avg_pass_att":32}}]')
    monkeypatch.chdir(tmp_path)
    prior = wr._prior_baseline_map(2026)
    assert prior["qb"]["snap_pct"] is None
    assert prior["qb"]["pass_att_pg"] == pytest.approx(32.0)


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


def test_weekly_api_payload_is_card_complete_without_offseason_projections():
    from dashboard_services.breakout_api import _weekly_row_to_candidate
    evidence = wb.score_player(
        WR, [wk(1, 25, 8, 2), wk(2, 65, 24, 8)], cutoff_week=2)
    row = {**evidence, "as_of_week": 2, "evidence": evidence,
           "reasons": "\n".join(evidence["reasons"]),
           "risks": "\n".join(evidence["risks"])}
    card = _weekly_row_to_candidate(row)
    required = {"breakout_score", "confidence", "classification",
                "classification_label", "provisional", "baseline_source",
                "as_of_week", "reasons", "risks", "usage_comparison", "sample"}
    assert required <= card.keys()
    assert card["weekly"] is True and card["mode"] == "weekly"
    assert card["usage_comparison"][0]["points"] >= card["usage_comparison"][-1]["points"]
    assert not ({"season1_ppr", "hit_probability", "opportunity_opened_score"} & card.keys())


def test_weekly_card_has_explicit_branch_and_plain_reason_support():
    source = open("app.py", encoding="utf-8").read()
    assert "candidate.weekly === true || candidate.mode === 'weekly'" in source
    assert "Array.isArray(candidate.reasons)" in source
    assert "Initial role:" in source
    # Offseason fallback remains, but is behind the explicit weekly branch.
    assert "No projection available" in source
    weekly_block = source[source.index("const ppgHtml = isWeekly"):source.index("const hitHtml", source.index("const ppgHtml = isWeekly"))]
    assert "No projection available" not in weekly_block.split(": range", 1)[0]


def test_observed_zero_and_unknown_baselines_are_distinct():
    observed = wb._share_growth(0.0, 20.0, allow_initial=False)
    unknown = wb._share_growth(None, 20.0, allow_initial=False)
    assert observed["delta"] == 20.0 and observed["points"] > 0
    assert unknown["baseline"] is None and unknown["delta"] is None
    assert unknown["points"] == 0


def test_partial_prior_only_compares_observed_signals():
    prior = {"snap_pct": None, "target_share": 8.0, "targets_pg": None,
             "carries_pg": None, "pass_att_pg": None}
    res = wb.score_player({**WR, "years_exp": 2}, [wk(1, 70, 24, 8)],
                          prior_baseline=prior, cutoff_week=1)
    assert res["baseline_source"] == "prior_season"
    assert res["signals"]["snap_share"]["baseline"] is None
    assert res["signals"]["snap_share"]["delta"] is None
    assert res["signals"]["snap_share"]["points"] == 0
    assert res["signals"]["target_share"]["delta"] == 16.0


def test_elite_rookie_debut_is_initial_role_watchlist():
    rookie = {**WR, "years_exp": 0, "season": 2026, "draft_year": 2026,
              "draft_round": 1}
    res = wb.score_player(rookie, [wk(1, 90, 35, 13)], cutoff_week=1)
    assert res["score_basis"] == "initial_role"
    assert res["role_change_score"] is None
    assert res["classification"] == "early_watch"
    assert res["breakout_score"] <= wb.INITIAL_ONE_GAME_CAP


def test_two_persistent_rookie_games_can_be_provisional_emerging():
    rookie = {**WR, "years_exp": 0, "season": 2026, "draft_year": 2026,
              "draft_round": 5}
    rows = [wk(1, 72, 23, 8), wk(2, 75, 25, 9)]
    res = wb.score_player(rookie, rows, cutoff_week=2)
    assert res["classification"] == "emerging_breakout"
    assert res["main_board_eligible"] is True
    assert res["score_basis"] == "initial_role"
    # The persistent cap is evidence-scaled now (45 + 5 per supporting signal,
    # max 60): strong multi-signal rookies earn headroom above the old fixed 45
    # instead of every rookie pinning at exactly the same number.
    expected_cap = wb._initial_persistent_cap(res["supporting_signal_count"])
    assert res["breakout_score"] <= expected_cap
    assert res["breakout_score"] > wb.INITIAL_PERSISTENT_CAP  # differentiated


def test_evidence_scaled_cap_rewards_stronger_rookie_evidence():
    rookie = {**WR, "years_exp": 0, "season": 2026, "draft_year": 2026,
              "draft_round": 5}
    strong = wb.score_player(rookie, [wk(1, 72, 23, 8), wk(2, 75, 25, 9)],
                             cutoff_week=2)
    weak = wb.score_player(rookie, [wk(1, 55, 12, 4), wk(2, 58, 13, 4)],
                           cutoff_week=2)
    assert strong["breakout_score"] > weak["breakout_score"]
    assert wb._initial_persistent_cap(0) == 45.0
    assert wb._initial_persistent_cap(3) == 60.0
    assert wb._initial_persistent_cap(10) == 60.0  # capped at the max


def test_rookie_transitions_to_real_change_window_in_week_three():
    rookie = {**WR, "years_exp": 0, "season": 2026, "draft_year": 2026}
    rows = [wk(1, 20, 6, 2), wk(2, 25, 8, 3), wk(3, 65, 24, 8)]
    res = wb.score_player(rookie, rows, cutoff_week=3)
    assert res["score_basis"] == "role_change"
    assert res["role_change_score"] is not None
    assert res["baseline_weeks"] == [1, 2]


def test_missing_cache_veteran_is_not_assumed_rookie():
    veteran = {**WR, "years_exp": 4, "season": 2026}
    res = wb.score_player(veteran, [wk(1, 75, 25, 9)], cutoff_week=1)
    assert res["is_rookie"] is False
    assert res["classification"] == "watchlist"
    assert res["breakout_score"] < wb.WATCHLIST_MIN_SCORE + 2


def test_target_spike_without_route_growth_is_conflicting():
    rows = [wk(1, 50, 12, 4), wk(2, 52, 13, 4), wk(3, 60, 25, 10)]
    rows[0]["routes"], rows[1]["routes"], rows[2]["routes"] = 28, 29, 24
    res = wb.score_player(WR, rows, cutoff_week=3)
    assert "targets_up_without_route_growth" in res["conflicting_signals"]


def test_te_blocking_snap_increase_does_not_clear_emerging_quality():
    te = {"player_id": "te", "position": "TE", "years_exp": 2}
    rows = [wk(1, 35, 8, 2), wk(2, 38, 8, 2), wk(3, 75, 8, 2)]
    for row, routes in zip(rows, (18, 19, 12)):
        row["routes"] = routes
    res = wb.score_player(te, rows, cutoff_week=3)
    assert "snaps_up_routes_down" in res["conflicting_signals"]
    assert res["classification"] == "watchlist"


def test_team_volume_count_spike_without_share_growth_stays_watchlist():
    rows = [wk(1, 60, 20, 6), wk(2, 62, 20, 6), wk(3, 63, 20, 10)]
    res = wb.score_player(WR, rows, cutoff_week=3)
    assert res["signals"]["target_share"]["points"] == 0
    assert res["supporting_signal_count"] < wb.MIN_SUPPORTING_SIGNALS
    assert res["classification"] == "early_watch"
    assert res["main_board_eligible"] is False


def test_role_held_after_starter_return_increases_sustainability():
    rows = [wk(1, 25, tgt=2, car=4), wk(2, 65, tgt=5, car=14),
            wk(3, 66, tgt=5, car=15)]
    normal = wb.score_player(RB, rows, cutoff_week=3,
                             injury_context={"vacated": True, "multi_week": True})
    held = wb.score_player(RB, rows, cutoff_week=3,
                           injury_context={"vacated": True, "multi_week": True,
                                           "starter_returned": True})
    assert held["sustainability_score"] > normal["sustainability_score"]


def test_garbage_time_qb_is_watchlist_and_discounted():
    qb = {"player_id": "q", "position": "QB", "years_exp": 2}
    rows = [wk(1, 10, pa=3, car=1), wk(2, 95, pa=35, car=6)]
    rows[-1]["garbage_time"] = True
    res = wb.score_player(qb, rows, cutoff_week=2)
    assert res["opportunity_source"] == "garbage_time"
    assert res["classification"] == "watchlist"


def test_ranking_confidence_cannot_overpower_signal_magnitude():
    def rank(score, confidence):
        return score * (.75 + .25 * confidence / 100)
    assert rank(40, 25) < rank(38, 80)


def test_lifecycle_transitions_across_snapshots():
    previous = {"breakout_score": 42, "evidence": {"lifecycle": {
        "first_detected_week": 2, "consecutive_flagged_weeks": 2}}}
    confirmed = wr.derive_lifecycle(
        {"breakout_score": 48, "classification": "emerging_breakout"},
        previous, 4, wb.WATCHLIST_MIN_SCORE)
    assert confirmed["lifecycle_state"] == "confirmed"
    cooled = wr.derive_lifecycle(
        {"breakout_score": 12, "classification": "watchlist"},
        previous, 4, wb.WATCHLIST_MIN_SCORE)
    assert cooled["lifecycle_state"] == "invalidated"


def test_breakout_page_does_not_send_offseason_floor_for_weekly_mode():
    source = open("app.py", encoding="utf-8").read()
    request_line = next(line for line in source.splitlines()
                        if "fetch('/api/breakout/candidates?season=" in line)
    assert "min_score=50" not in request_line


def test_meaningful_rookie_early_watch_is_separate_from_default_board(monkeypatch):
    import dashboard_services.breakout_api as api
    from data_building.breakout_engine import weekly_store
    rookie = wb.score_player(
        {**WR, "years_exp": 0, "season": 2026, "draft_year": 2026},
        [wk(1, 90, 35, 13)], cutoff_week=1)
    assert rookie["classification"] == "early_watch" and rookie["breakout_score"] < 50
    row = {**rookie, "season": 2026, "as_of_week": 1, "evidence": rookie}
    monkeypatch.setattr(weekly_store, "load_weekly_candidates", lambda *a, **k: {
        "candidates": [row], "as_of_week": 1, "as_of_date": "2026-09-15",
        "data_status": "ok", "scoring_version": wb.SCORING_VERSION})
    payload = api.get_weekly_breakout_candidates(2026)
    assert [candidate["player_id"] for candidate in payload["candidates"]] == [WR["player_id"]]
    assert [candidate["player_id"] for candidate in payload["early_watch"]] == [WR["player_id"]]


def test_current_version_filter_rejects_incompatible_snapshots(monkeypatch):
    from data_building.breakout_engine import weekly_store
    captured = {}

    class Result:
        def fetchone(self):
            return {"w": None}

    class Conn:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, query, params):
            captured["query"], captured["params"] = query, params
            return Result()

    monkeypatch.setattr(weekly_store, "init_weekly_breakout_db", lambda: None)
    monkeypatch.setattr(weekly_store, "get_conn", lambda: Conn())
    assert weekly_store.latest_scored_week(2026) is None
    assert "scoring_version = %s" in captured["query"]
    assert captured["params"][-1] == wb.SCORING_VERSION


def test_incompatible_weekly_history_does_not_fall_back_to_offseason(monkeypatch):
    import dashboard_services.breakout_api as api
    monkeypatch.setattr(api, "_weekly_breakout_available", lambda season: False)
    monkeypatch.setattr(api, "_weekly_history_exists", lambda season: True)
    payload = api.get_breakout_candidates(2026)
    assert payload["mode"] == "weekly"
    assert payload["data_status"] == "incompatible_snapshot"
    assert payload["required_scoring_version"] == wb.SCORING_VERSION


def test_weekly_contract_exposes_detection_and_lifecycle_fields():
    from dashboard_services.breakout_api import _weekly_row_to_candidate
    result = wb.score_player(
        {**WR, "years_exp": 0, "season": 2026, "draft_year": 2026},
        [wk(1, 72, 22, 8), wk(2, 75, 24, 9)], cutoff_week=2)
    result["lifecycle"] = {"previous_score": None, "score_change": None,
                           "first_detected_week": 1, "consecutive_flagged_weeks": 1,
                           "lifecycle_state": "new"}
    row = {**result, "as_of_week": 2, "evidence": result}
    card = _weekly_row_to_candidate(row)
    assert card["final_breakout_score"] == card["breakout_score"]
    for key in ("ranking_score", "score_basis", "role_change_score",
                "current_role_score", "sustainability_score",
                "breakout_novelty_score", "expectation_delta_score",
                "supporting_signal_count", "conflicting_signals",
                "opportunity_source", "lifecycle_state"):
        assert key in card
    assert card["role_change_score"] is None
    assert "hit_probability" not in card and "season1_ppr" not in card


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
    # Simulate full-suite collection having imported the real submodule first.
    # ``from data_building import weekly_metrics`` would incorrectly reuse this
    # stale package attribute even after sys.modules is replaced below.
    import data_building
    stale_wm = types.ModuleType("stale_weekly_metrics")
    stale_wm.build_weekly_metrics = lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("stale weekly_metrics module was used"))
    monkeypatch.setattr(data_building, "weekly_metrics", stale_wm, raising=False)
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
    monkeypatch.setattr(weekly_store, "publish_weekly_snapshot",
                        lambda season, week, results, **kwargs: (
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
    assert summary["status"] == "completed"
    assert calls["save"] and calls["save"][0][2] >= 1
    assert calls["record"] == []


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
    assert "rest_of_season_role_retention" in rep["methods"]["model"]
    assert "median_lead_time_games" in rep["methods"]["model"]
