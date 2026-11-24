"""weekly-v7 (predictive fix, v7.2 composition) behavior pins.

Covered here: the fast-track gate and its available-signal
renormalization (Fix 1), the rush-first RB role floor (Fix 2), the
two-tier assignment decoupled from call path (the Omar Cooper Jr.
case), staleness decay + hard expiry, the lifecycle "stale" state, and
the 6-week re-surfacing cooldown with its new-evidence release in the
canonical surfaced walk. The strengthened Recap Risk Index from the
v7.1 draft is deliberately NOT part of v7; the index in the scorer is
v7's original (see the version note in weekly_breakout.py).
"""
import pytest

pytest.importorskip("flask")

from data_building.breakout_engine import weekly_breakout as wb  # noqa: E402
from data_building.breakout_engine import weekly_grading as wg  # noqa: E402
from data_building.breakout_engine import weekly_runner as wr  # noqa: E402


def wk(week, snap, share, tgt, pts=6.0):
    return {"week": week, "snap_pct": snap, "target_share": share,
            "targets": tgt, "carries": 0, "ppr_pts": pts,
            "routes": None, "team_dropbacks": None}


WR = {"player_id": "ft", "position": "WR", "season": 2026, "years_exp": 1}

# A clean two-game usage jump in weeks 3-4 with NO routes data anywhere
# (route_participation is a 35% WR weight that is missing for data
# reasons, the exact Fix 1 shape from the backtest).
JUMP_ROWS = [wk(1, 28, 6, 2, 3.0), wk(2, 30, 7, 2, 4.0),
             wk(3, 62, 19, 6, 8.0), wk(4, 65, 20, 6, 9.0)]


def test_scoring_version_is_weekly_v7():
    assert wb.SCORING_VERSION == "weekly-v7"


def test_fast_track_gate_fires_and_promotes_the_jump():
    res = wb.score_player(WR, JUMP_ROWS, cutoff_week=4)
    assert res["fast_track_eligible"] is True
    assert res["call_path"] == "fast_track"
    assert res["breakout_score"] >= wb.EMERGING_MIN_SCORE
    assert res["main_board_eligible"] is True
    assert res["classification"] == "emerging_breakout"


def test_fast_track_renormalization_counts_missing_signal_as_unknown():
    # On the jump window, the fixed-weight score reads the missing
    # route_participation weight as a zero; the renormalized score (the
    # one the fast track blends) rescales over the available signals.
    recent, baseline = JUMP_ROWS[-2:], JUMP_ROWS[:-2]
    _sig_f, fixed = wb._position_signals("WR", recent, baseline)
    _sig_r, renorm = wb._position_signals("WR", recent, baseline,
                                          renormalize=True)
    assert fixed > 0
    assert renorm > fixed
    res = wb.score_player(WR, JUMP_ROWS, cutoff_week=4)
    assert res["change_score_jump"] == round(renorm, 1)


def test_fast_track_gate_blocked_by_established_evidence():
    prior = {"snap_pct": 70.0, "target_share": 20.0, "targets_pg": 6.0,
             "carries_pg": 0.0, "pass_att_pg": 0.0, "games": 16}
    res = wb.score_player(WR, JUMP_ROWS, cutoff_week=4,
                          prior_baseline=prior)
    assert res["fast_track_eligible"] is False
    assert res["call_path"] == "standard"


def _rb(week, snap, car, tgt, pts=8.0):
    return {"week": week, "snap_pct": snap, "target_share": 0,
            "targets": tgt, "carries": car, "ppr_pts": pts,
            "routes": None, "team_dropbacks": None}


RB = {"player_id": "rb", "position": "RB", "season": 2026, "years_exp": 1}


def _rb_rows(snap_hi):
    return [_rb(1, 25, 4, 1, 3.0), _rb(2, 28, 5, 1, 4.0),
            _rb(3, snap_hi, 16, 2, 14.0), _rb(4, snap_hi + 2, 17, 1, 15.0),
            _rb(5, snap_hi + 1, 16, 2, 13.0)]


def test_rush_first_rb_floor_accepts_a_rush_heavy_back():
    # 16+ carries/g with starter snap share but only ~1.5 targets/g:
    # the receiving second hit (targets_pg >= 3.0) fails, the rush-first
    # path (Fix 2) meets the floor.
    res = wb.score_player(RB, _rb_rows(70), cutoff_week=5)
    assert "position_specific_role_floor_not_met" not in \
        res["main_board_rejection_reasons"]
    assert res["main_board_eligible"] is True


def test_rush_first_rb_floor_still_needs_starter_snaps():
    # Same rushing volume on a 40% snap share is a committee role, not
    # a starter's rushing role: neither floor path is met.
    res = wb.score_player(RB, _rb_rows(40), cutoff_week=5)
    assert "position_specific_role_floor_not_met" in \
        res["main_board_rejection_reasons"]
    assert res["main_board_eligible"] is False


def test_tier_decoupled_from_call_path_cooper_case():
    # The Omar Cooper Jr. case from the v7.1 backtest: a fast-track
    # call that had already posted a starter-tier week must land in
    # Confirming - fast-track status never exempts a call from the
    # tier rule.
    res = wb.score_player(WR, JUMP_ROWS, cutoff_week=4)
    assert res["call_path"] == "fast_track"
    assert wb.assign_tier(True, res["recap_risk_score"]) == wb.TIER_CONFIRMING
    # The same call with no prior starter-tier week and a low index is
    # Breaking Out; the index alone at the flag line also confirms.
    assert wb.assign_tier(False, res["recap_risk_score"]) in (
        wb.TIER_BREAKING_OUT, wb.TIER_CONFIRMING)
    assert wb.assign_tier(False, 0.0) == wb.TIER_BREAKING_OUT
    assert wb.assign_tier(False, wb.RECAP_RISK_CONFIRMING_MIN) == \
        wb.TIER_CONFIRMING
    assert wb.TIER_LABELS[wb.TIER_CONFIRMING] == "Confirming"
    assert wb.TIER_LABELS[wb.TIER_BREAKING_OUT] == "Breaking Out"


def test_staleness_decay_scales_the_score():
    fresh = wb.score_player(WR, JUMP_ROWS, cutoff_week=4)
    stale = wb.score_player(WR, JUMP_ROWS, cutoff_week=6)
    assert stale["weeks_since_active"] == 2
    assert stale["stale_expired"] is False
    assert stale["breakout_score"] == round(
        fresh["breakout_score"] * wb.STALE_DECAY_PER_WEEK ** 2, 1)


def test_staleness_hard_expiry_at_three_weeks():
    res = wb.score_player(WR, JUMP_ROWS, cutoff_week=7)
    assert res["weeks_since_active"] == 3
    assert res["stale_expired"] is True
    assert "stale_evidence_expired" in res["main_board_rejection_reasons"]
    assert res["main_board_eligible"] is False
    assert res["classification"] in ("watchlist", "monitored")


def test_lifecycle_marks_frozen_evidence_stale():
    previous = {"breakout_score": 42, "recent_weeks": [3, 4],
                "evidence": {"lifecycle": {
                    "first_detected_week": 3,
                    "consecutive_flagged_weeks": 1}}}
    result = {"breakout_score": 36, "classification": "early_watch",
              "recent_weeks": [3, 4], "weeks_since_active": 2,
              "current_role_score": 60}
    life = wr.derive_lifecycle(result, previous, 6, wb.WATCHLIST_MIN_SCORE)
    assert life["evidence_frozen"] is True
    assert life["weeks_since_active"] == 2
    assert life["lifecycle_state"] == "stale"
    # A moved window is not frozen, even at the same staleness count.
    moved = dict(result, recent_weeks=[4, 5])
    life2 = wr.derive_lifecycle(moved, previous, 6, wb.WATCHLIST_MIN_SCORE)
    assert life2["evidence_frozen"] is False
    assert life2["lifecycle_state"] != "stale"


def _score_row(pid, classification, score, evidence=None):
    return {"player_id": pid, "classification": classification,
            "breakout_score": score, "evidence": evidence or {}}


def _patch_walk_store(monkeypatch, rows_by_run, weeks=(1, 2, 3, 4, 5)):
    from data_building.breakout_engine import weekly_store
    serving = {w: {"id": 100 + w} for w in weeks}
    monkeypatch.setattr(weekly_store, "list_completed_weeks",
                        lambda season: [{"as_of_week": w} for w in weeks])
    monkeypatch.setattr(weekly_store, "get_serving_run",
                        lambda season, week: serving.get(week))
    monkeypatch.setattr(weekly_store, "load_run_score_rows",
                        lambda run_id: list(rows_by_run.get(run_id, [])))
    # No starter-tier weeks unless a test says otherwise: the score-jump
    # leg alone decides.
    monkeypatch.setattr(wg, "starter_tier_weeks_by_player",
                        lambda season, through_week: {})


def test_cooldown_blocks_identical_repeat_inside_six_weeks(monkeypatch):
    rows = {101: [_score_row("ferg", "early_watch", 37.0)],
            104: [_score_row("ferg", "early_watch", 37.0)]}
    _patch_walk_store(monkeypatch, rows)
    by_week = wg.surfaced_player_ids_by_week(2026)
    assert by_week[1] == {"ferg"}
    # Week 4 is inside the 6-week cooldown and the call is identical
    # (score change 0, no starter-tier week): not a new call.
    assert by_week[4] == set()


def test_cooldown_releases_on_score_jump(monkeypatch):
    rows = {101: [_score_row("wayne", "early_watch", 38.0)],
            104: [_score_row("wayne", "early_watch", 44.0)]}
    _patch_walk_store(monkeypatch, rows)
    by_week = wg.surfaced_player_ids_by_week(2026)
    assert by_week[4] == {"wayne"}


def test_cooldown_releases_on_new_starter_tier_week(monkeypatch):
    rows = {101: [_score_row("wayne", "early_watch", 38.0)],
            104: [_score_row("wayne", "early_watch", 39.0)]}
    _patch_walk_store(monkeypatch, rows)
    monkeypatch.setattr(wg, "starter_tier_weeks_by_player",
                        lambda season, through_week: {"wayne": [3]})
    by_week = wg.surfaced_player_ids_by_week(2026)
    assert by_week[4] == {"wayne"}


def test_cooldown_expires_after_six_weeks(monkeypatch):
    rows = {101: [_score_row("ferg", "early_watch", 37.0)],
            108: [_score_row("ferg", "early_watch", 37.0)]}
    _patch_walk_store(monkeypatch, rows, weeks=(1, 8))
    by_week = wg.surfaced_player_ids_by_week(2026)
    assert by_week[8] == {"ferg"}


def test_board_payload_splits_two_tiers(monkeypatch):
    import dashboard_services.breakout_api as api
    from data_building.breakout_engine import weekly_store

    def _row(pid, tier):
        res = wb.score_player(WR, JUMP_ROWS, cutoff_week=4)
        res = {**res, "player_id": pid, "tier": tier,
               "starter_tier_before_call": tier == "confirming"}
        return {**res, "season": 2026, "as_of_week": 4, "evidence": res}

    rows = [_row("bo1", "breaking_out"), _row("cf1", "confirming")]
    monkeypatch.setattr(weekly_store, "load_weekly_candidates",
                        lambda *a, **k: {
                            "candidates": rows, "as_of_week": 4,
                            "as_of_date": "2026-10-01",
                            "data_status": "ok",
                            "scoring_version": wb.SCORING_VERSION})
    monkeypatch.setattr(api, "_recently_surfaced_player_ids",
                        lambda *a, **k: set())
    payload = api.get_weekly_breakout_candidates(2026, as_of_week=4)
    assert [c["player_id"] for c in payload["breaking_out"]] == ["bo1"]
    assert [c["player_id"] for c in payload["confirming"]] == ["cf1"]
    # Both tiers stay on the surfaced board (the split is a label).
    assert {c["player_id"] for c in payload["candidates"]} == {"bo1", "cf1"}
    by_id = {c["player_id"]: c for c in payload["candidates"]}
    assert by_id["cf1"]["tier_label"] == "Confirming"
    assert by_id["bo1"]["tier_label"] == "Breaking Out"
