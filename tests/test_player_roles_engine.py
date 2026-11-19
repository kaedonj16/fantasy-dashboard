"""Pure-engine tests for dashboard_services.player_roles (no flask/pandas)."""
from dashboard_services import player_roles as pr


def _row(week, **kw):
    base = {"week": week, "snap_pct": 50.0, "snaps": 30, "team_snaps": 60,
            "targets": 0, "receptions": 0, "carries": 0, "touches": 0,
            "target_share": None, "carry_share": None, "ppr_pts": 0.0,
            "rec_yards": 0.0, "rush_yards": 0.0, "pass_att": 0,
            "rec_tds": 0, "rush_tds": 0, "pass_tds": 0,
            "rz_targets": 0, "rz_carries": 0}
    base.update(kw)
    return base


def _wr_series(weeks, tgt_share, snap, targets=8):
    return [_row(w, snap_pct=snap, snaps=int(snap * 0.6), targets=targets,
                 receptions=int(targets * 0.7), target_share=tgt_share,
                 rec_yards=targets * 9.0, ppr_pts=targets * 1.6,
                 rz_targets=1)
            for w in weeks]


def _pool():
    weeks = [1, 2, 3, 4, 5, 6]
    pool = {
        "alpha": {"pos": "WR", "team": "DAL", "name": "Alpha Wideout",
                  "series": _wr_series(weeks, 32.0, 96.0, targets=11)},
        "mate": {"pos": "WR", "team": "DAL", "name": "Second Wideout",
                 "series": _wr_series(weeks, 14.0, 70.0, targets=5)},
        "other1": {"pos": "WR", "team": "NYG", "name": "Other One",
                   "series": _wr_series(weeks, 18.0, 75.0, targets=6)},
        "other2": {"pos": "WR", "team": "NYG", "name": "Other Two",
                   "series": _wr_series(weeks, 9.0, 55.0, targets=3)},
        "other3": {"pos": "WR", "team": "PHI", "name": "Other Three",
                   "series": _wr_series(weeks, 22.0, 85.0, targets=7)},
    }
    return pool


def test_alpha_wr1_tier():
    pool = _pool()
    computed = pr.compute_pool(pool)
    role = pr.build_role("alpha", pool, computed)
    assert role["job"] == "Alpha WR1"
    assert role["confidence"] == "High"
    assert role["axes"]["Volume"] >= 90


def test_provisional_when_few_games():
    pool = _pool()
    pool["alpha"] = {**pool["alpha"], "series": _wr_series([1, 2, 3], 32.0, 96.0, targets=11)}
    computed = pr.compute_pool(pool)
    role = pr.build_role("alpha", pool, computed)
    assert role["confidence"] == "Provisional"


def test_versatile_guard_single_tag():
    pool = _pool()
    feats = {
        "alpha": {"deep_rate": 45.0, "adot": 16.0, "explosive_rate": 30.0,
                  "yac_per_rec": 7.0},
        "mate": {"deep_rate": 10.0, "adot": 6.0, "explosive_rate": 5.0,
                 "yac_per_rec": 2.0},
        "other1": {"deep_rate": 15.0, "adot": 8.0, "explosive_rate": 8.0,
                   "yac_per_rec": 3.0},
        "other2": {"deep_rate": 5.0, "adot": 4.0, "explosive_rate": 2.0,
                   "yac_per_rec": 1.0},
        "other3": {"deep_rate": 20.0, "adot": 9.0, "explosive_rate": 10.0,
                   "yac_per_rec": 4.0},
    }
    computed = pr.compute_pool(pool, feats)
    role = pr.build_role("alpha", pool, computed)
    assert role["axes"]["Volume"] >= 80
    assert role["axes"]["Deep Work"] >= 80
    assert role["tags"] == ["Versatile"]


def _rb_pool_contingent():
    lead_series = [
        _row(w, snap_pct=72.0, snaps=45, carries=16, targets=3,
             carry_share=62.0, target_share=9.0, rush_yards=78.0,
             ppr_pts=13.0, rz_carries=2)
        for w in (1, 2, 3, 4)
    ]
    backup_series = []
    for w in range(1, 9):
        big = w >= 5
        backup_series.append(_row(
            w, snap_pct=55.0 if big else 18.0, snaps=35 if big else 11,
            carries=14 if big else 3, targets=4 if big else 1,
            carry_share=55.0 if big else 10.0,
            target_share=12.0 if big else 3.0,
            rush_yards=65.0 if big else 12.0,
            ppr_pts=19.0 if big else 2.0, rz_carries=2 if big else 0))
    return {
        "lead": {"pos": "RB", "team": "BUF", "name": "Larry Leadback",
                 "series": lead_series},
        "backup": {"pos": "RB", "team": "BUF", "name": "Barry Backup",
                   "series": backup_series},
        "rb_other": {"pos": "RB", "team": "MIA", "name": "Other Back",
                     "series": [_row(w, snap_pct=60.0, snaps=38, carries=12,
                                     targets=3, carry_share=45.0,
                                     target_share=10.0, rush_yards=50.0,
                                     ppr_pts=11.0, rz_carries=1)
                                for w in range(1, 9)]},
    }


def test_contingent_names_the_lead():
    pool = _rb_pool_contingent()
    computed = pr.compute_pool(pool)
    role = pr.build_role("backup", pool, computed)
    assert role["security"] == "Contingent"
    assert role["security_note"] is not None
    assert "Larry Leadback" in role["security_note"]


def test_lead_is_not_contingent():
    pool = _rb_pool_contingent()
    computed = pr.compute_pool(pool)
    role = pr.build_role("lead", pool, computed)
    assert role["security"] != "Contingent"


def test_passing_down_specialist():
    weeks = list(range(1, 7))
    pool = {
        "pd": {"pos": "RB", "team": "KC", "name": "Pass Down",
               "series": [_row(w, snap_pct=48.0, snaps=29, carries=3,
                               targets=7, receptions=6, carry_share=10.0,
                               target_share=20.0, rec_yards=52.0,
                               rush_yards=14.0, ppr_pts=10.0)
                          for w in weeks]},
        "bell": {"pos": "RB", "team": "KC", "name": "Bell Cow",
                 "series": [_row(w, snap_pct=70.0, snaps=44, carries=17,
                                 targets=2, carry_share=60.0, target_share=6.0,
                                 rush_yards=80.0, ppr_pts=14.0, rz_carries=2)
                            for w in weeks]},
        "mid": {"pos": "RB", "team": "DEN", "name": "Mid Back",
                "series": [_row(w, snap_pct=55.0, snaps=34, carries=10,
                                targets=3, carry_share=35.0, target_share=9.0,
                                rush_yards=44.0, ppr_pts=9.0, rz_carries=1)
                           for w in weeks]},
    }
    computed = pr.compute_pool(pool)
    role = pr.build_role("pd", pool, computed)
    assert role["job"] == "Passing-Down Specialist"


def test_trend_rising():
    series = []
    for i, w in enumerate(range(1, 9)):
        share = 10.0 + i * 3.0
        series.append(_row(w, snap_pct=60.0, snaps=36, targets=6,
                           target_share=share, rec_yards=50.0, ppr_pts=9.0))
    feats = pr.player_features(series, "WR")
    assert pr.trend_for(feats) == "Rising"


def test_timeline_segments_compress():
    pool = _pool()
    timeline = pr.build_timeline("alpha", pool)
    assert timeline
    assert timeline[0]["from_week"] == 1
    assert timeline[-1]["to_week"] == 6
    for seg in timeline:
        assert seg["from_week"] <= seg["to_week"]
        assert "job" in seg and "reason" in seg


def test_comparables_and_twin():
    pool = _pool()
    computed = pr.compute_pool(pool)
    jobs = {p: pr.job_for(e["pos"], computed[p]["score"], computed[p]["feats"])
            for p, e in pool.items()}
    comps, twin = pr.build_comparables("alpha", pool, computed, jobs)
    assert twin is not None and twin["player_id"] != "alpha"
    assert all(c["player_id"] != "alpha" for c in comps)
    assert len(comps) <= 3
    for c in comps:
        assert 0 <= c["similarity"] <= 100
        assert c["match_line"] and c["differ_line"]


def _qb_pool():
    weeks = [1, 2, 3, 4, 5, 6]
    pool = {}
    for pid, att, tds, rush in (("qb_a", 38, 2, 12.0), ("qb_b", 34, 2, 20.0),
                                ("qb_c", 30, 1, 8.0), ("qb_d", 26, 1, 30.0)):
        pool[pid] = {"pos": "QB", "team": "T" + pid[-1].upper(),
                     "name": pid.upper(),
                     "series": [_row(w, snap_pct=98.0, snaps=62, pass_att=att,
                                     pass_tds=tds, rush_yards=rush,
                                     ppr_pts=18.0) for w in weeks]}
    return pool


def _qb_pa_feats(rates, epas):
    return {pid: {"deep_att_rate": 10.0, "rating": 95.0, "rz_att_rate": 18.0,
                  "play_action_rate": rates[pid], "play_action_epa": epas[pid]}
            for pid in rates}


def test_play_action_merchant_tag_fires():
    pool = _qb_pool()
    feats = _qb_pa_feats(
        {"qb_a": 33.0, "qb_b": 24.0, "qb_c": 19.0, "qb_d": 13.0},
        {"qb_a": 0.34, "qb_b": 0.10, "qb_c": 0.02, "qb_d": -0.06})
    computed = pr.compute_pool(pool, feats)
    role = pr.build_role("qb_a", pool, computed)
    assert "Play-Action Merchant" in role["tags"]


def test_play_action_merchant_needs_epa_confirmation():
    pool = _qb_pool()
    # qb_a has the league's highest PA rate but its worst PA EPA: the
    # confirming signal fails, so the tag must not fire.
    feats = _qb_pa_feats(
        {"qb_a": 33.0, "qb_b": 24.0, "qb_c": 19.0, "qb_d": 13.0},
        {"qb_a": -0.08, "qb_b": 0.30, "qb_c": 0.12, "qb_d": 0.04})
    computed = pr.compute_pool(pool, feats)
    role = pr.build_role("qb_a", pool, computed)
    assert "Play-Action Merchant" not in role["tags"]


def test_play_action_merchant_absent_without_ftn_data():
    pool = _qb_pool()
    feats = {pid: {"deep_att_rate": 10.0, "rating": 95.0, "rz_att_rate": 18.0}
             for pid in pool}
    computed = pr.compute_pool(pool, feats)
    role = pr.build_role("qb_a", pool, computed)
    assert "Play-Action Merchant" not in role["tags"]


def test_aggregate_pa_rows_weighted_and_windowed():
    rows = [
        {"week": 1, "play_action_rate": 10.0, "play_action_epa": 0.0,
         "w_dropbacks": 40},
        {"week": 2, "play_action_rate": 30.0, "play_action_epa": 0.40,
         "w_dropbacks": 40},
        {"week": 3, "play_action_rate": None, "play_action_epa": None,
         "w_dropbacks": 40},
    ]
    agg = pr.aggregate_pa_rows(rows, [1, 2, 3])
    assert agg["play_action_rate"] == 20.0
    assert agg["play_action_epa"] == 0.2
    # Weeks outside the window are excluded entirely.
    assert pr.aggregate_pa_rows(rows, [2])["play_action_rate"] == 30.0
    # No charted weeks in window -> None (caller omits, never fakes).
    assert pr.aggregate_pa_rows(rows, [9]) is None
    assert pr.aggregate_pa_rows([], [1]) is None


def test_passer_rating_formula_bounds():
    # Perfect game clamps to 158.3; a dreadful one floors at 0.
    assert pr.passer_rating(20, 20, 400, 5, 0) == 158.3
    assert pr.passer_rating(20, 2, 30, 0, 4) == 0.0
    assert pr.passer_rating(0, 0, 0, 0, 0) == 0.0


def test_lane_and_zone_mapping():
    assert pr.lane_for("left", "end") == "LE"
    assert pr.lane_for("left", "tackle") == "LT"
    assert pr.lane_for("left", "guard") == "LG"
    assert pr.lane_for("middle", None) == "M"
    assert pr.lane_for("middle", "guard") == "M"
    assert pr.lane_for("right", "guard") == "RG"
    assert pr.lane_for("right", "tackle") == "RT"
    assert pr.lane_for("right", "end") == "RE"
    assert pr.zone_row_for_air_yards(-3) == "Behind"
    assert pr.zone_row_for_air_yards(0) == "Short"
    assert pr.zone_row_for_air_yards(9) == "Short"
    assert pr.zone_row_for_air_yards(10) == "Inter"
    assert pr.zone_row_for_air_yards(19) == "Inter"
    assert pr.zone_row_for_air_yards(20) == "Deep"


def test_success_rate_rule():
    assert pr.is_success(1, 10, 4) is True
    assert pr.is_success(1, 10, 3) is False
    assert pr.is_success(2, 10, 6) is True
    assert pr.is_success(3, 7, 7) is True
    assert pr.is_success(4, 7, 6) is False
