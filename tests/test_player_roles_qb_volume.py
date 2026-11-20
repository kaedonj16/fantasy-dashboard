"""QB job volume (plays per game) + Franchise QB1 games floor.

Pure-engine tests for dashboard_services.player_roles (no flask/pandas).

The QB job blend used to percentile dropbacks per game alone, which
prices a QB as a pocket passer: Josh Allen sat at QB1 (76.4) despite
66 rush attempts, because none of his rushing workload counted. The
volume input is now plays per game (pass attempts + rush attempts).
Franchise QB1 also requires >= 5 games in the window (the engine's
Provisional boundary): one-game fill-ins held the tier before this.
"""
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


def _qb_series(weeks, att, carries, rush_yd=10.0):
    return [_row(w, snap_pct=98.0, snaps=62, pass_att=att, carries=carries,
                 rush_yards=rush_yd, pass_tds=2, ppr_pts=18.0)
            for w in weeks]


def _qb_pool():
    """11 QBs, all 9 games, all start share 100 (score = 0.6*pct + 40).

    Allen-shaped: modest passing (28 att/g) but heavy rushing (8/g),
    so plays/g (36) outranks his dropbacks/g. Penix-shaped: more
    passing (30 att/g), almost no rushing (3/g), so dropbacks alone
    would have made him Franchise and plays/g does not.
    Plays/g ladder: 42, 40, 38, 36(allen), 33(penix), 31, 29, 27, 25,
    23, 21 -> Allen plays pctile 70 (score 82.0), Penix 60 (76.0).
    Att/g ladder:   40, 38, 36, 30(penix), 29, 28(allen), 26, 24, 22,
    20, 18 -> Allen dropback pctile 50 (old score 70.0, QB1).
    """
    weeks = list(range(1, 10))
    specs = {
        "top1": (40, 2), "top2": (38, 2), "top3": (36, 2),
        "allen": (28, 8), "penix": (30, 3),
        "mid1": (29, 2), "mid2": (26, 3), "mid3": (24, 3),
        "mid4": (22, 3), "mid5": (20, 3), "low": (18, 3),
    }
    pool = {}
    for pid, (att, car) in specs.items():
        rush = 36.0 if pid == "allen" else 10.0
        pool[pid] = {"pos": "QB", "team": "T" + pid[:2].upper(),
                     "name": pid.upper(),
                     "series": _qb_series(weeks, att, car, rush_yd=rush)}
    return pool


def test_plays_per_game_feature_counts_rush_attempts():
    # Allen's real shape: 257 pass att + 66 rush att over 9 games.
    series = []
    for i, w in enumerate(range(1, 10)):
        series.append(_row(w, snap_pct=98.0, snaps=62,
                           pass_att=29 if i < 5 else 28,
                           carries=8 if i < 3 else 7,
                           rush_yards=32.0, pass_tds=2, ppr_pts=24.0))
    feats = pr.player_features(series, "QB")
    assert feats["pass_att"] == 257
    assert feats["carries"] == 66
    assert round(feats["plays_per_game"], 1) == 35.9
    # The honest dropbacks feature is untouched by the new one.
    assert round(feats["dropbacks_per_game"], 1) == 28.6


def test_allen_rushing_volume_reaches_franchise():
    pool = _qb_pool()
    computed = pr.compute_pool(pool)
    role = pr.build_role("allen", pool, computed)
    assert round(computed["allen"]["feats"]["plays_per_game"], 1) == 36.0
    assert computed["allen"]["score"] == 82.0
    assert role["job"] == "Franchise QB1"
    # The Volume axis reads the same plays/g input as the job blend.
    assert role["axes"]["Volume"] == 70


def test_penix_passing_only_volume_drops_to_qb1():
    pool = _qb_pool()
    computed = pr.compute_pool(pool)
    role = pr.build_role("penix", pool, computed)
    assert computed["penix"]["score"] == 76.0
    assert role["job"] == "QB1"


def test_franchise_requires_five_games():
    assert pr.FRANCHISE_QB_MIN_GAMES == 5
    assert pr.job_for("QB", 85.0, {"games": 4}) == "QB1"
    assert pr.job_for("QB", 85.0, {"games": 5}) == "Franchise QB1"
    assert pr.job_for("QB", 100.0, {"games": 1}) == "QB1"
    # The floor only caps the top tier; lower tiers are untouched.
    assert pr.job_for("QB", 65.0, {"games": 1}) == "QB1"
    assert pr.job_for("QB", 45.0, {"games": 1}) == "QB2-Streamer"


def test_games_floor_is_qb_only():
    # Other positions' top tiers keep no games floor in this change.
    assert pr.job_for("WR", 90.0, {"games": 1}) == "Alpha WR1"
    assert pr.job_for("TE", 85.0, {"games": 1, "snap_pct": 80.0,
                                   "opp_share": 20.0}) == "Alpha TE1"


def test_one_game_flash_is_qb1_not_franchise():
    # A one-game fill-in with the pool's best plays/g scores 100 but
    # must land at QB1 with Provisional confidence, not Franchise.
    pool = {
        "flash": {"pos": "QB", "team": "TA", "name": "FLASH",
                  "series": _qb_series([10], 45, 5)},
    }
    for pid, att in (("reg1", 28), ("reg2", 26), ("reg3", 24), ("reg4", 22)):
        pool[pid] = {"pos": "QB", "team": "T" + pid[-1], "name": pid.upper(),
                     "series": _qb_series(list(range(1, 10)), att, 2)}
    computed = pr.compute_pool(pool)
    role = pr.build_role("flash", pool, computed)
    assert computed["flash"]["score"] == 100.0
    assert role["job"] == "QB1"
    assert role["confidence"] == "Provisional"
    # The same rates over a full 9-game window do earn the tier.
    pool["flash"] = {**pool["flash"],
                      "series": _qb_series(list(range(1, 10)), 45, 5)}
    computed = pr.compute_pool(pool)
    role = pr.build_role("flash", pool, computed)
    assert role["job"] == "Franchise QB1"
