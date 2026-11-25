"""QB job blend (plays per game + start share + efficiency) and the
Franchise QB1 bar and games floor.

Pure-engine tests for dashboard_services.player_roles (no flask/pandas).

The QB job blend used to percentile dropbacks per game alone, which
prices a QB as a pocket passer: Josh Allen sat at QB1 (76.4) despite
66 rush attempts, because none of his rushing workload counted. The
volume input is now plays per game (pass attempts + rush attempts).
The blend then grew a third leg: volume + start share alone crowned
pure volume merchants (Tyler Shough scored a perfect 100 on plays and
starts with a below-average efficiency percentile), so the score is
now 0.45 plays percentile + 0.25 start share + 0.30 efficiency
percentile (the Efficiency axis percentile itself), and Franchise
QB1 takes a 90 score plus >= 5 games in the window (the engine's
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


def _qb_series(weeks, att, carries, rush_yd=10.0, ppr=18.0):
    return [_row(w, snap_pct=98.0, snaps=62, pass_att=att, carries=carries,
                 rush_yards=rush_yd, pass_tds=2, ppr_pts=ppr)
            for w in weeks]


def _qb_pool():
    """11 QBs, all 9 games, all start share 100.

    Score = 0.45*plays pctile + 25 + 0.30*efficiency pctile, where the
    efficiency percentile is the ppr-per-dropback percentile here (no
    PBP slice in the fixture, so the axis fallback prices it).
    specs: pid -> (att/g, carries/g, ppr pts/g).

    Allen-shaped: second-best plays/g (38: modest passing, heavy
    rushing) and the pool's best efficiency (27 pts on 30 att), so he
    clears the 90 Franchise bar at 95.5. top1 owns the best plays/g
    (42) but the pool's worst efficiency (20 pts on 40 att): a pure
    volume merchant, QB1 at 70.0. Penix-shaped: mid plays/g (33) on
    passing alone with near-bottom efficiency, QB2-Streamer at 55.0.
    Plays/g ladder: 42, 38(allen), 36, 34, 33(penix), 31, 29, 27, 25,
    23, 21 -> Allen plays pctile 90, Penix 60.
    Efficiency ladder (ppr/att): allen .900, then .684, .667, .656,
    .652, .647, .640, .630, .621, penix .594, top1 .500 -> Allen
    efficiency pctile 100, Penix 10, top1 0.
    """
    weeks = list(range(1, 10))
    specs = {
        "top1": (40, 2, 20.0), "allen": (30, 8, 27.0),
        "qb3": (34, 2, 22.0), "qb4": (32, 2, 21.0),
        "penix": (32, 1, 19.0),
        "mid1": (29, 2, 18.0), "mid2": (27, 2, 17.0),
        "mid3": (25, 2, 16.0), "mid4": (23, 2, 15.0),
        "mid5": (21, 2, 14.0), "low": (19, 2, 13.0),
    }
    pool = {}
    for pid, (att, car, ppr) in specs.items():
        rush = 36.0 if pid == "allen" else 10.0
        pool[pid] = {"pos": "QB", "team": "T" + pid[:2].upper(),
                     "name": pid.upper(),
                     "series": _qb_series(weeks, att, car, rush_yd=rush,
                                          ppr=ppr)}
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
    assert round(computed["allen"]["feats"]["plays_per_game"], 1) == 38.0
    assert computed["allen"]["score"] == 95.5
    assert role["job"] == "Franchise QB1"
    # The Volume axis reads the same plays/g input as the job blend,
    # and the Efficiency axis IS the blend's efficiency percentile.
    assert role["axes"]["Volume"] == 90
    assert role["axes"]["Efficiency"] == 100


def test_efficiency_leg_demotes_pure_volume():
    # top1 owns the pool's best plays/g but its worst efficiency.
    # Volume + durability alone scored him a perfect 100; with the
    # efficiency leg he lands at QB1, below the slightly-lower-volume,
    # top-efficiency Allen shape.
    pool = _qb_pool()
    computed = pr.compute_pool(pool)
    assert computed["top1"]["score"] == 70.0
    assert computed["top1"]["score"] < computed["allen"]["score"]
    assert pr.build_role("top1", pool, computed)["job"] == "QB1"


def test_penix_passing_only_volume_drops():
    # Passing-only volume with near-bottom efficiency: plays/g alone
    # no longer carries him toward the top tier.
    pool = _qb_pool()
    computed = pr.compute_pool(pool)
    role = pr.build_role("penix", pool, computed)
    assert computed["penix"]["score"] == 55.0
    assert role["job"] == "QB2-Streamer"


def test_franchise_requires_five_games():
    assert pr.FRANCHISE_QB_MIN_GAMES == 5
    assert pr.FRANCHISE_QB_SCORE_MIN == 90.0
    assert pr.job_for("QB", 95.0, {"games": 4}) == "QB1"
    assert pr.job_for("QB", 95.0, {"games": 5}) == "Franchise QB1"
    assert pr.job_for("QB", 100.0, {"games": 1}) == "QB1"
    # The bar itself: 90 with the games, 89.9 without the tier.
    assert pr.job_for("QB", 90.0, {"games": 9}) == "Franchise QB1"
    assert pr.job_for("QB", 89.9, {"games": 9}) == "QB1"
    # The floor only caps the top tier; lower tiers are untouched.
    assert pr.job_for("QB", 65.0, {"games": 1}) == "QB1"
    assert pr.job_for("QB", 45.0, {"games": 1}) == "QB2-Streamer"


def test_games_floor_is_qb_only():
    # Other positions' top tiers keep no games floor in this change
    # (scores here clear the 92 top-tier bar, so only a games floor
    # could keep them out).
    assert pr.job_for("WR", 95.0, {"games": 1}) == "Alpha WR1"
    assert pr.job_for("TE", 95.0, {"games": 1, "snap_pct": 80.0,
                                   "opp_share": 20.0}) == "Alpha TE1"


def test_one_game_flash_is_qb1_not_franchise():
    # A one-game fill-in with the pool's best plays/g AND best
    # efficiency scores 100 but must land at QB1 with Provisional
    # confidence, not Franchise.
    pool = {
        "flash": {"pos": "QB", "team": "TA", "name": "FLASH",
                  "series": _qb_series([10], 45, 5, ppr=45.0)},
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
                      "series": _qb_series(list(range(1, 10)), 45, 5,
                                           ppr=45.0)}
    computed = pr.compute_pool(pool)
    role = pr.build_role("flash", pool, computed)
    assert role["job"] == "Franchise QB1"
