"""Room-dominance promotion + Home-Run Hitter confirm recalibration.

Room dominance: the Job blend percentiles opportunity share league-wide,
so a player who owns his own room (Ollie Gordon II: 65% of Miami's RB
opportunities at 3.1x the next back) still graded Lead 1A. A player one
tier below the top now promotes when he holds >= 50% of his team's
position-group opportunities at >= 1.8x the next teammate. Committees
must not promote.

Home-Run Hitter: the confirming signal used to be the percentile of
pbp_explosive_rate, which for an RB is the RECEIVING explosive rate
(20+ yard share of receptions) -- ~0 for pure runners, so it blocked
the trait the Explosiveness axis (explosive RUN rate) measures. It now
confirms on yards-per-carry percentile.
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


def _rb_series(weeks, carries, targets, snap, carry_share, target_share,
               ypc=4.5, rz=1):
    return [_row(w, snap_pct=snap, snaps=int(snap * 0.62), carries=carries,
                 targets=targets, receptions=int(targets * 0.8),
                 carry_share=carry_share, target_share=target_share,
                 rush_yards=carries * ypc, rec_yards=targets * 6.0,
                 ppr_pts=carries * 0.9 + targets * 1.2, rz_carries=rz)
            for w in weeks]


_W = list(range(1, 7))


def _league_pool(dom2_carries=6, dom2_targets=2, dom2_snap=30):
    """A small RB league. 'dom' grades Lead 1A on the league-wide blend;
    his room split depends on dom2's line."""
    return {
        "ace1": {"pos": "RB", "team": "AAA", "name": "Ace One",
                 "series": _rb_series(_W, 20, 4, 78, 68, 12, rz=3)},
        "ace2": {"pos": "RB", "team": "BBB", "name": "Ace Two",
                 "series": _rb_series(_W, 18, 4, 72, 62, 11, rz=3)},
        "dom": {"pos": "RB", "team": "TST", "name": "Dom Back",
                "series": _rb_series(_W, 15, 3, 62, 55, 10, rz=2)},
        "dom2": {"pos": "RB", "team": "TST", "name": "Second Back",
                 "series": _rb_series(_W, dom2_carries, dom2_targets,
                                      dom2_snap, 20, 6, rz=0)},
        "mid": {"pos": "RB", "team": "CCC", "name": "Mid Back",
                "series": _rb_series(_W, 11, 3, 52, 42, 9, rz=1)},
        "low": {"pos": "RB", "team": "DDD", "name": "Low Back",
                "series": _rb_series(_W, 6, 1, 28, 22, 4, rz=0)},
        "low2": {"pos": "RB", "team": "EEE", "name": "Low Two",
                 "series": _rb_series(_W, 4, 1, 20, 15, 3, rz=0)},
    }


def test_room_dominant_back_promotes_to_workhorse():
    pool = _league_pool()
    computed = pr.compute_pool(pool)
    # The league-wide blend alone leaves him one tier short...
    raw = pr.job_for("RB", computed["dom"]["score"], computed["dom"]["feats"])
    assert raw == "Lead 1A"
    # ...but he holds 69% of his room at 2.25x the next back.
    share, nxt, ratio = pr.room_dominance("dom", pool, computed)
    assert share >= pr.ROOM_DOM_SHARE_MIN
    assert ratio >= pr.ROOM_DOM_RATIO_MIN
    assert nxt < share
    role = pr.build_role("dom", pool, computed)
    assert role["job"] == "Workhorse"
    assert pr.job_for_player("dom", pool, computed) == "Workhorse"


def test_committee_back_does_not_promote():
    # dom2 grows into a real committee mate: dom keeps a share majority
    # (58%) but only 1.38x the next back -- no promotion.
    pool = _league_pool(dom2_carries=10, dom2_targets=3, dom2_snap=44)
    pool["dom2"]["series"] = _rb_series(_W, 10, 3, 44, 35, 9, rz=1)
    computed = pr.compute_pool(pool)
    share, _nxt, ratio = pr.room_dominance("dom", pool, computed)
    assert share >= pr.ROOM_DOM_SHARE_MIN
    assert ratio < pr.ROOM_DOM_RATIO_MIN
    role = pr.build_role("dom", pool, computed)
    assert role["job"] == "Lead 1A"


def test_near_split_does_not_promote():
    # 56.5% vs 43.5% (x1.30): a lead back in a split room, not a dominator.
    pool = _league_pool()
    pool["dom"] = {"pos": "RB", "team": "TST", "name": "Dom Back",
                   "series": _rb_series(_W, 11, 2, 58, 48, 8, rz=1)}
    pool["dom2"] = {"pos": "RB", "team": "TST", "name": "Second Back",
                    "series": _rb_series(_W, 8, 2, 46, 36, 7, rz=1)}
    computed = pr.compute_pool(pool)
    share, nxt, ratio = pr.room_dominance("dom", pool, computed)
    assert 55 <= share <= 58 and 42 <= nxt <= 45
    assert ratio < pr.ROOM_DOM_RATIO_MIN
    role = pr.build_role("dom", pool, computed)
    assert role["job"] == "Lead 1A"


def _totals_series(weeks, car_t, tgt_t, snap, carry_share, target_share):
    rows = []
    for i, w in enumerate(weeks):
        car = car_t // len(weeks) + (1 if i < car_t % len(weeks) else 0)
        tgt = tgt_t // len(weeks) + (1 if i < tgt_t % len(weeks) else 0)
        rows.append(_row(w, snap_pct=snap, snaps=int(snap * 0.62),
                         carries=car, targets=tgt,
                         receptions=int(tgt * 0.8),
                         carry_share=carry_share, target_share=target_share,
                         rush_yards=car * 4.4, rec_yards=tgt * 6.0,
                         ppr_pts=car * 0.9 + tgt * 1.2, rz_carries=1))
    return rows


def test_gordon_shaped_fixture_promotes():
    # Gordon-shaped room: 53% of the room's opportunities vs the next
    # back's 25% (x2.12), with the league-wide blend at Lead 1A.
    w10 = list(range(1, 11))
    pool = {
        "ace1": {"pos": "RB", "team": "AAA", "name": "Ace One",
                 "series": _rb_series(w10, 20, 4, 78, 68, 12, rz=3)},
        "ace2": {"pos": "RB", "team": "BBB", "name": "Ace Two",
                 "series": _rb_series(w10, 18, 4, 72, 62, 11, rz=3)},
        "gdom": {"pos": "RB", "team": "MIA", "name": "Gordon Shape",
                 "series": _totals_series(w10, 45, 8, 58, 40, 8)},
        "gmate": {"pos": "RB", "team": "MIA", "name": "Room Mate",
                  "series": _totals_series(w10, 20, 5, 34, 18, 5)},
        "gthird": {"pos": "RB", "team": "MIA", "name": "Room Third",
                   "series": _totals_series(w10, 18, 4, 26, 14, 4)},
        "mid": {"pos": "RB", "team": "CCC", "name": "Mid Back",
                "series": _rb_series(w10, 11, 3, 52, 42, 9, rz=1)},
        "low": {"pos": "RB", "team": "DDD", "name": "Low Back",
                "series": _rb_series(w10, 6, 1, 28, 22, 4, rz=0)},
    }
    computed = pr.compute_pool(pool)
    share, nxt, ratio = pr.room_dominance("gdom", pool, computed)
    assert share == 53.0 and nxt == 25.0
    assert ratio >= pr.ROOM_DOM_RATIO_MIN
    raw = pr.job_for("RB", computed["gdom"]["score"], computed["gdom"]["feats"])
    assert raw == "Lead 1A"
    role = pr.build_role("gdom", pool, computed)
    assert role["job"] == "Workhorse"


def test_promotion_never_demotes_or_skips_tiers():
    # A Committee 1B with a dominant room share of a thin room stays
    # Committee 1B: promotion only bridges the final tier.
    pool = _league_pool()
    computed = pr.compute_pool(pool)
    assert pr.job_for_player("mid", pool, computed) == "Committee 1B"
    # QB is exempt from room logic entirely.
    assert pr.room_dominance("ace1", pool, computed) is not None  # RB sanity
    qb_pool = {"q": {"pos": "QB", "team": "TST", "name": "Qb One",
                     "series": [_row(w, snap_pct=100.0, snaps=60, pass_att=35,
                                     ppr_pts=20.0) for w in _W]}}
    qb_computed = pr.compute_pool(qb_pool)
    assert pr.room_dominance("q", qb_pool, qb_computed) is None


# ---------------------------------------------------------------------------
# Home-Run Hitter confirm: yards per carry, not receiving explosive rate
# ---------------------------------------------------------------------------

def _hrh_pool():
    """Five backs. 'hr' breaks explosive runs at the pool's best rate
    and runs efficiently; his RECEIVING explosive rate is 0.0 (the old
    confirm's raw value). 'slug' also breaks runs but at 2.6 YPC."""
    def mk(carries, ypc, snap, cs):
        return _rb_series(_W, carries, 2, snap, cs, 6, ypc=ypc, rz=0)
    pool = {
        "hr": {"pos": "RB", "team": "AAA", "name": "Home Runner",
               "series": mk(12, 5.2, 55, 45)},
        "slug": {"pos": "RB", "team": "BBB", "name": "Slug Back",
                 "series": mk(12, 2.6, 55, 45)},
        "mid1": {"pos": "RB", "team": "CCC", "name": "Mid One",
                 "series": mk(11, 4.4, 52, 42)},
        "mid2": {"pos": "RB", "team": "DDD", "name": "Mid Two",
                 "series": mk(10, 4.0, 50, 40)},
        "low": {"pos": "RB", "team": "EEE", "name": "Low Back",
                "series": mk(8, 3.5, 40, 32)},
    }
    feats = {
        "hr": {"explosive_run_rate": 18.0, "explosive_rate": 0.0},
        "slug": {"explosive_run_rate": 16.0, "explosive_rate": 0.0},
        "mid1": {"explosive_run_rate": 10.0, "explosive_rate": 5.0},
        "mid2": {"explosive_run_rate": 6.0, "explosive_rate": 5.0},
        "low": {"explosive_run_rate": 3.0, "explosive_rate": 5.0},
    }
    return pool, feats


def test_home_run_hitter_confirms_on_ypc_not_receiving_explosives():
    pool, feats = _hrh_pool()
    computed = pr.compute_pool(pool, feats)
    hr = computed["hr"]
    assert hr["axes"]["Explosiveness"] >= 70
    # The old confirm's raw feature is 0.0 and its percentile < 50: the
    # old rule could never tag this player.
    assert hr["feats"].get("pbp_explosive_rate") == 0.0
    assert hr["feat_pct"]["pbp_explosive_rate"] < 50
    assert hr["feat_pct"]["rush_ypc"] >= 50
    role = pr.build_role("hr", pool, computed)
    assert "Home-Run Hitter" in role["tags"]


def test_home_run_hitter_blocked_when_ypc_is_pool_worst():
    pool, feats = _hrh_pool()
    computed = pr.compute_pool(pool, feats)
    slug = computed["slug"]
    assert slug["axes"]["Explosiveness"] >= 70
    assert slug["feat_pct"]["rush_ypc"] < 50
    role = pr.build_role("slug", pool, computed)
    assert "Home-Run Hitter" not in role["tags"]
