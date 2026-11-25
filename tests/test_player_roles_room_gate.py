"""Room gate: the top Job tier belongs to the room's rate winner.

The Job blend is league-wide, so a room's #2 could hold the top tier
on elite percentiles while never winning his own room (Denzel Boston
and Jordan Addison were both Alpha WR1 as their rooms' second
receivers). The gate caps the top tier (Alpha WR1 / Workhorse /
Alpha TE1) to the winner of the player's own team+position room, and
the winner is decided on opportunities PER GAME, not season totals,
so missed games do not cost a player his room: De'Von Achane leads
Miami's RB room per game in a 3-game sample, while Ollie Gordon II
only leads it on totals built up during Achane's absence.

Room-dominance promotion runs on the same rate basis, and the gate
runs after it: a promotion earned while the room's lead was out does
not override the gate. Claiming the room takes >= ROOM_GATE_MIN_GAMES
games; a 1-2 game spike can neither block the room nor take the tier.

The top tier also carries a 92 score bar at all three gated
positions (the tiers below widen to meet it). Fixture pools below
are sized so a room #2 can actually reach 92 the way he can in the
real 200+ player pools: pool-best snap percentile, second-best
opportunity-share percentile, and only the room term docking him.
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


def _wr_series(weeks, targets, snap, target_share, rz=1):
    return [_row(w, snap_pct=snap, snaps=int(snap * 0.62), targets=targets,
                 receptions=int(targets * 0.7), target_share=target_share,
                 rec_yards=targets * 8.0, ppr_pts=targets * 1.6,
                 rz_targets=rz)
            for w in weeks]


_W = list(range(1, 11))


def _raw(pid, pool, computed):
    return pr.job_for(pool[pid]["pos"], computed[pid]["score"],
                      computed[pid]["feats"])


def _rb_fillers(n, start=0):
    """Low-volume backs on their own teams: opportunity shares in the
    10-20% range, snaps 25-55, so the named backs' percentiles sit
    where a big real pool would put them."""
    out = {}
    for i in range(start, start + n):
        out[f"rb{i}"] = {
            "pos": "RB", "team": f"G{i:02d}", "name": f"Filler Back {i}",
            "series": _rb_series(_W, 4 + (i % 6), 1 + (i % 2),
                                 25 + (i * 7) % 31,
                                 [20, 24, 28, 32, 36][i % 5],
                                 [4, 5, 6, 7, 8][i % 5], rz=0)}
    return out


# ---------------------------------------------------------------------------
# The 92 bar itself
# ---------------------------------------------------------------------------

def test_top_tier_bar_is_92_at_gated_positions():
    assert pr.TOP_TIER_SCORE_MIN == 92.0
    assert pr.job_for("WR", 91.9, {}) == "WR1"
    assert pr.job_for("WR", 92.0, {}) == "Alpha WR1"
    assert pr.job_for("RB", 91.9, {}) == "Lead 1A"
    assert pr.job_for("RB", 92.0, {}) == "Workhorse"
    te_feats = {"snap_pct": 80.0, "opp_share": 20.0}
    assert pr.job_for("TE", 91.9, te_feats) == "TE1"
    assert pr.job_for("TE", 92.0, te_feats) == "Alpha TE1"
    # The tiers below simply widen; their floors do not move.
    assert pr.job_for("WR", 70.0, {}) == "WR1"
    assert pr.job_for("RB", 65.0, {}) == "Lead 1A"
    assert pr.job_for("TE", 60.0, te_feats) == "TE1"
    # QB keeps its own line for Franchise QB1 (90, plus the games floor).
    assert pr.job_for("QB", 90.0, {"games": 9}) == "Franchise QB1"
    assert pr.job_for("QB", 89.9, {"games": 9}) == "QB1"
    assert pr.job_for("QB", 80.0, {"games": 9}) == "QB1"


# ---------------------------------------------------------------------------
# The Achane/Gordon shape: rate winner keeps the tier, totals leader caps
# ---------------------------------------------------------------------------

def _mia_pool():
    """'ach' plays 3 games at 26 opp/game (78 total). 'gor' plays 10
    games at 19 opp/game (190 total) and scores exactly 92: pool-best
    snap percentile, second-best share percentile, docked only by the
    room term. On totals gor owns the room; on rates ach does."""
    pool = {
        "ach": {"pos": "RB", "team": "MIA", "name": "Rate Ace",
                "series": _rb_series([1, 2, 3], 22, 4, 62, 55, 13, rz=2)},
        "gor": {"pos": "RB", "team": "MIA", "name": "Totals Back",
                "series": _rb_series(_W, 16, 3, 85, 40, 10, rz=2)},
        "ace1": {"pos": "RB", "team": "AAA", "name": "Ace One",
                 "series": _rb_series(_W, 20, 4, 78, 40, 7, rz=3)},
        "ace2": {"pos": "RB", "team": "BBB", "name": "Ace Two",
                 "series": _rb_series(_W, 18, 4, 72, 38, 7, rz=3)},
        "mid": {"pos": "RB", "team": "CCC", "name": "Mid Back",
                "series": _rb_series(_W, 11, 3, 52, 30, 6, rz=1)},
        "mid2": {"pos": "RB", "team": "DDD", "name": "Mid Two",
                 "series": _rb_series(_W, 9, 2, 45, 28, 5, rz=1)},
        "low": {"pos": "RB", "team": "EEE", "name": "Low Back",
                "series": _rb_series(_W, 6, 1, 28, 15, 3, rz=0)},
        "low2": {"pos": "RB", "team": "FFF", "name": "Low Two",
                 "series": _rb_series(_W, 4, 1, 20, 12, 2, rz=0)},
    }
    pool.update(_rb_fillers(18, start=10))
    return pool


def test_rate_winner_keeps_workhorse_totals_leader_caps():
    pool = _mia_pool()
    computed = pr.compute_pool(pool)
    # Both men grade Workhorse on the league-wide blend...
    assert _raw("ach", pool, computed) == "Workhorse"
    assert _raw("gor", pool, computed) == "Workhorse"
    # ...gor has 2.4x ach's total opportunities but the lower rate.
    assert computed["gor"]["feats"]["opp"] > computed["ach"]["feats"]["opp"]
    assert (computed["ach"]["feats"]["opp_per_game"]
            > computed["gor"]["feats"]["opp_per_game"])
    assert pr.room_rate_winner("gor", pool, computed) == "ach"
    # The rate winner keeps the tier on 3 games; the totals leader caps.
    assert pr.job_for_player("ach", pool, computed) == "Workhorse"
    assert pr.job_for_player("gor", pool, computed) == "Lead 1A"
    role = pr.build_role("gor", pool, computed)
    assert role["job"] == "Lead 1A"


def test_gordon_shape_no_longer_dominates_his_room():
    # On totals, gor held 71% of the MIA room at 2.4x the next back:
    # the old dominance test promoted exactly this shape. On rates his
    # share is 42% at 0.73x, so there is nothing to promote.
    pool = _mia_pool()
    computed = pr.compute_pool(pool)
    share, _nxt, ratio = pr.room_dominance("gor", pool, computed)
    assert share < pr.ROOM_DOM_SHARE_MIN
    assert ratio < 1.0


# ---------------------------------------------------------------------------
# The Boston/Addison shape: a second Alpha in the room caps at WR1
# ---------------------------------------------------------------------------

def _two_alpha_wr_pool():
    """26 WRs. w2 is second in the whole pool in opportunity share
    and first in snap share, so even docked to the room's #2 term he
    scores exactly 92; w1 beats him on rate and share."""
    pool = {
        "w1": {"pos": "WR", "team": "TST", "name": "Room Winner",
               "series": _wr_series(_W, 11, 84, 25, rz=2)},
        "w2": {"pos": "WR", "team": "TST", "name": "Second Wideout",
               "series": _wr_series(_W, 10, 92, 24, rz=2)},
    }
    shares = [22, 20, 18, 16, 14, 12, 10, 9, 8, 7, 6, 5,
              4, 3, 15, 13, 11, 9, 8, 6, 5, 4, 3, 2]
    for i, share in enumerate(shares):
        pool[f"f{i}"] = {"pos": "WR", "team": f"F{i:02d}",
                         "name": f"Filler {i}",
                         "series": _wr_series(_W, 5, 80 - (i % 10) * 4,
                                              share, rz=0)}
    return pool


def test_second_alpha_in_room_caps_at_wr1():
    pool = _two_alpha_wr_pool()
    computed = pr.compute_pool(pool)
    # Both TST wideouts grade Alpha on the blend (the #2 rides a
    # pool-best snap share, the Boston/Addison shape)...
    assert _raw("w1", pool, computed) == "Alpha WR1"
    assert _raw("w2", pool, computed) == "Alpha WR1"
    # ...but only the room's rate winner keeps the title.
    assert pr.room_rate_winner("w2", pool, computed) == "w1"
    assert pr.job_for_player("w1", pool, computed) == "Alpha WR1"
    assert pr.job_for_player("w2", pool, computed) == "WR1"


def test_room_winner_under_92_is_not_alpha():
    # A smaller pool: the TST winner scores 91.5. Winning the room is
    # not enough on its own; under the 92 bar he is a WR1.
    pool = {
        "w1": {"pos": "WR", "team": "TST", "name": "Room Winner",
               "series": _wr_series(_W, 11, 84, 25, rz=2)},
        "w2": {"pos": "WR", "team": "TST", "name": "Second Wideout",
               "series": _wr_series(_W, 10, 92, 24, rz=2)},
        "wa": {"pos": "WR", "team": "AAA", "name": "Other Ace",
               "series": _wr_series(_W, 13, 86, 32, rz=2)},
        "wb": {"pos": "WR", "team": "BBB", "name": "Bee Wideout",
               "series": _wr_series(_W, 9, 80, 22, rz=1)},
    }
    for i, (tm, share) in enumerate([("CCC", 18), ("DDD", 16), ("EEE", 14),
                                     ("FFF", 12), ("GGG", 10), ("HHH", 8),
                                     ("III", 6), ("JJJ", 5), ("KKK", 4),
                                     ("LLL", 3)]):
        pool[f"f{i}"] = {"pos": "WR", "team": tm, "name": f"Filler {i}",
                         "series": _wr_series(_W, 5, 60 - i * 3, share, rz=0)}
    computed = pr.compute_pool(pool)
    assert 90 <= computed["w1"]["score"] < pr.TOP_TIER_SCORE_MIN
    assert pr.room_rate_winner("w1", pool, computed) == "w1"
    assert pr.job_for_player("w1", pool, computed) == "WR1"


def test_te_gate_caps_second_alpha_te1():
    pool = {
        "te1": {"pos": "TE", "team": "TST", "name": "Room Winner",
                "series": _wr_series(_W, 10, 80, 26, rz=2)},
        "te2": {"pos": "TE", "team": "TST", "name": "Second Tight End",
                "series": _wr_series(_W, 9, 88, 23, rz=2)},
    }
    shares = [19, 17, 15, 13, 11, 9, 7, 5, 4, 3,
              18, 16, 14, 12, 10, 8, 6, 5, 4, 3, 2, 9, 7, 5]
    for i, share in enumerate(shares):
        pool[f"t{i}"] = {"pos": "TE", "team": f"T{i:02d}",
                         "name": f"Filler {i}",
                         "series": _wr_series(_W, 4, 78 - (i % 9) * 5,
                                              share, rz=0)}
    computed = pr.compute_pool(pool)
    assert _raw("te1", pool, computed) == "Alpha TE1"
    assert _raw("te2", pool, computed) == "Alpha TE1"
    assert pr.job_for_player("te1", pool, computed) == "Alpha TE1"
    assert pr.job_for_player("te2", pool, computed) == "TE1"


# ---------------------------------------------------------------------------
# Short samples: a spike can neither claim the room nor block it
# ---------------------------------------------------------------------------

def test_two_game_rate_spike_neither_claims_nor_blocks():
    pool = {
        # The spike owns the room's best per-game rate (31.0) but in
        # only 2 games, on a team whose totals those weeks were huge
        # (his opportunity share is modest).
        "spike": {"pos": "RB", "team": "TST", "name": "Spike Back",
                  "series": _rb_series([1, 2], 26, 5, 50, 30, 6, rz=1)},
        "steady": {"pos": "RB", "team": "TST", "name": "Steady Back",
                   "series": _rb_series(_W, 18, 4, 76, 52, 12, rz=2)},
        "ace1": {"pos": "RB", "team": "AAA", "name": "Ace One",
                 "series": _rb_series(_W, 20, 4, 78, 68, 12, rz=3)},
        "ace2": {"pos": "RB", "team": "BBB", "name": "Ace Two",
                 "series": _rb_series(_W, 18, 4, 72, 62, 11, rz=3)},
        "mid": {"pos": "RB", "team": "CCC", "name": "Mid Back",
                "series": _rb_series(_W, 11, 3, 52, 42, 9, rz=1)},
        "mid2": {"pos": "RB", "team": "DDD", "name": "Mid Two",
                 "series": _rb_series(_W, 9, 2, 45, 36, 8, rz=1)},
        "g1": {"pos": "RB", "team": "GGG", "name": "Gee One",
               "series": _rb_series(_W, 8, 2, 40, 33, 7, rz=0)},
        "low": {"pos": "RB", "team": "EEE", "name": "Low Back",
                "series": _rb_series(_W, 6, 1, 28, 22, 4, rz=0)},
        "low2": {"pos": "RB", "team": "FFF", "name": "Low Two",
                 "series": _rb_series(_W, 4, 1, 20, 15, 3, rz=0)},
        "h1": {"pos": "RB", "team": "HHH", "name": "Aitch One",
               "series": _rb_series(_W, 5, 1, 24, 18, 4, rz=0)},
    }
    pool.update(_rb_fillers(8, start=30))
    computed = pr.compute_pool(pool)
    assert (computed["spike"]["feats"]["opp_per_game"]
            > computed["steady"]["feats"]["opp_per_game"])
    assert computed["spike"]["feats"]["games"] < pr.ROOM_GATE_MIN_GAMES
    # The spike cannot be the room winner, so he cannot block the
    # steady back's Workhorse, and he holds no top tier himself.
    assert pr.room_rate_winner("steady", pool, computed) == "steady"
    assert _raw("steady", pool, computed) == "Workhorse"
    assert pr.job_for_player("steady", pool, computed) == "Workhorse"
    assert pr.job_for_player("spike", pool, computed) != "Workhorse"


def test_gate_beats_promotion_for_short_sample_dominator():
    # The spike's RATE dominance is real (67% of the room's rates at
    # 2.07x), so promotion alone would hand him Workhorse from Lead
    # 1A. The gate runs after promotion and caps him anyway: with 2
    # games he is not the room winner.
    pool = {
        "spike": {"pos": "RB", "team": "TST", "name": "Spike Back",
                  "series": _rb_series([1, 2], 26, 5, 52, 55, 12, rz=2)},
        "steady": {"pos": "RB", "team": "TST", "name": "Steady Back",
                   "series": _rb_series(_W, 12, 3, 60, 45, 10, rz=1)},
        "ace1": {"pos": "RB", "team": "AAA", "name": "Ace One",
                 "series": _rb_series(_W, 20, 4, 78, 68, 12, rz=3)},
        "ace2": {"pos": "RB", "team": "BBB", "name": "Ace Two",
                 "series": _rb_series(_W, 18, 4, 72, 62, 11, rz=3)},
        "mid": {"pos": "RB", "team": "CCC", "name": "Mid Back",
                "series": _rb_series(_W, 11, 3, 52, 42, 9, rz=1)},
        "low": {"pos": "RB", "team": "DDD", "name": "Low Back",
                "series": _rb_series(_W, 6, 1, 28, 22, 4, rz=0)},
        "low2": {"pos": "RB", "team": "EEE", "name": "Low Two",
                 "series": _rb_series(_W, 4, 1, 20, 15, 3, rz=0)},
    }
    computed = pr.compute_pool(pool)
    raw = _raw("spike", pool, computed)
    assert raw == "Lead 1A"
    promoted = pr._room_promoted_job("spike", pool, computed, raw)
    assert promoted == "Workhorse"
    assert pr.room_rate_winner("spike", pool, computed) == "steady"
    assert pr.job_for_player("spike", pool, computed) == "Lead 1A"


def test_room_with_no_eligible_member_uses_plain_rate_leader():
    # Early-season window: nobody in the room has 3 games, so the
    # plain rate leader is the winner. He grades Lead 1A, promotes on
    # rate dominance, and the gate lets the promotion stand because
    # he IS the winner.
    pool = {
        "y1": {"pos": "RB", "team": "TST", "name": "Young One",
               "series": _rb_series([1, 2], 20, 4, 70, 60, 14, rz=2)},
        "y2": {"pos": "RB", "team": "TST", "name": "Young Two",
               "series": _rb_series([1, 2], 8, 2, 40, 28, 7, rz=0)},
        "ace1": {"pos": "RB", "team": "AAA", "name": "Ace One",
                 "series": _rb_series(_W, 20, 4, 78, 68, 12, rz=3)},
        "ace2": {"pos": "RB", "team": "BBB", "name": "Ace Two",
                 "series": _rb_series(_W, 18, 4, 72, 62, 11, rz=3)},
        "mid": {"pos": "RB", "team": "CCC", "name": "Mid Back",
                "series": _rb_series(_W, 11, 3, 52, 42, 9, rz=1)},
        "low": {"pos": "RB", "team": "DDD", "name": "Low Back",
                "series": _rb_series(_W, 6, 1, 28, 22, 4, rz=0)},
    }
    computed = pr.compute_pool(pool)
    assert computed["y1"]["feats"]["games"] < pr.ROOM_GATE_MIN_GAMES
    assert pr.room_rate_winner("y1", pool, computed) == "y1"
    raw = _raw("y1", pool, computed)
    assert raw == "Lead 1A"
    assert pr._room_promoted_job("y1", pool, computed, raw) == "Workhorse"
    assert pr.job_for_player("y1", pool, computed) == "Workhorse"


# ---------------------------------------------------------------------------
# Promotion is rate-based: equal totals, lopsided rates
# ---------------------------------------------------------------------------

def test_promotion_by_rate_not_totals():
    # dom and mate finish with IDENTICAL totals (80 opportunities):
    # on the old totals test dom's dominance ratio was 1.0x and he
    # never promoted. His rate is 2.5x mate's, so he promotes now.
    pool = {
        "dom": {"pos": "RB", "team": "TST", "name": "Rate Dominator",
                "series": _rb_series([1, 2, 3, 4], 17, 3, 60, 48, 11, rz=2)},
        "mate": {"pos": "RB", "team": "TST", "name": "Grinder Mate",
                 "series": _rb_series(_W, 7, 1, 35, 24, 5, rz=0)},
        "ace1": {"pos": "RB", "team": "AAA", "name": "Ace One",
                 "series": _rb_series(_W, 20, 4, 78, 68, 12, rz=3)},
        "ace2": {"pos": "RB", "team": "BBB", "name": "Ace Two",
                 "series": _rb_series(_W, 18, 4, 72, 62, 11, rz=3)},
        "mid": {"pos": "RB", "team": "CCC", "name": "Mid Back",
                "series": _rb_series(_W, 11, 3, 52, 42, 9, rz=1)},
        "low": {"pos": "RB", "team": "DDD", "name": "Low Back",
                "series": _rb_series(_W, 6, 1, 28, 22, 4, rz=0)},
        "low2": {"pos": "RB", "team": "EEE", "name": "Low Two",
                 "series": _rb_series(_W, 4, 1, 20, 15, 3, rz=0)},
    }
    computed = pr.compute_pool(pool)
    assert (computed["dom"]["feats"]["opp"]
            == computed["mate"]["feats"]["opp"])
    share, _nxt, ratio = pr.room_dominance("dom", pool, computed)
    assert share >= pr.ROOM_DOM_SHARE_MIN
    assert ratio >= pr.ROOM_DOM_RATIO_MIN
    assert _raw("dom", pool, computed) == "Lead 1A"
    assert pr.job_for_player("dom", pool, computed) == "Workhorse"


# ---------------------------------------------------------------------------
# QB is exempt: a roommate's higher rate cannot cap Franchise QB1
# ---------------------------------------------------------------------------

def test_qb_exempt_from_room_gate():
    def qb_series(weeks, att, car, ppr):
        return [_row(w, snap_pct=100.0, snaps=65, pass_att=att, carries=car,
                     ppr_pts=ppr) for w in weeks]
    # qb1 clears the 90 Franchise bar on the three-leg blend (plays
    # pctile 85.7, start share 100, efficiency pctile 100 -> 93.6);
    # qb2 owns the room's best plays-per-game rate but neither the
    # efficiency (74.3) nor the games (3) for the tier.
    pool = {
        "qb1": {"pos": "QB", "team": "TST", "name": "Starter",
                "series": qb_series(_W, 30, 2, 30.0)},
        "qb2": {"pos": "QB", "team": "TST", "name": "Relief Arm",
                "series": qb_series([1, 2, 3], 38, 4, 22.0)},
        "qb3": {"pos": "QB", "team": "AAA", "name": "Other Starter",
                "series": qb_series(_W, 25, 1, 22.0)},
        "qb4": {"pos": "QB", "team": "BBB", "name": "Fourth Starter",
                "series": qb_series(_W, 20, 0, 18.0)},
        "qb5": {"pos": "QB", "team": "CCC", "name": "Fifth Starter",
                "series": qb_series(_W, 18, 1, 9.0)},
        "qb6": {"pos": "QB", "team": "DDD", "name": "Sixth Starter",
                "series": qb_series(_W, 16, 1, 9.6)},
        "qb7": {"pos": "QB", "team": "EEE", "name": "Seventh Starter",
                "series": qb_series(_W, 14, 0, 9.1)},
        "qb8": {"pos": "QB", "team": "FFF", "name": "Eighth Starter",
                "series": qb_series(_W, 12, 0, 8.4)},
    }
    computed = pr.compute_pool(pool)
    # The relief arm owns the room's best plays-per-game rate...
    assert (computed["qb2"]["feats"]["plays_per_game"]
            > computed["qb1"]["feats"]["plays_per_game"])
    # ...but QB jobs have no room gate (and his 3 games trip the
    # separate Franchise games floor, not any room rule).
    assert pr.job_for_player("qb1", pool, computed) == "Franchise QB1"
    assert pr.job_for_player("qb2", pool, computed) == "QB1"
