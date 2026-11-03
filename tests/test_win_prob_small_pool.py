"""Small-pool tail, QB<->catcher covariance, and game-script tests.

The analytic win-probability model (per-position normals, variance =
frac * sigma_full^2 in progress, TEAM_CV_FLOOR) is pinned by
test_win_prob_calibration.py. This file pins the endgame layer added on
top of it:

* With <= 4 pending players in a full-slate matchup, compute_win_prob
  replaces the normal's *shape* with a deterministic Monte Carlo over
  skewed remaining distributions (lognormal / two-point spike), keeping
  the same means and spreads. Kaedon's JSN scenario is the regression
  case: the old double-shrunk late variance (sigma computed on the
  shrunken remaining projection, then scaled by frac again) read a
  one-big-play spot at ~1-6%; the corrected scaling plus the tail shape
  must land it in a plausible band.
* Same-team QB<->WR/TE pending pairs add 2 * rho * sigma_i * sigma_j
  (rho = 0.11, the empirical value from the TEAM_CV_FLOOR derivation).
* margin_lookup applies the small, capped game-script nudge to
  in-progress pass positions only; the default (no lookup) is unchanged.
"""
import pytest

from dashboard_services.matchups import (
    _QB_CATCHER_RHO,
    _script_mean_factor,
    compute_win_prob,
    STATUS_NOT_STARTED,
    STATUS_FINAL,
    STATUS_IN_PROGRESS,
)

_POS9 = ["QB", "RB", "RB", "WR", "WR", "TE", "WR", "K", "DEF"]


def _full_side(prefix, pts_list):
    return {"starters": [
        {"pid": "%s%d" % (prefix, i), "pos": _POS9[i], "pts": float(v)}
        for i, v in enumerate(pts_list)
    ]}


def _jsn_scenario():
    """Kaedon's case, priced by the real code path.

    JSN (WR, SEA) is the only player still playing: in progress with a
    quarter left (frac 0.25), 28 points banked, pregame projection 16.
    The pace blend in live_final_from_frac prices his remaining slice at
    +7 (live final 35, not the 30 of the original hypothetical -- 28 in
    three quarters is a scorching pace and the blend believes it). His
    team is 10 down on banked points, so he needs 11 more: expected to
    finish 3 short, with one big play winning it. Everyone else final.
    """
    left = _full_side("L", [22, 16, 14, 12, 11, 9, 7, 5, 4])  # 100 banked
    right = {"starters": [
        {"pid": "R%d" % i, "pos": _POS9[i], "pts": float(v)}
        for i, v in enumerate([14, 12, 10, 8, 7, 5, 4, 2])  # 62 banked
    ] + [{"pid": "JSN", "pos": "WR", "pts": 28.0, "nfl": "SEA"}]}
    status = {"L%d" % i: STATUS_FINAL for i in range(9)}
    status.update({"R%d" % i: STATUS_FINAL for i in range(8)})
    status["JSN"] = STATUS_IN_PROGRESS
    proj = {"L%d" % i: 12.0 for i in range(9)}
    proj.update({"R%d" % i: 11.0 for i in range(8)})
    proj["JSN"] = 16.0
    return right, left, status, proj


def test_jsn_late_game_tail_band():
    right, left, status, proj = _jsn_scenario()
    p = compute_win_prob(right, left, status, proj,
                         frac_lookup=lambda p_: 0.25)
    # The corrected analytic read is ~0.24 (remaining sigma
    # sqrt(0.25) * 8.36 = 4.18 around a -3 expected margin); the skewed
    # simulation reads ~0.18. The regressed double-shrunk scaling read
    # this at ~0.01-0.06. The band locks the scaling: plausible, not
    # near-zero, not a coin flip.
    assert 0.10 <= p <= 0.35


def test_small_pool_is_deterministic():
    right, left, status, proj = _jsn_scenario()
    p1 = compute_win_prob(right, left, status, proj,
                          frac_lookup=lambda p_: 0.25)
    p2 = compute_win_prob(right, left, status, proj,
                          frac_lookup=lambda p_: 0.25)
    assert p1 == p2


def test_small_pool_def_late_spike():
    # A DEF as the last player alive, down 4 with a quarter left: its
    # remaining slice is clock-only (no pace for DEF), mean +2.25 on a
    # 9.0 pregame projection, sigma sqrt(0.25) * 5.5 = 2.75. One
    # defensive TD wins it; the two-point/lognormal tail must price
    # that well above the old near-zero read but nowhere near a flip.
    left = _full_side("L", [22, 16, 14, 12, 11, 9, 7, 5, 4])  # 100 banked
    right = {"starters": [
        {"pid": "R%d" % i, "pos": _POS9[i], "pts": float(v)}
        for i, v in enumerate([16, 14, 13, 12, 11, 10, 8, 6])  # 90 banked
    ] + [{"pid": "D1", "pos": "DEF", "pts": 6.0, "nfl": "SEA"}]}
    status = {"L%d" % i: STATUS_FINAL for i in range(9)}
    status.update({"R%d" % i: STATUS_FINAL for i in range(8)})
    status["D1"] = STATUS_IN_PROGRESS
    proj = {"L%d" % i: 12.0 for i in range(9)}
    proj.update({"R%d" % i: 11.0 for i in range(8)})
    proj["D1"] = 9.0
    p = compute_win_prob(right, left, status, proj,
                         frac_lookup=lambda p_: 0.25)
    assert 0.05 < p < 0.40
    assert p == compute_win_prob(right, left, status, proj,
                                 frac_lookup=lambda p_: 0.25)


def test_partial_sides_keep_analytic_contract():
    # A one-player-per-side hypothetical is not a full-slate endgame:
    # the analytic result (the sqrt(frac) pin in the calibration file)
    # must be untouched by the simulation layer.
    from math import erf, sqrt
    st = {"1": STATUS_IN_PROGRESS, "2": STATUS_IN_PROGRESS}
    proj = {"1": 20.0, "2": 20.0}
    left = {"starters": [{"pid": "1", "pos": "QB", "pts": 15.0}]}
    right = {"starters": [{"pid": "2", "pos": "QB", "pts": 5.0}]}
    p = compute_win_prob(left, right, st, proj, frac_lookup=lambda p_: 0.5)
    # Pace-blended remaining: left 0.75*10 + 0.25*15 = 11.25 (total
    # 26.25), right 0.75*10 + 0.25*5 = 8.75 (total 13.75). Var per side
    # 0.5 * 7.5^2 = 28.125 (floor (0.2*20*sqrt(.5))^2 = 8 does not bind).
    # Phi(12.5 / sqrt(56.25)).
    expected = 0.5 * (1 + erf((12.5) / (sqrt(56.25) * sqrt(2.0))))
    assert p == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

def _cov_sides(stacked):
    left = {"starters": [
        {"pid": "a1", "pos": "QB", "pts": 0.0, "nfl": "KC"},
        {"pid": "a2", "pos": "WR", "pts": 0.0,
         "nfl": "KC" if stacked else "BUF"},
        {"pid": "a3", "pos": "RB", "pts": 0.0, "nfl": "KC"},
    ]}
    right = {"starters": [
        {"pid": "b1", "pos": "QB", "pts": 0.0, "nfl": "DAL"},
        {"pid": "b2", "pos": "WR", "pts": 0.0, "nfl": "NYJ"},
        {"pid": "b3", "pos": "RB", "pts": 0.0, "nfl": "DAL"},
    ]}
    status = {k: STATUS_NOT_STARTED
              for k in ("a1", "a2", "a3", "b1", "b2", "b3")}
    proj = {"a1": 25.0, "a2": 15.0, "a3": 13.0,
            "b1": 20.0, "b2": 12.0, "b3": 11.0}
    return left, right, status, proj


def test_qb_catcher_covariance_widens_stacked_team():
    # Six pending players: analytic path. The stacked side is favored on
    # projections, so widening its variance pulls its win probability
    # toward 50% relative to the identical unstacked lineup.
    left, right, status, proj = _cov_sides(stacked=True)
    p_stacked = compute_win_prob(left, right, status, proj)
    left, right, status, proj = _cov_sides(stacked=False)
    p_unstacked = compute_win_prob(left, right, status, proj)
    assert 0.5 < p_stacked < p_unstacked
    assert _QB_CATCHER_RHO == pytest.approx(0.11)


def test_covariance_needs_nfl_team_codes():
    # Recap/digest-style starters carry no NFL team: no pairs can form
    # and the 120 vs 110 read stays the pinned floor value.
    from dashboard_services.matchups import TEAM_CV_FLOOR  # noqa: F401
    slots = [("QB", 22), ("RB", 16), ("RB", 13), ("WR", 15), ("WR", 12),
             ("TE", 10), ("WR", 11), ("K", 8), ("DEF", 8)]

    def lineup(prefix, total):
        base = sum(v for _, v in slots)
        starters, pmap = [], {}
        for i, (pos, v) in enumerate(slots):
            pid = "%s%d" % (prefix, i)
            starters.append({"pid": pid, "pos": pos, "pts": 0.0})
            pmap[pid] = v * total / base
        return {"starters": starters}, pmap

    a, pa = lineup("a", 120.0)
    b, pb = lineup("b", 110.0)
    p = compute_win_prob(a, b, {}, {**pa, **pb})
    assert p == pytest.approx(0.6206, abs=0.001)


# ---------------------------------------------------------------------------
# Game script
# ---------------------------------------------------------------------------

def test_script_mean_factor_shape():
    assert _script_mean_factor(None, "WR") == 1.0
    assert _script_mean_factor(5.0, "WR") == 1.0       # inside deadband
    assert _script_mean_factor(-28.0, "WR") > 1.0      # trailing: fed
    assert _script_mean_factor(28.0, "WR") < 1.0       # leading: protected
    assert _script_mean_factor(-28.0, "RB") == 1.0     # pass positions only
    assert _script_mean_factor(-100.0, "QB") == pytest.approx(1.10)  # cap


def _script_prob(margin):
    left = {"starters": [{"pid": "Q1", "pos": "QB", "pts": 10.0, "nfl": "KC"}]}
    right = {"starters": [{"pid": "R1", "pos": "RB", "pts": 25.0}]}
    status = {"Q1": STATUS_IN_PROGRESS, "R1": STATUS_FINAL}
    proj = {"Q1": 20.0, "R1": 15.0}
    ml = (lambda p_: margin) if margin is not None else None
    return compute_win_prob(left, right, status, proj,
                            frac_lookup=lambda p_: 0.5, margin_lookup=ml)


def test_script_moves_prob_in_margin_direction():
    p_trail = _script_prob(-28.0)
    p_none = _script_prob(None)
    p_lead = _script_prob(28.0)
    assert p_trail > p_none > p_lead


def test_script_default_is_prior_behaviour():
    left = {"starters": [{"pid": "Q1", "pos": "QB", "pts": 10.0, "nfl": "KC"}]}
    right = {"starters": [{"pid": "R1", "pos": "RB", "pts": 25.0}]}
    status = {"Q1": STATUS_IN_PROGRESS, "R1": STATUS_FINAL}
    proj = {"Q1": 20.0, "R1": 15.0}
    assert compute_win_prob(left, right, status, proj,
                            frac_lookup=lambda p_: 0.5) == _script_prob(None)
