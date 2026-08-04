"""Two breakout-engine tuning changes:

1. The training breakout target requires BOTH a % jump (growth) AND an absolute
   PPG gain (min_delta), so trivial jumps (8->9.2) don't count as breakouts.
2. Vacated opportunity concentrates on the biggest incumbents (claim ** power)
   instead of splitting ~evenly across the whole room, so a WR2 inherits more of
   a departed WR1 than a deep-bench WR does.
"""
import pytest

T = pytest.importorskip("data_building.breakout_engine.train_hit_probability")
comp = pytest.importorskip("data_building.breakout_engine.components")


def test_breakout_target_requires_growth_and_delta():
    prior = {"a": {"ppr_ppg": 8.0, "games": 16},
             "b": {"ppr_ppg": 8.0, "games": 16},
             "c": {"ppr_ppg": 10.0, "games": 16}}
    out = {"a": {"ppr_ppg": 9.2,  "games": 16},   # +15%, +1.2 -> fails both
           "b": {"ppr_ppg": 11.6, "games": 16},   # +45%, +3.6 -> hit
           "c": {"ppr_ppg": 13.6, "games": 16}}   # +36% (<40%) -> fails growth
    hits = T._breakout_hits(out, prior, growth=1.40, min_ppg=7.0,
                            scratch_ppg=10.0, min_delta=3.5)
    assert hits == {"b"}


def test_breakout_delta_blocks_small_absolute_gain():
    # High baseline, clears the % but not the absolute gain floor.
    prior = {"x": {"ppr_ppg": 6.0, "games": 16}}
    out = {"x": {"ppr_ppg": 8.7, "games": 16}}  # +45% but only +2.7 PPG
    assert T._breakout_hits(out, prior, growth=1.40, min_ppg=7.0,
                            scratch_ppg=10.0, min_delta=3.5) == set()


def test_opportunity_share_concentrates_on_top_claimant():
    incumbents = {("IND", "WR"): (
        [{"player_id": "star", "last_season_targets": 100, "last_season_games": 16},
         {"player_id": "w2", "last_season_targets": 50, "last_season_games": 16}]
        + [{"player_id": f"bench{i}", "last_season_targets": 10, "last_season_games": 16}
           for i in range(6)]
    )}
    share, competitors = comp._opportunity_share("star", "IND", "WR", incumbents, {})
    proportional = 100 / (100 + 50 + 6 * 10)
    assert share > proportional      # top claimant gets MORE than an even split
    assert competitors == 8


def test_bounded_fit_clamps_negative_coef_to_zero():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(0, 1, (n, 6))
    # Opportunity (feat 0) is NEGATIVELY associated with y in the raw data, so an
    # unconstrained fit would give it a negative coefficient; readiness (feat 3)
    # helps. The constraint must clamp opportunity to ~0 and keep readiness positive.
    logit = -1.0 - 0.8 * X[:, 0] + 1.2 * X[:, 3]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    mask = [True, True, True, True, True, False]  # confidence (feat 5) free
    w, _ = T._fit_logit_bounded(X, y, C=0.3, nonneg_mask=mask)
    assert all(c >= -1e-6 for c, m in zip(w, mask) if m)  # no negative constrained coefs
    assert w[0] <= 1e-6   # opportunity clamped (would be negative unconstrained)
    assert w[3] > 0.0     # readiness stays positive

    w_free, _ = T._fit_logit_bounded(X, y, C=0.3, nonneg_mask=[False] * 6)
    assert w_free[0] < 0  # unconstrained really does want it negative


def test_opportunity_share_full_when_no_usage():
    # Nobody has prior usage -> don't dilute; scored player gets the full share.
    incumbents = {("IND", "WR"): [{"player_id": "rook", "last_season_targets": 0,
                                   "last_season_games": 0}]}
    share, _ = comp._opportunity_share("rook", "IND", "WR", incumbents, {})
    assert share == 1.0
