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


def test_partial_season_claim_not_inflated():
    # Regression: a 4-game player (targets projected to a full 17-game season
    # upstream) must not out-claim a full-season higher-usage player. Real rates:
    # McMillan 3.75/g (4 games), Egbuka 7.47/g (17 games) -> Egbuka claims more.
    mcmillan = {"last_season_targets": 64, "last_season_games": 4}   # 15 proj-> ~64
    egbuka = {"last_season_targets": 127, "last_season_games": 17}
    c_mcmillan = comp._competitor_claim("WR", mcmillan)
    c_egbuka = comp._competitor_claim("WR", egbuka)
    assert c_egbuka > c_mcmillan          # higher real usage claims more
    assert c_mcmillan < 5.0               # ~3.76/g, not the buggy 16/g


def test_partial_season_share_flows_to_higher_usage():
    incumbents = {("TB", "WR"): [
        {"player_id": "mcmillan", "last_season_targets": 64, "last_season_games": 4},
        {"player_id": "egbuka", "last_season_targets": 127, "last_season_games": 17},
    ]}
    s_egbuka, _ = comp._opportunity_share("egbuka", "TB", "WR", incumbents, {})
    s_mcmillan, _ = comp._opportunity_share("mcmillan", "TB", "WR", incumbents, {})
    assert s_egbuka > s_mcmillan  # vacated targets flow to the real volume earner


def test_readiness_discounts_small_sample_efficiency():
    # Identical elite efficiency, but a 4-game sample must score lower than a full
    # season — the projected-volume confidence used to let a hot 4 games count at
    # full weight (McMillan 92 readiness off 4 games). Games-based cap fixes it.
    meta = {"age": 24, "years_exp": 2}
    hot4 = {"yards_per_target": 11.0, "catch_rate": 0.72, "targets": 64, "games": 4}
    full17 = {"yards_per_target": 11.0, "catch_rate": 0.72, "targets": 127, "games": 17}
    s4, d4 = comp.calculate_player_readiness_score("x", "WR", 2025, meta, hot4)
    s17, d17 = comp.calculate_player_readiness_score("y", "WR", 2025, meta, full17)
    assert s4 < s17
    assert d4["efficiency_sample_multiplier"] < 1.0   # small sample discounted
    assert d4["skill_lift"] == 0.0                    # skill lift gated on real games
    assert d17["efficiency_sample_multiplier"] == 1.0  # full season unaffected


def test_opportunity_share_full_when_no_usage():
    # Nobody has prior usage -> don't dilute; scored player gets the full share.
    incumbents = {("IND", "WR"): [{"player_id": "rook", "last_season_targets": 0,
                                   "last_season_games": 0}]}
    share, _ = comp._opportunity_share("rook", "IND", "WR", incumbents, {})
    assert share == 1.0
