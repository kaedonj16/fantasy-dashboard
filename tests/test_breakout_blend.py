"""Board ranking blends the opportunity-weighted aggregate score with the model
hit probability, because the fitted model can't weight opportunity for WRs
(collinear with readiness + too few labeled breakouts), which buried
opportunity-rich players. blend = w*(score/100) + (1-w)*prob.
"""
import pytest

bapi = pytest.importorskip("dashboard_services.breakout_api")


def test_blend_lifts_opportunity_rich_over_pure_probability():
    # Egbuka: high aggregate score (opportunity), low model prob. A pure-probability
    # ranking buries him; the blend lifts him above a low-score/low-prob player.
    egbuka = {"breakout_opportunity_score": 83, "hit_probability": 0.10}
    weak = {"breakout_opportunity_score": 40, "hit_probability": 0.12}
    assert bapi._breakout_blend(egbuka) > bapi._breakout_blend(weak)


def test_blend_is_weighted_average(monkeypatch):
    monkeypatch.setattr(bapi, "BREAKOUT_RANK_BLEND_WEIGHT", 0.5)
    v = bapi._breakout_blend({"breakout_opportunity_score": 80, "hit_probability": 0.20})
    assert v == pytest.approx(0.5 * 0.80 + 0.5 * 0.20, abs=1e-6)  # 0.5


def test_blend_weight_shifts_toward_opportunity(monkeypatch):
    c = {"breakout_opportunity_score": 83, "hit_probability": 0.10}
    monkeypatch.setattr(bapi, "BREAKOUT_RANK_BLEND_WEIGHT", 0.5)
    lo = bapi._breakout_blend(c)
    monkeypatch.setattr(bapi, "BREAKOUT_RANK_BLEND_WEIGHT", 0.8)
    hi = bapi._breakout_blend(c)
    assert hi > lo  # heavier opportunity weight raises a high-score/low-prob player


def test_blend_handles_missing_fields():
    assert bapi._breakout_blend({}) == 0.0
