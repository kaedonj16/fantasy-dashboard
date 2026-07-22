"""Tests for keeper_page data helpers: ADP fallback and draft selection.

keeper_page imports lightly (no pandas/flask at module load), so these run in
the pure unit suite.
"""
import sys
import types

from dashboard_services.pages import keeper_page as kp


def _fake_adp_service(sleeper=None, fc=None):
    m = types.ModuleType("dashboard_services.adp_service")
    m.fetch_sleeper_adp = lambda season: (sleeper or {})
    m.fetch_fc_redraft_adp = lambda is_sf: (fc or {})
    return m


def test_adp_prefers_sleeper_and_picks_right_field(monkeypatch):
    fake = _fake_adp_service(
        sleeper={"1": {"adp_ppr": 3.0, "adp_2qb": 2.0}, "2": {"adp_ppr": 20.0}},
        fc={"9": {"avg_pick": 99.0}},
    )
    monkeypatch.setitem(sys.modules, "dashboard_services.adp_service", fake)
    assert kp._adp_map(is_sf=False, season=2026) == {"1": 3.0, "2": 20.0}   # 1QB -> adp_ppr; FC ignored
    assert kp._adp_map(is_sf=True, season=2026)["1"] == 2.0                 # SF -> adp_2qb


def test_adp_falls_back_to_fc_when_sleeper_empty(monkeypatch):
    fake = _fake_adp_service(sleeper={}, fc={"9": {"avg_pick": 99.0}})
    monkeypatch.setitem(sys.modules, "dashboard_services.adp_service", fake)
    assert kp._adp_map(is_sf=False, season=2026) == {"9": 99.0}


def test_value_rank_ranks_by_value_and_skips_zero():
    vr = kp._value_rank_map({"a": 1000.0, "b": 800.0, "c": 500.0, "d": 0.0})
    assert vr == {"a": 1.0, "b": 2.0, "c": 3.0}   # zero-value player omitted


def test_candidates_use_value_rank_when_adp_missing():
    vals = {"a": 1000.0, "b": 800.0}
    vr = kp._value_rank_map(vals)
    cands = kp._candidates_for_ids(
        ["a", "b"], {"a": {"name": "A", "pos": "WR"}}, vals, adp={}, drafted={}, value_rank=vr,
    )
    assert cands[0].adp_overall == 1.0 and cands[1].adp_overall == 2.0


def test_candidates_prefer_real_adp_over_rank():
    vals = {"a": 1000.0}
    vr = kp._value_rank_map(vals)   # a -> 1
    cands = kp._candidates_for_ids(["a"], {}, vals, adp={"a": 42.0}, drafted={}, value_rank=vr)
    assert cands[0].adp_overall == 42.0   # market ADP wins when present


def test_best_draft_prefers_completed_with_most_rounds():
    drafts = [
        {"draft_id": "rook", "status": "complete", "settings": {"rounds": 3}},
        {"draft_id": "startup", "status": "complete", "settings": {"rounds": 15}},
        {"draft_id": "pre", "status": "pre_draft", "settings": {"rounds": 20}},
    ]
    assert kp._best_draft(drafts)["draft_id"] == "startup"   # completed + most rounds


def test_best_draft_falls_back_when_none_complete():
    drafts = [{"draft_id": "x", "status": "pre_draft", "settings": {"rounds": 12}}]
    assert kp._best_draft(drafts)["draft_id"] == "x"
    assert kp._best_draft([]) is None
