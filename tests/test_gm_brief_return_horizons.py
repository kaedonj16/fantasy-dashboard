"""Frontier 5 in the GM brief: return horizons as weeks of coverage.

A fill-in claim is worth the weeks until the starter ahead of him
returns; the horizon comes only from a real ESPN return date
(dashboard_services.injury_return). No date -> no coverage block
and no horizon, never a per-class guess.
"""
import pytest

from dashboard_services import gm_brief, injury_return


@pytest.fixture(autouse=True)
def _no_fetch(monkeypatch):
    """weeks_out reads only the module cache in these tests."""
    monkeypatch.setattr(
        injury_return, "refresh_espn_return_dates", lambda **kw: {})


def _ctx():
    players = {
        "lamar": {"full_name": "Lamar Jackson", "team": "BAL",
                  "position": "QB", "depth_chart_order": 1,
                  "injury_status": "IR", "status": "Inactive"},
        "lock": {"full_name": "Drew Lock", "team": "BAL",
                 "position": "QB", "depth_chart_order": 2,
                 "injury_status": "", "status": "Active"},
        "cmc": {"full_name": "Christian McCaffrey", "team": "SF",
                "position": "RB", "depth_chart_order": 1,
                "injury_status": "IR", "status": "Inactive"},
        "wr1": {"full_name": "Healthy Wideout", "team": "SF",
                "position": "WR", "depth_chart_order": 1,
                "injury_status": "", "status": "Active"},
    }
    return {
        "players": players,
        "rosters": [{
            "roster_id": 1, "players": ["cmc", "wr1"],
            "starters": ["wr1"], "settings": {"wins": 3, "losses": 5},
        }],
        "current_week": 9,
        "league": {"settings": {}},
    }


def _dates(monkeypatch, mapping):
    monkeypatch.setattr(injury_return, "weeks_out_for_player",
                        lambda pid, today=None: mapping.get(pid))


def test_claim_coverage_names_the_starter_and_his_horizon(monkeypatch):
    _dates(monkeypatch, {"lamar": 3.2})
    actions = [{"kind": "waiver", "player_id": "lock",
                "action": "add_lock", "why": "QB2 hole"}]
    claims = gm_brief.build_claims(_ctx(), 1, actions, rules={})
    coverage = claims[0].get("coverage")
    assert coverage is not None
    assert coverage["for_name"] == "Lamar Jackson"
    assert coverage["weeks_out"] == 3.2
    assert coverage["return_week"] == 13     # Week 9 + ceil(3.2)
    assert coverage["source"] == "espn_return_date"


def test_claim_without_a_date_gets_no_coverage(monkeypatch):
    _dates(monkeypatch, {})
    actions = [{"kind": "waiver", "player_id": "lock",
                "action": "add_lock", "why": "QB2 hole"}]
    claims = gm_brief.build_claims(_ctx(), 1, actions, rules={})
    assert "coverage" not in claims[0]


def test_return_horizons_ledger_counts_the_dateless(monkeypatch):
    _dates(monkeypatch, {})   # CMC designated, no ESPN date
    out = gm_brief.build_return_horizons(_ctx(), 1)
    assert out == {"dated": [], "without_date": 1}
    _dates(monkeypatch, {"cmc": 1.1})
    out = gm_brief.build_return_horizons(_ctx(), 1)
    assert out["without_date"] == 0
    assert out["dated"][0]["name"] == "Christian McCaffrey"
    assert out["dated"][0]["return_week"] == 11


def test_return_horizons_none_when_nobody_is_designated(monkeypatch):
    _dates(monkeypatch, {"cmc": 2.0})
    ctx = _ctx()
    ctx["players"]["cmc"]["injury_status"] = ""
    assert gm_brief.build_return_horizons(ctx, 1) is None
