"""Frontier 3: keeper-cost pricing (utils/keeper_cost.py).

Costs come from the keeper page's own projection machinery; the
trade layer only reads them. No draft history / non-keeper format
-> no costs and no block, never an invented round.
"""
import pytest

from utils.keeper_cost import keeper_block_for_package, keeper_costs


@pytest.fixture()
def costs_ctx(monkeypatch):
    import dashboard_services.pages.keeper_page as kp

    def fake_compute(ctx, platform="sleeper", league_id="",
                     viewer_roster_id=None, **kw):
        return {"kept": [
            {"id": "harvey", "name": "RJ Harvey", "pos": "RB",
             "rosterId": "1", "costRound": 6, "adpOverall": 24,
             "projected": True},
            {"id": "kyren", "name": "Kyren Williams", "pos": "RB",
             "rosterId": "2", "costRound": 1, "adpOverall": 12,
             "projected": True},
            {"id": "plain", "name": "No Cost", "pos": "WR",
             "rosterId": "2", "costRound": None, "adpOverall": 60,
             "projected": True},
        ]}

    monkeypatch.setattr(kp, "compute_league_keepers", fake_compute)
    ctx = {"total_rosters": 12, "rosters": [{"roster_id": i}
                                            for i in range(12)],
           "league_type": "keeper", "platform": "fleaflicker",
           "league_id": "92916"}
    return ctx


def test_keeper_costs_reads_the_projection(costs_ctx):
    costs = keeper_costs(costs_ctx)
    assert costs["harvey"]["cost_round"] == 6
    assert costs["kyren"]["cost_round"] == 1
    assert "plain" not in costs          # no cost round, no entry
    # Memoized on the ctx.
    assert keeper_costs(costs_ctx) is costs


def test_block_prices_cheap_keeper_surplus(costs_ctx):
    block = keeper_block_for_package(
        costs_ctx,
        send=[{"id": "harvey", "value": 500.0}],
        receive=[{"id": "kyren", "value": 520.0}])
    assert block is not None
    # Harvey kept at Round 6 with a Round 2 ADP: real surplus.
    assert block["send"][0]["cost_round"] == 6
    assert block["send_surplus"] > 0
    # Kyren at a Round 1 cost carries almost no surplus.
    assert block["receive_surplus"] < block["send_surplus"]
    assert block["net_surplus_to_partner"] > 0
    assert any("Round 6" in n for n in block["notes"])


def test_block_absent_when_no_priced_keeper_moves(costs_ctx):
    assert keeper_block_for_package(
        costs_ctx, send=[{"id": "nobody", "value": 1.0}],
        receive=[]) is None


def test_redraft_league_has_no_costs():
    ctx = {"league_type": "redraft", "rosters": []}
    assert keeper_costs(ctx) == {}
    assert keeper_block_for_package(
        ctx, send=[{"id": "x"}], receive=[]) is None
