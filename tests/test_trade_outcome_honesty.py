"""Check Outcome honesty contracts for POST /api/trade-outcome.

- A player/pick with no current price is UNVALUED (value_now null,
  graded=false, verdict UNGRADED) - never a silent 0 and never an
  invented pick price.
- When every player has a trade-date value and the trade has no picks,
  the verdict grades the since-trade CHANGE (net_change), not the
  current net.
- The optional basis (is_sf / league_size / te_premium) selects the
  value column the calling surface displayed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


TABLE = [
    {"id": "1", "name": "Alpha WR", "position": "WR", "team": "NYJ",
     "value": 400.0, "sf_value": 450.0, "value_12": 410.0,
     "sf_value_12": 460.0},
    {"id": "2", "name": "Beta RB", "position": "RB", "team": "DEN",
     "value": 350.0, "sf_value": 320.0},
    {"id": "3", "name": "Gamma TE", "position": "TE", "team": "GB",
     "value": 200.0, "sf_value": 210.0},
]

# Trade-date (2026-06-19) values for the since-trade verdict test.
HISTORY = {
    "1": [{"as_of_date": "2026-06-19", "value": 500.0}],
    "2": [{"as_of_date": "2026-06-19", "value": 100.0}],
}


@pytest.fixture
def env(monkeypatch):
    import app as appmod
    import dashboard_services.picks as picks_mod
    import dashboard_services.player_value_history as pvh
    import dashboard_services.trade_time_values as ttv

    monkeypatch.setattr(
        appmod, "get_model_value_table_cached", lambda: [dict(r) for r in TABLE]
    )
    pick_table = {"2027_2_mid": 120.0}
    monkeypatch.setattr(picks_mod, "load_pick_value_table", lambda: pick_table)
    monkeypatch.setattr(ttv, "get_or_persist_trade_value_meta",
                        lambda pid, trade_date: None)
    monkeypatch.setattr(
        pvh, "get_player_value_history",
        lambda pid, days=800: list(HISTORY.get(str(pid), [])),
    )
    return appmod


@pytest.fixture
def client(env):
    from app import app as flask_app
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


def _post(client, **payload):
    res = client.post("/api/trade-outcome", json=payload)
    assert res.status_code == 200, res.get_data(as_text=True)
    return res.get_json()


def test_unvalued_player_is_ungraded_not_zero(client):
    data = _post(
        client,
        assets_received=[{"id": "999", "name": "Ghost Player"}],
        assets_sent=[{"id": "2", "name": "Beta RB"}],
        trade_date="",
    )
    assert data["graded"] is False
    assert data["verdict"] == "UNGRADED"
    assert data["received"][0]["value_now"] is None
    assert data["unvalued_assets"] == [
        {"id": "999", "name": "Ghost Player", "side": "received"}
    ]
    assert data["total_received_now"] == 0.0
    assert data["total_sent_now"] == 350.0
    assert data["verdict_basis"] == "current"


def test_verdict_follows_since_trade_change_not_current_net(client):
    # Now: received 400 vs sent 350 -> current net +50 (old logic: WIN).
    # Then: received 500 vs sent 100 -> net was +400, so the trade has
    # LOST 350 since it happened and the verdict must say so.
    data = _post(
        client,
        assets_received=[{"id": "1", "name": "Alpha WR"}],
        assets_sent=[{"id": "2", "name": "Beta RB"}],
        trade_date="2026-06-19",
    )
    assert data["graded"] is True
    assert data["then_complete"] is True
    assert data["verdict_basis"] == "since_trade"
    assert data["net_change"] == -350.0
    assert data["net_delta_now"] == 50.0
    assert data["verdict"] == "LOSS"


def test_sf_basis_and_te_premium_price_from_sf_columns(client):
    data = _post(
        client,
        assets_received=[{"id": "1", "name": "Alpha WR"}],
        assets_sent=[{"id": "2", "name": "Beta RB"}],
        trade_date="",
        is_sf=True,
    )
    assert data["received"][0]["value_now"] == 450.0
    assert data["sent"][0]["value_now"] == 320.0

    data = _post(
        client,
        assets_received=[{"id": "3", "name": "Gamma TE"}],
        assets_sent=[{"id": "2", "name": "Beta RB"}],
        trade_date="",
        is_sf=True,
        te_premium=1.0,
    )
    # SF value 210 with a full TE premium (+20%).
    assert data["received"][0]["value_now"] == 252.0


def test_pick_without_table_entry_is_unvalued_not_invented(client):
    data = _post(
        client,
        assets_received=[{
            "id": "2099 3.XX", "name": "2099 3rd", "asset_type": "pick",
            "pick_season": 2099, "pick_round": 3,
        }],
        assets_sent=[{"id": "2", "name": "Beta RB"}],
        trade_date="",
    )
    assert data["picks_in_trade"] is True
    pick_row = data["received"][0]
    assert pick_row["value_now"] is None  # never the old invented 10.0
    assert data["graded"] is False
    assert data["verdict"] == "UNGRADED"
    for row in data["received"] + data["sent"]:
        assert row["value_now"] != 10.0


def test_pick_with_table_entry_prices_and_blocks_then_complete(client):
    data = _post(
        client,
        assets_received=[{
            "id": "2027 2.XX", "name": "2027 2nd", "asset_type": "pick",
            "pick_season": 2027, "pick_round": 2,
        }],
        assets_sent=[{"id": "2", "name": "Beta RB"}],
        trade_date="2026-06-19",
    )
    assert data["received"][0]["value_now"] == 120.0
    assert data["graded"] is True
    # Picks have no trade-date value, so no since-trade verdict.
    assert data["then_complete"] is False
    assert data["verdict_basis"] == "current"
    assert data["net_change"] is None
