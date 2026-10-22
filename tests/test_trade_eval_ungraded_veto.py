"""POST /api/trade-eval honesty + veto-risk contracts.

- An unknown player pid or a pick the calibrated table doesn't cover for
  its own season is UNVALUED: graded=false, the asset is listed in
  unvalued_assets (the outcome analyzer's convention from PR #2748), and
  no fairness verdict is issued - never a silent 0 feeding a verdict.
- veto_risk reads the side totals build_side actually sets
  (effective_total). The old code read a "total_value" key that never
  exists, so the flag was permanently False; these tests exercise the
  real endpoint on both sides of the threshold
  (abs_diff > max(baseline * 0.30, 150)) so that class of break is
  caught. (The formula-only copy in test_trade_suggestion_realism.py
  could not catch it.)
- Pick pricing inside the calculator and in dashboard_services.trade_hub
  comes from the calibrated pick table, not flat 650/220/80 constants.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


TABLE = [
    {"id": "1", "name": "Alpha WR", "position": "WR", "team": "NYJ",
     "value": 900.0},
    {"id": "2", "name": "Beta RB", "position": "RB", "team": "DEN",
     "value": 300.0},
    {"id": "3", "name": "Gamma WR", "position": "WR", "team": "GB",
     "value": 850.0},
    {"id": "4", "name": "Delta RB", "position": "RB", "team": "KC",
     "value": 400.0},
    {"id": "5", "name": "Echo WR", "position": "WR", "team": "MIA",
     "value": 200.0},
    {"id": "6", "name": "Foxtrot WR", "position": "WR", "team": "BUF",
     "value": 100.0},
    {"id": "7", "name": "Zero RB", "position": "RB", "team": "FA",
     "value": 0.0},
]

PICK_TABLE = {
    "2026_1_01": 685.5,
    "2026_1_12": 213.2,
    "2028_1_mid": 218.5,
    "2028_1_early": 263.0,
    "2028_1_late": 199.9,
}


@pytest.fixture
def env(monkeypatch):
    import app as appmod
    import dashboard_services.player_value_history as pvh

    # api_trade_eval prefers the DB value source; feed it the fixture
    # table so the get_model_value_table_cached fallback is never needed.
    monkeypatch.setattr(
        pvh, "load_current_values_from_db",
        lambda: [dict(r) for r in TABLE],
    )
    monkeypatch.setattr(
        appmod, "load_pick_value_table", lambda **k: dict(PICK_TABLE)
    )
    monkeypatch.setattr(appmod, "_tier_thresholds_cached", lambda *a, **k: [])
    return appmod


@pytest.fixture
def client(env, offline_client):
    return offline_client


def _post(client, **payload):
    res = client.post("/api/trade-eval", json=payload)
    assert res.status_code == 200, res.get_data(as_text=True)
    return res.get_json()


# ---------------------------------------------------------------------------
# Silent-zero: unknown pid / unvalued pick -> UNGRADED, no verdict
# ---------------------------------------------------------------------------

def test_unknown_pid_is_ungraded_not_zero(client):
    data = _post(
        client,
        side_a_players=["999"],
        side_b_players=["2"],
    )
    assert data["graded"] is False
    assert data["unvalued_assets"] == [
        {"id": "999", "name": "Player 999", "side": "a"}
    ]
    # No fairness verdict may be issued off incomplete totals.
    assert "favored" not in data["verdict"]
    assert "looks very fair" not in data["verdict"]
    assert "Can't grade" in data["verdict"]
    assert "Player 999" in data["verdict"]
    # The unvalued asset is flagged per-asset and adds nothing to totals.
    assert data["side_a"]["breakdown"][0]["unvalued"] is True
    assert data["side_a"]["effective_total"] == 0.0
    assert data["side_b"]["effective_total"] == 300.0
    assert data["veto_risk"] is False


def test_zero_priced_player_is_ungraded(client):
    data = _post(
        client,
        side_a_players=["7"],
        side_b_players=["2"],
    )
    assert data["graded"] is False
    assert data["unvalued_assets"][0]["id"] == "7"
    assert data["unvalued_assets"][0]["name"] == "Zero RB"


def test_pick_missing_from_table_is_ungraded(client):
    # 2099 is not covered by the table for its own season, in any form.
    data = _post(
        client,
        side_a_players=["1"],
        side_a_picks=["2099_1_mid"],
        side_b_players=["2"],
    )
    assert data["graded"] is False
    assert {"id": "2099_1_mid", "name": "2099_1_mid", "side": "a"} in \
        data["unvalued_assets"]
    pick_row = [b for b in data["side_a"]["breakdown"]
                if b["type"] == "pick"][0]
    assert pick_row["unvalued"] is True
    # The valued player still totals; the pick contributes nothing.
    # (effective_total may sit below raw via the tier-stack depth
    # adjustment for the 2-for-1 count difference - raw is the check.)
    assert data["side_a"]["raw_total"] == 900.0
    assert data["side_a"]["raw_picks_total"] == 0.0


def test_valued_pick_prices_from_table_and_grades(client):
    data = _post(
        client,
        side_a_players=[],
        side_a_picks=["2026_1_01"],
        side_b_players=["2"],
    )
    assert data["graded"] is True
    assert data["unvalued_assets"] == []
    assert data["side_a"]["raw_picks_total"] == 685.5
    assert data["side_a"]["effective_total"] == 685.5


def test_fully_valued_trade_still_gets_verdict(client):
    data = _post(
        client,
        side_a_players=["1"],
        side_b_players=["3"],
    )
    assert data["graded"] is True
    assert "looks very fair" in data["verdict"]


# ---------------------------------------------------------------------------
# veto_risk through the real endpoint, both sides of the threshold
# ---------------------------------------------------------------------------

def test_veto_risk_true_for_900_vs_300(client):
    # diff 600 > max(900 * 0.30, 150) = 270
    data = _post(client, side_a_players=["1"], side_b_players=["2"])
    assert data["side_a"]["effective_total"] == 900.0
    assert data["side_b"]["effective_total"] == 300.0
    assert data["veto_risk"] is True


def test_veto_risk_false_for_close_deal(client):
    # diff 50, baseline 900 -> threshold 270
    data = _post(client, side_a_players=["1"], side_b_players=["3"])
    assert data["veto_risk"] is False


def test_veto_risk_raw_floor_side_of_threshold(client):
    # 400 vs 200: diff 200 > max(400 * 0.30, 150) = 150 -> True
    data = _post(client, side_a_players=["4"], side_b_players=["5"])
    assert data["veto_risk"] is True
    # 200 vs 100: diff 100, threshold is the 150 raw floor -> False
    data = _post(client, side_a_players=["5"], side_b_players=["6"])
    assert data["veto_risk"] is False


# ---------------------------------------------------------------------------
# Calibrated pick pricing (trade_hub) - no flat 650/220/80
# ---------------------------------------------------------------------------

def test_pick_value_for_uses_calibrated_table():
    from dashboard_services.trade_hub import pick_value_for

    # Early 2026 1st prices at the top of the table range.
    assert pick_value_for("2026_1_01", PICK_TABLE) == 685.5
    # A late 2026 1st prices well under the old flat 650.
    assert pick_value_for("2026_1_12", PICK_TABLE) == 213.2
    assert pick_value_for("2026_1_12", PICK_TABLE) < 650.0 / 2
    # Bucket pick prices from its bucket key, not a round constant.
    assert pick_value_for("2028_1_mid", PICK_TABLE) == 218.5
    # Round-only id: no slot keys exist for 2028, so the hierarchy lands
    # on the round's mid bucket (the round-average step of the hierarchy).
    assert pick_value_for("2028_1", PICK_TABLE) == 218.5


def test_pick_value_for_beyond_table_uses_nearest_season():
    from dashboard_services.trade_hub import pick_value_for

    # 2035 has no table coverage: nearest season (2028) round pricing,
    # well under the old flat 650 for a 1st.
    v = pick_value_for("2035_1_mid", PICK_TABLE)
    assert 0 < v < 400
    assert v != 650.0


def test_pick_value_for_missing_table_is_zero_not_flat():
    from dashboard_services.trade_hub import pick_value_for

    assert pick_value_for("2026_1_01", {}) == 0.0
    assert pick_value_for("not-a-pick", PICK_TABLE) == 0.0


def test_shop_package_prices_return_picks_from_table(monkeypatch):
    import dashboard_services.trade_hub as hub

    # One 2028 mid 1st as the only return asset that can reach the floor:
    # send value 250, floor 212.5; the pick must price at its table value
    # 218.5 (it did not exist as a candidate at all under round>=3 rules,
    # and a flat price would be 650).
    monkeypatch.setattr(hub, "_load_pick_table", lambda is_sf=False: dict(PICK_TABLE))

    def p(pid, name, pos, val):
        return {"id": pid, "name": name, "position": pos, "value": val,
                "team": "FAKE"}

    ctx = {
        "rosters": [
            {"roster_id": "1", "players": ["1"]},
            {"roster_id": "2", "players": ["9"]},
        ],
        "roster_map": {"1": "You", "2": "Team B"},
        "model_value_table": [
            p("1", "Sender RB", "RB", 250),
            p("9", "Cheap WR", "WR", 100),  # below the 150 candidate floor
        ],
        "players_index": {},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        "picks_by_roster": {
            "2": [{"season": 2028, "round": 1, "slot": None}],
        },
    }
    res = hub.shop_package(ctx, viewer_roster_id="1", send_ids=["1"])
    assert res["teams"], "expected Team B to be shoppable via its pick"
    get = res["teams"][0]["suggested_get"]
    pick_rows = [g for g in get if g["position"] == "PICK"]
    assert pick_rows, get
    assert pick_rows[0]["value"] == pytest.approx(218.5, abs=25.0)
    assert pick_rows[0]["value"] != 650.0
