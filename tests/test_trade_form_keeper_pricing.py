"""Wave 2 PR 4: form pricing (check 4) + keeper blend / Steal guard
(check 7).

- Jones fixture: a QB4-depth, negative-form focus player is priced at
  his discounted focus value — no starter-price package is generated.
- Kyren fixture: in a keeper league, sending a projected-keeper RB for
  a rental QB with no keeper value cannot grade "steal" and carries
  the keeper-core note; the same values in redraft can still steal.
- A designated keeper sent for a return that also carries keeper
  value grades normally.
"""
import pytest

pytest.importorskip("pandas")

from dashboard_services.keeper_core import keeper_core
from dashboard_services.trade_acquire_packages import (
    form_adjusted_focus_value,
    value_matched_acquire_packages,
)

# --- Check 4: form pricing -----------------------------------------------------

def test_form_adjustment_discounts_buried_benched_slumping():
    adj, info = form_adjusted_focus_value(
        1000.0, depth_rank=4, role_state="benched", form_swing=-0.40)
    assert info["factor"] < 0.70
    assert adj < 700.0
    assert "depth_rank_4" in info["reasons"]
    assert "benched" in info["reasons"]
    assert "form_slump" in info["reasons"]


def test_form_adjustment_prices_promoted_up():
    adj, info = form_adjusted_focus_value(500.0, role_state="promoted")
    assert adj > 500.0
    assert "promoted" in info["reasons"]


def test_form_adjustment_neutral_without_signals():
    adj, info = form_adjusted_focus_value(800.0, depth_rank=1,
                                          role_state="starter",
                                          form_swing=0.05)
    assert adj == 800.0
    assert info["reasons"] == []


def _viewer_assets():
    players = [
        {"player_id": "v-wr1", "name": "WR1", "position": "WR", "value": 950},
        {"player_id": "v-wr2", "name": "WR2", "position": "WR", "value": 700},
        {"player_id": "v-rb1", "name": "RB1", "position": "RB", "value": 620},
        {"player_id": "v-rb2", "name": "RB2", "position": "RB", "value": 430},
        {"player_id": "v-te1", "name": "TE1", "position": "TE", "value": 300},
    ]
    return players, []


def test_jones_qb4_packages_priced_at_discount():
    players, picks = _viewer_assets()
    # Daniel Jones shape: starter-priced at 1000 on season value, but
    # QB4 on his own roster with a -40% last-4-weeks form swing.
    pkgs = value_matched_acquire_packages(
        1000.0, players, picks, max_options=12, min_results=3,
        focus_meta={"depth_rank": 4, "role_state": None,
                    "form_swing": -0.40})
    assert pkgs, "discounted packages should still be generated"
    adjusted = pkgs[0]["focus_value_adjusted"]
    assert adjusted < 800.0  # 1000 * 0.80 (depth) * 0.85 (form) = 680
    # No starter-price package: nothing is priced against the raw 1000
    # (a fair package at raw price would total ~1000+).
    for pkg in pkgs:
        assert pkg["send_value"] <= 1000.0 * 0.95, pkg
    # Control: without the form meta, the same roster DOES generate a
    # starter-price package near the raw value.
    raw_pkgs = value_matched_acquire_packages(
        1000.0, players, picks, max_options=12, min_results=3)
    assert any(p["send_value"] >= 950.0 for p in raw_pkgs)


# --- Check 7: keeper core --------------------------------------------------------

def _row(pid, name, pos, value):
    return {"name": name, "position": pos, "value": float(value)}


def _keeper_ctx():
    """5-keeper league (92916 shape), values scaled to the 2-roster
    threshold scale the handlers derive. The viewer's RB (Kyren) is a
    projected keeper but NOT designated; his designated five are
    teammates. Burrow's roster has five stronger keeper candidates,
    so Burrow himself carries no keeper value."""
    viewer = {"roster_id": "1", "players": [
        "kyren", "cmc", "higgins", "jsn", "flowers", "waddle", "qb-mine"],
        "keepers": ["cmc", "higgins", "jsn", "flowers", "waddle"]}
    partner = {"roster_id": "2", "players": [
        "burrow", "qb2", "qb3", "p-star1", "p-wr2", "p-wr3", "p-star2",
        "p-rb2", "p-star4", "p-te2"]}
    values = {
        "kyren": _row("kyren", "Kyren Williams", "RB", 3050),
        "cmc": _row("cmc", "Christian McCaffrey", "RB", 3650),
        "higgins": _row("higgins", "Tee Higgins", "WR", 3550),
        "jsn": _row("jsn", "Jaxon Smith-Njigba", "WR", 2500),
        "flowers": _row("flowers", "Zay Flowers", "WR", 600),
        "waddle": _row("waddle", "Jaylen Waddle", "WR", 1300),
        "qb-mine": _row("qb-mine", "My QB", "QB", 2400),
        "burrow": _row("burrow", "Joe Burrow", "QB", 3400),
        "qb2": _row("qb2", "Partner QB2", "QB", 2700),
        "qb3": _row("qb3", "Partner QB3", "QB", 2500),
        "p-star1": _row("p-star1", "Star WR", "WR", 3600),
        "p-wr2": _row("p-wr2", "Partner WR2", "WR", 3520),
        "p-wr3": _row("p-wr3", "Partner WR3", "WR", 1500),
        "p-star2": _row("p-star2", "Star RB", "RB", 3500),
        "p-rb2": _row("p-rb2", "Partner RB2", "RB", 2400),
        "p-star4": _row("p-star4", "Star TE", "TE", 3480),
        "p-te2": _row("p-te2", "Partner TE2", "TE", 3450),
    }
    ctx = {
        "platform": "fleaflicker", "league_id": "92916", "season": 2026,
        "rosters": [viewer, partner],
        "roster_positions": ["QB", "SUPER_FLEX", "RB", "RB", "WR", "WR",
                             "TE"],
        "league_settings": {"type": 2, "max_keepers": 5},
        "_league_rules": {"source": "user_confirmed", "max_keepers": 5,
                          "keepers_known": True},
    }
    return ctx, [viewer, partner], values


def test_kyren_is_core_by_projection_not_designation():
    ctx, rosters, values = _keeper_ctx()
    core = keeper_core(ctx, rosters, values, is_redraft=False,
                       league_size=12)
    assert core["active"] is True
    assert "kyren" not in core["designated_ids"]
    assert "kyren" in core["projected_ids"]
    assert "kyren" in core["core_ids"]
    # Burrow is only his roster's 6th-best keeper candidate at limit 5:
    # no keeper value comes back.
    assert "burrow" not in core["core_ids"]


def test_keeper_core_inactive_in_redraft():
    ctx, rosters, values = _keeper_ctx()
    core = keeper_core(ctx, rosters, values, is_redraft=True,
                       league_size=12)
    assert core["active"] is False
    assert core["core_ids"] == set()


# --- Check 7: the Steal guard at the grade stamp -------------------------------

import app as app_mod  # noqa: E402


def _packages_call(monkeypatch, ctx, values_table, focus_id,
                   primary_packages):
    """Drive api_trade_intel_player_packages with the ML/trade-DB
    layers stubbed so primary_packages flow through _enrich_pkg and
    the PR 3 gate exactly as production packages do."""
    monkeypatch.setattr(app_mod, "has_premium_for_viewer",
                        lambda *a, **k: True)
    monkeypatch.setattr(app_mod, "get_league_ctx_from_cache",
                        lambda *a, **k: ctx)
    monkeypatch.setattr(app_mod, "get_model_value_table_cached",
                        lambda *a, **k: values_table)
    import data_building.trade_intel.trade_pattern_model as tpm
    monkeypatch.setattr(tpm, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(
        app_mod, "_real_trade_packages_for_target",
        lambda **k: {"packages": primary_packages,
                     "total_real_trades": 40, "sig_counts": {},
                     "velocity": {}, "buyer_split": {}})
    client = app_mod.app.test_client()
    return client.get(
        f"/api/trade-intel/player-packages/{focus_id}"
        "?season=2026&platform=fleaflicker&league_id=92916"
        "&viewer_roster_id=1&league_type=sf")


def _value_table(values):
    return [
        {"id": pid, "name": info["name"], "position": info["position"],
         "value": info["value"], "sf_value": info["value"],
         "age": 25, "team": "BUF", "pos_rank_label": "X1"}
        for pid, info in values.items()
    ]


def _asset(pid, name, pos, value):
    return {"player_id": pid, "id": pid, "name": name, "position": pos,
            "value": float(value), "send_value": float(value),
            "sf_value": float(value), "team": "BUF",
            "pos_rank_label": "", "is_pick": False}


def _kyren_for_burrow_pkg():
    # Kyren (3050, projected keeper — NOT designated, so tradeable)
    # for Burrow (3400): -10.3% on raw value — a "steal" that
    # surrenders keeper core for a rental with no keeper value.
    return {"send": [_asset("kyren", "Kyren Williams", "RB", 3050.0)],
            "sig": [], "trades_like_this": 3}


def test_kyren_steal_suppressed_in_keeper_league(monkeypatch):
    ctx, rosters, values = _keeper_ctx()
    resp = _packages_call(monkeypatch, ctx, _value_table(values),
                          "burrow", [_kyren_for_burrow_pkg()])
    assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
    pkgs = resp.get_json().get("real_packages") or []
    assert pkgs, "the package should still be offered, just not a steal"
    graded = pkgs[0]
    assert graded["value_grade"] != "steal"
    assert graded.get("grade_note") == "Costs keeper core"


def test_same_values_in_redraft_can_still_steal(monkeypatch):
    ctx, rosters, values = _keeper_ctx()
    ctx = dict(ctx)
    ctx["league_settings"] = {"type": 0}  # redraft
    ctx["_league_rules"] = {"source": "feed"}
    ctx["rosters"] = [dict(r) for r in ctx["rosters"]]
    for r in ctx["rosters"]:
        r.pop("keepers", None)
    resp = _packages_call(monkeypatch, ctx, _value_table(values),
                          "burrow", [_kyren_for_burrow_pkg()])
    assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
    pkgs = resp.get_json().get("real_packages") or []
    assert pkgs
    assert pkgs[0]["value_grade"] == "steal"
    assert not pkgs[0].get("grade_note")


def test_core_send_with_keeper_value_return_grades_normally(monkeypatch):
    ctx, rosters, values = _keeper_ctx()
    # Send projected-keeper Kyren (3050) for the partner's star WR
    # (3600, himself a projected keeper): the ratio is a steal
    # (-15.3%), but the return carries keeper value, so grading runs
    # normally — the steal stands, with no keeper-core note.
    pkg = {"send": [_asset("kyren", "Kyren Williams", "RB", 3050.0)],
           "sig": [], "trades_like_this": 3}
    resp = _packages_call(monkeypatch, ctx, _value_table(values),
                          "p-star1", [pkg])
    assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
    pkgs = resp.get_json().get("real_packages") or []
    assert pkgs
    assert pkgs[0]["value_grade"] == "steal"
    assert not pkgs[0].get("grade_note")
