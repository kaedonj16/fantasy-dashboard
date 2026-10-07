"""Need-aware distribute: the return package should favor the viewer's thinnest slots.

Distribute used to pick each owner's return combo purely on value closeness
(|package - stud|), so it could happily return three WRs to a team whose real
hole was at RB. _collect_owner_bests now scores combos with a need-adjusted
value: a combo at her weakest slots gets up to a 30% discount against one at
her strengths, so need wins ties and near-ties without overturning a clearly
fairer deal.

Pure functions only, so this runs in the base suite (no Flask/pandas).
"""
from dashboard_services.archetype_engine import (
    _build_distribute,
    _viewer_slot_need_weights,
)

_LINEUP = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]


def _v(pid, name, pos, val):
    return {
        "position": pos, "value": float(val), "name": name, "age": 26,
        "redraft_value": float(val), "team": "XX", "pos_rank_label": "",
    }


def _t(pid, name, pos, val):
    return {"player_id": pid, "name": name, "position": pos, "value": float(val)}


def _viewer():
    """Viewer with a clear RB need: weak RB2 behind the stud, strong WR room."""
    vals = {
        "v_stud": _v("v_stud", "Stud RB", "RB", 2000),
        "v_rb2":  _v("v_rb2", "Weak RB2", "RB", 500),
        "v_wr1":  _v("v_wr1", "WR1", "WR", 1500),
        "v_wr2":  _v("v_wr2", "WR2", "WR", 1400),
        "v_te1":  _v("v_te1", "TE1", "TE", 800),
        "v_qb1":  _v("v_qb1", "QB1", "QB", 1000),
    }
    return ["v_stud", "v_rb2", "v_wr1", "v_wr2", "v_te1", "v_qb1"], vals


def _run_distribute(viewer_players, values_by_id, pool, owner="2"):
    return _build_distribute(
        viewer_players, values_by_id, {owner: pool},
        owner_meta={owner: {}}, roster_map={owner: "Partner"},
        league_type="1qb", viewer_lineup_val=6000.0, league_avg=5500.0,
        viewer_pos_counts={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
        roster_positions=_LINEUP,
    )


def test_need_weights_rank_weakest_slot_first():
    players, vals = _viewer()
    w = _viewer_slot_need_weights(players, vals, _LINEUP)
    assert set(w) == {"RB", "WR", "TE"}
    # Weakest starter: RB 500, WR 1400, TE 800 -> RB neediest, WR not needy.
    assert w["RB"] > w["TE"] > w["WR"]
    assert w["RB"] > 0.9
    assert w["WR"] == 0.0


def test_need_weights_all_zero_when_balanced():
    vals = {
        "rb1": _v("rb1", "RB1", "RB", 1000), "rb2": _v("rb2", "RB2", "RB", 1000),
        "wr1": _v("wr1", "WR1", "WR", 1000), "wr2": _v("wr2", "WR2", "WR", 1000),
        "te1": _v("te1", "TE1", "TE", 1000), "qb1": _v("qb1", "QB1", "QB", 1000),
    }
    w = _viewer_slot_need_weights(list(vals), vals, _LINEUP)
    assert all(v == 0.0 for v in w.values())


def test_flex_assignee_counts_toward_its_position():
    # RB room: stud + weak RB2 + a flex-worthy RB3 who lands in the FLEX slot.
    vals = {
        "rb1": _v("rb1", "RB1", "RB", 2000), "rb2": _v("rb2", "RB2", "RB", 500),
        "rb3": _v("rb3", "RB3", "RB", 100),
        "wr1": _v("wr1", "WR1", "WR", 1500), "wr2": _v("wr2", "WR2", "WR", 1400),
        "te1": _v("te1", "TE1", "TE", 800), "qb1": _v("qb1", "QB1", "QB", 1000),
    }
    w = _viewer_slot_need_weights(list(vals), vals, _LINEUP)
    # The FLEX slot goes to RB3 (100), the weakest flex-eligible leftover, so
    # RB's weakest starter is 100, not 500: RB is needier than without flex.
    assert w["RB"] > 0.99


def test_equidistant_combos_need_filler_wins():
    players, vals = _viewer()
    pool = [
        _t("p_rb1", "P RB1", "RB", 1200), _t("p_rb2", "P RB2", "RB", 1100),
        _t("p_wr1", "P WR1", "WR", 1150), _t("p_wr2", "P WR2", "WR", 1150),
    ]
    for p in pool:
        vals[p["player_id"]] = _v(p["player_id"], p["name"], p["position"], p["value"])
    rows = _run_distribute(players, vals, pool)
    stud_rows = [r for r in rows if r["player_id"] == "v_stud"]
    assert stud_rows, "expected a distribute row for the stud"
    recv = {c["player_id"] for c in stud_rows[0]["suggested_receive"]}
    # RB pair: |2300-2000| = 300 -> ~30% need discount -> ~210.
    # WR pair: |2300-2000| = 300, no need -> 300.
    # Mixed:   |2250-2000| = 250 -> ~15% discount -> ~212.5.
    # The RB pair fills her thinnest slot and must win.
    assert recv == {"p_rb1", "p_rb2"}
    assert "Fills your thinnest spot: RB." in stud_rows[0]["why"]


def test_balanced_roster_closest_to_fair_wins():
    vals = {
        "v_stud": _v("v_stud", "Stud RB", "RB", 2000),
        "v_rb2":  _v("v_rb2", "RB2", "RB", 1000),
        "v_wr1":  _v("v_wr1", "WR1", "WR", 1000), "v_wr2": _v("v_wr2", "WR2", "WR", 1000),
        "v_te1":  _v("v_te1", "TE1", "TE", 1000), "v_qb1": _v("v_qb1", "QB1", "QB", 1000),
    }
    players = list(vals)
    pool = [
        _t("p_wr1", "P WR1", "WR", 1100), _t("p_wr2", "P WR2", "WR", 1100),
        _t("p_rb1", "P RB1", "RB", 1000), _t("p_rb2", "P RB2", "RB", 1000),
    ]
    for p in pool:
        vals[p["player_id"]] = _v(p["player_id"], p["name"], p["position"], p["value"])
    rows = _run_distribute(players, vals, pool)
    stud_rows = [r for r in rows if r["player_id"] == "v_stud"]
    assert stud_rows
    recv = {c["player_id"] for c in stud_rows[0]["suggested_receive"]}
    # No positional need anywhere: pure value closeness decides (2000 vs 2200).
    assert recv == {"p_rb1", "p_rb2"}
    assert "thinnest spot" not in stud_rows[0]["why"]


def test_picks_carry_no_need_weight():
    players, vals = _viewer()
    # Pass the pick through picks_by_owner so it flows through the real path.
    pool = [_t("p_rb1", "P RB1", "RB", 1050), _t("p_te1", "P TE1", "TE", 1050),
            _t("p_te2", "P TE2", "TE", 1050)]
    for p in pool:
        vals[p["player_id"]] = _v(p["player_id"], p["name"], p["position"], p["value"])
    rows = _build_distribute(
        players, vals, {"2": pool},
        owner_meta={"2": {}}, roster_map={"2": "Partner"},
        league_type="1qb", viewer_lineup_val=6000.0, league_avg=5500.0,
        viewer_pos_counts={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
        roster_positions=_LINEUP,
        picks_by_owner={"2": [{"player_id": "pk1", "name": "2027 1st",
                               "position": "PICK", "value": 1050.0,
                               "is_pick": True}]},
    )
    stud_rows = [r for r in rows if r["player_id"] == "v_stud"]
    assert stud_rows
    recv = {c["player_id"] for c in stud_rows[0]["suggested_receive"]}
    # [RB, TE]: |2100-2000| = 100, need ~1.67 -> adj ~75.
    # [RB, pick]: |2100-2000| = 100, need ~1.0 -> adj ~85.
    # If the pick wrongly contributed need, [RB, pick] would score ~70 and win.
    # The [RB, TE] win proves the pick adds no need of its own.
    assert recv == {"p_rb1", "p_te1"} or recv == {"p_rb1", "p_te2"}
    assert "Fills your thinnest spot: RB." in stud_rows[0]["why"]


def test_clearly_fairer_deal_not_overturned_by_need():
    players, vals = _viewer()
    pool = [
        _t("p_rb1", "P RB1", "RB", 1300), _t("p_rb2", "P RB2", "RB", 1200),
        _t("p_wr1", "P WR1", "WR", 1010), _t("p_wr2", "P WR2", "WR", 1010),
    ]
    for p in pool:
        vals[p["player_id"]] = _v(p["player_id"], p["name"], p["position"], p["value"])
    rows = _run_distribute(players, vals, pool)
    stud_rows = [r for r in rows if r["player_id"] == "v_stud"]
    assert stud_rows
    recv = {c["player_id"] for c in stud_rows[0]["suggested_receive"]}
    # WR pair: |2020-2000| = 20. RB pair: |2500-2000| = 500 -> adj ~350.
    # The 30% need discount must not overturn a clearly fairer deal.
    assert recv == {"p_wr1", "p_wr2"}
