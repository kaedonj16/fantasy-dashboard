"""Tests for the Wave 2 situation layer (utils/role_state.py).

Spec: Wave 2 section 2.2 (derivation precedence) and section 5 PR 1.
Pure module: no Flask, no network. Fixture players use the exact
Sleeper-shaped fields the global players payload carries
(``depth_chart_order``, ``depth_chart_position``, ``injury_status``,
``status``) - the Sleeper regression is that those fields, already
present in ctx players, are sufficient for role derivation with no
provider change.
"""
from utils.role_state import (
    ROLE_STATES,
    build_role_states,
    get_role_states,
    is_startable,
    role_chip,
)

WEEK = 9


def _player(team, pos, order, status="", name="", **extra):
    row = {
        "team": team,
        "position": pos,
        "depth_chart_order": order,
        "depth_chart_position": pos,
        "injury_status": status,
        "status": "Active" if not status else "Inactive",
        "full_name": name,
    }
    row.update(extra)
    return row


def _ctx(players, projections, prior=None):
    proj_by_week = {WEEK: {"projections": dict(projections)}}
    for wk, mp in (prior or {}).items():
        proj_by_week[wk] = {"projections": dict(mp)}
    return {"players": players, "proj_by_week": proj_by_week,
            "current_week": WEEK}


def test_depth1_qb_out_promotes_qb2_with_reason():
    # The Lock/Brissett case: depth-1 QB on IR, QB2 holds a starter
    # projection -> QB2 is promoted and the reason names the starter.
    players = {
        "qb1": _player("BAL", "QB", 1, "IR", "Lamar Jackson"),
        "qb2": _player("BAL", "QB", 2, "", "Jacoby Brissett"),
        "qb3": _player("BAL", "QB", 3, "", "Third String"),
    }
    states = build_role_states(_ctx(players, {"qb2": 19.5}), WEEK)
    assert states["qb1"]["state"] == "starter_out"
    promoted = states["qb2"]
    assert promoted["state"] == "promoted"
    assert promoted["signal"] == "depth_injury"
    assert "BAL" in promoted["reason"]
    assert "Lamar Jackson" in promoted["reason"]
    assert "IR" in promoted["reason"]
    assert promoted["week"] == WEEK
    # The QB3 behind the promoted QB2 is not promoted.
    assert states["qb3"]["state"] == "unknown"


def test_provider_designation_on_roster_promotes_backup():
    # Fleaflicker capture path: the designation captured on the ctx
    # roster (severity OUT / typeFull Injured Reserve) overlays the
    # global payload, whose own row for the starter is stale-clean.
    players = {
        "qb1": _player("SEA", "QB", 1, "", "Sam Darnold"),
        "qb2": _player("SEA", "QB", 2, "Questionable", "Drew Lock"),
    }
    ctx = _ctx(players, {"qb2": 20.67})
    ctx["rosters"] = [{
        "roster_id": 1, "players": ["qb1", "qb2"], "starters": ["qb1"],
        "player_designations": {
            "qb1": {"injury_severity": "OUT",
                    "injury_type": "Injured Reserve",
                    "injury_status": "IR",
                    "injury_description": "Thumb"},
        },
    }]
    states = build_role_states(ctx, WEEK)
    assert states["qb1"]["state"] == "starter_out"
    assert "Injured Reserve" in states["qb1"]["reason"]
    assert states["qb2"]["state"] == "promoted"


def test_healthy_demoted_qb_is_benched():
    # Healthy QB whose projection vanished while a teammate projects,
    # and who himself projected as a starter last week: he lost the job.
    players = {
        "qb_old": _player("PHI", "QB", 2, "", "Demoted Passer"),
        "qb_new": _player("PHI", "QB", 1, "", "New Passer"),
    }
    ctx = _ctx(players, {"qb_new": 21.0},
               prior={WEEK - 1: {"qb_old": 20.0, "qb_new": 0.0}})
    states = build_role_states(ctx, WEEK)
    assert states["qb_old"]["state"] == "benched"
    assert "QB2" in states["qb_old"]["reason"]
    assert "PHI" in states["qb_old"]["reason"]
    assert states["qb_new"]["state"] == "starter"


def test_ordinary_starters_are_starters():
    players = {
        "qb1": _player("KC", "QB", 1, "", "Star Passer"),
        "rb1": _player("KC", "RB", 1, "", "Star Runner"),
        "wr1": _player("KC", "WR", 1, "Questionable", "Star Catcher"),
    }
    states = build_role_states(
        _ctx(players, {"qb1": 24.0, "rb1": 16.0, "wr1": 14.0}), WEEK)
    assert states["qb1"]["state"] == "starter"
    assert states["rb1"]["state"] == "starter"
    assert states["wr1"]["state"] == "starter"


def test_no_signal_bench_player_is_unknown():
    # A deep bench player with no projection and no signals is a real
    # "unknown", never a negative state.
    players = {
        "rb1": _player("DAL", "RB", 1, "", "Lead Back"),
        "rb4": _player("DAL", "RB", 4, "", "Deep Bench"),
    }
    states = build_role_states(_ctx(players, {"rb1": 15.0}), WEEK)
    assert states["rb4"]["state"] == "unknown"
    assert states["rb4"]["signal"] == "none"


def test_returning_after_absence_run():
    players = {
        "wr1": _player("MIA", "WR", 1, "", "Comeback Catcher"),
    }
    ctx = _ctx(players, {"wr1": 13.5},
               prior={WEEK - 2: {"wr1": 0.0}, WEEK - 1: {"wr1": 0.0}})
    states = build_role_states(ctx, WEEK)
    assert states["wr1"]["state"] == "returning"


def test_bye_week_is_not_an_absence_or_a_state():
    players = {
        "wr1": _player("MIA", "WR", 1, "", "Bye Catcher", bye_week=WEEK),
    }
    ctx = _ctx(players, {}, prior={WEEK - 1: {"wr1": 12.0}})
    states = build_role_states(ctx, WEEK)
    assert states["wr1"]["state"] == "unknown"


def test_states_vocabulary_and_shape():
    players = {"rb1": _player("DAL", "RB", 1, "", "Lead Back")}
    states = build_role_states(_ctx(players, {"rb1": 15.0}), WEEK)
    assert set(ROLE_STATES) == {
        "starter", "promoted", "starter_out", "benched", "returning",
        "unknown",
    }
    entry = states["rb1"]
    assert set(entry) == {"state", "reason", "signal", "week"}
    assert entry["state"] in ROLE_STATES


def test_is_startable_definition():
    # Role + projection, both required (spec section 2.2).
    assert is_startable({"state": "promoted"}, 20.67, "QB")
    assert is_startable({"state": "starter"}, 15.0, "QB")
    assert is_startable({"state": "returning"}, 10.0, "RB")
    # The Rattler anti-example shape: healthy backup, no promotion
    # signal, no starter projection -> not startable.
    assert not is_startable({"state": "unknown"}, 20.67, "QB")
    assert not is_startable({"state": "promoted"}, 9.0, "QB")
    assert not is_startable({"state": "starter_out"}, 25.0, "QB")
    assert not is_startable({"state": "benched"}, 25.0, "QB")
    assert not is_startable({"state": "starter"}, None, "QB")
    assert not is_startable(None, 25.0, "QB")
    # String form is accepted for simple callers.
    assert is_startable("promoted", 20.67, "QB")


def test_role_chip_only_for_display_states():
    chip = role_chip({"state": "promoted", "reason": "Starting for BAL"})
    assert chip == {"label": "Starting role", "kind": "promoted",
                    "note": "Starting for BAL"}
    assert role_chip({"state": "benched", "reason": "Benched"})["label"] \
        == "Benched"
    assert role_chip({"state": "returning", "reason": "Back"})["label"] \
        == "Returning"
    assert role_chip({"state": "starter", "reason": "x"}) is None
    assert role_chip({"state": "unknown", "reason": ""}) is None
    assert role_chip(None) is None


def test_get_role_states_memoizes_on_ctx_serve_time_only():
    players = {"rb1": _player("DAL", "RB", 1, "", "Lead Back")}
    ctx = _ctx(players, {"rb1": 15.0})
    assert "_role_states" not in ctx
    first = get_role_states(ctx, WEEK)
    # The memo is a serve-time overlay on the ctx dict, keyed by week.
    assert ctx["_role_states"]["week"] == WEEK
    assert ctx["_role_states"]["states"] is first
    assert get_role_states(ctx, WEEK) is first
    # A different week recomputes.
    other = get_role_states(ctx, WEEK + 1)
    assert other is not first
    # build_role_states itself never writes the memo (pure).
    ctx2 = _ctx(players, {"rb1": 15.0})
    build_role_states(ctx2, WEEK)
    assert "_role_states" not in ctx2


def test_missing_data_fails_soft():
    assert build_role_states({}, WEEK) == {}
    assert build_role_states(None, WEEK) == {}
    states = build_role_states({"players": {"x": {"team": "KC"}}}, WEEK)
    assert states["x"]["state"] == "unknown"
