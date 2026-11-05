"""Parity: Front Office potential targets run the shared checks.

_potential_trade_targets historically priced its own packages with
no legality pass: it could pitch a designated keeper as a target,
offer the viewer's own designated keeper as payment, and pitch a
partner's sole starter as a Reach. These tests pin the shared
layers' vetoes (keeper locks + trade_partner_checks.evaluate_package).
"""
import pytest

import dashboard_services.ai.front_office_report as fmod
from dashboard_services.ai.front_office_report import (
    _potential_trade_targets,
)


@pytest.fixture(autouse=True)
def _real_bindings(monkeypatch):
    # Whichever FO harness imports front_office_report first binds
    # its import-time stubs into the module: the espn safe_float
    # stub returns the default for every input (zeroing all values)
    # and utils.roster_strength may be stubbed to the int 0. Pin the
    # real semantics, as the sibling FO harnesses do, and let
    # monkeypatch restore them afterwards.
    monkeypatch.setattr(
        fmod, "safe_float",
        lambda x, default=0.0: default if x is None else float(x))
    monkeypatch.setattr(
        fmod, "STARTER_THRESHOLD",
        {"QB": 500, "RB": 350, "WR": 350, "TE": 200})


def _mk(players):
    """players: {pid: (pos, value)} -> (players_index, value_lookup)."""
    index, lookup = {}, {}
    for pid, (pos, value) in players.items():
        index[pid] = {"position": pos, "full_name": pid.upper(),
                      "team": "BUF", "age": 25}
        lookup[pid] = {"value": float(value), "position": pos,
                       "name": pid.upper(), "age": 25}
    return index, lookup


def _ctx():
    players = {
        # Viewer: one QB (not sendable), deep skill room.
        "vq1": ("QB", 900),
        "vrb1": ("RB", 800), "vrb2": ("RB", 700), "vrb3": ("RB", 650),
        "vrb4": ("RB", 600),
        "vwr1": ("WR", 780), "vwr2": ("WR", 720), "vwr3": ("WR", 680),
        "vwr4": ("WR", 620),
        "vte1": ("TE", 500),
        # Partner 2: a SOLE startable QB.
        "pq1": ("QB", 1000), "pq2": ("QB", 100),
        "prb1": ("RB", 850), "prb2": ("RB", 640),
        "pwr1": ("WR", 810), "pwr2": ("WR", 660), "pte1": ("TE", 480),
        # Partner 3: two startable QBs; the backup is gettable.
        "tq1": ("QB", 950), "tq2": ("QB", 800),
        "trb1": ("RB", 840), "trb2": ("RB", 630),
        "twr1": ("WR", 820), "twr2": ("WR", 670), "tte1": ("TE", 470),
    }
    index, lookup = _mk(players)

    def roster(rid, pids, **kw):
        row = {"roster_id": rid, "players": pids,
               "starters": pids[:6], "settings": {}}
        row.update(kw)
        return row

    ctx = {
        "players_index": index,
        "model_value_table": [],
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE",
                             "FLEX"],
        "rosters": [
            roster(1, ["vq1", "vrb1", "vrb2", "vrb3", "vrb4",
                       "vwr1", "vwr2", "vwr3", "vwr4", "vte1"]),
            roster(2, ["pq1", "pq2", "prb1", "prb2", "pwr1", "pwr2",
                       "pte1"]),
            roster(3, ["tq1", "tq2", "trb1", "trb2", "twr1", "twr2",
                       "tte1"]),
        ],
        "roster_map": {"1": "Viewer", "2": "Sole QB FC",
                       "3": "Two QB FC"},
        "platform": "sleeper", "league_id": "L", "season": 2026,
    }
    return ctx, lookup


def test_sole_starter_is_vetoed_by_the_shared_check():
    ctx, lookup = _ctx()
    out = _potential_trade_targets(ctx, "1", lookup, ["QB"])
    ids = {c["id"] for c in out}
    # Partner 2's only QB would leave him with nobody: the shared
    # check drops the pitch entirely.
    assert "pq1" not in ids
    # Partner 3's backup survives the same check and is annotated.
    assert "tq2" in ids
    kept = next(c for c in out if c["id"] == "tq2")
    assert kept["partner_check"]["viable"] is True
    assert "_pkg" not in kept      # internal package never leaks


def test_designated_keeper_is_never_a_target():
    ctx, lookup = _ctx()
    ctx["rosters"][2]["keepers"] = ["tq2"]   # designated: untradeable
    out = _potential_trade_targets(ctx, "1", lookup, ["QB"])
    ids = {c["id"] for c in out}
    assert "tq2" not in ids
    # The starter in front of him is still a legal (Reach) target.
    assert "tq1" in ids


def test_viewers_designated_keeper_is_never_the_payment():
    ctx, lookup = _ctx()
    open_out = _potential_trade_targets(ctx, "1", lookup, ["QB"])
    open_gap = next(c for c in open_out
                    if c["id"] == "tq2")["gap_pct"]
    # The viewer's cheapest sendable skill piece (vte1, 500) is a
    # designated keeper: he leaves the surplus pool, so the fairest
    # package for tq2 (800) is repriced from [500, 600] (73% gap)
    # to [600] alone (75% gap). The keeper was not spent.
    ctx["rosters"][0]["keepers"] = ["vte1"]
    out = _potential_trade_targets(ctx, "1", lookup, ["QB"])
    kept = next(c for c in out if c["id"] == "tq2")
    assert kept["partner_check"]["viable"] is True
    assert open_gap == 73
    assert kept["gap_pct"] == 75
