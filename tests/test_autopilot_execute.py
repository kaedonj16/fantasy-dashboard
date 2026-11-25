"""Execution-service tests over fakes: guardrails re-checked live.

The fake transport holds mutable roster state exactly like the probe's
fake: POSTs apply LINEUP items to the payload, so verify-by-re-read sees
real post-write state. Store and gate are in-memory fakes.
"""
import copy

import pytest

from dashboard_services import autopilot_execute as exec_mod
from dashboard_services import autopilot_preview, autopilot_store as store

SWID = "{11111111-2222-3333-4444-555555555555}"
ACCOUNT = 42
LEAGUE = "887776065"
SEASON = 2026


def make_payload(slots, *, locked=()):
    entries = []
    for pid, slot in slots.items():
        entries.append({
            "playerId": pid,
            "lineupSlotId": slot,
            "lineupLocked": pid in locked,
            "playerPoolEntry": {
                "id": pid,
                "lineupSlotId": slot,
                "lineupLocked": pid in locked,
                "player": {
                    "id": pid,
                    "fullName": f"Player {pid}",
                    "eligibleSlots": [2, 4, 23, 20],
                    "defaultPositionId": 2,
                    "injuryStatus": "ACTIVE",
                    "proTeamId": 1,
                },
            },
        })
    return {
        "id": int(LEAGUE),
        "scoringPeriodId": 13,
        "teams": [{
            "id": 2, "name": "Team", "owners": [SWID],
            "roster": {"entries": entries},
        }],
        "settings": {},
    }


def swap_group(in_pid, in_from, in_to, out_pid, delta=10.0):
    items = [
        {"player_id": in_pid, "from_slot": in_from, "to_slot": in_to},
        {"player_id": out_pid, "from_slot": in_to, "to_slot": in_from},
    ]
    return {
        "items": items,
        "in": [{"player_id": in_pid, "name": f"Player {in_pid}",
                "from_slot": in_from, "to_slot": in_to, "score": 18.0}],
        "out": [{"player_id": out_pid, "name": f"Player {out_pid}",
                 "from_slot": in_to, "to_slot": in_from, "score": 8.0}],
        "delta": delta,
    }


BASE_SLOTS = {1: 2, 2: 20, 3: 23, 4: 20}
GROUP_A = swap_group(2, 20, 2, 1)          # bench 2 starts at RB, 1 benches
GROUP_B = swap_group(4, 20, 23, 3)         # bench 4 takes FLEX from 3


class FakeTransport:
    def __init__(self, payload, *, reject=None, apply=True):
        self.payload = payload
        self.reject = reject
        self.apply = apply
        self.posts = []
        self.gets = 0

    def get_league(self, season, league_id, params):
        self.gets += 1
        return copy.deepcopy(self.payload)

    def post_transaction(self, season, league_id, payload):
        self.posts.append(payload)
        if self.reject is not None:
            return self.reject
        if self.apply:
            team = self.payload["teams"][0]
            by_pid = {e["playerId"]: e for e in team["roster"]["entries"]}
            for item in payload["items"]:
                entry = by_pid[item["playerId"]]
                entry["lineupSlotId"] = item["toLineupSlotId"]
                entry["playerPoolEntry"]["lineupSlotId"] = item["toLineupSlotId"]
        return 200, {"status": "EXECUTED", "id": "txn"}


@pytest.fixture
def harness(monkeypatch):
    h = {
        "settings": dict(store.DEFAULT_SETTINGS, mode="approve"),
        "kill": False,
        "access": True,
        "plan": None,
        "plan_status": {},
        "audit": [],
        "paused_calls": [],
    }

    def make_plan(groups, snapshot=None, mode="approve"):
        h["plan"] = {
            "id": 7, "account_id": ACCOUNT, "platform": "espn",
            "league_id": LEAGUE, "season": SEASON, "team_id": 2,
            "scoring_period": 13, "mode": mode, "status": "open",
            "snapshot": dict(snapshot or BASE_SLOTS),
            "swaps": groups, "lineup_items": [], "issues": [],
        }
        return h["plan"]

    h["make_plan"] = make_plan
    monkeypatch.setattr(
        "dashboard_services.accounts.get_espn_league_credentials",
        lambda *a: {"espn_s2": "s2", "swid": SWID})
    monkeypatch.setattr(
        "dashboard_services.accounts.get_any_espn_account_credentials",
        lambda *a: {"espn_s2": "s2", "swid": SWID})
    monkeypatch.setattr(autopilot_preview, "autopilot_access",
                        lambda *a: h["access"])
    monkeypatch.setattr(store, "get_settings", lambda *a: dict(h["settings"]))
    monkeypatch.setattr(store, "get_kill_switch", lambda *a: h["kill"])
    monkeypatch.setattr(store, "get_plan", lambda pid: h["plan"])
    monkeypatch.setattr(
        store, "set_plan_status",
        lambda pid, status: h["plan_status"].__setitem__(pid, status))
    monkeypatch.setattr(
        store, "set_paused",
        lambda *a: h["paused_calls"].append(a[-1]))
    monkeypatch.setattr(
        store, "record_audit", lambda *a, **k: h["audit"].append((a, k)))
    return h


def run(h, transport, **kw):
    return exec_mod.execute_plan(
        ACCOUNT, "espn", LEAGUE, SEASON, 7, transport=transport, **kw)


def audit_actions(h):
    return [a[4] for a, _k in [(x[0], x[1]) for x in h["audit"]]]


def test_approve_executes_accepted_and_skips_rest(harness):
    harness["make_plan"]([GROUP_A, GROUP_B])
    transport = FakeTransport(make_payload(BASE_SLOTS))
    out = run(harness, transport, accepted_groups=[0])
    assert out["status"] == "executed"
    assert len(transport.posts) == 1
    posted = {(i["playerId"], i["toLineupSlotId"])
              for i in transport.posts[0]["items"]}
    assert posted == {(2, 2), (1, 20)}  # only group A
    actions = audit_actions(harness)
    assert "approved" in actions and "executed" in actions
    assert "skipped" in actions  # group B audited as skipped
    assert harness["plan_status"][7] == "executed"


def test_auto_mode_executes_whole_plan(harness):
    harness["settings"]["mode"] = "auto"
    harness["make_plan"]([GROUP_A, GROUP_B], mode="auto")
    transport = FakeTransport(make_payload(BASE_SLOTS))
    out = run(harness, transport)
    assert out["status"] == "executed"
    posted = {(i["playerId"], i["toLineupSlotId"])
              for i in transport.posts[0]["items"]}
    assert posted == {(2, 2), (1, 20), (4, 23), (3, 20)}


def test_snapshot_mismatch_supersedes_without_writing(harness):
    harness["make_plan"]([GROUP_A])
    changed = dict(BASE_SLOTS)
    changed[3] = 20  # someone already moved the FLEX player manually
    changed[4] = 23
    transport = FakeTransport(make_payload(changed))
    out = run(harness, transport, accepted_groups=[0])
    assert out["status"] == "superseded"
    assert transport.posts == []
    assert harness["plan_status"][7] == "superseded"


def test_player_locked_between_plan_and_execute(harness):
    harness["make_plan"]([GROUP_A])
    transport = FakeTransport(make_payload(BASE_SLOTS, locked=(2,)))
    out = run(harness, transport, accepted_groups=[0])
    assert out["status"] == "failed"
    assert transport.posts == []
    assert "failed" in audit_actions(harness)


def test_espn_rejection_stops_with_verbatim_detail(harness):
    harness["make_plan"]([GROUP_A])
    body = {"status": "FAILED",
            "messages": ["TRAN_LINEUP_LOCKED: lineup is locked"]}
    transport = FakeTransport(make_payload(BASE_SLOTS), reject=(409, body))
    out = run(harness, transport, accepted_groups=[0])
    assert out["status"] == "rejected"
    assert "TRAN_LINEUP_LOCKED" in out["detail"]
    assert len(transport.posts) == 1  # stopped, no retry
    assert harness["plan_status"][7] == "failed"


def test_kill_switch_blocks_before_any_read_or_write(harness):
    harness["kill"] = True
    harness["make_plan"]([GROUP_A])
    transport = FakeTransport(make_payload(BASE_SLOTS))
    out = run(harness, transport, accepted_groups=[0])
    assert out["status"] == "blocked_kill_switch"
    assert transport.gets == 0 and transport.posts == []


def test_paused_league_blocks(harness):
    harness["settings"]["paused"] = True
    harness["make_plan"]([GROUP_A])
    transport = FakeTransport(make_payload(BASE_SLOTS))
    out = run(harness, transport, accepted_groups=[0])
    assert out["status"] == "blocked_paused"
    assert transport.posts == []


def test_access_gate_refuses(harness):
    harness["access"] = False
    harness["make_plan"]([GROUP_A])
    transport = FakeTransport(make_payload(BASE_SLOTS))
    out = run(harness, transport, accepted_groups=[0])
    assert out["status"] == "refused"
    assert transport.gets == 0 and transport.posts == []


def test_rerun_is_idempotent_noop(harness):
    harness["make_plan"]([GROUP_A])
    final = dict(BASE_SLOTS)
    final[1], final[2] = 20, 2  # plan already reflected in the roster
    transport = FakeTransport(make_payload(final))
    out = run(harness, transport, accepted_groups=[0])
    assert out["status"] == "already_applied"
    assert transport.posts == []
    assert harness["plan_status"][7] == "already_applied"


def test_verification_mismatch_pauses_league(harness):
    harness["make_plan"]([GROUP_A])
    transport = FakeTransport(make_payload(BASE_SLOTS), apply=False)
    out = run(harness, transport, accepted_groups=[0])
    assert out["status"] == "verification_failed"
    assert harness["paused_calls"] == [True]
    assert "failed" in audit_actions(harness)
