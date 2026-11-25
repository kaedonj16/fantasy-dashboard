"""Plan-builder tests: real diff logic over a fake ESPN transport.

Roster shapes mirror espn_api's mRoster parsing; start-score bundles are
crafted so each player's blended score equals his proj_pts (proj == recent
== season == last4 makes every factor neutral), which keeps the expected
optimal lineup computable by hand. The score engine itself is the real
utils.start_sit_score.compute_start_score.
"""
import copy

import pytest

# The plan builder's dependency chain (espn_api, start-score bundles)
# pulls pandas; the pure-Python lint shard does not install it. This
# guard also auto-marks the module integration (tests/conftest.py), so
# it runs in the full-stack CI job.
pytest.importorskip("pandas")

from dashboard_services import autopilot_plan as plan_mod
from dashboard_services import autopilot_store as store

SWID = "{11111111-2222-3333-4444-555555555555}"
ACCOUNT = 42
LEAGUE = "887776065"
SEASON = 2026

# espn pid -> (canon pid, position)
CANON = {
    "101": "q1", "102": "rb1", "103": "rb2", "104": "wr1", "105": "wr2",
    "106": "te1", "107": "wr3", "108": "k1", "-16007": "KC",
    "110": "rb3", "111": "wr4", "112": "qb2", "113": "rb4", "114": "te2",
    "115": "wr5", "116": "wr6",
}
POS = {
    "q1": "QB", "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB",
    "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR", "wr5": "WR",
    "wr6": "WR", "te1": "TE", "te2": "TE", "k1": "K", "KC": "DEF",
    "qb2": "QB",
}
POS_ID = {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "K": 5, "DEF": 16}
ELIGIBLE = {
    "QB": [0, 20], "RB": [2, 23, 20], "WR": [4, 23, 20], "TE": [6, 23, 20],
    "K": [17, 20], "DEF": [16, 20],
}
SCORES = {
    "q1": 20, "rb1": 8, "rb2": 15, "wr1": 14, "wr2": 13, "te1": 10,
    "wr3": 12, "k1": 8, "KC": 7, "rb3": 18, "wr4": 9, "qb2": 18,
    "rb4": 5, "te2": 6, "wr5": 11,
}
# espn pid -> slot
BASE_SLOTS = {
    101: 0, 102: 2, 103: 2, 104: 4, 105: 4, 106: 6, 107: 23, 108: 17,
    -16007: 16, 110: 20, 111: 20, 112: 20, 113: 20, 114: 20, 115: 20,
    116: 21,
}
SLOT_COUNTS = {"0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "17": 1, "16": 1,
               "20": 6, "21": 1}


def make_payload(slots=None, *, locked=(), injuries=None):
    slots = dict(BASE_SLOTS if slots is None else slots)
    injuries = injuries or {}
    entries = []
    for pid, slot in slots.items():
        canon = CANON[str(pid)]
        pos = POS[canon]
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
                    "fullName": f"Player {canon}",
                    "eligibleSlots": ELIGIBLE[pos],
                    "defaultPositionId": POS_ID[pos],
                    "injuryStatus": injuries.get(pid, "ACTIVE"),
                    "proTeamId": 1,
                },
            },
        })
    return {
        "id": int(LEAGUE),
        "scoringPeriodId": 13,
        "teams": [{
            "id": 2, "name": "Cook and Ball Torture", "owners": [SWID],
            "roster": {"entries": entries},
        }],
        "settings": {
            "scoringSettings": {"scoringItems": []},
            "rosterSettings": {"lineupSlotCounts": SLOT_COUNTS},
        },
    }


def make_bundles(scores=None, *, statuses=None):
    scores = dict(SCORES if scores is None else scores)
    statuses = statuses or {}
    out = {}
    for canon, score in scores.items():
        out[canon] = {
            "pos": POS[canon], "proj_pts": float(score),
            "recent_ppg": float(score), "season_ppg": float(score),
            "last4_ppg": float(score), "played_games": 8,
            "on_bye": False, "injury_status": statuses.get(canon),
            "team": "KC", "opponent": "DEN",
        }
    return out


class FakeTransport:
    def __init__(self, payload):
        self.payload = payload
        self.posts = []

    def get_league(self, season, league_id, params):
        return copy.deepcopy(self.payload)

    def post_transaction(self, season, league_id, payload):
        self.posts.append(payload)
        return 200, {"status": "EXECUTED", "id": "txn"}


@pytest.fixture
def harness(monkeypatch):
    h = {
        "settings": dict(store.DEFAULT_SETTINGS, mode="approve"),
        "saved_plans": [],
        "audit": [],
        "bundles": make_bundles(),
        "kill": False,
    }
    monkeypatch.setattr(
        "dashboard_services.accounts.get_espn_league_credentials",
        lambda *a: {"espn_s2": "s2", "swid": SWID})
    monkeypatch.setattr(
        "dashboard_services.accounts.get_any_espn_account_credentials",
        lambda *a: {"espn_s2": "s2", "swid": SWID})
    monkeypatch.setattr(
        "dashboard_services.providers.espn_api._espn_to_canon_cached",
        lambda: {})
    monkeypatch.setattr(
        "dashboard_services.providers.espn_api.resolve_espn_player_id",
        lambda pid, m, player=None: CANON.get(str(pid)))
    monkeypatch.setattr(
        "data_building.start_score_bundle.load_start_score_bundles_for_scoring",
        lambda season, week, scoring, pids: {
            p: h["bundles"][p] for p in pids if p in h["bundles"]})
    monkeypatch.setattr(plan_mod, "_def_rank", lambda *a: (None, 32))
    monkeypatch.setattr(store, "get_settings", lambda *a: dict(h["settings"]))
    monkeypatch.setattr(store, "get_kill_switch", lambda *a: h["kill"])

    def fake_save_plan(account_id, platform, league_id, season, plan):
        saved = dict(plan, id=100 + len(h["saved_plans"]), status="open",
                     account_id=account_id, platform=platform,
                     league_id=league_id, season=season)
        h["saved_plans"].append(saved)
        return saved

    monkeypatch.setattr(store, "save_plan", fake_save_plan)
    monkeypatch.setattr(
        store, "record_audit",
        lambda *a, **k: h["audit"].append((a, k)))
    return h


def build(h, payload):
    return plan_mod.build_plan(
        ACCOUNT, "espn", LEAGUE, SEASON, transport=FakeTransport(payload))


def test_diff_yields_expected_swap(harness):
    out = build(harness, make_payload())
    assert out["status"] == "planned"
    plan = out["plan"]
    assert plan["scoring_period"] == 13
    assert len(plan["swaps"]) == 1
    swap = plan["swaps"][0]
    assert [e["player_id"] for e in swap["in"]] == [110]   # rb3 in
    assert [e["player_id"] for e in swap["out"]] == [102]  # rb1 out
    assert swap["delta"] == pytest.approx(10.0)
    assert swap["in"][0]["score"] == pytest.approx(18.0)
    items = {(i["player_id"], i["from_slot"], i["to_slot"])
             for i in plan["lineup_items"]}
    assert items == {(110, 20, 2), (102, 2, 20)}
    # The plan was persisted and audited as planned.
    assert harness["saved_plans"] and harness["saved_plans"][0]["id"] == plan["id"]
    assert any(a[4] == "planned" for a, _k in [(x[0], x[1]) for x in harness["audit"]])


def test_locked_player_is_never_moved(harness):
    # rb2 locked AND weak: without the pin the optimizer would bench him
    # for rb3. He must not appear in any item.
    harness["bundles"] = make_bundles({**SCORES, "rb2": 3})
    out = build(harness, make_payload(locked=(103,)))
    plan = out["plan"]
    assert all(i["player_id"] != 103 for i in plan["lineup_items"])
    assert plan["snapshot"][103] == 2


def test_protected_starter_is_never_benched(harness):
    harness["settings"]["protected_player_ids"] = [102]  # weak RB starter
    out = build(harness, make_payload())
    plan = out["plan"]
    assert all(i["player_id"] != 102 for i in plan["lineup_items"])
    # The improvement still happens through the other RB slot.
    assert any(i["player_id"] == 110 for i in plan["lineup_items"])


def test_problem_slots_only_keeps_only_the_fix(harness):
    harness["settings"]["problem_slots_only"] = True
    harness["bundles"] = make_bundles(statuses={"wr2": "Out"})
    out = build(harness, make_payload(injuries={105: "OUT"}))
    plan = out["plan"]
    # Only the injured-WR fix survives; the +10 RB upgrade is dropped.
    assert len(plan["swaps"]) == 1
    swap = plan["swaps"][0]
    assert [e["player_id"] for e in swap["in"]] == [115]
    assert [e["player_id"] for e in swap["out"]] == [105]
    assert any(i["kind"] == "injury" for i in plan["issues"])


def test_problem_slots_off_keeps_both_upgrades(harness):
    harness["bundles"] = make_bundles(statuses={"wr2": "Out"})
    out = build(harness, make_payload(injuries={105: "OUT"}))
    assert len(out["plan"]["swaps"]) == 2


def test_mode_off_plans_nothing(harness):
    harness["settings"]["mode"] = "off"

    class Exploding:
        def get_league(self, *a):
            raise AssertionError("transport must not be touched when off")

    out = plan_mod.build_plan(ACCOUNT, "espn", LEAGUE, SEASON,
                              transport=Exploding())
    assert out["status"] == "off"
    assert harness["saved_plans"] == []


def test_already_optimal_is_no_moves(harness):
    harness["bundles"] = make_bundles({**SCORES, "rb3": 4, "wr5": 4, "qb2": 4})
    out = build(harness, make_payload())
    assert out["status"] == "no_moves"
    assert out["plan"]["swaps"] == []
