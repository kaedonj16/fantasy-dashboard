"""Autopilot store tests against an in-memory fake connection.

The fake dispatches on the store's SQL surface (psycopg3 style:
conn.execute returns a cursor). It models just enough Postgres behavior
(upserts, supersede-on-insert, append-only audit) to pin the store's
contract without a live database.
"""
import json

import pytest

from dashboard_services import autopilot_store as store

ACCOUNT = 42
LEAGUE = "887776065"
OTHER_LEAGUE = "2129982245"
SEASON = 2026


class FakeCursor:
    def __init__(self, one=None, all_=None):
        self._one = one
        self._all = all_ or []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._all)


class FakeConn:
    def __init__(self):
        self.settings = {}
        self.account_state = {}
        self.plans = {}
        self.audit = []
        self._next_plan_id = 1
        self.commits = 0

    def commit(self):
        self.commits += 1

    def execute(self, sql, params=None):
        sql_n = " ".join(str(sql).split())
        params = params or ()
        if sql_n.startswith(("CREATE TABLE", "CREATE INDEX")):
            return FakeCursor()
        if "INSERT INTO autopilot_settings" in sql_n:
            (account_id, platform, league_id, season, mode, lineups,
             problem_only, push, email, protected, paused) = params
            self.settings[(account_id, platform, league_id, season)] = {
                "account_id": account_id, "platform": platform,
                "league_id": league_id, "season": season, "mode": mode,
                "lineups_enabled": lineups, "problem_slots_only": problem_only,
                "notify_push": push, "notify_email": email,
                "protected_player_ids": json.loads(protected),
                "paused": paused, "waivers_enabled": False,
                "faab_cap_per_claim": None, "faab_cap_per_week": None,
            }
            return FakeCursor()
        if "SELECT * FROM autopilot_settings" in sql_n:
            row = self.settings.get(tuple(params))
            return FakeCursor(one=dict(row) if row else None)
        if "SELECT kill_switch FROM autopilot_account_state" in sql_n:
            on = self.account_state.get(params[0])
            return FakeCursor(one={"kill_switch": on} if on is not None else None)
        if "INSERT INTO autopilot_account_state" in sql_n:
            self.account_state[params[0]] = params[1]
            return FakeCursor()
        if "UPDATE autopilot_plans SET status = 'superseded'" in sql_n:
            for plan in self.plans.values():
                if (plan["account_id"], plan["platform"], plan["league_id"],
                        plan["season"]) == tuple(params) and plan["status"] == "open":
                    plan["status"] = "superseded"
            return FakeCursor()
        if "INSERT INTO autopilot_plans" in sql_n:
            (account_id, platform, league_id, season, team_id, period, mode,
             snapshot, snap_hash, swaps, items, issues) = params
            pid = self._next_plan_id
            self._next_plan_id += 1
            row = {
                "id": pid, "account_id": account_id, "platform": platform,
                "league_id": league_id, "season": season, "team_id": team_id,
                "scoring_period": period, "mode": mode, "status": "open",
                "snapshot": json.loads(snapshot), "snapshot_hash": snap_hash,
                "swaps": json.loads(swaps), "lineup_items": json.loads(items),
                "issues": json.loads(issues),
            }
            self.plans[pid] = row
            return FakeCursor(one=dict(row))
        if "SELECT * FROM autopilot_plans WHERE id = %s" in sql_n:
            row = self.plans.get(params[0])
            return FakeCursor(one=dict(row) if row else None)
        if "SELECT * FROM autopilot_plans" in sql_n:
            open_ = [p for p in self.plans.values()
                     if (p["account_id"], p["platform"], p["league_id"],
                         p["season"]) == tuple(params[:4]) and p["status"] == "open"]
            row = max(open_, key=lambda p: p["id"]) if open_ else None
            return FakeCursor(one=dict(row) if row else None)
        if "UPDATE autopilot_plans SET status = %s" in sql_n:
            self.plans[params[1]]["status"] = params[0]
            return FakeCursor()
        if "INSERT INTO autopilot_audit" in sql_n:
            keys = ("account_id", "platform", "league_id", "season", "plan_id",
                    "scoring_period", "action", "mode", "player_id",
                    "player_name", "from_slot", "to_slot", "score_delta",
                    "detail", "verification", "espn_status")
            row = dict(zip(keys, params))
            row["id"] = len(self.audit) + 1
            self.audit.append(row)
            return FakeCursor()
        if "SELECT * FROM autopilot_audit" in sql_n:
            rows = [r for r in self.audit
                    if (r["account_id"], r["platform"], r["league_id"],
                        r["season"]) == tuple(params[:4])]
            rows.sort(key=lambda r: -r["id"])
            return FakeCursor(all_=rows[:params[4]])
        raise AssertionError(f"Unexpected SQL: {sql_n}")


class _Ctx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_db(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(store, "_conn", lambda: _Ctx(conn))
    monkeypatch.setattr(store, "_TABLES_READY", False)
    return conn


def test_settings_defaults_when_unset(fake_db):
    s = store.get_settings(ACCOUNT, "espn", LEAGUE, SEASON)
    assert s["mode"] == "off"
    assert s["notify_push"] is True and s["notify_email"] is True
    assert s["protected_player_ids"] == []
    assert s["paused"] is False


def test_settings_round_trip_and_league_independence(fake_db):
    store.save_settings(ACCOUNT, "espn", LEAGUE, SEASON, {
        "mode": "approve", "protected_player_ids": [101, 205, 99],
    })
    store.save_settings(ACCOUNT, "espn", OTHER_LEAGUE, SEASON, {"mode": "auto"})
    a = store.get_settings(ACCOUNT, "espn", LEAGUE, SEASON)
    b = store.get_settings(ACCOUNT, "espn", OTHER_LEAGUE, SEASON)
    assert a["mode"] == "approve"
    assert a["protected_player_ids"] == [101, 205, 99]  # order preserved
    assert b["mode"] == "auto"
    assert b["protected_player_ids"] == []


def test_mode_validation_and_phase15_refusal(fake_db):
    with pytest.raises(ValueError):
        store.save_settings(ACCOUNT, "espn", LEAGUE, SEASON, {"mode": "turbo"})
    with pytest.raises(ValueError):
        store.save_settings(ACCOUNT, "espn", LEAGUE, SEASON, {"faab_cap_per_week": 50})
    with pytest.raises(ValueError):
        store.save_settings(ACCOUNT, "espn", LEAGUE, SEASON, {"waivers_enabled": True})


def test_kill_switch_defaults_off(fake_db):
    assert store.get_kill_switch(ACCOUNT) is False
    assert store.set_kill_switch(ACCOUNT, True) is True
    assert store.get_kill_switch(ACCOUNT) is True
    assert store.get_kill_switch(999) is False


def test_new_plan_supersedes_open_plan(fake_db):
    p1 = store.save_plan(ACCOUNT, "espn", LEAGUE, SEASON, {
        "team_id": 2, "scoring_period": 13, "mode": "approve",
        "snapshot": {1: 2, 2: 20}, "swaps": [], "lineup_items": [], "issues": [],
    })
    p2 = store.save_plan(ACCOUNT, "espn", LEAGUE, SEASON, {
        "team_id": 2, "scoring_period": 13, "mode": "approve",
        "snapshot": {1: 2, 2: 20}, "swaps": [], "lineup_items": [], "issues": [],
    })
    assert store.get_plan(p1["id"])["status"] == "superseded"
    assert store.get_open_plan(ACCOUNT, "espn", LEAGUE, SEASON)["id"] == p2["id"]
    assert p1["snapshot_hash"] == p2["snapshot_hash"]


def test_audit_is_append_only_and_ordered(fake_db):
    store.record_audit(ACCOUNT, "espn", LEAGUE, SEASON, "planned", player_id=1)
    store.record_audit(ACCOUNT, "espn", LEAGUE, SEASON, "executed", player_id=1,
                       verification="verified")
    rows = store.list_audit(ACCOUNT, "espn", LEAGUE, SEASON)
    assert [r["action"] for r in rows] == ["executed", "planned"]
    assert not hasattr(store, "update_audit") and not hasattr(store, "delete_audit")
    with pytest.raises(ValueError):
        store.record_audit(ACCOUNT, "espn", LEAGUE, SEASON, "edited")
