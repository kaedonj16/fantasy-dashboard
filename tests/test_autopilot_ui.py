"""Autopilot Phase 1 UI tests: real page, actions, surfaces, scheduler.

Everything runs on fakes shaped like the backend suites: a FakeTransport
holding mutable per-league payloads (POSTs apply LINEUP items, so
verify-by-re-read sees real post-write state) and the store-test
FakeConn behind the real autopilot_store. autopilot_access is patched
to the pilot matrix (account 42, both usable ESPN leagues); the real
gate has its own suite (test_autopilot_access.py).
"""
import copy
import json
import re

import pytest

pytest.importorskip("flask")
# The plan builder's dependency chain pulls pandas; this guard also
# auto-marks the module integration (tests/conftest.py).
pytest.importorskip("pandas")

from dashboard_services import autopilot_preview, autopilot_store as store

SWID = "{11111111-2222-3333-4444-555555555555}"
ACCOUNT = 42
LEAGUE = "887776065"
OTHER_LEAGUE = "2129982245"
SEASON = 2026
OWNER_EMAIL = "owner@example.com"

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
BASE_SLOTS = {
    101: 0, 102: 2, 103: 2, 104: 4, 105: 4, 106: 6, 107: 23, 108: 17,
    -16007: 16, 110: 20, 111: 20, 112: 20, 113: 20, 114: 20, 115: 20,
    116: 21,
}
SLOT_COUNTS = {"0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "17": 1, "16": 1,
               "20": 6, "21": 1}


def make_payload(league_id=LEAGUE):
    entries = []
    for pid, slot in BASE_SLOTS.items():
        canon = CANON[str(pid)]
        pos = POS[canon]
        entries.append({
            "playerId": pid,
            "lineupSlotId": slot,
            "lineupLocked": False,
            "playerPoolEntry": {
                "id": pid,
                "lineupSlotId": slot,
                "lineupLocked": False,
                "player": {
                    "id": pid,
                    "fullName": f"Player {canon}",
                    "eligibleSlots": ELIGIBLE[pos],
                    "defaultPositionId": POS_ID[pos],
                    "injuryStatus": "ACTIVE",
                    "proTeamId": 1,
                },
            },
        })
    return {
        "id": int(league_id),
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


def make_bundles():
    out = {}
    for canon, score in SCORES.items():
        out[canon] = {
            "pos": POS[canon], "proj_pts": float(score),
            "recent_ppg": float(score), "season_ppg": float(score),
            "last4_ppg": float(score), "played_games": 8,
            "on_bye": False, "injury_status": None,
            "team": "KC", "opponent": "DEN",
        }
    return out


class FakeTransport:
    """One shared mutable state across every constructed instance."""

    state = None  # {"payloads": {league_id: payload}, "posts": [...]}

    def __init__(self, *a, **k):
        pass

    @classmethod
    def reset(cls):
        cls.state = {
            "payloads": {LEAGUE: make_payload(LEAGUE),
                         OTHER_LEAGUE: make_payload(OTHER_LEAGUE)},
            "posts": [],
        }

    def get_league(self, season, league_id, params):
        return copy.deepcopy(self.state["payloads"][str(league_id)])

    def post_transaction(self, season, league_id, payload):
        self.state["posts"].append((str(league_id), payload))
        team = self.state["payloads"][str(league_id)]["teams"][0]
        by_pid = {e["playerId"]: e for e in team["roster"]["entries"]}
        for item in payload["items"]:
            entry = by_pid[item["playerId"]]
            entry["lineupSlotId"] = item["toLineupSlotId"]
            entry["playerPoolEntry"]["lineupSlotId"] = item["toLineupSlotId"]
        return 200, {"status": "EXECUTED", "id": "txn"}


# ── Store fake (mirrors tests/test_autopilot_store.py) ─────────────────

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

    def commit(self):
        pass

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
def env(monkeypatch):
    """Real store over FakeConn + fake ESPN + pilot access matrix."""
    conn = FakeConn()
    monkeypatch.setattr(store, "_conn", lambda: _Ctx(conn))
    monkeypatch.setattr(store, "_TABLES_READY", False)
    FakeTransport.reset()
    monkeypatch.setattr(
        "dashboard_services.providers.espn_writes.EspnTransport",
        FakeTransport)
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
    bundles = make_bundles()
    monkeypatch.setattr(
        "data_building.start_score_bundle.load_start_score_bundles_for_scoring",
        lambda season, week, scoring, pids: {
            p: bundles[p] for p in pids if p in bundles})
    from dashboard_services import autopilot_plan as plan_mod
    monkeypatch.setattr(plan_mod, "_def_rank", lambda *a: (None, 32))
    monkeypatch.setattr(
        autopilot_preview, "autopilot_access",
        lambda account_id, platform, league_id, season: (
            account_id == ACCOUNT and platform == "espn"
            and str(league_id) in (LEAGUE, OTHER_LEAGUE)))
    notes = {"executed": [], "waiting": []}
    from dashboard_services import autopilot_notify
    monkeypatch.setattr(
        autopilot_notify, "notify_execution",
        lambda *a: notes["executed"].append(a))
    monkeypatch.setattr(
        autopilot_notify, "notify_plan_waiting",
        lambda *a: notes["waiting"].append(a))
    return {"conn": conn, "notes": notes}


def _session(client, account_id=ACCOUNT):
    with client.session_transaction() as sess:
        sess["account_id"] = account_id
        sess["account_email"] = OWNER_EMAIL


def _approve_mode(env):
    store.save_settings(ACCOUNT, "espn", LEAGUE, SEASON, {"mode": "approve"})


# ── Gate matrix ────────────────────────────────────────────────────────

def test_page_gate_matrix(offline_client, env):
    _session(offline_client)
    assert offline_client.get(f"/espn/{SEASON}/{LEAGUE}/autopilot").status_code == 200
    assert offline_client.get(f"/espn/{SEASON}/{OTHER_LEAGUE}/autopilot").status_code == 200


def test_page_404_for_non_pilot_and_other_platforms(offline_client, env):
    _session(offline_client, account_id=7)
    assert offline_client.get(f"/espn/{SEASON}/{LEAGUE}/autopilot").status_code == 404
    _session(offline_client)
    assert offline_client.get(f"/sleeper/{SEASON}/{LEAGUE}/autopilot").status_code == 404
    assert offline_client.get("/yahoo/2026/1307110/autopilot").status_code == 404


def test_api_404_for_non_pilot(offline_client, env):
    _session(offline_client, account_id=7)
    resp = offline_client.post(
        f"/api/autopilot/espn/{SEASON}/{LEAGUE}/mode", json={"mode": "auto"})
    assert resp.status_code == 404


# ── Real page content ──────────────────────────────────────────────────

def test_page_renders_real_plan_without_sample_wording(offline_client, env):
    _approve_mode(env)
    _session(offline_client)
    resp = offline_client.get(f"/espn/{SEASON}/{LEAGUE}/autopilot")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # Real fixture plan: rb3 (18) starts over rb1 (8) for +10.0.
    assert "Player rb3" in body
    assert "+10.0" in body
    assert "Cook and Ball Torture" in body
    assert "data-ap-accept" in body
    # Kaedon directive: no sample / preview-mock wording anywhere, and
    # no page header block (the page starts with the hero card).
    assert "sample" not in body.lower()
    assert "preview mock" not in body.lower()
    assert 'class="ap-h1"' not in body


def test_page_off_mode_shows_no_queue(offline_client, env):
    _session(offline_client)
    resp = offline_client.get(f"/espn/{SEASON}/{LEAGUE}/autopilot")
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "Autopilot is off" in body
    assert ">Accept</button>" not in body


# ── Actions ────────────────────────────────────────────────────────────

def test_accept_executes_group_and_audits(offline_client, env):
    _approve_mode(env)
    _session(offline_client)
    offline_client.get(f"/espn/{SEASON}/{LEAGUE}/autopilot")  # builds plan
    plan_id = max(env["conn"].plans)
    resp = offline_client.post(
        f"/api/autopilot/espn/{SEASON}/{LEAGUE}/execute",
        json={"plan_id": plan_id, "groups": [0]})
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["result"]["status"] == "executed"
    assert len(FakeTransport.state["posts"]) == 1
    actions = [r["action"] for r in env["conn"].audit]
    assert "executed" in actions
    assert env["notes"]["executed"], "execution notification not sent"


def test_mode_post_persists_and_surface_mode_reflects_it(offline_client, env, monkeypatch):
    _session(offline_client)
    resp = offline_client.post(
        f"/api/autopilot/espn/{SEASON}/{LEAGUE}/mode", json={"mode": "auto"})
    assert resp.status_code == 200
    assert store.get_settings(ACCOUNT, "espn", LEAGUE, SEASON)["mode"] == "auto"
    # The shared surface helper (portfolio control, hero chip) reads
    # the same stored mode for the signed-in pilot account.
    monkeypatch.setenv("AUTOPILOT_PREVIEW_EMAILS", OWNER_EMAIL)
    import app as appmod
    with appmod.app.test_request_context("/"):
        from flask import session
        session["account_id"] = ACCOUNT
        session["account_email"] = OWNER_EMAIL
        assert autopilot_preview.autopilot_surface_mode(
            "espn", LEAGUE, SEASON) == "auto"
        assert autopilot_preview.autopilot_surface_mode(
            "sleeper", LEAGUE, SEASON) is None


def test_protected_players_editor_persists(offline_client, env):
    _approve_mode(env)
    _session(offline_client)
    resp = offline_client.post(
        f"/api/autopilot/espn/{SEASON}/{LEAGUE}/settings",
        json={"protected_player_ids": [110, 102]})
    assert resp.status_code == 200
    assert store.get_settings(
        ACCOUNT, "espn", LEAGUE, SEASON)["protected_player_ids"] == [110, 102]
    body = offline_client.get(
        f"/espn/{SEASON}/{LEAGUE}/autopilot").get_data(as_text=True)
    assert 'data-ap-protected="[110, 102]"' in body
    assert "Player rb3" in body  # roster renders in the editor


def test_kill_switch_endpoint_blocks_execution(offline_client, env, monkeypatch):
    monkeypatch.setenv("AUTOPILOT_PREVIEW_EMAILS", OWNER_EMAIL)
    _approve_mode(env)
    _session(offline_client)
    offline_client.get(f"/espn/{SEASON}/{LEAGUE}/autopilot")  # builds plan
    plan_id = max(env["conn"].plans)
    resp = offline_client.post(
        "/api/autopilot/kill-switch", json={"on": True})
    assert resp.status_code == 200
    assert store.get_kill_switch(ACCOUNT) is True
    resp = offline_client.post(
        f"/api/autopilot/espn/{SEASON}/{LEAGUE}/execute",
        json={"plan_id": plan_id, "groups": [0]})
    assert resp.get_json()["result"]["status"] == "blocked_kill_switch"
    assert FakeTransport.state["posts"] == []


# ── Scheduler ──────────────────────────────────────────────────────────

def _run_scheduler(env, monkeypatch):
    from dashboard_services import autopilot_schedule
    rows = [
        {"account_id": s["account_id"], "platform": s["platform"],
         "league_id": s["league_id"], "season": s["season"]}
        for s in env["conn"].settings.values() if s["mode"] != "off"
    ]
    monkeypatch.setattr(autopilot_schedule, "_candidate_leagues", lambda: rows)
    return autopilot_schedule.run_scheduled_runs(
        transport_factory=lambda *a: FakeTransport())


def test_scheduler_auto_executes_approve_only_queues(env, monkeypatch):
    store.save_settings(ACCOUNT, "espn", LEAGUE, SEASON, {"mode": "auto"})
    store.save_settings(ACCOUNT, "espn", OTHER_LEAGUE, SEASON, {"mode": "approve"})
    summary = _run_scheduler(env, monkeypatch)
    assert summary["executed"] == 1
    assert summary["queued"] == 1
    posts = FakeTransport.state["posts"]
    assert len(posts) == 1 and posts[0][0] == LEAGUE  # only the Auto league wrote
    assert len(env["notes"]["executed"]) == 1
    assert len(env["notes"]["waiting"]) == 1


def test_scheduler_kill_switch_blocks_everything(env, monkeypatch):
    store.save_settings(ACCOUNT, "espn", LEAGUE, SEASON, {"mode": "auto"})
    store.set_kill_switch(ACCOUNT, True)
    summary = _run_scheduler(env, monkeypatch)
    assert summary["executed"] == 0 and summary["queued"] == 0
    assert FakeTransport.state["posts"] == []


# ── Nav surfaces ───────────────────────────────────────────────────────

@pytest.fixture()
def _nav_env(monkeypatch):
    import app
    monkeypatch.setattr(
        app, "get_nfl_state",
        lambda: {"season": "2026", "season_type": "reg", "week": 4})
    monkeypatch.setattr(app, "get_league_ctx_from_cache", lambda *a, **k: {})
    monkeypatch.setattr(app, "has_draft_ended", lambda *a, **k: True)
    monkeypatch.setattr(app, "_games_live_or_imminent", lambda *a, **k: False)
    monkeypatch.setattr(app, "_breakouts_are_new", lambda: False)
    monkeypatch.setattr(app, "_nav_show_keeper", lambda *a, **k: True)
    monkeypatch.setattr(app, "_nav_is_best_ball", lambda *a, **k: False)
    return app


def test_nav_item_for_both_usable_leagues_only(_nav_env, monkeypatch):
    monkeypatch.setattr(
        autopilot_preview, "autopilot_surface_visible",
        lambda p, l, s: p == "espn" and str(l) in (LEAGUE, OTHER_LEAGUE))
    app = _nav_env
    for league_id in (LEAGUE, OTHER_LEAGUE):
        with app.app.test_request_context(f"/espn/2026/{league_id}/dashboard"):
            html = app.build_nav(league_id, "dashboard", "espn", 2026)
        menu = re.search(r"id='weeklyNavMenu'.*?</div>", html, re.S)
        assert menu and "Autopilot" in menu.group(0), league_id
    with app.app.test_request_context("/sleeper/2026/123456789/dashboard"):
        html = app.build_nav("123456789", "dashboard", "sleeper", 2026)
    assert "Autopilot" not in html
