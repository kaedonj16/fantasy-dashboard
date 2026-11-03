"""Tuesday weekly-recap push: completed-week math, per-league dedupe, wiring.

The push fires from the weekly-recap-push Render cron (Tuesdays ~9am ET)
through /api/cron/notifications kind='recap' -> notify_recap_ready(). These
tests pin: the just-completed week (Sleeper previous_week, else week - 1),
the Tuesday gate, one send per league with the recap copy/link, and the
atomic app_state claim that makes a same-Tuesday re-run a no-op.
"""
import sys
import types
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import utils.push_notifications as pn

ROOT = Path(__file__).resolve().parents[1]
TUESDAY = date(2026, 10, 13)     # weekday() == 1
MONDAY = date(2026, 10, 12)
WEDNESDAY = date(2026, 10, 14)


# ── Completed-week math ──────────────────────────────────────────────────────

@pytest.mark.parametrize("state,expected", [
    ({"week": 7, "previous_week": 6}, 6),    # normal Tuesday after flip
    ({"week": 2, "previous_week": 1}, 1),    # Week 1 recap
    ({"week": 7}, 6),                        # no previous_week: week - 1
    ({"week": 7, "previous_week": 0}, 6),    # unusable previous_week
    ({"week": 7, "previous_week": 7}, 6),    # prev not < week: fall back
    ({"week": 1, "previous_week": 0}, 0),    # nothing completed yet
    ({"week": 1}, 0),
    ({"week": 0}, 0),
    ({}, 0),
    ({"week": "7", "previous_week": "6"}, 6),  # string state values
])
def test_recap_completed_week(state, expected):
    assert pn._recap_completed_week(state) == expected


# ── Fake app_state DB (claim semantics only) ─────────────────────────────────

class _FakeCursor:
    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [self._row] if self._row else []


class _FakeAppStateConn:
    """Emulates the app_state claim insert used by _app_state_claim."""

    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=None):
        up = " ".join(sql.split()).upper()
        if "INSERT INTO APP_STATE" in up and "DO NOTHING" in up:
            key, value = params
            if key in self.store:
                return _FakeCursor(None)
            self.store[key] = value
            return _FakeCursor({"key": key})
        raise AssertionError(f"unexpected SQL in recap test: {sql}")


@pytest.fixture
def recap_env(monkeypatch):
    store = {}
    conn = _FakeAppStateConn(store)
    db_stub = types.ModuleType("dashboard_services.db")
    db_stub.get_conn = lambda: conn
    monkeypatch.setitem(sys.modules, "dashboard_services.db", db_stub)

    state = {"season": 2026, "week": 7, "previous_week": 6,
             "season_type": "reg"}
    api_stub = types.ModuleType("dashboard_services.api")
    api_stub.get_nfl_state = lambda: state
    monkeypatch.setitem(sys.modules, "dashboard_services.api", api_stub)

    sends = []

    def fake_broadcast(league_id, title=None, body=None, url=None, tag=None,
                       notif_type=None):
        sends.append({"league_id": league_id, "title": title, "body": body,
                      "url": url, "tag": tag, "notif_type": notif_type})
        return 2

    monkeypatch.setattr(pn, "_get_subscribed_leagues",
                        lambda: [("111", "sleeper"), ("222", "espn")])
    monkeypatch.setattr(pn, "_get_subscribed_account_keys", lambda: [])
    monkeypatch.setattr(pn, "_get_legacy_subscribed_leagues",
                        lambda: [("111", "sleeper"), ("222", "espn")])
    monkeypatch.setattr(
        pn, "_league_display_name",
        lambda platform, league_id, season: {"111": "Blackedraw"}.get(
            str(league_id), ""))
    monkeypatch.setattr(pn, "_broadcast_league", fake_broadcast)
    return {"store": store, "state": state, "sends": sends}


def test_tuesday_sends_one_push_per_league(recap_env):
    sent = pn.notify_recap_ready(today=TUESDAY)
    assert sent == 4  # 2 leagues x fake 2 recipients
    assert len(recap_env["sends"]) == 2
    by_league = {s["league_id"]: s for s in recap_env["sends"]}
    blackedraw = by_league["111"]
    assert blackedraw["title"] == "Your Week 6 recap is ready"
    assert blackedraw["body"] == (
        "See how your matchup shook out in Blackedraw.")
    assert blackedraw["url"] == "/sleeper/2026/111/recap"
    assert blackedraw["tag"] == "recap-ready-2026-6"
    assert blackedraw["notif_type"] == "recap_ready"
    assert by_league["222"]["url"] == "/espn/2026/222/recap"
    assert by_league["222"]["body"] == (
        "See how your matchup shook out.")
    # No em dashes anywhere in the copy.
    for s in recap_env["sends"]:
        assert "—" not in s["title"] and "—" not in s["body"]


def test_rerun_same_tuesday_is_noop(recap_env):
    assert pn.notify_recap_ready(today=TUESDAY) == 4
    assert pn.notify_recap_ready(today=TUESDAY) == 0
    assert len(recap_env["sends"]) == 2
    assert set(recap_env["store"]) == {
        "recap_ready:111:2026:6", "recap_ready:222:2026:6"}


def test_next_week_sends_again(recap_env):
    assert pn.notify_recap_ready(today=TUESDAY) == 4
    recap_env["state"].update({"week": 8, "previous_week": 7})
    assert pn.notify_recap_ready(today=TUESDAY) == 4
    assert recap_env["sends"][-1]["title"] == "Your Week 7 recap is ready"


def test_non_tuesday_sends_nothing(recap_env):
    assert pn.notify_recap_ready(today=MONDAY) == 0
    assert pn.notify_recap_ready(today=WEDNESDAY) == 0
    assert recap_env["sends"] == []


def test_no_completed_week_sends_nothing(recap_env):
    recap_env["state"].update({"week": 1, "previous_week": 0})
    assert pn.notify_recap_ready(today=TUESDAY) == 0
    assert recap_env["sends"] == []


def test_offseason_sends_nothing(recap_env):
    recap_env["state"].update({"season_type": "pre"})
    assert pn.notify_recap_ready(today=TUESDAY) == 0
    assert recap_env["sends"] == []


def test_no_subscribed_leagues_sends_nothing(recap_env, monkeypatch):
    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: [])
    monkeypatch.setattr(pn, "_get_legacy_subscribed_leagues", lambda: [])
    assert pn.notify_recap_ready(today=TUESDAY) == 0
    assert recap_env["sends"] == []


def test_missing_previous_week_uses_week_minus_one(recap_env):
    del recap_env["state"]["previous_week"]
    assert pn.notify_recap_ready(today=TUESDAY) == 4
    assert recap_env["sends"][0]["title"] == "Your Week 6 recap is ready"


# ── Wiring guards (cron service, trigger kind, hook dispatch) ────────────────

def test_render_yaml_has_weekly_recap_push_cron():
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    block = render.split("name: weekly-recap-push", 1)[1].split("- type: cron", 1)[0]
    assert 'schedule: "0 13,14,15,16 * * 2"' in block
    assert "python scripts/trigger_notifications.py recap" in block
    assert "key: APP_URL" in block
    assert "key: CRON_SECRET" in block
    assert "value: America/New_York" in block


def test_trigger_script_accepts_recap_kind():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trigger_notifications", ROOT / "scripts" / "trigger_notifications.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "recap" in mod.ALLOWED
    # No credentials -> clean skip, same as the other kinds.
    assert mod.trigger("recap", app_url="", secret="") == 1


def test_push_bp_dispatches_recap_kind():
    push_bp = (ROOT / "routes" / "push_bp.py").read_text(encoding="utf-8")
    assert 'kind == "recap"' in push_bp
    assert "notify_recap_ready" in push_bp


def test_daily_cron_no_longer_sends_recap_push():
    cron_daily = (ROOT / "cron_daily.py").read_text(encoding="utf-8")
    assert "notify_recap_ready()" not in cron_daily


# ── Stagger gates: week_results 9am ET, recap_ready 11am ET ─────────────────

ET = ZoneInfo("America/New_York")


def _tue_dt(hour):
    return datetime(2026, 10, 13, hour, 0, tzinfo=ET)  # a Tuesday


def test_recap_ready_holds_at_9am_and_consumes_no_claim(recap_env):
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(9)) == 0
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(10)) == 0
    assert recap_env["sends"] == []
    assert recap_env["store"] == {}


def test_recap_ready_releases_at_11am(recap_env):
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(11)) == 4
    assert len(recap_env["sends"]) == 2


# ── Aggregated week_results push ─────────────────────────────────────────────

def _result(league_id, name, wins, losses, ties=0, h2h="W", my_pts=100.0,
            opp_pts=90.0, opp_name="Rival Team", sw=None, sl=None,
            platform="sleeper"):
    return {
        "platform": platform, "league_id": str(league_id), "season": 2026,
        "name": name, "wins": wins, "losses": losses, "ties": ties,
        "h2h": h2h, "my_pts": my_pts, "opp_pts": opp_pts,
        "opp_name": opp_name,
        "season_wins": sw, "season_losses": sl,
        "season_ties": 0 if sw is not None else None,
    }


@pytest.fixture
def results_env(monkeypatch):
    store = {}
    conn = _FakeAppStateConn(store)
    db_stub = types.ModuleType("dashboard_services.db")
    db_stub.get_conn = lambda: conn
    monkeypatch.setitem(sys.modules, "dashboard_services.db", db_stub)

    state = {"season": 2026, "week": 7, "previous_week": 6,
             "season_type": "reg"}
    api_stub = types.ModuleType("dashboard_services.api")
    api_stub.get_nfl_state = lambda: state
    monkeypatch.setitem(sys.modules, "dashboard_services.api", api_stub)

    env = {"store": store, "state": state, "sends": [],
           "leagues": [], "results": {}}

    def fake_broadcast_account(account_key, title=None, body=None, url=None,
                               tag=None, notif_type=None, league_id=None,
                               platform=None):
        env["sends"].append({
            "account_key": account_key, "title": title, "body": body,
            "url": url, "tag": tag, "notif_type": notif_type,
            "league_id": league_id, "platform": platform})
        return 3

    monkeypatch.setattr(pn, "_get_subscribed_account_keys", lambda: ["42"])
    monkeypatch.setattr(pn, "_recap_account_leagues",
                        lambda account_key, season: list(env["leagues"]))
    monkeypatch.setattr(
        pn, "_recap_league_result",
        lambda platform, league_id, season, week, roster_id:
            dict(env["results"][str(league_id)])
            if env["results"].get(str(league_id)) is not None else None)
    monkeypatch.setattr(pn, "_broadcast_account", fake_broadcast_account)
    return env


def _lg(league_id, platform="sleeper"):
    return {"platform": platform, "league_id": str(league_id),
            "season": 2026, "roster_id": "1", "name": ""}


def test_week_results_aggregates_all_leagues(results_env):
    results_env["leagues"] = [_lg("111"), _lg("222"), _lg("333")]
    results_env["results"] = {
        "111": _result("111", "Blackedraw", 1, 0, sw=7, sl=1),
        "222": _result("222", "League of Greatness", 0, 1, h2h="L",
                       sw=5, sl=3),
        "333": _result("333", "Third League", 1, 0, sw=3, sl=5),
    }
    sent = pn.notify_week_results(today=TUESDAY, now=_tue_dt(9))
    assert sent == 3
    assert len(results_env["sends"]) == 1
    push = results_env["sends"][0]
    assert push["account_key"] == "42"
    assert push["title"] == "Week 6: You went 2-1"
    assert push["body"] == (
        "Blackedraw is now 7-1. League of Greatness is now 5-3. "
        "Tap for all 3 leagues.")
    assert push["url"] == "/portfolio"
    assert push["tag"] == "week-results-2026-6"
    assert push["notif_type"] == "week_results"
    assert "—" not in push["title"] and "—" not in push["body"]


def test_week_results_title_includes_ties_only_when_present(results_env):
    results_env["leagues"] = [_lg("111"), _lg("222"), _lg("333")]
    results_env["results"] = {
        "111": _result("111", "A League", 1, 0, sw=6, sl=2),
        "222": _result("222", "B League", 0, 1, h2h="L", sw=4, sl=4),
        "333": _result("333", "C League", 0, 0, ties=1, h2h="T",
                       sw=3, sl=3),
    }
    results_env["results"]["333"]["season_ties"] = 1
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(9)) == 3
    assert results_env["sends"][0]["title"] == "Week 6: You went 1-1-1"


def test_week_results_partial_counts_only_resolved(results_env):
    results_env["leagues"] = [_lg("111"), _lg("222"), _lg("333"), _lg("444")]
    results_env["results"] = {
        "111": _result("111", "Blackedraw", 1, 0, sw=7, sl=1),
        "222": _result("222", "League of Greatness", 1, 0, sw=6, sl=2),
        "333": _result("333", "Third League", 0, 1, h2h="L", sw=2, sl=6),
        "444": None,
    }
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(9)) == 3
    push = results_env["sends"][0]
    assert push["title"] == "Week 6: You went 2-1"
    assert push["body"].endswith("Tap for all 3 leagues.")


def test_week_results_single_league_score_copy(results_env):
    results_env["leagues"] = [_lg("111")]
    results_env["results"] = {
        "111": _result("111", "Blackedraw", 1, 0, my_pts=128.4,
                       opp_pts=112.7, sw=7, sl=1),
    }
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(9)) == 3
    push = results_env["sends"][0]
    assert push["title"] == "Week 6: You won in Blackedraw"
    assert push["body"] == (
        "128.4 to 112.7 vs Rival Team. Blackedraw is now 7-1.")
    assert push["url"] == "/sleeper/2026/111/weekly"
    assert push["league_id"] == "111"


def test_week_results_single_league_loss_copy(results_env):
    results_env["leagues"] = [_lg("111")]
    results_env["results"] = {
        "111": _result("111", "Blackedraw", 0, 1, h2h="L", my_pts=98.2,
                       opp_pts=121.0, sw=6, sl=2),
    }
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(9)) == 3
    push = results_env["sends"][0]
    assert push["title"] == "Week 6: You lost in Blackedraw"
    assert push["body"] == (
        "98.2 to 121.0 vs Rival Team. Blackedraw is now 6-2.")


def test_week_results_zero_resolved_sends_nothing(results_env):
    results_env["leagues"] = [_lg("111"), _lg("222")]
    results_env["results"] = {"111": None, "222": None}
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(9)) == 0
    assert results_env["sends"] == []
    # The per-account claim is still consumed (claimed before computing),
    # so a later same-Tuesday run cannot retry this account.
    assert set(results_env["store"]) == {"recap_results_acct:42:2026:6"}


def test_week_results_claim_prevents_double_send(results_env):
    results_env["leagues"] = [_lg("111")]
    results_env["results"] = {
        "111": _result("111", "Blackedraw", 1, 0, sw=7, sl=1)}
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(9)) == 3
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(11)) == 0
    assert len(results_env["sends"]) == 1
    assert set(results_env["store"]) == {"recap_results_acct:42:2026:6"}


def test_week_results_holds_before_9am_and_on_non_tuesday(results_env):
    results_env["leagues"] = [_lg("111")]
    results_env["results"] = {
        "111": _result("111", "Blackedraw", 1, 0, sw=7, sl=1)}
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(8)) == 0
    assert results_env["store"] == {}
    monday_9am = datetime(2026, 10, 12, 9, 0, tzinfo=ET)
    assert pn.notify_week_results(today=MONDAY, now=monday_9am) == 0
    assert results_env["sends"] == []


def test_tuesday_stagger_9am_results_only_11am_recap_only(
        recap_env, results_env, monkeypatch):
    # One account in two leagues: both pushes are wired off the same
    # subscriptions, but the stagger keeps them in separate hours and each
    # dedupes on its own claim.
    monkeypatch.setattr(pn, "_get_subscribed_account_keys", lambda: ["42"])
    results_env["leagues"] = [_lg("111"), _lg("222")]
    results_env["results"] = {
        "111": _result("111", "Blackedraw", 1, 0, sw=7, sl=1),
        "222": _result("222", "Other League", 0, 1, h2h="L", sw=4, sl=4),
    }
    # 8am: neither is due, and no claim of either kind is consumed.
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(8)) == 0
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(8)) == 0
    assert results_env["store"] == {} and recap_env["store"] == {}
    # 9am: ONLY the aggregated results push goes.
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(9)) == 3
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(9)) == 0
    assert len(results_env["sends"]) == 1 and recap_env["sends"] == []
    # 11am: ONLY recap-ready goes (the account's combined push plus the
    # legacy per-league sends from the other fixture); results is claimed.
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(11)) > 0
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(11)) == 0
    assert len(recap_env["sends"]) == 2
    assert [s["notif_type"] for s in results_env["sends"]] == [
        "week_results", "recap_ready"]
    # Later runs: nothing double-sends.
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(12)) == 0
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(12)) == 0


def test_week_results_no_account_subscriptions_sends_nothing(
        results_env, monkeypatch):
    monkeypatch.setattr(pn, "_get_subscribed_account_keys", lambda: [])
    results_env["leagues"] = [_lg("111")]
    results_env["results"] = {
        "111": _result("111", "Blackedraw", 1, 0, sw=7, sl=1)}
    assert pn.notify_week_results(today=TUESDAY, now=_tue_dt(9)) == 0
    assert results_env["sends"] == []


# ── _recap_league_result: real matchup resolution ────────────────────────────

@pytest.fixture
def platform_env(monkeypatch):
    data = {"matchups": [], "rosters": [], "users": [], "league": {}}
    stub = types.ModuleType("dashboard_services.platform_api")
    stub.get_matchups = lambda p, lid, week, season: list(data["matchups"])
    stub.get_rosters = lambda p, lid, season: list(data["rosters"])
    stub.get_users = lambda p, lid, season: list(data["users"])
    stub.get_league = lambda p, lid, season: dict(data["league"])
    monkeypatch.setitem(sys.modules, "dashboard_services.platform_api", stub)
    return data


def _matchup(roster_id, matchup_id, points):
    return {"roster_id": roster_id, "matchup_id": matchup_id,
            "points": points}


def _roster(roster_id, owner_id, wins=0, losses=0, ties=0):
    return {"roster_id": roster_id, "owner_id": owner_id,
            "settings": {"wins": wins, "losses": losses, "ties": ties}}


def test_recap_league_result_head_to_head_win(platform_env):
    platform_env["matchups"] = [_matchup(1, 10, 120.5), _matchup(2, 10, 100.0)]
    platform_env["rosters"] = [_roster(1, "u1", 6, 1), _roster(2, "u2", 3, 4)]
    platform_env["users"] = [
        {"user_id": "u1", "display_name": "My Team"},
        {"user_id": "u2", "display_name": "Rival Team"},
    ]
    platform_env["league"] = {"name": "Blackedraw", "settings": {}}
    res = pn._recap_league_result("sleeper", "111", 2026, 6, "1")
    assert res["h2h"] == "W"
    assert (res["wins"], res["losses"], res["ties"]) == (1, 0, 0)
    assert res["my_pts"] == 120.5 and res["opp_pts"] == 100.0
    assert res["opp_name"] == "Rival Team"
    assert (res["season_wins"], res["season_losses"]) == (6, 1)
    assert res["name"] == "Blackedraw"


def test_recap_league_result_median_game_folds_in(platform_env):
    platform_env["matchups"] = [
        _matchup(1, 10, 120.5), _matchup(2, 10, 100.0),
        _matchup(3, 11, 90.0), _matchup(4, 11, 80.0),
    ]
    platform_env["rosters"] = [
        _roster(1, "u1", 6, 1), _roster(2, "u2", 3, 4),
        _roster(3, "u3", 2, 5), _roster(4, "u4", 1, 6),
    ]
    platform_env["league"] = {
        "name": "Median League",
        "settings": {"league_average_match": 1},
    }
    # Roster 1 wins head-to-head AND beats the league median (95.0).
    res = pn._recap_league_result("sleeper", "111", 2026, 6, "1")
    assert (res["wins"], res["losses"]) == (2, 0)
    # Roster 2 loses head-to-head but beats the median: 1-1 week.
    res2 = pn._recap_league_result("sleeper", "111", 2026, 6, "2")
    assert (res2["wins"], res2["losses"]) == (1, 1)


# ── recap_ready combined per account (legacy rows keep per-league) ──────────

@pytest.fixture
def acct_env(monkeypatch):
    store = {}
    conn = _FakeAppStateConn(store)
    db_stub = types.ModuleType("dashboard_services.db")
    db_stub.get_conn = lambda: conn
    monkeypatch.setitem(sys.modules, "dashboard_services.db", db_stub)

    state = {"season": 2026, "week": 7, "previous_week": 6,
             "season_type": "reg"}
    api_stub = types.ModuleType("dashboard_services.api")
    api_stub.get_nfl_state = lambda: state
    monkeypatch.setitem(sys.modules, "dashboard_services.api", api_stub)

    env = {"store": store, "state": state, "acct_sends": [],
           "league_sends": [], "acct_leagues": [], "legacy": []}

    def fake_broadcast_account(account_key, title=None, body=None, url=None,
                               tag=None, notif_type=None, league_id=None,
                               platform=None):
        env["acct_sends"].append({
            "account_key": account_key, "title": title, "body": body,
            "url": url, "tag": tag, "notif_type": notif_type,
            "league_id": league_id, "platform": platform})
        return 2

    def fake_broadcast_league(league_id, title=None, body=None, url=None,
                              tag=None, notif_type=None):
        env["league_sends"].append({
            "league_id": league_id, "title": title, "body": body,
            "url": url, "tag": tag, "notif_type": notif_type})
        return 2

    monkeypatch.setattr(pn, "_get_subscribed_account_keys", lambda: ["42"])
    monkeypatch.setattr(pn, "_recap_account_leagues",
                        lambda account_key, season: list(env["acct_leagues"]))
    monkeypatch.setattr(pn, "_get_legacy_subscribed_leagues",
                        lambda: list(env["legacy"]))
    monkeypatch.setattr(pn, "_broadcast_account", fake_broadcast_account)
    monkeypatch.setattr(pn, "_broadcast_league", fake_broadcast_league)
    monkeypatch.setattr(pn, "_league_display_name",
                        lambda platform, league_id, season: "")
    return env


def _acct_lg(league_id, name, platform="sleeper"):
    return {"platform": platform, "league_id": str(league_id),
            "season": 2026, "roster_id": "1", "name": name}


def test_recap_ready_multi_league_account_gets_one_combined_push(acct_env):
    names = ["Blackedraw", "League of Greatness"] + [
        f"League {i}" for i in range(3, 12)]
    acct_env["acct_leagues"] = [
        _acct_lg(str(100 + i), n) for i, n in enumerate(names)]
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(11)) == 2
    assert len(acct_env["acct_sends"]) == 1
    assert acct_env["league_sends"] == []
    push = acct_env["acct_sends"][0]
    assert push["title"] == "Your Week 6 recaps are ready"
    assert push["body"] == (
        "Blackedraw, League of Greatness and 9 more leagues.")
    assert push["url"] == "/portfolio"
    assert push["tag"] == "recap-ready-2026-6"
    assert push["notif_type"] == "recap_ready"
    assert "Scores are final" not in push["body"]
    assert set(acct_env["store"]) == {"recap_ready_acct:42:2026:6"}
    # Re-run: the account claim blocks a second combined push.
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(11)) == 0
    assert len(acct_env["acct_sends"]) == 1


def test_recap_ready_three_league_account_names_all_three(acct_env):
    acct_env["acct_leagues"] = [
        _acct_lg("111", "Blackedraw"), _acct_lg("222", "League of Greatness"),
        _acct_lg("333", "Third League")]
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(11)) == 2
    assert acct_env["acct_sends"][0]["body"] == (
        "Blackedraw, League of Greatness and Third League.")


def test_recap_ready_two_league_account_body(acct_env):
    acct_env["acct_leagues"] = [
        _acct_lg("111", "Blackedraw"), _acct_lg("222", "Other League")]
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(11)) == 2
    assert acct_env["acct_sends"][0]["body"] == (
        "Blackedraw and Other League.")


def test_recap_ready_single_league_account_keeps_classic_push(acct_env):
    acct_env["acct_leagues"] = [_acct_lg("111", "Blackedraw")]
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(11)) == 2
    assert len(acct_env["acct_sends"]) == 1
    assert acct_env["league_sends"] == []
    push = acct_env["acct_sends"][0]
    assert push["title"] == "Your Week 6 recap is ready"
    assert push["body"] == "See how your matchup shook out in Blackedraw."
    assert push["url"] == "/sleeper/2026/111/recap"
    assert push["league_id"] == "111"


def test_recap_ready_account_and_legacy_paths_never_overlap(acct_env):
    # League 111 is covered by the account send; only the account-less
    # league 222 may take the per-league path.
    acct_env["acct_leagues"] = [_acct_lg("111", "Blackedraw")]
    acct_env["legacy"] = [("222", "espn")]
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(11)) == 4
    assert [s["league_id"] for s in acct_env["acct_sends"]] == ["111"]
    assert [s["league_id"] for s in acct_env["league_sends"]] == ["222"]
    assert set(acct_env["store"]) == {
        "recap_ready_acct:42:2026:6", "recap_ready:222:2026:6"}


def test_recap_ready_account_path_holds_before_11am(acct_env):
    acct_env["acct_leagues"] = [_acct_lg("111", "Blackedraw")]
    acct_env["legacy"] = [("222", "espn")]
    assert pn.notify_recap_ready(today=TUESDAY, now=_tue_dt(9)) == 0
    assert acct_env["acct_sends"] == [] and acct_env["league_sends"] == []
    assert acct_env["store"] == {}


def test_recap_league_result_unresolved_returns_none(platform_env):
    platform_env["matchups"] = [_matchup(2, 10, 100.0), _matchup(3, 10, 90.0)]
    assert pn._recap_league_result("sleeper", "111", 2026, 6, "1") is None
    platform_env["matchups"] = [
        {"roster_id": 1, "matchup_id": 10, "points": None},
        _matchup(2, 10, 90.0),
    ]
    assert pn._recap_league_result("sleeper", "111", 2026, 6, "1") is None
    assert pn._recap_league_result("sleeper", "111", 2026, 6, "") is None
