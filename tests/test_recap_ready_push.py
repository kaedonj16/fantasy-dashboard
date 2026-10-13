"""Tuesday weekly-recap push: completed-week math, per-league dedupe, wiring.

The push fires from the weekly-recap-push Render cron (Tuesdays ~9am ET)
through /api/cron/notifications kind='recap' -> notify_recap_ready(). These
tests pin: the just-completed week (Sleeper previous_week, else week - 1),
the Tuesday gate, one send per league with the recap copy/link, and the
atomic app_state claim that makes a same-Tuesday re-run a no-op.
"""
import sys
import types
from datetime import date
from pathlib import Path

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
        "Scores are final. See how your matchup shook out in Blackedraw.")
    assert blackedraw["url"] == "/sleeper/2026/111/weekly"
    assert blackedraw["tag"] == "recap-ready-2026-6"
    assert blackedraw["notif_type"] == "recap_ready"
    assert by_league["222"]["url"] == "/espn/2026/222/weekly"
    assert by_league["222"]["body"] == (
        "Scores are final. See how your matchup shook out.")
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
    assert 'schedule: "0 13 * * 2"' in block
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
