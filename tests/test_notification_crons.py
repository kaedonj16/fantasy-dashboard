"""Guards for Render notification crons and the lightweight trigger script."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
RENDER = (ROOT / "render.yaml").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
PUSH_BP = (ROOT / "routes" / "push_bp.py").read_text(encoding="utf-8")


def _load_trigger():
    spec = importlib.util.spec_from_file_location(
        "trigger_notifications", ROOT / "scripts" / "trigger_notifications.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_web_service_has_cron_secret_for_notification_hook():
    web = RENDER.split("name: brfantasy", 1)[1].split("- type: cron", 1)[0]
    assert "key: CRON_SECRET" in web
    assert "key: ADMIN_SECRET" in web
    hourly = RENDER.split("name: hourly-notifications", 1)[1].split("- type: cron", 1)[0]
    weekly = RENDER.split("name: weekly-email", 1)[1].split("- type: cron", 1)[0]
    assert "schedule: 5 * * * *" in hourly
    assert "python scripts/trigger_notifications.py hourly" in hourly
    assert "key: APP_URL" in hourly
    assert "key: CRON_SECRET" in hourly
    assert "value: America/New_York" in hourly
    assert 'schedule: "0 13 * * 2"' in weekly
    assert "python scripts/trigger_notifications.py weekly" in weekly
    assert "value: America/New_York" in weekly
    # Render cron is UTC; 13:00 UTC is 9am EDT / 8am EST. Do not regress to 09:00 UTC.
    assert "schedule: 0 9 * * 2" not in weekly
    assert "UTC" in weekly
    assert 'data.get("email")' in PUSH_BP
    assert "force=force" in PUSH_BP


def test_production_does_not_start_inprocess_notify_scheduler_by_default():
    assert "ENABLE_INPROCESS_NOTIFY_SCHEDULER" in APP_PY
    assert '!= "production"' in APP_PY[APP_PY.index("_inprocess_notify"):]


def test_notifications_hook_accepts_cron_secret():
    assert "def _notifications_cron_authorized" in PUSH_BP
    assert "X-Cron-Secret" in PUSH_BP
    assert "hmac.compare_digest" in PUSH_BP


def test_trigger_skips_without_credentials(monkeypatch, capsys):
    mod = _load_trigger()
    monkeypatch.delenv("APP_URL", raising=False)
    monkeypatch.delenv("CRON_SECRET", raising=False)
    assert mod.trigger("hourly", app_url="", secret="") == 1
    assert "APP_URL or CRON_SECRET not set" in capsys.readouterr().out


def test_trigger_rejects_unknown_type():
    mod = _load_trigger()
    assert mod.trigger("monthly") == 2


def test_trigger_posts_type_and_secret(monkeypatch):
    mod = _load_trigger()
    captured = {}

    class _Resp:
        status = 200

        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = req.data
        captured["headers"] = dict(req.header_items())
        return _Resp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", _urlopen)
    assert mod.trigger("weekly", app_url="https://brfantasyfootball.com", secret="s3cret") == 0
    assert captured["url"] == "https://brfantasyfootball.com/api/cron/notifications"
    assert captured["timeout"] == 900
    assert b'"type": "weekly"' in captured["body"]
    assert b'"secret": "s3cret"' in captured["body"]
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers.get("x-cron-secret") == "s3cret"


def test_trigger_nonzero_on_http_error(monkeypatch):
    mod = _load_trigger()

    class _FP:
        def read(self, n=-1):
            return b'{"error":"Forbidden"}'

        def close(self):
            return None

    def _urlopen(req, timeout=0):
        raise HTTPError(req.full_url, 403, "Forbidden", hdrs=None, fp=_FP())

    monkeypatch.setattr(mod.urllib.request, "urlopen", _urlopen)
    assert mod.trigger("hourly", app_url="https://example.test", secret="x") == 1


def test_run_hourly_returns_sent_counts(monkeypatch):
    import utils.push_notifications as pn

    monkeypatch.setattr(pn, "notify_lineup_lock", lambda: 3)
    monkeypatch.setattr(pn, "notify_lineup_lock_sunday", lambda: 1)
    monkeypatch.setattr(pn, "notify_close_game", lambda: 0)
    monkeypatch.setattr(pn, "notify_transaction_drops", lambda: 2)
    monkeypatch.setattr(pn, "notify_injury_alert", lambda: None)
    monkeypatch.setattr(pn, "notify_breakout_weekly", lambda: 5)
    monkeypatch.setattr(pn, "_flush_digest", lambda: 4)

    counts = pn.run_hourly()
    assert counts == {
        "lineup_lock": 3,
        "lineup_lock_sunday": 1,
        "close_game": 0,
        "transaction_drops": 2,
        "injury_alert": 0,
        "breakout_weekly": 5,
        "digest": 4,
        "total": 15,
    }


def test_trigger_logs_sent_count(monkeypatch, capsys):
    import json as _json

    mod = _load_trigger()

    class _Resp:
        status = 200

        def read(self):
            return _json.dumps({
                "ok": True, "sent": 7,
                "breakdown": {"lineup_lock": 3, "close_game": 0,
                              "transaction_drops": 0, "injury_alert": 0,
                              "digest": 4, "total": 7},
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp())
    assert mod.trigger("hourly", app_url="https://example.test", secret="x") == 0
    out = capsys.readouterr().out
    assert "[notify-cron] hourly: HTTP 200 sent=7" in out
    assert "lineup_lock=3" in out
    assert "digest=4" in out


def _stub_module(name, **attrs):
    import types
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_run_redzone_td_poll_no_new_tds(monkeypatch):
    import sys

    import utils.push_notifications as pn
    import utils.redzone_store as rs

    fake_api = _stub_module(
        "dashboard_services.api",
        get_nfl_state=lambda: {"season": 2026, "week": 4},
        get_nfl_players=lambda: {},
    )
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)
    monkeypatch.setattr(rs, "get_watermark", lambda: 1000.0)
    monkeypatch.setattr(rs, "get_td_plays_since", lambda season, since: [])

    assert pn.run_redzone_td_poll() == {"games": 0, "leagues": 0, "sent": 0}


def test_run_redzone_td_poll_cold_start_sets_watermark_without_sending(monkeypatch):
    import sys

    import utils.push_notifications as pn
    import utils.redzone_store as rs

    fake_api = _stub_module(
        "dashboard_services.api",
        get_nfl_state=lambda: {"season": 2026, "week": 4},
        get_nfl_players=lambda: {},
    )
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)
    monkeypatch.setattr(rs, "get_watermark", lambda: 0.0)
    wm = {}
    monkeypatch.setattr(rs, "set_watermark", lambda ts: wm.setdefault("ts", ts))
    monkeypatch.setattr(pn, "notify_redzone_scores",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not notify on cold start")))
    monkeypatch.setattr(pn, "_flush_digest", lambda: 0)

    assert pn.run_redzone_td_poll() == {"games": 0, "leagues": 0, "sent": 0}
    assert wm["ts"] > 0


def test_run_redzone_td_poll_sends_and_flushes_digest(monkeypatch):
    import sys

    import utils.push_notifications as pn
    import utils.redzone_store as rs
    import dashboard_services.platform_api as papi

    fake_api = _stub_module(
        "dashboard_services.api",
        get_nfl_state=lambda: {"season": 2026, "week": 4},
        get_nfl_players=lambda: {
            "1": {"full_name": "Test Back", "position": "RB"},
        },
        get_normalized_scoring_settings=lambda platform: {"rec": 1.0},
    )
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)
    monkeypatch.setattr(papi, "get_rosters", lambda *a, **k: [{"roster_id": 1}])
    monkeypatch.setattr(papi, "sync_league_globals", lambda *a, **k: None)
    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: [("L1", "sleeper")])

    td = ("g1", {"play_id": "p1", "is_td": True, "pid": "1",
                 "name": "Test Back"}, 1200.0)
    monkeypatch.setattr(rs, "get_watermark", lambda: 1000.0)
    monkeypatch.setattr(rs, "get_td_plays_since", lambda season, since: [td])
    wm = {}
    monkeypatch.setattr(rs, "set_watermark", lambda ts: wm.setdefault("ts", ts))

    calls = {}

    def fake_notify(league_id, platform, pbp_by_game, player_info,
                    rosters, scoring, **kw):
        calls["notify"] = (league_id, platform, pbp_by_game, player_info,
                           rosters, scoring, kw)
        return 2

    monkeypatch.setattr(pn, "notify_redzone_scores", fake_notify)
    monkeypatch.setattr(pn, "_flush_digest", lambda: 1)

    assert pn.run_redzone_td_poll() == {"games": 1, "leagues": 1, "sent": 3}
    # watermark advances to the max observed play ts, not wall-clock now
    assert wm["ts"] == 1200.0
    lid, plat, pbp, pinfo, rosters, scoring, kw = calls["notify"]
    assert (lid, plat) == ("L1", "sleeper")
    assert list(pbp.keys()) == ["g1"]
    assert pbp["g1"][0]["play_id"] == "p1"
    assert pinfo["1"] == {"name": "Test Back", "pos": "RB"}
    assert scoring.get("rec") == 1.0
    assert kw == {"season": 2026, "week": 4}


def test_trigger_redzone_logs_sent_count(monkeypatch, capsys):
    import json as _json

    mod = _load_trigger()

    class _Resp:
        status = 200

        def read(self):
            return _json.dumps({
                "ok": True, "sent": 3,
                "breakdown": {"games": 5, "leagues": 2, "sent": 3},
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp())
    assert mod.trigger("redzone", app_url="https://example.test", secret="x") == 0
    out = capsys.readouterr().out
    assert "[notify-cron] redzone: HTTP 200 sent=3" in out
    assert "games=5" in out


def test_redzone_poller_cron_service_registered():
    rz = RENDER.split("name: redzone-td-poller", 1)[1].split("- type: cron", 1)[0]
    assert 'schedule: "* * * * *"' in rz
    assert "python scripts/trigger_notifications.py redzone" in rz
    assert "key: APP_URL" in rz
    assert "key: CRON_SECRET" in rz
    assert "value: America/New_York" in rz


def _next_weekday_ep(weekday, hour=12):
    """Epoch for the next given weekday (0=Mon..6=Sun) at hour UTC."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    days_ahead = (weekday - now.weekday()) % 7 or 7
    dt = (now + timedelta(days=days_ahead)).replace(
        hour=hour, minute=0, second=0, microsecond=0)
    return dt.timestamp()


def test_lineup_lock_window_bounds():
    import time

    import utils.push_notifications as pn

    now = time.time()
    assert pn._in_lineup_lock_window(now + 60 * 60) is True
    assert pn._in_lineup_lock_window(now + 41 * 60) is True
    assert pn._in_lineup_lock_window(now + 99 * 60) is True
    assert pn._in_lineup_lock_window(now + 101 * 60) is False
    assert pn._in_lineup_lock_window(now + 39 * 60) is False
    assert pn._in_lineup_lock_window(now - 10 * 60) is False


def test_lineup_lock_sunday_uses_first_sunday_kickoff(monkeypatch):
    import utils.push_notifications as pn

    monday_ep = _next_weekday_ep(0)   # not Sunday in ET
    sunday_ep = _next_weekday_ep(6)   # Sunday in ET
    games = [
        {"gameTime_epoch": str(monday_ep), "home": "KC", "away": "BUF"},
        {"gameTime_epoch": str(sunday_ep), "home": "DAL", "away": "PHI"},
    ]
    monkeypatch.setattr(pn, "_lineup_lock_base",
                        lambda: (games, 2026, 4))

    seen = {}
    monkeypatch.setattr(pn, "_in_lineup_lock_window",
                        lambda ep: seen.setdefault("ep", ep) or True)
    monkeypatch.setattr(pn, "_lineup_lock_send",
                        lambda games, season, week, **kw: seen.update(kw) or 5)

    assert pn.notify_lineup_lock_sunday() == 5
    assert seen["ep"] == sunday_ep
    assert seen["dedupe_key"] == "lineup_lock_sunday"
    assert seen["tag"] == "lineup-lock-sunday-2026-4"
    assert "Sunday" in seen["kickoff_line"]


def test_lineup_lock_sunday_skips_when_no_sunday_games(monkeypatch):
    import utils.push_notifications as pn

    games = [{"gameTime_epoch": str(_next_weekday_ep(0)),
              "home": "KC", "away": "BUF"}]
    monkeypatch.setattr(pn, "_lineup_lock_base",
                        lambda: (games, 2026, 4))
    called = []
    monkeypatch.setattr(pn, "_lineup_lock_send",
                        lambda *a, **k: called.append(1))

    assert pn.notify_lineup_lock_sunday() is None
    assert called == []


def test_lineup_lock_thursday_uses_week_first_kickoff(monkeypatch):
    import utils.push_notifications as pn

    monday_ep = _next_weekday_ep(0)
    sunday_ep = _next_weekday_ep(6)
    games = [
        {"gameTime_epoch": str(sunday_ep), "home": "DAL", "away": "PHI"},
        {"gameTime_epoch": str(monday_ep), "home": "KC", "away": "BUF"},
    ]
    monkeypatch.setattr(pn, "_lineup_lock_base",
                        lambda: (games, 2026, 4))

    seen = {}
    monkeypatch.setattr(pn, "_in_lineup_lock_window",
                        lambda ep: seen.setdefault("ep", ep) or True)
    monkeypatch.setattr(pn, "_lineup_lock_send",
                        lambda games, season, week, **kw: seen.update(kw) or 3)

    assert pn.notify_lineup_lock() == 3
    assert seen["ep"] == min(monday_ep, sunday_ep)
    assert seen["dedupe_key"] == "lineup_lock_week"
    assert seen["tag"] == "lineup-lock-2026-4"


def test_run_hourly_includes_sunday_lineup_lock(monkeypatch):
    import utils.push_notifications as pn

    monkeypatch.setattr(pn, "notify_lineup_lock", lambda: 3)
    monkeypatch.setattr(pn, "notify_lineup_lock_sunday", lambda: 1)
    monkeypatch.setattr(pn, "notify_close_game", lambda: 0)
    monkeypatch.setattr(pn, "notify_transaction_drops", lambda: 2)
    monkeypatch.setattr(pn, "notify_injury_alert", lambda: None)
    monkeypatch.setattr(pn, "_flush_digest", lambda: 4)

    counts = pn.run_hourly()
    assert counts["lineup_lock"] == 3
    assert counts["lineup_lock_sunday"] == 1
    assert counts["total"] == 10
