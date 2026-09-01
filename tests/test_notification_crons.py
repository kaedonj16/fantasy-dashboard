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
