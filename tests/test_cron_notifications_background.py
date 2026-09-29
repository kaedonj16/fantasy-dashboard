"""Background dispatch for /api/cron/notifications (hourly crash fix).

The endpoint must return 202 immediately and run the notification work on a
background daemon thread, with a non-blocking overlap guard so per-minute
cron hits can never pile up inside the gunicorn request threads.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

pytest.importorskip("flask_limiter")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with the real push_bp (light module-level imports)."""
    import routes.push_bp as push_mod
    from extensions import limiter

    test_app = flask.Flask(__name__)
    test_app.config["TESTING"] = True
    limiter.init_app(test_app)
    test_app.register_blueprint(push_mod.push_bp)
    return test_app.test_client()


@pytest.fixture()
def push_mod():
    import routes.push_bp as push_mod

    return push_mod


@pytest.fixture(autouse=True)
def _unlock_cron_guard(push_mod):
    """Keep the module-level overlap guard unlocked between tests.

    The dispatch test's fake thread never runs the worker, so the lock it
    acquired would otherwise stay held and break test independence.
    """
    if push_mod._cron_notifications_lock.locked():
        push_mod._cron_notifications_lock.release()
    yield
    if push_mod._cron_notifications_lock.locked():
        push_mod._cron_notifications_lock.release()


def _authed_post(client, monkeypatch, payload):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    return client.post("/api/cron/notifications", json=payload)


class _FakeThread:
    """Stands in for threading.Thread: records dispatch, never runs the target."""

    instances = []

    def __init__(self, target=None, kwargs=None, daemon=None):
        self.target = target
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = False
        _FakeThread.instances.append(self)

    def start(self):
        self.started = True


@pytest.fixture()
def fake_thread(monkeypatch, push_mod):
    _FakeThread.instances.clear()
    monkeypatch.setattr(push_mod.threading, "Thread", _FakeThread)
    return _FakeThread


def test_cron_notifications_returns_202_without_running_work_inline(
    client, monkeypatch, push_mod, fake_thread
):
    """The request thread dispatches the worker; it must not execute it."""
    calls = []
    monkeypatch.setattr(
        push_mod, "_run_cron_notifications", lambda **kw: calls.append(kw)
    )

    resp = _authed_post(client, monkeypatch, {"secret": "s3cret", "type": "hourly"})

    assert resp.status_code == 202
    assert resp.get_json() == {"ok": True, "queued": True}
    # Dispatched to exactly one daemon thread with the parsed params...
    assert len(fake_thread.instances) == 1
    thread = fake_thread.instances[0]
    assert thread.started is True
    assert thread.daemon is True
    assert thread.kwargs["kind"] == "hourly"
    # ...but the work itself did not run inline in the request thread.
    assert calls == []


def test_cron_notifications_skips_when_previous_run_in_progress(
    client, monkeypatch, push_mod, fake_thread
):
    """Overlap guard: a held lock means 202 + skipped, no new thread."""
    assert push_mod._cron_notifications_lock.acquire(blocking=False)
    try:
        resp = _authed_post(
            client, monkeypatch, {"secret": "s3cret", "type": "scorezone"}
        )
    finally:
        push_mod._cron_notifications_lock.release()

    assert resp.status_code == 202
    assert resp.get_json() == {"ok": True, "skipped": "already_running"}
    assert fake_thread.instances == []


def test_cron_notifications_still_rejects_bad_auth(
    client, monkeypatch, push_mod, fake_thread
):
    """Auth stays in the request thread, before any dispatch."""
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.post("/api/cron/notifications", json={"secret": "wrong"})
    assert resp.status_code == 403
    assert fake_thread.instances == []


def test_worker_runs_hourly_and_releases_lock_on_success(push_mod, monkeypatch):
    import utils.push_notifications as pn

    monkeypatch.setattr(pn, "run_hourly", lambda: {"total": 0})
    assert push_mod._cron_notifications_lock.acquire(blocking=False)
    push_mod._run_cron_notifications(kind="hourly")
    assert not push_mod._cron_notifications_lock.locked()


def test_worker_releases_lock_and_swallows_exceptions(push_mod, monkeypatch):
    """A background failure must never propagate or wedge the guard lock."""
    import utils.push_notifications as pn

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(pn, "run_scorezone_td_poll", _boom)
    assert push_mod._cron_notifications_lock.acquire(blocking=False)
    push_mod._run_cron_notifications(kind="scorezone")  # must not raise
    assert not push_mod._cron_notifications_lock.locked()


def test_worker_passes_weekly_params_through(push_mod, monkeypatch):
    import utils.weekly_email as we

    seen = {}
    monkeypatch.setattr(
        we, "send_weekly_digests", lambda **kw: seen.update(kw) or {"ok": True}
    )
    assert push_mod._cron_notifications_lock.acquire(blocking=False)
    push_mod._run_cron_notifications(
        kind="weekly", account_id=42, email="a@example.test", force=True
    )
    assert seen == {"account_id": 42, "email": "a@example.test", "force": True}
    assert not push_mod._cron_notifications_lock.locked()


def _load_trigger():
    spec = importlib.util.spec_from_file_location(
        "trigger_notifications", ROOT / "scripts" / "trigger_notifications.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trigger_strips_whitespace_from_app_url(monkeypatch):
    """A pasted trailing newline in APP_URL must not produce an invalid URL."""
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
        return _Resp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", _urlopen)
    monkeypatch.setenv("APP_URL", "https://www.brfantasyfootball.com\n")
    assert mod.trigger("hourly", secret="s3cret") == 0
    assert captured["url"] == "https://www.brfantasyfootball.com/api/cron/notifications"
