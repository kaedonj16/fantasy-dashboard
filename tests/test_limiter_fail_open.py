"""Rate limiter must fail open when its storage backend raises (Class D audit).

Proven from the installed Flask-Limiter 3.11.0 source
(flask_limiter/extension.py):

* ``Limiter.__init__`` takes ``swallow_errors=None``; in ``init_app`` that
  falls back to the ``RATELIMIT_SWALLOW_ERRORS`` config value, default
  ``False``. extensions.py passed no argument and app.py sets no such
  config, so the limiter ran with swallow_errors=False.
* ``Limiter._check_request_limit`` re-raises any non-RateLimitExceeded
  exception when swallow_errors is False. A Redis storage raise during
  ``__evaluate_limits`` (``self.limiter.hit(...)``) therefore propagated
  out of the before-request chain and Flask returned 500 for EVERY
  rate-limited endpoint at once, even though buckets are per-endpoint
  (flask_limiter/wrappers.py ``Limit.scope_for``: scope = endpoint for
  non-shared limits, and this app uses no shared_limit / meta limits /
  default limits).

The fix is ``swallow_errors=True`` in extensions.py: the storage error is
logged and the view still executes; with healthy storage the limit trips
exactly as before.

These tests run against the real shared limiter from extensions.py. The
storage failure is stubbed per-test via monkeypatch on the live
RateLimiter's ``hit`` (scoped, auto-undone; per AGENTS.md never leave
stubs installed at import time).
"""
from __future__ import annotations

from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("flask_limiter")

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


@pytest.fixture()
def fail_open_app():
    from extensions import limiter

    test_app = flask.Flask(__name__)
    test_app.config["TESTING"] = True
    limiter.init_app(test_app)

    @test_app.route("/audit-d-fail-open")
    @limiter.limit("2 per minute")
    def audit_d_fail_open():
        return flask.jsonify({"ok": True})

    return test_app, limiter


def test_limiter_is_configured_to_swallow_storage_errors():
    src = (ROOT / "extensions.py").read_text(encoding="utf-8")
    assert "swallow_errors=True" in src


def test_storage_failure_still_executes_the_view(fail_open_app, monkeypatch):
    test_app, limiter = fail_open_app

    def _raising_hit(*args, **kwargs):
        raise ConnectionError("redis storage is down (stubbed)")

    # limiter.limiter is the live limits-library RateLimiter (a property
    # on the Flask-Limiter extension, resolved after init_app).
    monkeypatch.setattr(limiter.limiter, "hit", _raising_hit)

    client = test_app.test_client()
    for _ in range(3):
        resp = client.get("/audit-d-fail-open")
        assert resp.status_code == 200, (
            "a storage outage must fail open, not 500 the request"
        )
        assert resp.get_json() == {"ok": True}


def test_healthy_storage_still_enforces_the_limit(fail_open_app):
    test_app, _limiter = fail_open_app
    client = test_app.test_client()
    # A distinct REMOTE_ADDR gives this test its own bucket even if other
    # tests in the same process already spent the 127.0.0.1 bucket.
    env = {"REMOTE_ADDR": "10.30.0.9"}
    assert client.get("/audit-d-fail-open", environ_overrides=env).status_code == 200
    assert client.get("/audit-d-fail-open", environ_overrides=env).status_code == 200
    resp = client.get("/audit-d-fail-open", environ_overrides=env)
    assert resp.status_code == 429, "healthy storage must still trip the limit"


def test_trade_caps_match_the_counted_client_flow():
    """app.js costs up to 3 trade-eval POSTs and 2 playoff-impact POSTs
    per single user action (renderChips ends in recomputeTrade and is
    called for both sides; suggestion/sample flows add analyzeTrade;
    every successful recompute spawns one playoff-impact fetch). The
    caps must allow the same 30 actions/min on both endpoints."""
    app_py = (ROOT / "app.py").read_text(encoding="utf-8")

    eval_block = app_py.split('@app.route("/api/trade-eval", methods=["POST"])', 1)[1]
    eval_block = eval_block.split("def api_trade_eval", 1)[0]
    assert '@limiter.limit("90 per minute")' in eval_block

    impact_block = app_py.split(
        '@app.route("/api/trade-eval/playoff-impact", methods=["POST"])', 1
    )[1]
    impact_block = impact_block.split("def api_trade_eval_playoff_impact", 1)[0]
    assert '@limiter.limit("60 per minute")' in impact_block

    # The counted flow itself: renderChips funnels into recomputeTrade,
    # and recomputeTrade fetches both endpoints.
    assert "recomputeTrade();" in APP_JS
    assert 'fetch("/api/trade-eval"' in APP_JS
    assert 'fetch("/api/trade-eval/playoff-impact"' in APP_JS
