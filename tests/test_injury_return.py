"""ESPN injury return cache behavior."""
import json
import time
from urllib.error import HTTPError

import pytest

from dashboard_services import injury_return as ir


@pytest.fixture(autouse=True)
def _reset_injury_return_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "espn_return_dates.json"
    monkeypatch.setattr(ir, "_CACHE_FILE", cache_file)
    monkeypatch.setattr(ir, "_CACHE", {"ts": 0.0, "by_pid": {}})
    monkeypatch.setattr(ir, "_FETCH_BLOCKED_UNTIL", 0.0)
    monkeypatch.setattr(ir, "_LAST_FETCH_WARN", 0.0)
    yield


def test_refresh_uses_stale_disk_when_fetch_blocked(monkeypatch, tmp_path):
    stale = {"p1": {"player_id": "p1", "return_date": "2026-09-01"}}
    ir._CACHE_FILE.write_text(json.dumps(stale), encoding="utf-8")
    # Backdate file so it is outside the fresh TTL.
    old = time.time() - ir._TTL - 60
    import os

    os.utime(ir._CACHE_FILE, (old, old))

    def _boom():
        raise HTTPError(ir._ESPN_INJURIES_URL, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr(ir, "_fetch_espn_json", _boom)
    out = ir.refresh_espn_return_dates()
    assert out == stale
    assert ir._FETCH_BLOCKED_UNTIL > time.time()


def test_refresh_skips_network_while_blocked(monkeypatch):
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise HTTPError(ir._ESPN_INJURIES_URL, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr(ir, "_fetch_espn_json", _boom)
    ir._FETCH_BLOCKED_UNTIL = time.time() + 3600
    assert ir.refresh_espn_return_dates() == {}
    assert ir.refresh_espn_return_dates() == {}
    assert calls["n"] == 0


def test_refresh_force_bypasses_block(monkeypatch):
    payload = {"injuries": []}
    calls = {"n": 0}

    def _ok():
        calls["n"] += 1
        return payload

    monkeypatch.setattr(ir, "_fetch_espn_json", _ok)
    monkeypatch.setattr(
        "dashboard_services.providers.global_adp.espn_id_to_canonical",
        lambda: {},
    )
    ir._FETCH_BLOCKED_UNTIL = time.time() + 3600
    ir.refresh_espn_return_dates(force=True)
    assert calls["n"] == 1
