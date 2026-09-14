"""fetch_week_stats must self-heal an empty (pre-kickoff) cache.

Regression guard: an empty ``{}`` weekly stats file used to be returned forever
because the loader only refetched when the file was missing/corrupt. That left a
week whose games had been played permanently blank, so the player-modal game log
(and anything reading these files) kept showing projections for played games.
"""
import json
import os
import time

import pytest

# Importing pulls utils.utils (pandas etc.); skip cleanly in the lint-only shard.
mod = pytest.importorskip("data_building.external_data.sleeper_bulk_stats")


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_populated_cache_returned_without_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    p = tmp_path / "sleeper_stats" / "sleeper_stats_s2026_w1.json"
    _write(p, {"8112": {"rec": 5}})

    def _boom(*a, **k):
        raise AssertionError("a populated cache must not trigger a fetch")

    monkeypatch.setattr(mod.requests, "get", _boom)
    assert mod.fetch_week_stats(2026, 1) == {"8112": {"rec": 5}}


def test_empty_fresh_cache_not_refetched(tmp_path, monkeypatch):
    # A just-written empty week (no games yet) is served as-is so future/bye
    # weeks don't hammer Sleeper on every call.
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    p = tmp_path / "sleeper_stats" / "sleeper_stats_s2026_w2.json"
    _write(p, {})

    def _boom(*a, **k):
        raise AssertionError("a fresh empty week must not be refetched")

    monkeypatch.setattr(mod.requests, "get", _boom)
    assert mod.fetch_week_stats(2026, 2) == {}


def test_empty_stale_cache_refetches_and_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    p = tmp_path / "sleeper_stats" / "sleeper_stats_s2026_w1.json"
    _write(p, {})
    stale = time.time() - (mod.WEEK_CACHE_TTL + 60)
    os.utime(p, (stale, stale))

    monkeypatch.setattr(mod.requests, "get",
                        lambda *a, **k: _FakeResp({"8112": {"rec": 7}}))
    out = mod.fetch_week_stats(2026, 1)
    assert out == {"8112": {"rec": 7}}
    assert json.loads(p.read_text()) == {"8112": {"rec": 7}}  # written back


def test_missing_cache_fetches(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(mod.requests, "get",
                        lambda *a, **k: _FakeResp({"1": {"rush_yd": 10}}))
    assert mod.fetch_week_stats(2026, 5) == {"1": {"rush_yd": 10}}
