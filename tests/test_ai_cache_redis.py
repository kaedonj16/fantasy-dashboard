"""Redis-backed durable copy for the AI text cache.

Regression coverage: the file cache lives on the web service's ephemeral
disk, so every deploy/restart wiped Front Office reports (and every other
AI artifact) and the next view repaid the full build + AI call. Entries are
now written to Redis as well; reads consult the file first, then Redis,
and a Redis hit backfills the file verbatim. All Redis failures degrade to
the pre-existing file-only behavior.
"""

import json
import time

import pytest

from dashboard_services.ai import cache as ai_cache


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.setex_calls = []

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.setex_calls.append((key, ttl))
        self.store[key] = value
        return True


class BrokenRedis:
    def get(self, key):
        raise ConnectionError("redis down")

    def setex(self, key, ttl, value):
        raise ConnectionError("redis down")


@pytest.fixture()
def fake_redis(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_cache, "AI_CACHE_DIR", tmp_path)
    fake = FakeRedis()
    monkeypatch.setattr(ai_cache, "_redis_client", lambda: fake)
    return fake


def test_save_writes_redis_with_default_ttl_and_file(fake_redis, tmp_path):
    ai_cache.save_cached_ai_text("k1", "hello")
    key = ai_cache._redis_key("k1")
    assert key in fake_redis.store
    assert fake_redis.setex_calls == [(key, ai_cache.AI_CACHE_TTL)]
    obj = json.loads(fake_redis.store[key])
    assert obj["content"] == "hello"
    assert obj["ttl"] == ai_cache.AI_CACHE_TTL
    assert (tmp_path / "k1.json").exists()


def test_load_survives_file_wipe_via_redis(fake_redis, tmp_path):
    # The deploy scenario: the entry was saved, then the disk was wiped.
    ai_cache.save_cached_ai_text("k2", "durable")
    saved_ts = json.loads((tmp_path / "k2.json").read_text(encoding="utf-8"))["ts"]
    (tmp_path / "k2.json").unlink()
    assert ai_cache.load_cached_ai_text("k2") == "durable"
    # The Redis hit backfills the file verbatim: same ts, so the entry's
    # expiry does not move.
    backfilled = json.loads((tmp_path / "k2.json").read_text(encoding="utf-8"))
    assert backfilled["ts"] == saved_ts
    assert backfilled["content"] == "durable"


def test_redis_entry_respects_fallback_ttl(fake_redis, tmp_path):
    ai_cache.save_cached_ai_text(
        "k3", "fallback", ttl=ai_cache.AI_CACHE_FALLBACK_TTL)
    key = ai_cache._redis_key("k3")
    assert fake_redis.setex_calls == [(key, ai_cache.AI_CACHE_FALLBACK_TTL)]
    # Age the Redis envelope past the short TTL (the fake does not expire
    # keys itself, so the reader's own envelope check must reject it).
    obj = json.loads(fake_redis.store[key])
    obj["ts"] = time.time() - ai_cache.AI_CACHE_FALLBACK_TTL - 1
    fake_redis.store[key] = json.dumps(obj)
    (tmp_path / "k3.json").unlink()
    assert ai_cache.load_cached_ai_text("k3") is None


def test_broken_redis_degrades_to_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_cache, "AI_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ai_cache, "_redis_client", lambda: BrokenRedis())
    ai_cache.save_cached_ai_text("k4", "file-only")  # must not raise
    assert ai_cache.load_cached_ai_text("k4") == "file-only"


def test_oversize_entry_skips_redis_but_saves_file(fake_redis, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_CACHE_REDIS_MAX_BYTES", "1024")
    ai_cache.save_cached_ai_text("k5", "x" * 5000)
    assert fake_redis.store == {}
    assert ai_cache.load_cached_ai_text("k5") == "x" * 5000


def test_redis_client_disabled_without_url_or_by_kill_switch(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.delenv("AI_CACHE_REDIS", raising=False)
    assert ai_cache._redis_client() is None
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("AI_CACHE_REDIS", "0")
    assert ai_cache._redis_client() is None


def test_no_redis_configured_is_file_only(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_cache, "AI_CACHE_DIR", tmp_path)
    monkeypatch.setenv("REDIS_URL", "")
    ai_cache.save_cached_ai_text("k6", "plain")
    assert ai_cache.load_cached_ai_text("k6") == "plain"
