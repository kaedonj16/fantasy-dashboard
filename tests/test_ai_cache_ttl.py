"""Per-entry TTL for the AI text cache.

Regression coverage: a transient AI failure must not poison the cache for the
full 12-hour TTL. Failure fallbacks are saved with AI_CACHE_FALLBACK_TTL so a
later view retries the AI call and the "unavailable" notice clears itself.
"""

import json
import time

import pytest

from dashboard_services.ai import cache as ai_cache


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_cache, "AI_CACHE_DIR", tmp_path)
    return tmp_path


def test_default_ttl_still_12h(cache_dir):
    ai_cache.save_cached_ai_text("k1", "hello")
    obj = json.loads((cache_dir / "k1.json").read_text(encoding="utf-8"))
    assert obj["ttl"] == ai_cache.AI_CACHE_TTL == 12 * 60 * 60
    assert ai_cache.load_cached_ai_text("k1") == "hello"


def test_short_ttl_entry_expires_early(cache_dir):
    ai_cache.save_cached_ai_text("k2", "fallback", ttl=ai_cache.AI_CACHE_FALLBACK_TTL)
    assert ai_cache.load_cached_ai_text("k2") == "fallback"
    obj = json.loads((cache_dir / "k2.json").read_text(encoding="utf-8"))
    # Simulate the entry aging past the short TTL but well within 12h.
    obj["ts"] = time.time() - ai_cache.AI_CACHE_FALLBACK_TTL - 1
    (cache_dir / "k2.json").write_text(json.dumps(obj), encoding="utf-8")
    assert ai_cache.load_cached_ai_text("k2") is None


def test_short_ttl_does_not_affect_default_entries(cache_dir):
    ai_cache.save_cached_ai_text("k3", "normal")
    obj = json.loads((cache_dir / "k3.json").read_text(encoding="utf-8"))
    # Aged past the fallback TTL but within the default TTL: still served.
    obj["ts"] = time.time() - ai_cache.AI_CACHE_FALLBACK_TTL - 1
    (cache_dir / "k3.json").write_text(json.dumps(obj), encoding="utf-8")
    assert ai_cache.load_cached_ai_text("k3") == "normal"


def test_old_entries_without_stored_ttl_use_default(cache_dir):
    # Entries written before per-entry TTL existed have no "ttl" key.
    (cache_dir / "k4.json").write_text(
        json.dumps({"ts": time.time(), "content": "legacy"}), encoding="utf-8"
    )
    assert ai_cache.load_cached_ai_text("k4") == "legacy"
