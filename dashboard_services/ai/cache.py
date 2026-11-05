from __future__ import annotations

import logging
import hashlib
import json
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

AI_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "ai_cache"
AI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

AI_CACHE_TTL = 60 * 60 * 12  # 12 hours

# Short TTL for AI-failure fallbacks. A transient OpenAI failure (rate limit,
# outage, unexpected error) must not poison the cache for 12 hours; the next
# view a few minutes later retries the AI call and the notice clears itself.
AI_CACHE_FALLBACK_TTL = 5 * 60  # 5 minutes

# The file cache lives on the web service's ephemeral disk (render.yaml gives
# brfantasy no persistent disk), so every deploy, restart or OOM wiped every
# AI artifact -- Front Office reports, GM memos, trade analyses -- and the
# next view repaid the full data build plus the AI call. Redis is already
# provisioned (REDIS_URL, see league_ctx_store / player_details_store) and
# survives deploys, so it holds the durable copy of every entry; the file
# remains the fast local copy. Fail-soft throughout, mirroring those stores:
# Redis down, slow, unconfigured, oversized or disabled via AI_CACHE_REDIS=0
# degrades to the pre-existing file-only behavior, and no cache operation
# ever raises into a caller.
AI_CACHE_REDIS_PREFIX = "ai_cache:"
AI_CACHE_REDIS_MAX_BYTES = 2 * 1024 * 1024


def _hash_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_ai_cache_key(artifact_type: str, payload: dict, prompt_version: str = "v1") -> str:
    payload_hash = _hash_payload(payload)
    return f"{artifact_type}_{prompt_version}_{payload_hash}"


# ── Redis layer ──────────────────────────────────────────────────────────────

def _redis_enabled() -> bool:
    if (os.environ.get("AI_CACHE_REDIS") or "1").strip() == "0":
        return False
    return bool((os.environ.get("REDIS_URL") or "").strip())


def _redis_max_bytes() -> int:
    try:
        return max(1024, int(os.environ.get(
            "AI_CACHE_REDIS_MAX_BYTES", str(AI_CACHE_REDIS_MAX_BYTES))))
    except (TypeError, ValueError):
        return AI_CACHE_REDIS_MAX_BYTES


def _redis_client():
    """Fresh client per operation (fork-safe), mirroring league_ctx_store.

    Bounded timeouts so a hung Redis can never stall a report view for long;
    entries are at most a few hundred KB, so 1.5s is generous.
    """
    if not _redis_enabled():
        return None
    url = (os.environ.get("REDIS_URL") or "").strip()
    try:
        import redis  # type: ignore

        return redis.from_url(url, socket_timeout=1.5, socket_connect_timeout=1.0)
    except Exception:
        return None


def _redis_key(cache_key: str) -> str:
    return f"{AI_CACHE_REDIS_PREFIX}{cache_key}"


def _entry_fresh(obj: dict) -> bool:
    try:
        ts = float(obj.get("ts") or 0)
    except (TypeError, ValueError):
        return False
    ttl = obj.get("ttl") or AI_CACHE_TTL
    return time.time() - ts <= ttl


def _save_to_redis(cache_key: str, obj: dict) -> None:
    client = _redis_client()
    if client is None:
        return
    try:
        blob = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return
    if len(blob.encode("utf-8")) > _redis_max_bytes():
        logger.debug("[ai-cache] entry over Redis size cap; file copy only")
        return
    try:
        # Redis enforces the same per-entry TTL the file copy checks by hand.
        client.setex(_redis_key(cache_key), int(obj.get("ttl") or AI_CACHE_TTL), blob)
    except Exception:
        logger.debug("[ai-cache] redis save skipped", exc_info=True)


def _load_from_redis(cache_key: str) -> dict | None:
    """The stored envelope from Redis, or None on any miss/failure."""
    client = _redis_client()
    if client is None:
        return None
    try:
        raw = client.get(_redis_key(cache_key))
    except Exception:
        logger.debug("[ai-cache] redis load skipped", exc_info=True)
        return None
    if not raw:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict) or "content" not in obj:
        return None
    # Redis expiry should already have dropped stale entries; check the
    # envelope too so a fallback entry can never outlive its short TTL.
    if not _entry_fresh(obj):
        return None
    return obj


# ── File layer ───────────────────────────────────────────────────────────────

def _write_file_entry(cache_key: str, obj: dict) -> None:
    path = AI_CACHE_DIR / f"{cache_key}.json"
    # Write to a temp file then atomically rename so a crash mid-write can't
    # leave a half-written (corrupt) cache entry.
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)


def _load_from_file(cache_key: str) -> dict | None:
    path = AI_CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict) or not _entry_fresh(obj):
        return None
    return obj


# ── Public API (unchanged signatures) ────────────────────────────────────────

def load_cached_ai_text(cache_key: str) -> str | None:
    # Every store the save path writes to is consulted: the local file first
    # (fast, same-process), then the durable Redis copy. A Redis hit backfills
    # the file verbatim (same ts/ttl, so the entry's expiry does not move) so
    # later reads on this disk stay local.
    obj = _load_from_file(cache_key)
    if obj is not None:
        return obj.get("content")
    obj = _load_from_redis(cache_key)
    if obj is None:
        return None
    _write_file_entry(cache_key, obj)
    return obj.get("content")


def save_cached_ai_text(cache_key: str, content: str, *, metadata: dict | None = None,
                        ttl: int | None = None) -> None:
    obj = {
        "ts": time.time(),
        "ttl": ttl if ttl is not None else AI_CACHE_TTL,
        "content": content,
    }
    if metadata is not None:
        obj["metadata"] = metadata
    _write_file_entry(cache_key, obj)
    _save_to_redis(cache_key, obj)
