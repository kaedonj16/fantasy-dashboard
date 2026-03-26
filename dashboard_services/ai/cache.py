from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

AI_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "ai_cache"
AI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

AI_CACHE_TTL = 60 * 60 * 12  # 12 hours


def _hash_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_ai_cache_key(artifact_type: str, payload: dict, prompt_version: str = "v1") -> str:
    payload_hash = _hash_payload(payload)
    return f"{artifact_type}_{prompt_version}_{payload_hash}"


def load_cached_ai_text(cache_key: str) -> str | None:
    path = AI_CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        ts = float(obj.get("ts") or 0)
        if time.time() - ts > AI_CACHE_TTL:
            return None
        return obj.get("content")
    except Exception:
        return None


def save_cached_ai_text(cache_key: str, content: str) -> None:
    path = AI_CACHE_DIR / f"{cache_key}.json"
    obj = {
        "ts": time.time(),
        "content": content,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
