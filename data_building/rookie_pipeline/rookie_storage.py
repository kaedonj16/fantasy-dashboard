from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils.paths import DATA_DIR


def _json_default(obj):
    """
    Custom JSON serializer for types json.dumps can't handle natively.
    Mirrors the encoder in dashboard_services/db.py for consistency.
    """
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


UTC = timezone.utc


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class CacheReadResult:
    payload: Optional[Dict[str, Any]]
    is_stale: bool
    cache_path: Path


class RookieDiskCache:
    """Simple disk cache with TTL and stale fallback support."""

    def __init__(
        self,
        root: Optional[Path] = None,
        historical_ttl_seconds: int = 60 * 60 * 24 * 30,
        live_market_ttl_seconds: int = 60 * 60 * 6,
    ) -> None:
        self.root = (root or (DATA_DIR / "cache" / "rookie_sources")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.historical_ttl_seconds = historical_ttl_seconds
        self.live_market_ttl_seconds = live_market_ttl_seconds

    @staticmethod
    def _safe_component(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)

    def _cache_path(self, source_name: str, season: int, player_key: str) -> Path:
        source = self._safe_component(source_name)
        player = self._safe_component(player_key)
        return self.root / source / str(season) / f"{player}.json"

    def _ttl_for_source_type(self, source_type: str) -> int:
        return self.live_market_ttl_seconds if source_type == "draft_market" else self.historical_ttl_seconds

    def read(
        self,
        source_name: str,
        season: int,
        player_key: str,
        source_type: str,
    ) -> CacheReadResult:
        path = self._cache_path(source_name, season, player_key)
        if not path.exists():
            return CacheReadResult(payload=None, is_stale=False, cache_path=path)

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return CacheReadResult(payload=None, is_stale=False, cache_path=path)

        cached_at = raw.get("cached_at")
        if not cached_at:
            return CacheReadResult(payload=raw, is_stale=True, cache_path=path)

        try:
            ts = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
            age_seconds = (datetime.now(UTC) - ts).total_seconds()
        except ValueError:
            return CacheReadResult(payload=raw, is_stale=True, cache_path=path)

        ttl = self._ttl_for_source_type(source_type)
        return CacheReadResult(payload=raw, is_stale=age_seconds > ttl, cache_path=path)

    def write(
        self,
        source_name: str,
        season: int,
        player_key: str,
        payload: Dict[str, Any],
    ) -> Path:
        path = self._cache_path(source_name, season, player_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(payload)
        data["cached_at"] = utc_now_iso()
        path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        return path


def write_rookie_snapshot(file_prefix: str, as_of_date: str, payload: Dict[str, Any]) -> Tuple[Path, Path]:
    """Write the latest rookie snapshot file under data/.

    Returns (latest, latest) for backwards compatibility with callers that
    unpack two paths.
    """
    import time as _time
    latest = DATA_DIR / f"{file_prefix}_latest.json"

    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)

    # Only refresh _latest once per calendar day
    already_fresh = (
        latest.exists()
        and _time.time() - latest.stat().st_mtime < 86400
        and latest.stat().st_mtime // 86400 == _time.time() // 86400
    )
    if not already_fresh:
        latest.write_text(text, encoding="utf-8")
    return latest, latest

