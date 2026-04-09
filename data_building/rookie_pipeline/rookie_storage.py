from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from data_building.paths import DATA_DIR


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
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return path


def write_rookie_snapshot(file_prefix: str, as_of_date: str, payload: Dict[str, Any]) -> Tuple[Path, Path]:
    """Write dated and latest rookie snapshot files under data/."""
    dated = DATA_DIR / f"{file_prefix}_{as_of_date}.json"
    latest = DATA_DIR / f"{file_prefix}_latest.json"

    text = json.dumps(payload, indent=2, sort_keys=True)
    dated.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    return dated, latest


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
