"""Low-overhead, opt-in process memory snapshots for startup logs."""
from __future__ import annotations

import os
import sys


def current_rss_mb() -> float:
    """Read current RSS without adding a psutil dependency."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
        except (ImportError, OSError):
            return 0.0


def cache_entry_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    api = sys.modules.get("dashboard_services.api")
    if api is not None and hasattr(api, "ttl_cache_entry_counts"):
        counts["dashboard_ttl"] = sum(api.ttl_cache_entry_counts().values())
    utils = sys.modules.get("utils.utils")
    if utils is not None and hasattr(utils, "_JSON_CACHE"):
        lock = getattr(utils, "_JSON_CACHE_LOCK", None)
        if lock:
            with lock:
                counts["parsed_json"] = len(utils._JSON_CACHE)
        else:
            counts["parsed_json"] = len(utils._JSON_CACHE)
    return counts


def format_memory_snapshot(label: str) -> str:
    counts = cache_entry_counts()
    cache_text = ",".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"
    return f"[memory] {label} pid={os.getpid()} rss={current_rss_mb():.1f}MiB caches={cache_text}"
