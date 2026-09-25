"""Shared pipeline-health persistence for cron_daily and the web app.

The cron container's disk is invisible to the web container on Render, so
``cron_daily`` POSTs each step's status to the CRON_SECRET-authenticated
``/api/cron/pipeline-health`` web endpoint, and the web process persists it
with :func:`write_step_health` into its own ``CACHE_DIR/pipeline_health.json``,
which ``/api/health/pipeline`` reads back.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from utils.paths import CACHE_DIR

HEALTH_FILENAME = "pipeline_health.json"

_VALID_STATUSES = ("ok", "error", "timeout", "skipped")


def health_path(cache_dir: Path | None = None) -> Path:
    return (cache_dir or CACHE_DIR) / HEALTH_FILENAME


def write_step_health(
    step_name: str,
    status: str,
    at: str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Merge one step's status into pipeline_health.json; return the full payload."""
    dest = health_path(cache_dir)
    data: dict = {}
    try:
        if dest.exists():
            data = json.loads(dest.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    now = at or datetime.now(timezone.utc).isoformat()
    entry: dict = {"status": str(status), "at": now}
    if status == "ok":
        entry["last_success"] = now
    else:
        # Keep the previous last_success so "last succeeded at" survives errors.
        prev = data.get(str(step_name)) or {}
        if isinstance(prev, dict) and prev.get("last_success"):
            entry["last_success"] = prev["last_success"]
    data[str(step_name)] = entry
    data["_updated"] = now
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def read_health(cache_dir: Path | None = None) -> dict:
    """Read the pipeline-health payload; {} when missing or unreadable."""
    dest = health_path(cache_dir)
    try:
        if dest.exists():
            data = json.loads(dest.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}
