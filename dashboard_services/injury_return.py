"""ESPN injury return dates, overlaid on waiver vacancy duration.

Sleeper only publishes a status class (IR / OUT / …), so vacancy length used
to be a fixed guess per class. ESPN's public injury report includes an optional
``returnDate``; we scrape that league-wide report, map ESPN athlete ids to
canonical (Sleeper) ids, and expose weeks-remaining for the waiver model.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from utils.paths import CACHE_DIR

logger = logging.getLogger(__name__)

_ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}
_TTL = 6 * 3600
_CACHE: Dict[str, Any] = {"ts": 0.0, "by_pid": {}}
_CACHE_FILE = CACHE_DIR / "injury_return" / "espn_return_dates.json"


def _as_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def weeks_until_return(return_date, today: Optional[date] = None) -> Optional[float]:
    """Weeks from ``today`` until ``return_date``. ``None`` when unknown/past."""
    day = _as_date(return_date)
    if not day:
        return None
    now = today or date.today()
    delta = (day - now).days
    if delta <= 0:
        return 0.4  # listed return is due — still a this-week absence
    return round(delta / 7.0, 2)


def _injury_type_label(raw) -> str:
    if isinstance(raw, dict):
        return str(raw.get("description") or raw.get("name") or raw.get("id") or "").strip()
    return str(raw or "").strip()


def parse_espn_injuries_payload(payload: dict, espn_to_canon: Optional[dict] = None) -> Dict[str, dict]:
    """Flatten ESPN's team-grouped injury report to canonical_id -> row."""
    xwalk = {str(k): str(v) for k, v in (espn_to_canon or {}).items() if k and v}
    out: Dict[str, dict] = {}
    groups = []
    if isinstance(payload, dict):
        groups = payload.get("injuries") or payload.get("items") or []
    for group in groups or []:
        if not isinstance(group, dict):
            continue
        entries = group.get("injuries") or group.get("items") or []
        if isinstance(group.get("athlete"), dict) and not entries:
            entries = [group]
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            athlete = entry.get("athlete") or entry.get("player") or {}
            if not isinstance(athlete, dict):
                athlete = {}
            espn_id = str(athlete.get("id") or entry.get("id") or "").strip()
            details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
            return_raw = (
                details.get("returnDate")
                or details.get("return_date")
                or details.get("date")
                or entry.get("returnDate")
                or entry.get("date")
            )
            status = str(
                entry.get("status")
                or (details.get("fantasyStatus") or {}).get("description")
                or ""
            ).strip()
            pid = xwalk.get(espn_id) or ""
            if not pid and espn_id:
                pid = espn_id
            if not pid:
                continue
            row = {
                "espn_id": espn_id,
                "player_id": pid,
                "name": athlete.get("displayName") or athlete.get("fullName") or "",
                "status": status,
                "return_date": str(return_raw)[:10] if return_raw else None,
                "type": _injury_type_label(details.get("type") or entry.get("type")),
            }
            # Prefer a row that actually has a return date when ESPN lists duplicates.
            prev = out.get(pid)
            if prev and prev.get("return_date") and not row["return_date"]:
                continue
            out[pid] = row
    return out


def _load_disk() -> Dict[str, dict]:
    try:
        if _CACHE_FILE.exists() and (time.time() - _CACHE_FILE.stat().st_mtime) < _TTL:
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        logger.debug("injury_return disk load failed", exc_info=True)
    return {}


def _save_disk(by_pid: Dict[str, dict]) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(by_pid), encoding="utf-8")
    except Exception:
        logger.debug("injury_return disk save failed", exc_info=True)


def _fetch_espn_json() -> dict:
    import urllib.request

    req = urllib.request.Request(_ESPN_INJURIES_URL, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=12) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def refresh_espn_return_dates(*, force: bool = False) -> Dict[str, dict]:
    """Fetch ESPN's league injury report and cache canonical-id rows."""
    now = time.time()
    if not force and _CACHE.get("by_pid") and (now - float(_CACHE.get("ts") or 0)) < _TTL:
        return _CACHE["by_pid"]
    disk = _load_disk()
    if not force and disk:
        _CACHE["by_pid"] = disk
        _CACHE["ts"] = now
        return disk
    try:
        from dashboard_services.providers.global_adp import espn_id_to_canonical
        xwalk = espn_id_to_canonical()
    except Exception:
        xwalk = {}
    try:
        payload = _fetch_espn_json()
    except Exception:
        logger.warning("espn injury fetch failed", exc_info=True)
        if disk:
            _CACHE["by_pid"] = disk
            _CACHE["ts"] = now
            return disk
        return {}
    by_pid = parse_espn_injuries_payload(payload, xwalk)
    _CACHE["by_pid"] = by_pid
    _CACHE["ts"] = now
    _save_disk(by_pid)
    return by_pid


def get_return_date(player_id: str) -> Optional[str]:
    pid = str(player_id or "").strip()
    if not pid:
        return None
    row = (refresh_espn_return_dates() or {}).get(pid) or {}
    return row.get("return_date")


def weeks_out_for_player(player_id: str, today: Optional[date] = None) -> Optional[float]:
    """ESPN-derived weeks remaining, or ``None`` when ESPN has no return date."""
    return weeks_until_return(get_return_date(player_id), today=today)
