"""Fresh waiver pool: transactions delta over a cached league ctx (spec §5).

The waiver pool is derived as (model value table minus ctx rosters), and
the league ctx can be up to 12 hours old, so a player dropped an hour ago
still reads as rostered and never surfaces. This module fetches the
league's recent transactions through the provider's existing
transactions surface, keeps only the ones newer than the ctx build
timestamp (``ctx["_cache_synced_at"]``), and nets them into a small delta:

- ``adds``: canonical player ids that became rostered after the ctx build
  (they must leave the pool -- suggesting a player another team just added
  is worse than missing a drop),
- ``drops``: ``{player_id: dropped_at_iso}`` for players who became free
  after the ctx build (they enter the pool, with the drop time the
  "claim by" chip renders from).

The delta is cached per (platform, league_id, season, ctx build) for
120 s, matching the live-score cache TTL, so repeat loads cost at most
one provider sweep per league per two minutes. Every failure is soft:
any error yields an empty delta and callers derive exactly as before.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DELTA_TTL_SECONDS = 120.0
_DELTA_CACHE: dict = {}  # key -> (monotonic ts, delta dict)
_DELTA_CACHE_MAX = 128

_EMPTY_DELTA = {"adds": frozenset(), "drops": {}, "hash": ""}


def _parse_ts(value) -> Optional[datetime]:
    """Parse a ctx build timestamp (ISO string / datetime / epoch) to UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        try:
            v = float(value)
            if v > 1e11:  # epoch ms
                v /= 1000.0
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _tx_ts(tx: dict) -> Optional[datetime]:
    """Transaction timestamp as UTC datetime (providers use epoch ms)."""
    raw = tx.get("created")
    if raw is None:
        raw = tx.get("status_updated")
    if raw is None:
        return None
    return _parse_ts(raw)


def _default_fetch(platform: str, league_id: str, season: int,
                   weeks: list) -> dict:
    """Provider transactions via the shared service sweep (its own cache)."""
    from dashboard_services.service import get_transactions_by_week

    return get_transactions_by_week(
        league_id, list(weeks), platform=platform, season=int(season)) or {}


def _delta_weeks(ctx: dict) -> list:
    """Fantasy weeks that can hold transactions newer than the ctx build.

    Transactions are bucketed by fantasy week, and a move made today can
    land in the current or the just-turned week, so sweep a small window
    around the ctx's current week instead of the whole season.
    """
    try:
        cur = int(ctx.get("current_week") or 0)
    except (TypeError, ValueError):
        cur = 0
    if not cur:
        try:
            from dashboard_services.api import get_nfl_state

            state = get_nfl_state() or {}
            cur = int(state.get("week") or state.get("display_week") or 1)
        except Exception:
            cur = 1
    return [w for w in (cur - 1, cur, cur + 1) if 1 <= w <= 18]


def build_delta(transactions, ctx_ts: Optional[datetime]) -> dict:
    """Net a flat transaction list into adds/drops newer than ``ctx_ts``.

    Pure (no I/O) so the netting rules are unit-testable: events apply in
    timestamp order, net state wins, failed/pending transactions and
    transactions at or before the ctx build are ignored. Without a ctx
    build timestamp there is no freshness baseline, so the delta is
    empty (callers keep today's behavior).
    """
    if ctx_ts is None:
        return {"adds": set(), "drops": {}, "hash": ""}
    events = []  # (ts, seq, pid, kind)
    seq = 0
    for tx in transactions or []:
        if not isinstance(tx, dict):
            continue
        status = str(tx.get("status") or "").strip().lower()
        if status and status not in ("complete", "completed", "successful",
                                     "success"):
            continue
        ts = _tx_ts(tx)
        if ts is None or ts <= ctx_ts:
            continue
        for pid in (tx.get("adds") or {}):
            events.append((ts, seq, str(pid), "add"))
            seq += 1
        for pid in (tx.get("drops") or {}):
            events.append((ts, seq, str(pid), "drop"))
            seq += 1
    events.sort(key=lambda e: (e[0], e[1]))
    adds: set = set()
    drops: dict = {}
    for ts, _seq, pid, kind in events:
        if kind == "add":
            adds.add(pid)
            drops.pop(pid, None)
        else:
            adds.discard(pid)
            drops[pid] = ts.isoformat()
    digest = hashlib.sha256(
        repr((sorted(adds), sorted(drops.items()))).encode("utf-8")
    ).hexdigest()
    return {"adds": adds, "drops": drops, "hash": digest}


def get_transactions_delta(platform: str, league_id: str, season: int,
                           ctx: dict, *,
                           fetch_fn: Optional[Callable] = None,
                           weeks: Optional[list] = None) -> dict:
    """Cached transactions delta for one league ctx. Never raises."""
    try:
        ctx_ts = _parse_ts((ctx or {}).get("_cache_synced_at"))
        if ctx_ts is None or not league_id:
            return {"adds": set(), "drops": {}, "hash": ""}
        key = (str(platform), str(league_id), int(season),
               ctx_ts.isoformat())
        now = time.monotonic()
        hit = _DELTA_CACHE.get(key)
        if hit and (now - hit[0]) < DELTA_TTL_SECONDS:
            cached = hit[1]
            return {"adds": set(cached["adds"]),
                    "drops": dict(cached["drops"]), "hash": cached["hash"]}
        sweep = list(weeks) if weeks else _delta_weeks(ctx or {})
        fetch = fetch_fn or _default_fetch
        by_week = fetch(str(platform), str(league_id), int(season), sweep)
        flat = []
        if isinstance(by_week, dict):
            for w in sweep:
                flat.extend(by_week.get(w) or by_week.get(str(w)) or [])
        elif isinstance(by_week, list):
            flat = list(by_week)
        delta = build_delta(flat, ctx_ts)
        if len(_DELTA_CACHE) >= _DELTA_CACHE_MAX:
            _DELTA_CACHE.clear()
        _DELTA_CACHE[key] = (now, {"adds": set(delta["adds"]),
                                   "drops": dict(delta["drops"]),
                                   "hash": delta["hash"]})
        return delta
    except Exception:
        logger.debug("[waiver-freshness] delta failed for %s:%s",
                     platform, league_id, exc_info=True)
        return {"adds": set(), "drops": {}, "hash": ""}


def apply_delta_to_rostered(rostered_ids, delta: dict) -> set:
    """Rostered set corrected by the delta: drops out, adds in."""
    out = {str(p) for p in (rostered_ids or set())}
    if not delta:
        return out
    out -= set((delta.get("drops") or {}).keys())
    out |= {str(p) for p in (delta.get("adds") or set())}
    return out


def dropped_within(delta: dict, hours: float = 48.0,
                   now: Optional[datetime] = None) -> dict:
    """``{player_id: dropped_at_iso}`` for drops inside the last ``hours``."""
    out = {}
    if not delta:
        return out
    ref = now or datetime.now(timezone.utc)
    cutoff = ref - timedelta(hours=float(hours))
    for pid, iso in (delta.get("drops") or {}).items():
        ts = _parse_ts(iso)
        if ts is not None and ts >= cutoff:
            out[str(pid)] = iso
    return out


# ── Waiver process ("claim by") ─────────────────────────────────────────────

_WEEKDAY_INDEX = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3,
    "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def _parse_process_time(raw) -> Optional[tuple]:
    """(hour, minute) from '03:00', '3:00 AM', or an ISO/time string."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.hour, raw.minute
    text = str(raw).strip().upper()
    if not text:
        return None
    meridiem = None
    if text.endswith("AM"):
        meridiem, text = "AM", text[:-2].strip()
    elif text.endswith("PM"):
        meridiem, text = "PM", text[:-2].strip()
    if "T" in text:  # ISO datetime: take the time part
        text = text.split("T", 1)[1]
    parts = text.replace(":", " ").split()
    try:
        hour = int(float(parts[0]))
        minute = int(float(parts[1])) if len(parts) > 1 else 0
    except (TypeError, ValueError, IndexError):
        return None
    if meridiem == "PM" and hour < 12:
        hour += 12
    elif meridiem == "AM" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _process_weekdays(raw) -> Optional[set]:
    """Allowed weekday ints (Mon=0) from 'daily', a list, or a CSV string.

    None means every day (daily / unspecified)."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        days = set()
        for item in raw:
            sub = _process_weekdays(item)
            if sub is None:
                return None
            days |= sub
        return days or None
    text = str(raw).strip().lower()
    if not text or text in ("daily", "every day", "all"):
        return None
    days = set()
    for token in text.replace(",", " ").split():
        if token in _WEEKDAY_INDEX:
            days.add(_WEEKDAY_INDEX[token])
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if 0 <= value <= 6:
            days.add(value)
    return days or None


def waiver_process_info(settings: Optional[dict],
                        now: Optional[datetime] = None) -> Optional[dict]:
    """{type, next_process_at} from league waiver settings, or None.

    Only computed when the settings actually carry a waiver type and a
    process time (Workstream 1's rules overlay supplies them as
    ``waiver_type`` / ``waiver_process_time`` / ``waiver_process_days`` /
    ``waiver_process_tz``); process times are never guessed, so absent
    keys yield None and the chip renders without a time.
    """
    settings = settings or {}
    raw_type = str(settings.get("waiver_type") or "").strip().lower()
    if not raw_type:
        return None
    wtype = "faab" if "faab" in raw_type else "rolling"
    parsed = _parse_process_time(
        settings.get("waiver_process_time") or settings.get("waiver_time"))
    if parsed is None:
        return None
    hour, minute = parsed
    tz_name = str(settings.get("waiver_process_tz")
                  or settings.get("waiver_tz") or "America/New_York")
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name)
    except Exception:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    local = ref.astimezone(tz)
    days = _process_weekdays(
        settings.get("waiver_process_days") or settings.get("waiver_days"))
    candidate = local.replace(hour=hour, minute=minute, second=0,
                              microsecond=0)
    for _ in range(9):
        if candidate > local and (days is None
                                  or candidate.weekday() in days):
            return {"type": wtype,
                    "next_process_at": candidate.isoformat()}
        candidate += timedelta(days=1)
    return {"type": wtype, "next_process_at": None}
