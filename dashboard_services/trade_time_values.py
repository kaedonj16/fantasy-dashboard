"""Persist player values at the moment a trade happened.

Trade Outcome used to pick the closest daily snapshot within 30 days, so a
gap in ``player_value_history`` quietly mis-priced the deal. This table stores
the value we actually observed (or backfilled) for each player on the trade
date, and Outcome reads it first.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, Iterable, Optional

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)


def init_trade_time_values_db() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_time_values (
                    player_id TEXT NOT NULL,
                    as_of_date DATE NOT NULL,
                    value NUMERIC,
                    value_sf NUMERIC,
                    source TEXT NOT NULL DEFAULT 'snapshot',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (player_id, as_of_date)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trade_time_values_date
                ON trade_time_values (as_of_date DESC)
                """
            )


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


def snapshot_trade_values(
    player_ids: Iterable[str],
    as_of,
    *,
    values: Optional[Dict[str, float]] = None,
    values_sf: Optional[Dict[str, float]] = None,
    source: str = "snapshot",
) -> int:
    """Upsert one row per player for ``as_of``. Skips empty ids and non-positive values."""
    day = _as_date(as_of)
    if not day:
        return 0
    values = values or {}
    values_sf = values_sf or {}
    rows = []
    for raw in player_ids or []:
        pid = str(raw or "").strip()
        if not pid:
            continue
        try:
            val = float(values.get(pid) or 0)
        except (TypeError, ValueError):
            val = 0.0
        try:
            sf = float(values_sf.get(pid) or 0) or None
        except (TypeError, ValueError):
            sf = None
        if val <= 0 and not sf:
            continue
        rows.append((pid, day, val if val > 0 else None, sf, source))
    if not rows:
        return 0
    init_trade_time_values_db()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO trade_time_values (player_id, as_of_date, value, value_sf, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (player_id, as_of_date) DO UPDATE SET
                    value = COALESCE(EXCLUDED.value, trade_time_values.value),
                    value_sf = COALESCE(EXCLUDED.value_sf, trade_time_values.value_sf),
                    source = EXCLUDED.source
                """,
                rows,
            )
    return len(rows)


def get_trade_time_value(player_id: str, as_of) -> Optional[float]:
    """Exact-date lookup. ``None`` when we have no persisted row."""
    row = get_trade_time_row(player_id, as_of)
    return None if row is None else row[0]


def get_trade_time_row(player_id: str, as_of) -> Optional[tuple]:
    """Exact-date lookup as ``(value, source)`` or ``None``."""
    day = _as_date(as_of)
    pid = str(player_id or "").strip()
    if not day or not pid:
        return None
    try:
        init_trade_time_values_db()
        with get_conn() as conn:
            row = conn.execute(
                "SELECT value, value_sf, source FROM trade_time_values "
                "WHERE player_id = %s AND as_of_date = %s",
                (pid, day),
            ).fetchone()
        if not row:
            return None
        source = str(row["source"] or "snapshot")
        for key in ("value", "value_sf"):
            try:
                v = float(row[key] or 0)
            except (TypeError, ValueError, KeyError):
                continue
            if v > 0:
                return round(v, 1), source
        return None
    except Exception:
        logger.debug("trade_time_values lookup failed", exc_info=True)
        return None


def get_or_persist_trade_value(player_id: str, as_of) -> Optional[float]:
    """Outcome lookup: persisted row, then today's live table, then history."""
    hit = get_or_persist_trade_value_meta(player_id, as_of)
    return None if hit is None else hit[0]


def get_or_persist_trade_value_meta(player_id: str, as_of) -> Optional[tuple]:
    """Like ``get_or_persist_trade_value`` but returns ``(value, source)``."""
    val = get_trade_time_value(player_id, as_of)
    if val is not None:
        row = get_trade_time_row(player_id, as_of)
        return row if row is not None else (val, "snapshot")
    day = _as_date(as_of)
    pid = str(player_id or "").strip()
    if not day or not pid:
        return None
    if abs((date.today() - day).days) <= 1:
        ones, sfs = current_model_values()
        if pid in ones or pid in sfs:
            snapshot_trade_values(
                [pid], day, values=ones, values_sf=sfs, source="snapshot",
            )
            val = get_trade_time_value(pid, day)
            if val is not None:
                row = get_trade_time_row(pid, day)
                return row if row is not None else (val, "snapshot")
    persisted = persist_from_history(pid, day)
    if persisted is None:
        return None
    row = get_trade_time_row(pid, day)
    return row if row is not None else (persisted, "backfill")


def persist_from_history(player_id: str, as_of, *, max_gap_days: int = 30) -> Optional[float]:
    """Copy the closest daily history row onto ``trade_time_values`` and return it.

    Exact-date hits are stored as ``backfill``; a nearest-neighbor within
    ``max_gap_days`` is stored as ``backfill-nearest`` so Outcome stops rolling
    the dice on later history gaps.
    """
    day = _as_date(as_of)
    pid = str(player_id or "").strip()
    if not day or not pid:
        return None
    hit = get_trade_time_value(pid, day)
    if hit is not None:
        return hit
    try:
        from dashboard_services.player_value_history import get_player_value_history
        history = get_player_value_history(pid, days=800) or []
    except Exception:
        logger.debug("trade_time_values history load failed", exc_info=True)
        return None
    best_val = None
    best_diff = None
    for snap in history:
        snap_day = _as_date(snap.get("as_of_date"))
        if not snap_day:
            continue
        try:
            val = float(snap.get("value") or 0)
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        diff = abs((snap_day - day).days)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_val = val
    if best_val is None or best_diff is None or best_diff > int(max_gap_days):
        return None
    source = "backfill" if best_diff == 0 else "backfill-nearest"
    snapshot_trade_values([pid], day, values={pid: best_val}, source=source)
    return round(best_val, 1)


def backfill_from_trade_intel(limit: int = 5000) -> int:
    """Fill missing trade-date rows from crawled trades + daily value history."""
    init_trade_time_values_db()
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT a.player_id, t.created_at::date AS as_of_date
                FROM trade_intel_assets a
                JOIN trade_intel_trades t ON t.id = a.trade_id
                WHERE a.asset_type = 'player'
                  AND a.player_id IS NOT NULL
                  AND t.created_at IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM trade_time_values v
                      WHERE v.player_id = a.player_id
                        AND v.as_of_date = t.created_at::date
                  )
                ORDER BY t.created_at DESC
                LIMIT %s
                """,
                (int(limit),),
            ).fetchall()
    except Exception:
        logger.debug("trade_time_values backfill query failed", exc_info=True)
        return 0
    n = 0
    for row in rows or []:
        if persist_from_history(row["player_id"], row["as_of_date"]) is not None:
            n += 1
    return n


def current_model_values() -> tuple[Dict[str, float], Dict[str, float]]:
    """Live 1QB / Superflex values for snapshotting a trade that just happened."""
    ones: Dict[str, float] = {}
    sfs: Dict[str, float] = {}
    try:
        from dashboard_services.player_value_history import load_current_values_from_db
        rows = load_current_values_from_db() or []
    except Exception:
        logger.debug("current_model_values load failed", exc_info=True)
        return ones, sfs
    for p in rows:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if not pid:
            continue
        try:
            v = float(p.get("value") or 0)
        except (TypeError, ValueError):
            v = 0.0
        try:
            sf = float(p.get("sf_value") or p.get("value_sf") or 0)
        except (TypeError, ValueError):
            sf = 0.0
        if v > 0:
            ones[pid] = v
        if sf > 0:
            sfs[pid] = sf
    return ones, sfs
