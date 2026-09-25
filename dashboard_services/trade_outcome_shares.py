"""Shareable public links for trade-outcome results ("who won the trade").

A share stores the *frozen* outcome payload the /api/trade-outcome endpoint
returned (verdict, totals, per-asset then/now values) plus the two team names,
so the public card keeps showing the verdict exactly as shared. The stored
payload exposes only team names and asset names/values -- no league internals
or user identity.

Lifecycle mirrors the trade-card shares (shared_trades): short random ids and
a probabilistic 5-day prune.

Payloads live in Postgres, not in process memory.
"""
from __future__ import annotations

import logging
import random
import secrets

logger = logging.getLogger(__name__)

_TABLES_READY = False

#: Shares older than this are pruned (probabilistically, on save).
SHARE_TTL_SQL = "INTERVAL '5 days'"

#: Max rows per side accepted on save -- outcome payloads are small by nature.
MAX_ASSETS_PER_SIDE = 30


def init_outcome_shares_table() -> None:
    """Create the trade_outcome_shares table once per process."""
    global _TABLES_READY
    if _TABLES_READY:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_outcome_shares (
                    share_id   TEXT PRIMARY KEY,
                    params     TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.commit()
        _TABLES_READY = True
    except Exception as exc:
        logger.warning("[trade-outcome-share] table init failed: %s", exc)


def _sanitize_row(row: dict) -> dict:
    row = row if isinstance(row, dict) else {}
    def _num(v):
        try:
            return round(float(v), 1)
        except (TypeError, ValueError):
            return None
    return {
        "name": str(row.get("name") or "")[:80],
        "is_pick": bool(row.get("is_pick")),
        "value_then": _num(row.get("value_then")),
        "value_now": _num(row.get("value_now")) or 0.0,
        "delta": _num(row.get("delta")),
    }


def sanitize_outcome_params(data: dict) -> dict:
    """Whitelist/shape the client payload into the stored share params."""
    data = data if isinstance(data, dict) else {}
    def _num(v, default=0.0):
        try:
            return round(float(v), 1)
        except (TypeError, ValueError):
            return default
    received = data.get("received") or []
    sent = data.get("sent") or []
    verdict = str(data.get("verdict") or "EVEN").upper()
    if verdict not in ("WIN", "LOSS", "EVEN"):
        verdict = "EVEN"
    return {
        "team_a": str(data.get("team_a") or "Team A")[:60],
        "team_b": str(data.get("team_b") or "Team B")[:60],
        "trade_date": str(data.get("trade_date") or "")[:10],
        "verdict": verdict,
        "net_delta_now": _num(data.get("net_delta_now")),
        "total_a_now": _num(data.get("total_a_now", data.get("total_received_now"))),
        "total_b_now": _num(data.get("total_b_now", data.get("total_sent_now"))),
        "then_estimated": bool(data.get("then_estimated")),
        "a_rows": [_sanitize_row(r) for r in received[:MAX_ASSETS_PER_SIDE]],
        "b_rows": [_sanitize_row(r) for r in sent[:MAX_ASSETS_PER_SIDE]],
    }


def create_outcome_share(params: dict) -> str:
    """Store sanitized outcome params and return the share id. Raises on DB errors."""
    import json as _json
    from dashboard_services.db import get_conn
    init_outcome_shares_table()
    share_id = secrets.token_urlsafe(6)
    payload = _json.dumps(params)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO trade_outcome_shares (share_id, params) VALUES (%s, %s) "
            "ON CONFLICT (share_id) DO NOTHING",
            (share_id, payload),
        )
        # Probabilistic cleanup: prune shares older than 5 days (~10% of saves)
        if random.random() < 0.10:
            conn.execute(
                "DELETE FROM trade_outcome_shares WHERE created_at < NOW() - " + SHARE_TTL_SQL
            )
        conn.commit()
    return share_id


def get_outcome_share(share_id: str) -> dict | None:
    """Fetch a share by id; None when unknown. Returns the parsed params dict."""
    import json as _json
    sid = (share_id or "").strip()
    if not sid or len(sid) > 128:
        return None
    try:
        init_outcome_shares_table()
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            row = conn.execute(
                "SELECT params, created_at FROM trade_outcome_shares WHERE share_id = %s",
                (sid,),
            ).fetchone()
    except Exception as exc:
        logger.warning("[trade-outcome-share] lookup error for %s: %s", sid, exc)
        return None
    if not row:
        return None
    try:
        _row = row if hasattr(row, "__getitem__") else {"params": row[0], "created_at": row[1]}
        return {"params": _json.loads(_row["params"]), "created_at": _row.get("created_at") if hasattr(_row, "get") else row[1]}
    except Exception:
        return None
