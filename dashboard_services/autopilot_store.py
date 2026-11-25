"""Persistence for Autopilot Phase 1: settings, kill switch, plans, audit.

Tables (migration 060_autopilot.sql; also created idempotently here,
following the accounts.py / start_score_bundle pattern):

- ``autopilot_settings``: one row per user_leagues identity
  (account_id, platform, league_id, season): mode, guardrail toggles,
  notification preferences, protected player ids, paused flag. Waiver and
  FAAB columns exist now so Phase 1.5 needs no second migration, but the
  Phase 1 write path refuses to set them.
- ``autopilot_account_state``: the account-wide kill switch.
- ``autopilot_plans``: one open plan per league at a time (guardrail 9);
  building a new plan supersedes the old one. Swap groups and the raw
  LINEUP items live as JSONB on the plan row.
- ``autopilot_audit``: append-only event log (guardrail 10). There is no
  update or delete function for it in this module, by design.

All functions take the acting account's id explicitly; nothing here reads
Flask session state.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TABLES_READY = False

MODES = ("off", "approve", "auto")
PLAN_STATUSES = ("open", "executed", "partial", "failed", "superseded", "already_applied")
AUDIT_ACTIONS = ("planned", "approved", "executed", "skipped", "failed")

#: Settings a caller may write in Phase 1. Waivers / FAAB keys are refused
#: (Phase 1.5) so a stored value can never imply claims are supported.
_WRITABLE_SETTINGS = {
    "mode",
    "lineups_enabled",
    "problem_slots_only",
    "notify_push",
    "notify_email",
    "protected_player_ids",
    "paused",
}
_PHASE_15_SETTINGS = {"waivers_enabled", "faab_cap_per_claim", "faab_cap_per_week"}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "mode": "off",
    "lineups_enabled": True,
    "problem_slots_only": False,
    "notify_push": True,
    "notify_email": True,
    "protected_player_ids": [],
    "paused": False,
    "waivers_enabled": False,
    "faab_cap_per_claim": None,
    "faab_cap_per_week": None,
}


def _conn():
    from dashboard_services.db import get_conn

    return get_conn()


def init_autopilot_tables() -> None:
    """Create the Autopilot tables once per process. Never raises."""
    global _TABLES_READY
    if _TABLES_READY:
        return
    try:
        with _conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autopilot_settings (
                    id                  SERIAL PRIMARY KEY,
                    account_id          INTEGER NOT NULL,
                    platform            TEXT NOT NULL,
                    league_id           TEXT NOT NULL,
                    season              INTEGER NOT NULL,
                    mode                TEXT NOT NULL DEFAULT 'off',
                    lineups_enabled     BOOLEAN NOT NULL DEFAULT TRUE,
                    problem_slots_only  BOOLEAN NOT NULL DEFAULT FALSE,
                    notify_push         BOOLEAN NOT NULL DEFAULT TRUE,
                    notify_email        BOOLEAN NOT NULL DEFAULT TRUE,
                    protected_player_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                    paused              BOOLEAN NOT NULL DEFAULT FALSE,
                    waivers_enabled     BOOLEAN NOT NULL DEFAULT FALSE,
                    faab_cap_per_claim  INTEGER,
                    faab_cap_per_week   INTEGER,
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT valid_autopilot_mode CHECK (mode IN ('off', 'approve', 'auto')),
                    UNIQUE (account_id, platform, league_id, season)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autopilot_settings_account "
                "ON autopilot_settings(account_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autopilot_account_state (
                    account_id  INTEGER PRIMARY KEY,
                    kill_switch BOOLEAN NOT NULL DEFAULT FALSE,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autopilot_plans (
                    id              SERIAL PRIMARY KEY,
                    account_id      INTEGER NOT NULL,
                    platform        TEXT NOT NULL,
                    league_id       TEXT NOT NULL,
                    season          INTEGER NOT NULL,
                    team_id         INTEGER NOT NULL,
                    scoring_period  INTEGER NOT NULL,
                    mode            TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'open',
                    snapshot        JSONB NOT NULL DEFAULT '{}'::jsonb,
                    snapshot_hash   TEXT NOT NULL,
                    swaps           JSONB NOT NULL DEFAULT '[]'::jsonb,
                    lineup_items    JSONB NOT NULL DEFAULT '[]'::jsonb,
                    issues          JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT valid_autopilot_plan_status CHECK (
                        status IN ('open', 'executed', 'partial', 'failed',
                                   'superseded', 'already_applied')
                    )
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autopilot_plans_league "
                "ON autopilot_plans(account_id, platform, league_id, season, status)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autopilot_audit (
                    id              SERIAL PRIMARY KEY,
                    account_id      INTEGER NOT NULL,
                    platform        TEXT NOT NULL,
                    league_id       TEXT NOT NULL,
                    season          INTEGER NOT NULL,
                    plan_id         INTEGER,
                    scoring_period  INTEGER,
                    action          TEXT NOT NULL,
                    mode            TEXT,
                    player_id       INTEGER,
                    player_name     TEXT,
                    from_slot       INTEGER,
                    to_slot         INTEGER,
                    score_delta     DOUBLE PRECISION,
                    detail          TEXT,
                    verification    TEXT,
                    espn_status     TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT valid_autopilot_audit_action CHECK (
                        action IN ('planned', 'approved', 'executed', 'skipped', 'failed')
                    )
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autopilot_audit_league "
                "ON autopilot_audit(account_id, platform, league_id, season, created_at DESC)"
            )
            conn.commit()
        _TABLES_READY = True
    except Exception:
        logger.debug("init_autopilot_tables failed", exc_info=True)


def snapshot_hash(snapshot: Dict[Any, Any], scoring_period: int) -> str:
    """Stable hash of a {player_id: slot_id} lineup snapshot + period."""
    items = sorted((int(k), int(v)) for k, v in (snapshot or {}).items())
    raw = f"{int(scoring_period)}|" + ",".join(f"{p}:{s}" for p, s in items)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _as_json_list(value) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    return list(value) if isinstance(value, (list, tuple)) else []


def _settings_from_row(row) -> Dict[str, Any]:
    out = dict(DEFAULT_SETTINGS)
    if row is None:
        return out
    for key in DEFAULT_SETTINGS:
        if key in row and row[key] is not None:
            out[key] = row[key]
    out["protected_player_ids"] = [
        int(p) for p in _as_json_list(out["protected_player_ids"])
        if str(p).lstrip("-").isdigit()
    ]
    return out


# ---------------------------------------------------------------------------
# Settings + kill switch
# ---------------------------------------------------------------------------

def get_settings(account_id, platform, league_id, season) -> Dict[str, Any]:
    """Stored settings for one league identity, defaults when unset."""
    init_autopilot_tables()
    with _conn() as conn:
        row = conn.execute(
            """SELECT * FROM autopilot_settings
               WHERE account_id = %s AND platform = %s AND league_id = %s
                 AND season = %s""",
            (int(account_id), str(platform), str(league_id), int(season)),
        ).fetchone()
    return _settings_from_row(row)


def save_settings(account_id, platform, league_id, season, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Upsert settings. Unknown keys raise; Phase 1.5 keys are refused."""
    updates = dict(updates or {})
    refused = _PHASE_15_SETTINGS & set(updates)
    if refused:
        raise ValueError(
            f"Phase 1 cannot set waiver/FAAB settings: {sorted(refused)} (Phase 1.5)"
        )
    unknown = set(updates) - _WRITABLE_SETTINGS
    if unknown:
        raise ValueError(f"Unknown autopilot settings: {sorted(unknown)}")
    if "mode" in updates and updates["mode"] not in MODES:
        raise ValueError(f"Invalid autopilot mode: {updates['mode']!r}")
    if "protected_player_ids" in updates:
        updates["protected_player_ids"] = [
            int(p) for p in (updates["protected_player_ids"] or [])
        ]
    current = get_settings(account_id, platform, league_id, season)
    current.update(updates)
    init_autopilot_tables()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO autopilot_settings
                   (account_id, platform, league_id, season, mode,
                    lineups_enabled, problem_slots_only, notify_push,
                    notify_email, protected_player_ids, paused, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, now())
               ON CONFLICT (account_id, platform, league_id, season)
               DO UPDATE SET mode = EXCLUDED.mode,
                             lineups_enabled = EXCLUDED.lineups_enabled,
                             problem_slots_only = EXCLUDED.problem_slots_only,
                             notify_push = EXCLUDED.notify_push,
                             notify_email = EXCLUDED.notify_email,
                             protected_player_ids = EXCLUDED.protected_player_ids,
                             paused = EXCLUDED.paused,
                             updated_at = now()""",
            (
                int(account_id), str(platform), str(league_id), int(season),
                current["mode"], bool(current["lineups_enabled"]),
                bool(current["problem_slots_only"]), bool(current["notify_push"]),
                bool(current["notify_email"]),
                json.dumps(current["protected_player_ids"]),
                bool(current["paused"]),
            ),
        )
        conn.commit()
    return get_settings(account_id, platform, league_id, season)


def get_kill_switch(account_id) -> bool:
    """Account-wide Autopilot kill switch. Defaults to off."""
    init_autopilot_tables()
    with _conn() as conn:
        row = conn.execute(
            "SELECT kill_switch FROM autopilot_account_state WHERE account_id = %s",
            (int(account_id),),
        ).fetchone()
    return bool(row and row["kill_switch"])


def set_kill_switch(account_id, on: bool) -> bool:
    init_autopilot_tables()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO autopilot_account_state (account_id, kill_switch, updated_at)
               VALUES (%s, %s, now())
               ON CONFLICT (account_id)
               DO UPDATE SET kill_switch = EXCLUDED.kill_switch, updated_at = now()""",
            (int(account_id), bool(on)),
        )
        conn.commit()
    return get_kill_switch(account_id)


def set_paused(account_id, platform, league_id, season, paused: bool) -> Dict[str, Any]:
    """Per-league pause (used by the panel and by verification failures)."""
    return save_settings(
        account_id, platform, league_id, season, {"paused": bool(paused)}
    )


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

def _plan_from_row(row) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    out = dict(row)
    for key in ("snapshot", "swaps", "lineup_items", "issues"):
        value = out.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                value = {} if key == "snapshot" else []
        out[key] = value
    return out


def save_plan(account_id, platform, league_id, season, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Store a new open plan, superseding any open plan for the league."""
    init_autopilot_tables()
    snap = {int(k): int(v) for k, v in (plan.get("snapshot") or {}).items()}
    period = int(plan["scoring_period"])
    with _conn() as conn:
        conn.execute(
            """UPDATE autopilot_plans SET status = 'superseded', updated_at = now()
               WHERE account_id = %s AND platform = %s AND league_id = %s
                 AND season = %s AND status = 'open'""",
            (int(account_id), str(platform), str(league_id), int(season)),
        )
        row = conn.execute(
            """INSERT INTO autopilot_plans
                   (account_id, platform, league_id, season, team_id,
                    scoring_period, mode, status, snapshot, snapshot_hash,
                    swaps, lineup_items, issues)
               VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s::jsonb, %s,
                       %s::jsonb, %s::jsonb, %s::jsonb)
               RETURNING *""",
            (
                int(account_id), str(platform), str(league_id), int(season),
                int(plan["team_id"]), period, str(plan.get("mode") or "approve"),
                json.dumps(snap), snapshot_hash(snap, period),
                json.dumps(plan.get("swaps") or []),
                json.dumps(plan.get("lineup_items") or []),
                json.dumps(plan.get("issues") or []),
            ),
        ).fetchone()
        conn.commit()
    return _plan_from_row(row)


def get_plan(plan_id) -> Optional[Dict[str, Any]]:
    init_autopilot_tables()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM autopilot_plans WHERE id = %s", (int(plan_id),)
        ).fetchone()
    return _plan_from_row(row)


def get_open_plan(account_id, platform, league_id, season) -> Optional[Dict[str, Any]]:
    init_autopilot_tables()
    with _conn() as conn:
        row = conn.execute(
            """SELECT * FROM autopilot_plans
               WHERE account_id = %s AND platform = %s AND league_id = %s
                 AND season = %s AND status = 'open'
               ORDER BY id DESC LIMIT 1""",
            (int(account_id), str(platform), str(league_id), int(season)),
        ).fetchone()
    return _plan_from_row(row)


def set_plan_status(plan_id, status: str) -> None:
    if status not in PLAN_STATUSES:
        raise ValueError(f"Invalid plan status: {status!r}")
    init_autopilot_tables()
    with _conn() as conn:
        conn.execute(
            "UPDATE autopilot_plans SET status = %s, updated_at = now() WHERE id = %s",
            (status, int(plan_id)),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Audit (append-only: insert + read, nothing else)
# ---------------------------------------------------------------------------

def record_audit(
    account_id, platform, league_id, season, action: str, *,
    plan_id=None, scoring_period=None, mode=None, player_id=None,
    player_name=None, from_slot=None, to_slot=None, score_delta=None,
    detail=None, verification=None, espn_status=None,
) -> None:
    """Append one audit entry. Never raises (auditing must not break a run;
    a failure here is logged and the caller's outcome stands)."""
    if action not in AUDIT_ACTIONS:
        raise ValueError(f"Invalid audit action: {action!r}")
    try:
        init_autopilot_tables()
        with _conn() as conn:
            conn.execute(
                """INSERT INTO autopilot_audit
                       (account_id, platform, league_id, season, plan_id,
                        scoring_period, action, mode, player_id, player_name,
                        from_slot, to_slot, score_delta, detail, verification,
                        espn_status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s)""",
                (
                    int(account_id), str(platform), str(league_id), int(season),
                    int(plan_id) if plan_id is not None else None,
                    int(scoring_period) if scoring_period is not None else None,
                    action, mode,
                    int(player_id) if player_id is not None else None,
                    player_name,
                    int(from_slot) if from_slot is not None else None,
                    int(to_slot) if to_slot is not None else None,
                    float(score_delta) if score_delta is not None else None,
                    detail, verification, espn_status,
                ),
            )
            conn.commit()
    except Exception:
        logger.warning("autopilot audit write failed", exc_info=True)


def list_audit(account_id, platform, league_id, season, limit: int = 50) -> List[Dict[str, Any]]:
    init_autopilot_tables()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT * FROM autopilot_audit
               WHERE account_id = %s AND platform = %s AND league_id = %s
                 AND season = %s
               ORDER BY id DESC LIMIT %s""",
            (int(account_id), str(platform), str(league_id), int(season),
             int(limit)),
        ).fetchall()
    return [dict(r) for r in rows or []]
