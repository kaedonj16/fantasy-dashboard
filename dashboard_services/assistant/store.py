"""Postgres storage for the assistant: threads, advice log, preferences.

Lazy table init mirrors the user_ui_prefs pattern in routes/user_pages_bp.py:
CREATE TABLE IF NOT EXISTS on first use, guarded by a module flag, failures
logged and swallowed so a DB hiccup never breaks unrelated pages.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)

_TABLE_READY = False


def get_conn(*args, **kwargs):
    # Local indirection so tests can monkeypatch store.get_conn.
    from dashboard_services.db import get_conn as _fn
    return _fn(*args, **kwargs)


def _init_tables():
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_threads (
                    thread_id TEXT PRIMARY KEY,
                    account_key TEXT NOT NULL,
                    platform TEXT,
                    league_id TEXT,
                    season INT,
                    last_entities JSONB NOT NULL DEFAULT '{}'::jsonb,
                    turn_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_advice_log (
                    log_id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    account_key TEXT NOT NULL,
                    league_id TEXT,
                    platform TEXT,
                    season INT,
                    week INT,
                    thread_id TEXT,
                    turn_index INT,
                    question_text TEXT,
                    intent TEXT,
                    entities JSONB NOT NULL DEFAULT '{}'::jsonb,
                    assumed_context JSONB NOT NULL DEFAULT '{}'::jsonb,
                    channel TEXT NOT NULL DEFAULT 'site',
                    proactive BOOLEAN NOT NULL DEFAULT FALSE,
                    trigger TEXT,
                    input_mode TEXT NOT NULL DEFAULT 'typed',
                    tools_called JSONB NOT NULL DEFAULT '[]'::jsonb,
                    evidence_pack JSONB NOT NULL DEFAULT '{}'::jsonb,
                    evidence_hash TEXT,
                    evidence_as_of JSONB NOT NULL DEFAULT '{}'::jsonb,
                    answer_text TEXT,
                    verdict JSONB,
                    numbers_shown JSONB NOT NULL DEFAULT '{}'::jsonb,
                    model TEXT,
                    router_model TEXT,
                    prompt_version TEXT,
                    input_tokens INT,
                    output_tokens INT,
                    cached_tokens INT,
                    est_cost_usd NUMERIC,
                    latency_ms INT,
                    thumb TEXT,
                    correction_text TEXT,
                    corrected_claim TEXT,
                    correction_category TEXT,
                    reask_flag BOOLEAN NOT NULL DEFAULT FALSE,
                    revises_log_id BIGINT,
                    grade TEXT,
                    grader TEXT,
                    graded_at TIMESTAMPTZ,
                    outcome_detail JSONB,
                    process_note TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_preferences (
                    id BIGSERIAL PRIMARY KEY,
                    account_key TEXT NOT NULL,
                    text TEXT NOT NULL,
                    league_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_assistant_log_account ON assistant_advice_log (account_key, created_at)"
            )
            conn.commit()
        _TABLE_READY = True
    except Exception as exc:
        logger.warning("[assistant] table init failed: %s", exc)


def _j(value) -> str:
    return json.dumps(value if value is not None else {}, default=str)


def evidence_hash(pack) -> str:
    raw = json.dumps(pack or {}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# -- threads -----------------------------------------------------------------

def get_or_create_thread(account_key: str, thread_id: str | None = None, *, platform=None, league_id=None, season=None) -> dict:
    _init_tables()
    tid = str(thread_id or "").strip() or uuid.uuid4().hex
    with get_conn() as conn:
        row = None
        if thread_id:
            row = conn.execute(
                "SELECT thread_id, account_key, platform, league_id, season, last_entities, turn_count FROM assistant_threads WHERE thread_id = %s",
                (tid,),
            ).fetchone()
        if row:
            d = dict(row) if isinstance(row, dict) else {
                "thread_id": row[0], "account_key": row[1], "platform": row[2],
                "league_id": row[3], "season": row[4], "last_entities": row[5], "turn_count": row[6],
            }
            if isinstance(d.get("last_entities"), str):
                try:
                    d["last_entities"] = json.loads(d["last_entities"])
                except Exception:
                    d["last_entities"] = {}
            return d
        conn.execute(
            "INSERT INTO assistant_threads (thread_id, account_key, platform, league_id, season, last_entities, turn_count) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)",
            (tid, account_key, platform, league_id, season, _j({}), 0),
        )
        conn.commit()
    return {"thread_id": tid, "account_key": account_key, "platform": platform,
            "league_id": league_id, "season": season, "last_entities": {}, "turn_count": 0}


def update_thread(thread_id: str, *, last_entities=None, turn_count=None):
    _init_tables()
    try:
        with get_conn() as conn:
            conn.execute(
                "UPDATE assistant_threads SET last_entities = %s::jsonb, turn_count = %s, updated_at = NOW() WHERE thread_id = %s",
                (_j(last_entities or {}), int(turn_count or 0), thread_id),
            )
            conn.commit()
    except Exception:
        logger.debug("[assistant] thread update failed", exc_info=True)


def get_last_log_for_thread(thread_id: str) -> dict | None:
    _init_tables()
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT log_id, intent, entities, verdict, answer_text FROM assistant_advice_log WHERE thread_id = %s ORDER BY log_id DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        if not row:
            return None
        return dict(row) if isinstance(row, dict) else {
            "log_id": row[0], "intent": row[1], "entities": row[2], "verdict": row[3], "answer_text": row[4]}
    except Exception:
        return None


# -- advice log ---------------------------------------------------------------

_LOG_COLS = [
    "account_key", "league_id", "platform", "season", "week", "thread_id", "turn_index",
    "question_text", "intent", "entities", "assumed_context", "channel", "proactive",
    "trigger", "input_mode", "tools_called", "evidence_pack", "evidence_hash", "evidence_as_of",
    "answer_text", "verdict", "numbers_shown", "model", "router_model", "prompt_version",
    "input_tokens", "output_tokens", "cached_tokens", "est_cost_usd", "latency_ms",
    "revises_log_id",
]
_JSON_COLS = {"entities", "assumed_context", "tools_called", "evidence_pack", "evidence_as_of", "verdict", "numbers_shown"}


def write_log(**fields) -> int | None:
    """Insert one advice-log row; returns log_id or None on failure."""
    _init_tables()
    pack = fields.get("evidence_pack") or {}
    if not fields.get("evidence_hash"):
        fields["evidence_hash"] = evidence_hash(pack)
    cols = [c for c in _LOG_COLS if c in fields or c in ("account_key",)]
    placeholders = []
    values = []
    for c in cols:
        v = fields.get(c)
        if c in _JSON_COLS:
            placeholders.append("%s::jsonb")
            values.append(_j(v if v is not None else ({} if c != "tools_called" else [])))
        else:
            placeholders.append("%s")
            values.append(v)
    sql = f"INSERT INTO assistant_advice_log ({', '.join(cols)}) VALUES ({', '.join(placeholders)}) RETURNING log_id"
    try:
        with get_conn() as conn:
            row = conn.execute(sql, tuple(values)).fetchone()
            conn.commit()
        if row is None:
            return None
        return int(row["log_id"] if isinstance(row, dict) else row[0])
    except Exception:
        logger.warning("[assistant] write_log failed", exc_info=True)
        return None


def get_log(log_id, account_key: str | None = None) -> dict | None:
    _init_tables()
    try:
        with get_conn() as conn:
            if account_key is not None:
                row = conn.execute(
                    "SELECT log_id, account_key, intent, answer_text, verdict FROM assistant_advice_log WHERE log_id = %s AND account_key = %s",
                    (int(log_id), account_key),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT log_id, account_key, intent, answer_text, verdict FROM assistant_advice_log WHERE log_id = %s",
                    (int(log_id),),
                ).fetchone()
        if not row:
            return None
        return dict(row) if isinstance(row, dict) else {
            "log_id": row[0], "account_key": row[1], "intent": row[2], "answer_text": row[3], "verdict": row[4]}
    except Exception:
        return None


def set_thumb(log_id, account_key: str, thumb: str) -> bool:
    if thumb not in ("up", "down"):
        return False
    _init_tables()
    if get_log(log_id, account_key) is None:
        return False
    with get_conn() as conn:
        conn.execute(
            "UPDATE assistant_advice_log SET thumb = %s WHERE log_id = %s AND account_key = %s",
            (thumb, int(log_id), account_key),
        )
        conn.commit()
    return True


def set_correction(log_id, account_key: str, *, category: str, text: str | None = None,
                   corrected_claim: str | None = None, reask_flag: bool = False) -> bool:
    _init_tables()
    if get_log(log_id, account_key) is None:
        return False
    with get_conn() as conn:
        conn.execute(
            "UPDATE assistant_advice_log SET correction_category = %s, correction_text = %s, corrected_claim = %s, reask_flag = %s WHERE log_id = %s AND account_key = %s",
            (category, text, corrected_claim, bool(reask_flag), int(log_id), account_key),
        )
        conn.commit()
    return True


_LOG_FULL_COLS = ("log_id", "account_key", "league_id", "platform", "season", "week",
                  "thread_id", "turn_index", "question_text", "intent", "entities",
                  "answer_text", "verdict", "numbers_shown")


def get_log_full(log_id, account_key: str | None = None) -> dict | None:
    """The full advice-log row (entities/verdict/numbers parsed), for the
    correction re-check: everything needed to rebuild the answer's evidence
    and compare the fresh call against the one on record."""
    _init_tables()
    try:
        with get_conn() as conn:
            if account_key is not None:
                row = conn.execute(
                    f"SELECT {', '.join(_LOG_FULL_COLS)} FROM assistant_advice_log WHERE log_id = %s AND account_key = %s",
                    (int(log_id), account_key),
                ).fetchone()
            else:
                row = conn.execute(
                    f"SELECT {', '.join(_LOG_FULL_COLS)} FROM assistant_advice_log WHERE log_id = %s",
                    (int(log_id),),
                ).fetchone()
        if not row:
            return None
        d = dict(row) if isinstance(row, dict) else dict(zip(_LOG_FULL_COLS, row))
        for col in ("entities", "verdict", "numbers_shown"):
            if isinstance(d.get(col), str):
                try:
                    d[col] = json.loads(d[col])
                except Exception:
                    d[col] = {}
        return d
    except Exception:
        return None


def count_monthly_asked(account_key: str) -> int:
    _init_tables()
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM assistant_advice_log WHERE account_key = %s AND proactive = FALSE AND created_at >= date_trunc('month', NOW())",
                (account_key,),
            ).fetchone()
        if not row:
            return 0
        return int(row["n"] if isinstance(row, dict) else row[0])
    except Exception:
        return 0


def find_proactive_digest(account_key: str, league_id, week) -> dict | None:
    _init_tables()
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT log_id, answer_text, verdict, numbers_shown FROM assistant_advice_log WHERE account_key = %s AND league_id = %s AND week = %s AND proactive = TRUE AND trigger = %s ORDER BY log_id DESC LIMIT 1",
                (account_key, str(league_id) if league_id is not None else None, week, "digest"),
            ).fetchone()
        if not row:
            return None
        d = dict(row) if isinstance(row, dict) else {
            "log_id": row[0], "answer_text": row[1], "verdict": row[2], "numbers_shown": row[3]}
        # verdict column stores the full answer dict for proactive rows (see service).
        return d
    except Exception:
        return None


# -- outcome grading (stage 2) -------------------------------------------------

_LOG_GRADE_COLS = ("log_id", "created_at", "account_key", "league_id", "platform", "season",
                   "week", "question_text", "intent", "entities", "evidence_pack",
                   "answer_text", "verdict", "numbers_shown", "thumb", "correction_text",
                   "corrected_claim", "correction_category", "reask_flag", "proactive",
                   "trigger", "grade", "grader", "graded_at", "outcome_detail")
_LOG_JSON_COLS = ("entities", "evidence_pack", "verdict", "numbers_shown", "outcome_detail")


def _grade_row(row) -> dict:
    d = dict(row) if isinstance(row, dict) else dict(zip(_LOG_GRADE_COLS, row))
    for col in _LOG_JSON_COLS:
        if isinstance(d.get(col), str):
            try:
                d[col] = json.loads(d[col])
            except Exception:
                d[col] = {}
    return d


def fetch_ungraded_rows(intent: str, limit: int = 500) -> list[dict]:
    """Ungraded advice-log rows for one intent, oldest first (grader input)."""
    _init_tables()
    try:
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_LOG_GRADE_COLS)} FROM assistant_advice_log "
                "WHERE intent = %s AND grade IS NULL ORDER BY log_id ASC LIMIT %s",
                (intent, int(limit)),
            ).fetchall()
        return [_grade_row(r) for r in (rows or [])]
    except Exception:
        logger.warning("[assistant] fetch_ungraded_rows failed", exc_info=True)
        return []


def write_grade(log_id, grade: str, grader: str, outcome_detail) -> bool:
    """Write one outcome grade onto a log row.

    The grade IS NULL guard is the idempotency rule: a row that already
    carries a grade is never rewritten, so re-running a grader is safe and
    a version bump never rewrites history (workshop section 4).
    """
    _init_tables()
    try:
        with get_conn() as conn:
            cur = conn.execute(
                "UPDATE assistant_advice_log SET grade = %s, grader = %s, graded_at = NOW(), "
                "outcome_detail = %s::jsonb WHERE log_id = %s AND grade IS NULL",
                (grade, grader, _j(outcome_detail or {}), int(log_id)),
            )
            conn.commit()
        return bool(getattr(cur, "rowcount", 0))
    except Exception:
        logger.warning("[assistant] write_grade failed", exc_info=True)
        return False


def fetch_log_rows_between(start, end) -> list[dict]:
    """All advice-log rows created in [start, end), for the review packet."""
    _init_tables()
    try:
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_LOG_GRADE_COLS)} FROM assistant_advice_log "
                "WHERE created_at >= %s AND created_at < %s ORDER BY log_id ASC",
                (start, end),
            ).fetchall()
        return [_grade_row(r) for r in (rows or [])]
    except Exception:
        logger.warning("[assistant] fetch_log_rows_between failed", exc_info=True)
        return []


def fetch_graded_rows_between(start, end) -> list[dict]:
    """Advice-log rows whose grade was written in [start, end)."""
    _init_tables()
    try:
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_LOG_GRADE_COLS)} FROM assistant_advice_log "
                "WHERE graded_at >= %s AND graded_at < %s ORDER BY log_id ASC",
                (start, end),
            ).fetchall()
        return [_grade_row(r) for r in (rows or [])]
    except Exception:
        logger.warning("[assistant] fetch_graded_rows_between failed", exc_info=True)
        return []


# -- preferences ---------------------------------------------------------------

def _pref_dict(row) -> dict:
    if isinstance(row, dict):
        d = dict(row)
    else:
        d = {"id": row[0], "account_key": row[1], "text": row[2], "league_id": row[3], "created_at": row[4] if len(row) > 4 else None}
    if d.get("created_at") is not None and not isinstance(d["created_at"], str):
        d["created_at"] = str(d["created_at"])
    return {"id": d.get("id"), "text": d.get("text"), "league_id": d.get("league_id"), "created_at": d.get("created_at")}


def list_preferences(account_key: str) -> list[dict]:
    _init_tables()
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT id, account_key, text, league_id, created_at FROM assistant_preferences WHERE account_key = %s ORDER BY id ASC",
                (account_key,),
            ).fetchall()
        return [_pref_dict(r) for r in (rows or [])]
    except Exception:
        return []


def add_preference(account_key: str, text: str, league_id=None) -> dict:
    _init_tables()
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO assistant_preferences (account_key, text, league_id) VALUES (%s, %s, %s) RETURNING id, account_key, text, league_id, created_at",
            (account_key, text, league_id),
        ).fetchone()
        conn.commit()
    return _pref_dict(row) if row else {"id": None, "text": text, "league_id": league_id, "created_at": None}


def delete_preference(account_key: str, pref_id) -> bool:
    _init_tables()
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM assistant_preferences WHERE id = %s AND account_key = %s",
            (int(pref_id), account_key),
        )
        conn.commit()
    return True


def delete_all_preferences(account_key: str) -> bool:
    _init_tables()
    with get_conn() as conn:
        conn.execute("DELETE FROM assistant_preferences WHERE account_key = %s", (account_key,))
        conn.commit()
    return True
