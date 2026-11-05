"""Per-league rules record (Site Knowledge Wave 1).

One shared record per (platform, league_id, season) holding the league's
ground-truth rules: playoff format, waiver system, trade deadline and
designated keepers. Records are resolved with precedence

    user_confirmed > feed > derived > default_unverified

and applied to league contexts as a serve-time overlay (see
``apply_rules_overlay``, called from ``app.get_league_ctx_from_cache``) --
never baked into the pickled Redis ctx, so a corrected record takes
effect on the next serve instead of whenever the ctx happens to rebuild.

Why this exists: some provider adapters silently default the values their
API never publishes (Fleaflicker serves ``playoff_teams=6`` /
``playoff_week_start=15``; MFL has an ``or 6`` fallback), and every
consumer downstream treated those defaults as facts. The record labels
each league's values with their source so surfaces can trust, derive,
confirm, or honestly mark them unverified.

Storage is ``league_rules`` (migrations/053_league_rules.sql), created
idempotently here as well. Every read/write fails soft: a missing table
or DB error degrades to today's behavior (provider defaults in the ctx,
championship odds hidden where the format gate already fails closed).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, datetime, time as _time

logger = logging.getLogger(__name__)

SOURCE_USER_CONFIRMED = "user_confirmed"
SOURCE_FEED = "feed"
SOURCE_DERIVED = "derived"
SOURCE_DEFAULT = "default_unverified"

_PRECEDENCE = {
    SOURCE_DEFAULT: 0,
    SOURCE_DERIVED: 1,
    SOURCE_FEED: 2,
    SOURCE_USER_CONFIRMED: 3,
}

# Championship odds trust only a format the provider actually published
# or a user confirmed. A derived format (prior-season playoff
# scoreboards) is good enough for the regular-season odds, the
# standings picture and the projected bracket, but not for this flag.
_CHAMPIONSHIP_SOURCES = frozenset({SOURCE_FEED, SOURCE_USER_CONFIRMED})

WAIVER_TYPES = frozenset({"faab", "rolling", "reverse_standings", "free"})

_RULE_FIELDS = (
    "playoff_teams",
    "playoff_week_start",
    "playoff_round_type",
    "playoff_reseed",
    "waiver_type",
    "waiver_process_time",
    "waiver_process_days",
    "waiver_process_tz",
    "trade_deadline",
    "max_keepers",
)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS league_rules (
    platform           TEXT NOT NULL,
    league_id          TEXT NOT NULL,
    season             INTEGER NOT NULL,
    playoff_teams      INTEGER,
    playoff_week_start INTEGER,
    playoff_round_type INTEGER,
    playoff_reseed     BOOLEAN,
    waiver_type        TEXT,
    waiver_process_time TIME,
    waiver_process_days TEXT,
    waiver_process_tz  TEXT DEFAULT 'America/New_York',
    trade_deadline     DATE,
    max_keepers        INTEGER,
    designated_keepers JSONB NOT NULL DEFAULT '{}'::jsonb,
    keepers_known      BOOLEAN NOT NULL DEFAULT FALSE,
    source             TEXT NOT NULL,
    confirmed_by       INTEGER,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (platform, league_id, season)
)
"""

_MEMO: dict = {}
_MEMO_TTL_SECONDS = 60.0
_MEMO_LOCK = threading.Lock()
_TABLE_READY = False
_DERIVE_ATTEMPTED: set = set()


# ── small coercion helpers ──────────────────────────────────────────────

def _as_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_time_str(value):
    """Normalize a TIME / 'HH:MM' value to 'HH:MM' (or None)."""
    if value is None:
        return None
    if isinstance(value, _time):
        return value.strftime("%H:%M")
    if isinstance(value, datetime):
        return value.strftime("%H:%M")
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        return None
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return f"{hh:02d}:{mm:02d}"


def _as_date_str(value):
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) \
            else value.isoformat()
    text = str(value).strip()
    return text or None


def _clamp_playoff_teams(value, total_rosters):
    teams = _as_int(value)
    if teams is None:
        return None
    teams = max(2, teams)
    total = _as_int(total_rosters)
    if total is not None and total >= 2:
        teams = min(teams, total)
    return teams


def _blank_record(platform, league_id, season):
    rec = {
        "platform": platform,
        "league_id": str(league_id),
        "season": int(season),
        "designated_keepers": {},
        "keepers_known": False,
        "source": SOURCE_DEFAULT,
        "confirmed_by": None,
        "updated_at": None,
    }
    for field in _RULE_FIELDS:
        rec[field] = None
    rec["waiver_process_tz"] = "America/New_York"
    return rec


# ── storage (fail soft) ─────────────────────────────────────────────────

def _conn():
    from dashboard_services.db import get_conn

    return get_conn(autocommit=True)


def _ensure_table() -> bool:
    global _TABLE_READY
    if _TABLE_READY:
        return True
    try:
        with _conn() as conn:
            conn.execute(_CREATE_SQL)
        _TABLE_READY = True
        return True
    except Exception:
        logger.debug("[league-rules] table ensure failed", exc_info=True)
        return False


def _row_to_record(row) -> dict:
    rec = _blank_record(row.get("platform"), row.get("league_id"),
                        row.get("season"))
    for field in _RULE_FIELDS:
        rec[field] = row.get(field)
    rec["playoff_teams"] = _as_int(rec["playoff_teams"])
    rec["playoff_week_start"] = _as_int(rec["playoff_week_start"])
    rec["playoff_round_type"] = _as_int(rec["playoff_round_type"])
    rec["max_keepers"] = _as_int(rec["max_keepers"])
    if rec["playoff_reseed"] is not None:
        rec["playoff_reseed"] = bool(rec["playoff_reseed"])
    rec["waiver_process_time"] = _as_time_str(rec["waiver_process_time"])
    rec["trade_deadline"] = _as_date_str(rec["trade_deadline"])
    keepers = row.get("designated_keepers") or {}
    if isinstance(keepers, str):
        try:
            keepers = json.loads(keepers)
        except (TypeError, ValueError):
            keepers = {}
    rec["designated_keepers"] = {
        str(k): [str(p) for p in (v or [])] for k, v in dict(keepers).items()
    }
    rec["keepers_known"] = bool(row.get("keepers_known"))
    rec["source"] = row.get("source") or SOURCE_DEFAULT
    rec["confirmed_by"] = _as_int(row.get("confirmed_by"))
    updated = row.get("updated_at")
    rec["updated_at"] = (
        updated.isoformat() if isinstance(updated, datetime)
        else (str(updated) if updated else None)
    )
    return rec


def _fetch_record_db(platform, league_id, season):
    if not _ensure_table():
        return None
    try:
        with _conn() as conn:
            cur = conn.execute(
                "SELECT * FROM league_rules"
                " WHERE platform = %s AND league_id = %s AND season = %s",
                (platform, str(league_id), int(season)),
            )
            row = cur.fetchone()
        return _row_to_record(row) if row else None
    except Exception:
        logger.debug("[league-rules] fetch failed", exc_info=True)
        return None


def _fetch_record(platform, league_id, season):
    """Stored record (or None), memoized for 60s per league."""
    key = (platform, str(league_id), int(season))
    now = time.monotonic()
    with _MEMO_LOCK:
        hit = _MEMO.get(key)
        if hit and now - hit[0] < _MEMO_TTL_SECONDS:
            return hit[1]
    rec = _fetch_record_db(platform, league_id, season)
    with _MEMO_LOCK:
        _MEMO[key] = (time.monotonic(), rec)
    return rec


def _store_record(record) -> bool:
    if not _ensure_table():
        return False
    try:
        with _conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO league_rules (
                    platform, league_id, season, playoff_teams,
                    playoff_week_start, playoff_round_type, playoff_reseed,
                    waiver_type, waiver_process_time, waiver_process_days,
                    waiver_process_tz, trade_deadline, max_keepers,
                    designated_keepers, keepers_known, source, confirmed_by,
                    updated_at
                ) VALUES (
                    %(platform)s, %(league_id)s, %(season)s, %(playoff_teams)s,
                    %(playoff_week_start)s, %(playoff_round_type)s,
                    %(playoff_reseed)s, %(waiver_type)s,
                    %(waiver_process_time)s, %(waiver_process_days)s,
                    %(waiver_process_tz)s, %(trade_deadline)s, %(max_keepers)s,
                    %(designated_keepers)s::jsonb, %(keepers_known)s,
                    %(source)s, %(confirmed_by)s, now()
                )
                ON CONFLICT (platform, league_id, season) DO UPDATE SET
                    playoff_teams = EXCLUDED.playoff_teams,
                    playoff_week_start = EXCLUDED.playoff_week_start,
                    playoff_round_type = EXCLUDED.playoff_round_type,
                    playoff_reseed = EXCLUDED.playoff_reseed,
                    waiver_type = EXCLUDED.waiver_type,
                    waiver_process_time = EXCLUDED.waiver_process_time,
                    waiver_process_days = EXCLUDED.waiver_process_days,
                    waiver_process_tz = EXCLUDED.waiver_process_tz,
                    trade_deadline = EXCLUDED.trade_deadline,
                    max_keepers = EXCLUDED.max_keepers,
                    designated_keepers = EXCLUDED.designated_keepers,
                    keepers_known = EXCLUDED.keepers_known,
                    source = EXCLUDED.source,
                    confirmed_by = EXCLUDED.confirmed_by,
                    updated_at = now()
                RETURNING updated_at
                """,
                {
                    "platform": record["platform"],
                    "league_id": str(record["league_id"]),
                    "season": int(record["season"]),
                    "playoff_teams": record.get("playoff_teams"),
                    "playoff_week_start": record.get("playoff_week_start"),
                    "playoff_round_type": record.get("playoff_round_type"),
                    "playoff_reseed": record.get("playoff_reseed"),
                    "waiver_type": record.get("waiver_type"),
                    "waiver_process_time": record.get("waiver_process_time"),
                    "waiver_process_days": record.get("waiver_process_days"),
                    "waiver_process_tz": record.get("waiver_process_tz"),
                    "trade_deadline": record.get("trade_deadline"),
                    "max_keepers": record.get("max_keepers"),
                    "designated_keepers": json.dumps(
                        record.get("designated_keepers") or {}
                    ),
                    "keepers_known": bool(record.get("keepers_known")),
                    "source": record.get("source") or SOURCE_DEFAULT,
                    "confirmed_by": record.get("confirmed_by"),
                },
            )
            row = cur.fetchone()
        if row:
            updated = row.get("updated_at") if isinstance(row, dict) \
                else row[0]
            record["updated_at"] = (
                updated.isoformat() if isinstance(updated, datetime)
                else (str(updated) if updated else None)
            )
        key = (record["platform"], str(record["league_id"]),
               int(record["season"]))
        with _MEMO_LOCK:
            _MEMO[key] = (time.monotonic(), record)
        return True
    except Exception:
        logger.debug("[league-rules] store failed", exc_info=True)
        return False


# ── resolution ──────────────────────────────────────────────────────────

def _feed_values(platform, settings):
    """Rule values the provider actually publishes, or None.

    Fleaflicker's API never publishes playoff settings (its adapter
    defaults are explicitly unverified), and MFL counts only when its
    payload carried ``playoffTeams`` (the adapter flags that via
    ``playoff_teams_from_feed``). Sleeper / ESPN / Yahoo read the fields
    straight from their payloads.
    """
    if not settings:
        return None
    if platform == "fleaflicker":
        return None
    if platform == "mfl" and not settings.get("playoff_teams_from_feed"):
        return None
    teams = _as_int(settings.get("playoff_teams"))
    if teams is None or teams < 2:
        return None
    out = {"playoff_teams": teams}
    pws = _as_int(
        settings.get("playoff_week_start")
        or settings.get("playoff_start_week")
    )
    if pws:
        out["playoff_week_start"] = pws
    round_type = _as_int(settings.get("playoff_round_type"))
    if round_type is not None:
        out["playoff_round_type"] = round_type
    if settings.get("playoff_reseed") is not None:
        out["playoff_reseed"] = bool(settings["playoff_reseed"])
    for field in (
        "waiver_process_time", "waiver_process_days", "waiver_process_tz",
        "trade_deadline", "max_keepers",
    ):
        if settings.get(field) is not None:
            out[field] = settings[field]
    if isinstance(settings.get("waiver_type"), str) \
            and settings["waiver_type"] in WAIVER_TYPES:
        out["waiver_type"] = settings["waiver_type"]
    return out


def _settings_values(settings):
    """Whatever the provider settings carry (possibly silent defaults)."""
    out = {}
    if not settings:
        return out
    for field in _RULE_FIELDS:
        if settings.get(field) is not None:
            out[field] = settings[field]
    pws = _as_int(
        settings.get("playoff_week_start")
        or settings.get("playoff_start_week")
    )
    if pws:
        out["playoff_week_start"] = pws
    return out


def _merge(base, values, source, total_rosters):
    rec = dict(base) if base else None
    if rec is None:
        return None
    for field in _RULE_FIELDS:
        if values.get(field) is not None:
            rec[field] = values[field]
    rec["playoff_teams"] = _clamp_playoff_teams(
        rec.get("playoff_teams"), total_rosters
    )
    rec["waiver_process_time"] = _as_time_str(rec.get("waiver_process_time"))
    rec["trade_deadline"] = _as_date_str(rec.get("trade_deadline"))
    rec["source"] = source
    return rec


def _rules_equal(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if a.get("source") != b.get("source"):
        return False
    return all(a.get(f) == b.get(f) for f in _RULE_FIELDS)


def resolve_league_rules(platform, league_id, season, provider_settings=None,
                         total_rosters=None, deriver=None) -> dict:
    """Resolve (and persist) the league's rules record.

    ``deriver`` is an optional callable ``(platform, league_id, season) ->
    dict | None`` supplying derived values (prior-season playoff field /
    week start); it is attempted at most once per league per process and
    only while no record above ``default_unverified`` exists.
    """
    platform = (platform or "").strip().lower()
    league_id = str(league_id)
    season = int(season)
    settings = provider_settings or {}
    stored = _fetch_record(platform, league_id, season)
    if stored and stored["source"] == SOURCE_USER_CONFIRMED:
        return stored

    feed = _feed_values(platform, settings)
    if feed:
        base = stored or _blank_record(platform, league_id, season)
        merged = _merge(base, feed, SOURCE_FEED, total_rosters)
        if not _rules_equal(merged, stored):
            _store_record(merged)
        return merged

    if stored and stored["source"] in (SOURCE_FEED, SOURCE_DERIVED):
        return stored

    key = (platform, league_id, season)
    if deriver is not None and key not in _DERIVE_ATTEMPTED:
        _DERIVE_ATTEMPTED.add(key)
        try:
            derived = deriver(platform, league_id, season)
        except Exception:
            logger.debug("[league-rules] deriver failed", exc_info=True)
            derived = None
        if derived:
            base = stored or _blank_record(platform, league_id, season)
            if stored is None:
                # Carry the provider's (defaulted) week start etc. as the
                # base the derived field lands on.
                base = _merge(
                    base, _settings_values(settings), SOURCE_DERIVED,
                    total_rosters,
                ) or base
            merged = _merge(base, derived, SOURCE_DERIVED, total_rosters)
            if not _rules_equal(merged, stored):
                _store_record(merged)
            return merged

    if stored:
        return stored
    base = _blank_record(platform, league_id, season)
    default = _merge(
        base, _settings_values(settings), SOURCE_DEFAULT, total_rosters
    )
    _store_record(default)
    return default


def confirm_league_rules(platform, league_id, season, *, playoff_teams=None,
                         playoff_week_start=None, waiver_type=None,
                         waiver_process_time=None, waiver_process_days=None,
                         account_id=None, total_rosters=None) -> dict:
    """Persist a user-confirmed record (highest precedence).

    Raises ValueError on values that cannot describe a real league.
    """
    platform = (platform or "").strip().lower()
    league_id = str(league_id)
    season = int(season)
    teams = _as_int(playoff_teams)
    total = _as_int(total_rosters)
    if teams is None or teams < 2 or (total is not None and teams > total):
        raise ValueError("playoff_teams out of range")
    week_start = _as_int(playoff_week_start)
    if week_start is None or not (2 <= week_start <= 25):
        raise ValueError("playoff_week_start out of range")
    wtype = (waiver_type or "").strip().lower() or None
    if wtype is not None and wtype not in WAIVER_TYPES:
        raise ValueError("unknown waiver_type")
    base = _fetch_record(platform, league_id, season) \
        or _blank_record(platform, league_id, season)
    rec = _merge(
        base,
        {
            "playoff_teams": teams,
            "playoff_week_start": week_start,
            "waiver_type": wtype,
            "waiver_process_time": _as_time_str(waiver_process_time),
            "waiver_process_days": (waiver_process_days or None),
        },
        SOURCE_USER_CONFIRMED,
        total_rosters,
    )
    rec["confirmed_by"] = _as_int(account_id)
    _store_record(rec)
    return rec


# ── overlay / fingerprints ──────────────────────────────────────────────

def apply_rules_overlay(ctx, record=None, deriver=None):
    """Merge the league's rules record into ``ctx["league_settings"]``.

    Serve-time only: the cached/pickled ctx keeps the provider values,
    and the record (the source of truth) is re-applied on every serve, so
    a confirmation or correction cannot be resurrected by a stale ctx.
    Also stamps ``playoff_format_verified`` (feed or user-confirmed only;
    a derived format does not earn it) and a ``ctx["_league_rules"]``
    summary for renderers and cache signatures. No-ops when no record
    can be resolved (e.g. DB unavailable).
    """
    if not isinstance(ctx, dict):
        return ctx
    platform = str(ctx.get("platform") or "").strip().lower()
    league_id = ctx.get("resolved_league_id") or ctx.get("league_id")
    season = _as_int(ctx.get("season") or ctx.get("current_season"))
    if not platform or league_id is None or not season:
        return ctx
    if record is None:
        settings = ctx.get("league_settings") \
            or (ctx.get("league") or {}).get("settings") or {}
        total = _as_int(ctx.get("total_rosters"))
        if total is None:
            total = len({str(r.get("roster_id"))
                         for r in (ctx.get("rosters") or [])}) or None
        try:
            record = resolve_league_rules(
                platform, league_id, season,
                provider_settings=settings, total_rosters=total,
                deriver=deriver,
            )
        except Exception:
            logger.debug("[league-rules] resolve failed", exc_info=True)
            return ctx
    if not record:
        return ctx
    settings = ctx.setdefault("league_settings", {})
    for field in _RULE_FIELDS:
        value = record.get(field)
        if value is not None:
            settings[field] = value
    settings["playoff_format_verified"] = \
        record.get("source") in _CHAMPIONSHIP_SOURCES
    ctx["_league_rules"] = {
        "source": record.get("source"),
        "playoff_teams": record.get("playoff_teams"),
        "playoff_week_start": record.get("playoff_week_start"),
        "updated_at": record.get("updated_at"),
        "keepers_known": bool(record.get("keepers_known")),
    }
    return ctx


def rules_fingerprint(record) -> str:
    """Cache-signature component: changes when the rules record does."""
    if not record:
        return "none"
    return (
        f"{record.get('playoff_teams')}-{record.get('playoff_week_start')}"
        f"-{record.get('updated_at') or ''}"
    )


def ctx_rules_fingerprint(ctx) -> str:
    if not isinstance(ctx, dict):
        return "none"
    return rules_fingerprint(ctx.get("_league_rules"))
