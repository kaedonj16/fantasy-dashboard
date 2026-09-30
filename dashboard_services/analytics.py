"""First-party product analytics for the fantasy dashboard (Phase 1).

A single ``analytics_events`` Postgres table records two kinds of signal:

* pageviews, logged automatically by an ``after_request`` hook in app.py
  (HTML GET page renders only; static assets, health checks, /api/ traffic
  and bots are skipped), and
* explicit product moments via :func:`track_event` at the call sites that
  matter: signup, login, league linked, trade evaluated, waivers viewed,
  breakout board viewed, front office generated, push subscribed, checkout
  started, PRO subscribed / cancelled.

Design rules, from the plan Kaedon approved:

* Aggregate by default. The admin page shows counts and cohorts, never
  per-user click timelines.
* No PII in the events table. ``account_id`` is a joinable key, never an
  email or name; ``path`` is the request path only.
* First-party only. No third-party trackers, so the AdSense consent setup
  is untouched.
* Never raise, never slow a request. Every write path is wrapped; a failed
  insert is a debug log, not a 500.
* Self-exclusion. Admin sessions (ADMIN_KEY or password login) never log
  events, /admin/ pages are never pageviewed, and account ids listed in
  ANALYTICS_EXCLUDE_ACCOUNT_IDS are filtered out of every aggregation
  query, so the operator's own usage never pollutes the stats.

Table creation follows the lazy ``init_accounts_tables`` pattern: a
per-process ``_TABLES_READY`` guard, called at the top of every public
function, so the table exists on long-lived production databases without a
fresh-DB initialization run.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import uuid
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

logger = logging.getLogger(__name__)

_TABLES_READY = False

# Account ids excluded from every aggregation query, from the
# ANALYTICS_EXCLUDE_ACCOUNT_IDS env var (comma-separated). Parsed once per
# process; tests reset _EXCLUDED_ACCOUNT_IDS to re-parse.
_EXCLUDED_ACCOUNT_IDS: Optional[FrozenSet[int]] = None

# Event vocabulary. Centralized so the write call sites, the admin page and
# the tests share the same names.
EVENT_PAGEVIEW = "pageview"
EVENT_SIGNUP = "signup"
EVENT_LOGIN = "login"
EVENT_LEAGUE_LINKED = "league_linked"
EVENT_TRADE_EVALUATED = "trade_evaluated"
EVENT_WAIVERS_VIEWED = "waivers_viewed"
EVENT_BREAKOUT_VIEWED = "breakout_viewed"
EVENT_FRONT_OFFICE_GENERATED = "front_office_generated"
EVENT_PUSH_SUBSCRIBED = "push_subscribed"
EVENT_CHECKOUT_STARTED = "checkout_started"
EVENT_PRO_SUBSCRIBED = "pro_subscribed"
EVENT_PRO_CANCELLED = "pro_cancelled"

# Substrings that mark a request as bot traffic. Bots are not users for
# DAU/WAU purposes, so their pageviews are skipped at the hook.
_BOT_UA_HINTS = ("bot", "crawl", "spider", "slurp", "mediapartners", "ahrefs", "semrush")


def excluded_account_ids() -> FrozenSet[int]:
    """Account ids excluded from all analytics aggregations.

    Read from ANALYTICS_EXCLUDE_ACCOUNT_IDS (comma-separated), tolerant of
    blanks and bad values; parsed once per process.
    """
    global _EXCLUDED_ACCOUNT_IDS
    if _EXCLUDED_ACCOUNT_IDS is None:
        raw = os.environ.get("ANALYTICS_EXCLUDE_ACCOUNT_IDS", "") or ""
        ids = set()
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
        _EXCLUDED_ACCOUNT_IDS = frozenset(ids)
    return _EXCLUDED_ACCOUNT_IDS


def _exclusion_clause(column: str = "account_id") -> str:
    """SQL fragment excluding ANALYTICS_EXCLUDE_ACCOUNT_IDS.

    Ids are validated ints at parse time, so interpolating them is safe.
    Empty string when nothing is configured.
    """
    ids = excluded_account_ids()
    if not ids:
        return ""
    return "AND COALESCE(%s, -1) NOT IN (%s)" % (
        column,
        ", ".join(str(i) for i in sorted(ids)),
    )


def init_analytics_tables() -> None:
    """Create the analytics_events table and its index. Never raises."""
    global _TABLES_READY
    if _TABLES_READY:
        return
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id         BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
                    account_id INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
                    session_id TEXT,
                    event      TEXT NOT NULL,
                    path       TEXT,
                    props      JSONB DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_analytics_events_created_event
                    ON analytics_events (created_at, event)
                """
            )
            try:
                conn.commit()
            except Exception:
                pass
        _TABLES_READY = True
    except Exception:
        logger.debug("[analytics] init_analytics_tables failed", exc_info=True)


def _jsonb(value: Any) -> Any:
    """Adapt a props dict for the JSONB column, tolerating a missing driver."""
    payload = dict(value or {})
    try:
        from psycopg.types.json import Json as _Json

        return _Json(payload)
    except Exception:
        # psycopg absent (or fake conn in tests): psycopg3 adapts plain
        # dicts to JSONB natively, so pass it through.
        return payload


def track_event(
    event: str,
    account_id: Optional[int] = None,
    session_id: Optional[str] = None,
    path: Optional[str] = None,
    props: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert one analytics event. Cheap single INSERT; never raises.

    Skipped entirely for admin sessions: Kaedon's own usage must not
    pollute the stats, so any request authenticated via ADMIN_KEY or the
    admin password login records nothing here.
    """
    try:
        from flask import has_request_context

        if has_request_context():
            from dashboard_services.admin_auth import is_admin

            if is_admin():
                return
        init_analytics_tables()
        aid: Optional[int] = None
        if account_id is not None:
            try:
                aid = int(account_id)
            except (TypeError, ValueError):
                aid = None
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            conn.execute(
                "INSERT INTO analytics_events (account_id, session_id, event, path, props)"
                " VALUES (%s, %s, %s, %s, %s)",
                (aid, session_id, str(event), path, _jsonb(props)),
            )
            try:
                conn.commit()
            except Exception:
                pass
    except Exception:
        logger.debug("[analytics] track_event failed event=%s", event, exc_info=True)


def account_id_from_session() -> Optional[int]:
    """Return the signed-in Google account id, or None outside a session."""
    try:
        from flask import session

        raw = session.get("account_id")
        return int(raw) if raw is not None else None
    except Exception:
        return None


def ensure_anon_session_id() -> Optional[str]:
    """Stable per-browser id for signed-out guests, stored in the session."""
    try:
        from flask import session

        sid = session.get("_analytics_sid")
        if not sid:
            sid = uuid.uuid4().hex
            session["_analytics_sid"] = sid
        return sid
    except Exception:
        return None


def account_id_from_subscriber_token(token: Any) -> Optional[int]:
    """Parse the billing subscriber token ("acct:123") to an account id."""
    try:
        text = str(token or "")
        if text.startswith("acct:"):
            return int(text.split(":", 1)[1])
        return None
    except (TypeError, ValueError):
        return None


def should_log_pageview(
    method: str,
    path: str,
    content_type: Optional[str],
    status_code: int,
    user_agent: str = "",
) -> bool:
    """Pure decision: is this response a user-facing HTML pageview?

    Kept pure (no Flask globals) so it is unit-testable without the app.
    """
    if (method or "").upper() != "GET":
        return False
    p = path or ""
    if p.startswith(("/static/", "/healthz", "/api/", "/admin/")):
        return False
    if status_code != 200:
        return False
    if "text/html" not in (content_type or ""):
        return False
    ua = (user_agent or "").lower()
    if any(hint in ua for hint in _BOT_UA_HINTS):
        return False
    return True


# ── Aggregation queries (admin page) ──────────────────────────────────────────

_IDENT = "COALESCE(account_id::text, 's:' || COALESCE(session_id, '-'))"


def _fetchall(sql: str, args: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    """Run a read query and return rows.

    get_conn() uses psycopg's dict_row, so rows are dicts keyed by column
    name, never tuples. Raises on DB failure (route catches)."""
    init_analytics_tables()
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        return list(conn.execute(sql, args).fetchall())


def dau_last_30_days() -> List[Dict[str, Any]]:
    """Distinct users per day for the last 30 days (pageviews only)."""
    rows = _fetchall(
        f"""
        SELECT date_trunc('day', created_at)::date AS d,
               COUNT(DISTINCT {_IDENT}) AS users
        FROM analytics_events
        WHERE event = 'pageview' AND created_at >= now() - interval '30 days'
        {_exclusion_clause()}
        GROUP BY 1 ORDER BY 1
        """
    )
    return [{"date": str(r["d"]), "users": int(r["users"])} for r in rows]


# ── DAU breakdown (diagnostic only; does not change the headline DAU) ───────
#
# The headline DAU counts distinct _IDENT values, which mixes three very
# different things: signed-in accounts (deduplicated, trustworthy),
# anonymous browser sessions (one per browser/device, and a brand new one
# for every cookie-less hit), and browsers that appear as BOTH in one day
# (counted twice). dau_breakdown() decomposes today's headline number so
# the admin page can show what it is actually made of, under both the
# current UTC day definition and the New York day.

_BREAKDOWN_KEYS = (
    "headline",
    "signed_in",
    "anon_sessions",
    "anon_one_and_done",
    "anon_engaged",
    "anon_linked_sessions",
    "realistic_preview",
    "total_pageviews",
)

# Day filters are fixed literals (never user input), so interpolating the
# chosen one into the SQL is safe.
_DAY_FILTERS = {
    "utc": "date_trunc('day', created_at) = date_trunc('day', now())",
    "ny": (
        "(created_at AT TIME ZONE 'America/New_York')::date"
        " = (now() AT TIME ZONE 'America/New_York')::date"
    ),
}


def _breakdown_for_day(day_filter: str) -> Dict[str, int]:
    """One day's DAU decomposition. Session classification is disjoint:

    a session (non-null session_id) with any signed-in pageview that day is
    "linked"; a purely anonymous session is "one-and-done" with exactly 1
    pageview and "engaged" with 2+. So for anonymous sessions:
    anon_sessions = anon_one_and_done + anon_engaged + anon_linked_sessions.
    """
    rows = _fetchall(
        f"""
        WITH base AS (
            SELECT account_id, session_id
            FROM analytics_events
            WHERE event = 'pageview' AND {day_filter}
            {_exclusion_clause()}
        ),
        sessions AS (
            SELECT session_id,
                   COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views,
                   COUNT(*) FILTER (WHERE account_id IS NOT NULL) AS acct_views
            FROM base
            WHERE session_id IS NOT NULL
            GROUP BY session_id
        )
        SELECT
            (SELECT COUNT(DISTINCT {_IDENT}) FROM base) AS headline,
            (SELECT COUNT(DISTINCT account_id) FROM base) AS signed_in,
            (SELECT COUNT(*) FROM base) AS total_pageviews,
            COUNT(*) FILTER (WHERE anon_views > 0) AS anon_sessions,
            COUNT(*) FILTER (WHERE acct_views = 0 AND anon_views = 1)
                AS anon_one_and_done,
            COUNT(*) FILTER (WHERE acct_views = 0 AND anon_views >= 2)
                AS anon_engaged,
            COUNT(*) FILTER (WHERE anon_views > 0 AND acct_views > 0)
                AS anon_linked_sessions,
            (SELECT COUNT(DISTINCT account_id) FROM base)
                + COUNT(*) FILTER (WHERE acct_views = 0 AND anon_views >= 2)
                AS realistic_preview
        FROM sessions
        """
    )
    if not rows:
        return {k: 0 for k in _BREAKDOWN_KEYS}
    r = rows[0]
    return {k: int(r[k] or 0) for k in _BREAKDOWN_KEYS}


def dau_breakdown() -> Dict[str, Dict[str, int]]:
    """Today's DAU decomposed, for the UTC day and the New York day.

    Returns {"utc": {...}, "ny": {...}} with the _BREAKDOWN_KEYS metrics in
    each. The "utc" headline uses exactly the _IDENT definition and day
    grouping of dau_last_30_days(), so it matches the chart's today bar.
    "realistic_preview" previews a stricter definition: distinct signed-in
    accounts plus engaged anonymous sessions that never signed in that day.
    Diagnostic only: the headline DAU/WAU definitions are unchanged.
    """
    return {
        "utc": _breakdown_for_day(_DAY_FILTERS["utc"]),
        "ny": _breakdown_for_day(_DAY_FILTERS["ny"]),
    }


def one_and_done_top_paths(limit: int = 5) -> List[Dict[str, Any]]:
    """Top paths among today's (UTC) one-and-done anonymous sessions.

    Aggregate counts only (path + session count), no session ids or other
    per-visitor detail. Shows where single-hit anonymous traffic lands,
    which is where cookie-less bots and preview fetchers show up.
    """
    try:
        n = max(1, int(limit))
    except (TypeError, ValueError):
        n = 5
    rows = _fetchall(
        f"""
        WITH base AS (
            SELECT account_id, session_id, path
            FROM analytics_events
            WHERE event = 'pageview' AND {_DAY_FILTERS["utc"]}
            {_exclusion_clause()}
        ),
        sessions AS (
            SELECT session_id,
                   COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views,
                   COUNT(*) FILTER (WHERE account_id IS NOT NULL) AS acct_views
            FROM base
            WHERE session_id IS NOT NULL
            GROUP BY session_id
        ),
        one_done AS (
            SELECT session_id FROM sessions
            WHERE acct_views = 0 AND anon_views = 1
        )
        SELECT b.path AS path, COUNT(*) AS n
        FROM base b
        JOIN one_done o ON o.session_id = b.session_id
        WHERE b.account_id IS NULL AND b.path IS NOT NULL
        GROUP BY b.path
        ORDER BY n DESC, b.path
        LIMIT %s
        """,
        (n,),
    )
    return [
        {"path": str(r["path"]), "count": int(r["n"])}
        for r in rows
        if r["path"]
    ]


def wau_last_12_weeks() -> List[Dict[str, Any]]:
    """Distinct users per ISO week for the last 12 weeks."""
    rows = _fetchall(
        f"""
        SELECT date_trunc('week', created_at)::date AS w,
               COUNT(DISTINCT {_IDENT}) AS users
        FROM analytics_events
        WHERE event = 'pageview' AND created_at >= now() - interval '12 weeks'
        {_exclusion_clause()}
        GROUP BY 1 ORDER BY 1
        """
    )
    return [{"week": str(r["w"]), "users": int(r["users"])} for r in rows]


def signups_per_day() -> List[Dict[str, Any]]:
    """New accounts per day for the last 30 days (existing accounts table)."""
    rows = _fetchall(
        f"""
        SELECT created_at::date AS d, COUNT(*) AS n
        FROM accounts
        WHERE created_at >= now() - interval '30 days'
        {_exclusion_clause("id")}
        GROUP BY 1 ORDER BY 1
        """
    )
    return [{"date": str(r["d"]), "signups": int(r["n"])} for r in rows]


def feature_usage_by_week(weeks: int = 8) -> List[Dict[str, Any]]:
    """Non-pageview event counts per week, newest week first."""
    rows = _fetchall(
        f"""
        SELECT date_trunc('week', created_at)::date AS w, event, COUNT(*) AS n
        FROM analytics_events
        WHERE event <> 'pageview' AND created_at >= now() - make_interval(weeks => %s)
        {_exclusion_clause()}
        GROUP BY 1, 2 ORDER BY 1 DESC, 3 DESC
        """,
        (int(weeks),),
    )
    return [{"week": str(r["w"]), "event": str(r["event"]), "count": int(r["n"])} for r in rows]


def week_over_week_return() -> List[Dict[str, Any]]:
    """Per week: active users and how many were also active the prior week."""
    rows = _fetchall(
        f"""
        WITH weekly AS (
            SELECT date_trunc('week', created_at)::date AS w,
                   {_IDENT} AS ident
            FROM analytics_events
            WHERE event = 'pageview' AND created_at >= now() - interval '9 weeks'
            {_exclusion_clause()}
            GROUP BY 1, 2
        ),
        flagged AS (
            SELECT w, ident,
                   (LAG(w) OVER (PARTITION BY ident ORDER BY w)
                        = (w - interval '7 days')::date) AS returned
            FROM weekly
        )
        SELECT w, COUNT(*) AS active,
               COUNT(*) FILTER (WHERE returned) AS returned
        FROM flagged GROUP BY 1 ORDER BY 1
        """
    )
    out = []
    for r in rows:
        active = int(r["active"])
        returned = int(r["returned"])
        out.append({
            "week": str(r["w"]),
            "active": active,
            "returned": returned,
            "rate": round(100.0 * returned / active, 1) if active else 0.0,
        })
    return out


def funnel_last_30_days() -> Dict[str, int]:
    """Visitor -> signup -> league linked -> PRO counts for the last 30 days.

    Stages are period totals, not a strict cohort funnel; the page labels the
    period so the numbers read honestly.
    """
    def _one(sql: str, args: Sequence[Any] = ()) -> int:
        try:
            rows = _fetchall(sql, args)
            # get_conn() uses psycopg's dict_row: rows are dicts keyed by
            # column name. COUNT(*) with no alias comes back as "count".
            return int(rows[0]["count"]) if rows else 0
        except Exception:
            logger.debug("[analytics] funnel query failed", exc_info=True)
            return 0

    visitors = _one(
        f"""
        SELECT COUNT(DISTINCT {_IDENT})
        FROM analytics_events
        WHERE event = 'pageview' AND created_at >= now() - interval '30 days'
        {_exclusion_clause()}
        """
    )
    signups = _one(
        f"""SELECT COUNT(*) FROM accounts
        WHERE created_at >= now() - interval '30 days'
        {_exclusion_clause("id")}"""
    )
    # Linked stage reads user_leagues, not the league_linked event: the event
    # stream only exists from deploy day, while user_leagues has the full
    # history (added_at was backfilled at migration time for older rows, so
    # very old links may read as linked on the migration date; self-heals as
    # the window moves past it).
    linked = _one(
        f"""
        SELECT COUNT(DISTINCT account_id) FROM user_leagues
        WHERE added_at >= now() - interval '30 days'
        {_exclusion_clause()}
        """
    )
    # PRO stage: user_league_subscriptions.user_id is a provider-side TEXT id,
    # not an accounts.id, so the exclusion list does not apply here.
    pro = _one(
        """
        SELECT COUNT(DISTINCT user_id) FROM user_league_subscriptions
        WHERE created_at >= now() - interval '30 days'
        """
    )
    return {"visitors": visitors, "signups": signups, "linked": linked, "pro": pro}


# ── Gap filling (continuous axes for the charts) ────────────────────────────

def _utc_today() -> _dt.date:
    return _dt.datetime.now(_dt.timezone.utc).date()


def fill_daily_gaps(
    rows: List[Dict[str, Any]],
    date_key: str,
    value_key: str,
    days: int = 30,
    today: Optional[_dt.date] = None,
) -> List[tuple]:
    """rows: [{date_key: 'YYYY-MM-DD', value_key: n}] -> [(label 'MM-DD', value)]
    covering every day of the trailing `days`-day window, zeros filled."""
    day = today or _utc_today()
    by_date = {str(r[date_key]): r[value_key] for r in rows}
    out = []
    for i in range(days - 1, -1, -1):
        d = day - _dt.timedelta(days=i)
        iso = d.isoformat()
        out.append((iso[5:], int(by_date.get(iso, 0))))
    return out


def fill_weekly_gaps(
    rows: List[Dict[str, Any]],
    date_key: str,
    value_key: str,
    weeks: int = 12,
    today: Optional[_dt.date] = None,
) -> List[tuple]:
    """Same as fill_daily_gaps for Monday-anchored weeks (matches Postgres
    date_trunc('week', ...))."""
    day = today or _utc_today()
    monday = day - _dt.timedelta(days=day.weekday())
    by_date = {str(r[date_key]): r[value_key] for r in rows}
    out = []
    for i in range(weeks - 1, -1, -1):
        d = monday - _dt.timedelta(weeks=i)
        iso = d.isoformat()
        out.append((iso[5:], int(by_date.get(iso, 0))))
    return out


def events_table_ready() -> bool:
    """Whether any events have been recorded yet (drives empty states)."""
    try:
        rows = _fetchall("SELECT COUNT(*) FROM analytics_events")
        return int(rows[0]["count"]) > 0 if rows else False
    except Exception:
        return False
