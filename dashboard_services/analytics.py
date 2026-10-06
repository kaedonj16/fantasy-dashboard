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
from urllib.parse import urlparse

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
EVENT_PAYWALL_VIEWED = "paywall_viewed"

# Paywall surfaces that fire EVENT_PAYWALL_VIEWED from the client (see
# POST /api/analytics/paywall). Stable ids only; the beacon endpoint and
# the admin aggregation share this allowlist.
PAYWALL_SURFACES = (
    "plan_modal",
    "locked_metric",
    "breakout_nudge",
    "wrapped_finale",
    "player_modal_nudge",
    "movers_nudge",
)

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


def external_ref_host(referrer: Optional[str], request_host: Optional[str]) -> Optional[str]:
    """Hostname of an external referrer, or None.

    Pure helper for the pageview hook: returns the referrer's hostname
    (lowercased, no port) only when the referrer parses to a host that
    differs from the request's own host. Direct visits, internal
    navigation, missing and malformed referrers all return None. Host
    only: no referrer path or query is ever recorded.
    """
    try:
        ref = (referrer or "").strip()
        if not ref:
            return None
        host = urlparse(ref).hostname
        if not host:
            return None
        own = (request_host or "").strip().lower().split(":")[0]
        host = host.lower()
        if own and host == own:
            return None
        return host[:255]
    except Exception:
        return None


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
    """Distinct users per day for the last 30 days (pageviews only).

    Realistic definition, on a New York day: distinct signed-in accounts
    (any pageview count) plus anonymous sessions with 2+ pageviews that
    never signed in that day. Sessions that also signed in are counted
    only via their account, so nobody is double-counted. One-and-done
    anonymous sessions (the bulk of bot / bounce traffic) are excluded.
    """
    rows = _fetchall(
        f"""
        WITH base AS (
            SELECT account_id, session_id,
                   (created_at AT TIME ZONE 'America/New_York')::date AS d
            FROM analytics_events
            WHERE event = 'pageview' AND created_at >= now() - interval '30 days'
            {_exclusion_clause()}
        ),
        sessions AS (
            SELECT d, session_id,
                   COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views,
                   COUNT(*) FILTER (WHERE account_id IS NOT NULL) AS acct_views
            FROM base
            WHERE session_id IS NOT NULL
            GROUP BY 1, 2
        ),
        per_day AS (
            SELECT d, COUNT(DISTINCT account_id) AS signed_in
            FROM base
            GROUP BY 1
        ),
        engaged AS (
            SELECT d, COUNT(*) AS engaged
            FROM sessions
            WHERE acct_views = 0 AND anon_views >= 2
            GROUP BY 1
        )
        SELECT p.d AS d, p.signed_in + COALESCE(e.engaged, 0) AS users
        FROM per_day p
        LEFT JOIN engaged e ON e.d = p.d
        ORDER BY 1
        """
    )
    return [{"date": str(r["d"]), "users": int(r["users"])} for r in rows]


# ── DAU breakdown (reconciliation view for the headline DAU) ────────────────
#
# The raw distinct _IDENT count mixes three very different things:
# signed-in accounts (deduplicated, trustworthy), anonymous browser
# sessions (one per browser/device, and a brand new one for every
# cookie-less hit), and browsers that appear as BOTH in one day (counted
# twice). The headline DAU now uses the realistic definition (signed-in
# accounts plus engaged anonymous sessions, New York day); dau_breakdown()
# decomposes today's numbers so the admin page can reconcile the chart
# value against the raw old-definition count, under both the UTC day and
# the New York day.

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
    each. "headline" is the raw distinct _IDENT count (the old definition,
    kept for comparison). "realistic_preview" is the current headline
    definition: distinct signed-in accounts plus engaged anonymous
    sessions that never signed in that day; its New York value matches
    the chart's today bar.
    """
    return {
        "utc": _breakdown_for_day(_DAY_FILTERS["utc"]),
        "ny": _breakdown_for_day(_DAY_FILTERS["ny"]),
    }


def one_and_done_top_paths(limit: int = 5) -> List[Dict[str, Any]]:
    """Top paths among today's (New York day) one-and-done anonymous sessions.

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
            WHERE event = 'pageview' AND {_DAY_FILTERS["ny"]}
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
    """Distinct users per week for the last 12 weeks.

    Same realistic definition as DAU, per Monday-anchored New York week:
    distinct signed-in accounts plus anonymous sessions with 2+ pageviews
    that never signed in that week (sessions that also signed in count
    only via their account).
    """
    rows = _fetchall(
        f"""
        WITH base AS (
            SELECT account_id, session_id,
                   date_trunc('week', created_at AT TIME ZONE 'America/New_York')::date AS w
            FROM analytics_events
            WHERE event = 'pageview' AND created_at >= now() - interval '12 weeks'
            {_exclusion_clause()}
        ),
        sessions AS (
            SELECT w, session_id,
                   COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views,
                   COUNT(*) FILTER (WHERE account_id IS NOT NULL) AS acct_views
            FROM base
            WHERE session_id IS NOT NULL
            GROUP BY 1, 2
        ),
        per_week AS (
            SELECT w, COUNT(DISTINCT account_id) AS signed_in
            FROM base
            GROUP BY 1
        ),
        engaged AS (
            SELECT w, COUNT(*) AS engaged
            FROM sessions
            WHERE acct_views = 0 AND anon_views >= 2
            GROUP BY 1
        )
        SELECT p.w AS w, p.signed_in + COALESCE(e.engaged, 0) AS users
        FROM per_week p
        LEFT JOIN engaged e ON e.w = p.w
        ORDER BY 1
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
    """Per week: active users and how many were also active the prior week.

    The weekly active set uses the same realistic identities as WAU, so
    the Active column matches the WAU chart: signed-in accounts as
    'a:'||account_id and engaged anonymous sessions (2+ pageviews, never
    signed in that week) as 's:'||session_id.
    """
    rows = _fetchall(
        f"""
        WITH base AS (
            SELECT account_id, session_id,
                   date_trunc('week', created_at AT TIME ZONE 'America/New_York')::date AS w
            FROM analytics_events
            WHERE event = 'pageview' AND created_at >= now() - interval '9 weeks'
            {_exclusion_clause()}
        ),
        sessions AS (
            SELECT w, session_id,
                   COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views,
                   COUNT(*) FILTER (WHERE account_id IS NOT NULL) AS acct_views
            FROM base
            WHERE session_id IS NOT NULL
            GROUP BY 1, 2
        ),
        weekly AS (
            SELECT w, 'a:' || account_id::text AS ident
            FROM base
            WHERE account_id IS NOT NULL
            GROUP BY 1, 2
            UNION
            SELECT w, 's:' || session_id AS ident
            FROM sessions
            WHERE acct_views = 0 AND anon_views >= 2
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


# ── Owner metrics: paywalls, traffic, activation, revenue, retention ───────

def _window_days(days: Any, default: int = 30) -> int:
    try:
        return max(1, int(days))
    except (TypeError, ValueError):
        return default


def paywall_summary(days: int = 30) -> Dict[str, Any]:
    """Paywall views and their downstream conversion for the last `days`.

    Views are EVENT_PAYWALL_VIEWED rows (fired client-side via
    POST /api/analytics/paywall; counting rule lives in static/paywall.js:
    one event per plan-modal render and one per inline nudge render).
    Viewer identity is _IDENT (account when signed in, else session).
    Conversion joins on that identity and requires the checkout /
    subscription event to be timestamped AFTER the viewer's first
    paywall view in the window.
    """
    days = _window_days(days)
    ident = _IDENT
    rows = _fetchall(
        f"""
        WITH views AS (
            SELECT {ident} AS viewer, created_at
            FROM analytics_events
            WHERE event = 'paywall_viewed'
              AND created_at >= now() - make_interval(days => %s)
            {_exclusion_clause()}
        ),
        first_view AS (
            SELECT viewer, MIN(created_at) AS first_at
            FROM views GROUP BY viewer
        )
        SELECT
            (SELECT COUNT(*) FROM views) AS total_views,
            (SELECT COUNT(*) FROM first_view) AS viewers,
            (SELECT COUNT(*) FROM first_view f WHERE EXISTS (
                SELECT 1 FROM analytics_events e
                WHERE e.event = 'checkout_started'
                  AND COALESCE(e.account_id::text, 's:' || COALESCE(e.session_id, '-')) = f.viewer
                  AND e.created_at > f.first_at
                  {_exclusion_clause("e.account_id")}
            )) AS checkout_viewers,
            (SELECT COUNT(*) FROM first_view f WHERE EXISTS (
                SELECT 1 FROM analytics_events e
                WHERE e.event = 'pro_subscribed'
                  AND COALESCE(e.account_id::text, 's:' || COALESCE(e.session_id, '-')) = f.viewer
                  AND e.created_at > f.first_at
                  {_exclusion_clause("e.account_id")}
            )) AS subscribed_viewers
        """,
        (days,),
    )
    r = rows[0] if rows else {}
    total_views = int(r.get("total_views") or 0)
    viewers = int(r.get("viewers") or 0)
    checkout_viewers = int(r.get("checkout_viewers") or 0)
    subscribed_viewers = int(r.get("subscribed_viewers") or 0)

    surface_rows = _fetchall(
        f"""
        SELECT COALESCE(NULLIF(props->>'surface', ''), 'unknown') AS surface,
               COUNT(*) AS views,
               COUNT(DISTINCT {ident}) AS viewers
        FROM analytics_events
        WHERE event = 'paywall_viewed'
          AND created_at >= now() - make_interval(days => %s)
        {_exclusion_clause()}
        GROUP BY 1 ORDER BY views DESC, surface
        """,
        (days,),
    )
    metric_rows = _fetchall(
        f"""
        SELECT props->>'metric' AS metric,
               COUNT(*) AS views,
               COUNT(DISTINCT {ident}) AS viewers
        FROM analytics_events
        WHERE event = 'paywall_viewed'
          AND created_at >= now() - make_interval(days => %s)
          AND COALESCE(props->>'metric', '') <> ''
        {_exclusion_clause()}
        GROUP BY 1 ORDER BY views DESC, metric
        LIMIT 10
        """,
        (days,),
    )
    return {
        "days": days,
        "total_views": total_views,
        "viewers": viewers,
        "checkout_viewers": checkout_viewers,
        "subscribed_viewers": subscribed_viewers,
        "checkout_pct": round(100.0 * checkout_viewers / viewers, 1) if viewers else 0.0,
        "subscribed_pct": round(100.0 * subscribed_viewers / viewers, 1) if viewers else 0.0,
        "by_surface": [
            {"surface": str(x["surface"]), "views": int(x["views"]), "viewers": int(x["viewers"])}
            for x in surface_rows
        ],
        "by_metric": [
            {"metric": str(x["metric"]), "views": int(x["views"]), "viewers": int(x["viewers"])}
            for x in metric_rows
            if x["metric"]
        ],
    }


# First-touch CTE shared by the two traffic aggregations: each session's
# earliest pageview in the window gives its landing path and referrer
# host ('direct' when the pageview recorded no external ref_host).
_TRAFFIC_CTES = """
    WITH pv AS (
        SELECT account_id, session_id, path, props, created_at
        FROM analytics_events
        WHERE event = 'pageview' AND session_id IS NOT NULL
          AND created_at >= now() - make_interval(days => %s)
        {excl}
    ),
    sessions AS (
        SELECT session_id,
               COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views,
               COUNT(*) FILTER (WHERE account_id IS NOT NULL) AS acct_views
        FROM pv
        GROUP BY session_id
    ),
    first_touch AS (
        SELECT DISTINCT ON (session_id)
               session_id, path AS landing_path,
               COALESCE(NULLIF(props->>'ref_host', ''), 'direct') AS source
        FROM pv
        ORDER BY session_id, created_at ASC
    )
"""


def traffic_sources(days: int = 30) -> List[Dict[str, Any]]:
    """Sessions by first-touch referrer host (or 'direct'), last `days`.

    Engaged uses the realistic DAU definition per session (2+ pageviews,
    never signed in); signed_in counts sessions with any signed-in
    pageview in the window.
    """
    days = _window_days(days)
    rows = _fetchall(
        _TRAFFIC_CTES.format(excl=_exclusion_clause())
        + """
        SELECT f.source AS source,
               COUNT(*) AS sessions,
               COUNT(*) FILTER (WHERE s.acct_views = 0 AND s.anon_views >= 2) AS engaged,
               COUNT(*) FILTER (WHERE s.acct_views > 0) AS signed_in
        FROM first_touch f
        JOIN sessions s ON s.session_id = f.session_id
        GROUP BY f.source
        ORDER BY sessions DESC, f.source
        LIMIT 10
        """,
        (days,),
    )
    return [
        {
            "source": str(r["source"]),
            "sessions": int(r["sessions"]),
            "engaged": int(r["engaged"]),
            "signed_in": int(r["signed_in"]),
        }
        for r in rows
    ]


def top_landing_paths(days: int = 30) -> List[Dict[str, Any]]:
    """Sessions by first-touch landing path, last `days` (top 10)."""
    days = _window_days(days)
    rows = _fetchall(
        _TRAFFIC_CTES.format(excl=_exclusion_clause())
        + """
        SELECT f.landing_path AS path,
               COUNT(*) AS sessions,
               COUNT(*) FILTER (WHERE s.acct_views = 0 AND s.anon_views >= 2) AS engaged
        FROM first_touch f
        JOIN sessions s ON s.session_id = f.session_id
        WHERE f.landing_path IS NOT NULL
        GROUP BY f.landing_path
        ORDER BY sessions DESC, f.landing_path
        LIMIT 10
        """,
        (days,),
    )
    return [
        {"path": str(r["path"]), "sessions": int(r["sessions"]), "engaged": int(r["engaged"])}
        for r in rows
        if r["path"]
    ]


def activation_cohort(days: int = 30) -> Dict[str, Any]:
    """Signup -> league-linked cohort for accounts created in the window.

    Unlike the period funnel, every stage counts the SAME accounts:
    signups in the window, and how many of them have a user_leagues row
    added within 24 hours / 7 days of their account creation, or at any
    point up to now. Provider split (when built) reads the provider of
    each 7-day-linked account's earliest link.
    """
    days = _window_days(days)
    rows = _fetchall(
        f"""
        WITH cohort AS (
            SELECT id, created_at
            FROM accounts
            WHERE created_at >= now() - make_interval(days => %s)
            {_exclusion_clause("id")}
        ),
        links AS (
            SELECT c.id,
                   EXISTS (
                       SELECT 1 FROM user_leagues ul
                       WHERE ul.account_id = c.id
                         AND ul.added_at >= c.created_at
                         AND ul.added_at < c.created_at + interval '24 hours'
                   ) AS within_24h,
                   EXISTS (
                       SELECT 1 FROM user_leagues ul
                       WHERE ul.account_id = c.id
                         AND ul.added_at >= c.created_at
                         AND ul.added_at < c.created_at + interval '7 days'
                   ) AS within_7d,
                   EXISTS (
                       SELECT 1 FROM user_leagues ul
                       WHERE ul.account_id = c.id
                   ) AS ever
            FROM cohort c
        )
        SELECT COUNT(*) AS signups,
               COUNT(*) FILTER (WHERE within_24h) AS linked_24h,
               COUNT(*) FILTER (WHERE within_7d) AS linked_7d,
               COUNT(*) FILTER (WHERE ever) AS linked_ever
        FROM links
        """,
        (days,),
    )
    r = rows[0] if rows else {}
    signups = int(r.get("signups") or 0)
    linked_24h = int(r.get("linked_24h") or 0)
    linked_7d = int(r.get("linked_7d") or 0)
    linked_ever = int(r.get("linked_ever") or 0)

    by_provider: List[Dict[str, Any]] = []
    provider_col = _user_leagues_provider_column()
    if provider_col:
        prov_rows = _fetchall(
            f"""
            WITH cohort AS (
                SELECT id, created_at
                FROM accounts
                WHERE created_at >= now() - make_interval(days => %s)
                {_exclusion_clause("id")}
            ),
            first_links AS (
                SELECT DISTINCT ON (ul.account_id)
                       ul.account_id, ul.{provider_col} AS provider
                FROM user_leagues ul
                JOIN cohort c ON c.id = ul.account_id
                WHERE ul.added_at >= c.created_at
                  AND ul.added_at < c.created_at + interval '7 days'
                ORDER BY ul.account_id, ul.added_at ASC
            )
            SELECT provider, COUNT(*) AS n
            FROM first_links
            GROUP BY provider
            ORDER BY n DESC, provider
            """,
            (days,),
        )
        by_provider = [
            {"provider": str(x["provider"]), "count": int(x["n"])}
            for x in prov_rows
            if x["provider"]
        ]
    return {
        "days": days,
        "signups": signups,
        "linked_24h": linked_24h,
        "linked_7d": linked_7d,
        "linked_ever": linked_ever,
        "pct_24h": round(100.0 * linked_24h / signups, 1) if signups else 0.0,
        "pct_7d": round(100.0 * linked_7d / signups, 1) if signups else 0.0,
        "pct_ever": round(100.0 * linked_ever / signups, 1) if signups else 0.0,
        "by_provider": by_provider,
    }


_PROVIDER_COLUMN_CANDIDATES = ("provider", "platform", "source")
_user_leagues_provider_col: Optional[str] = None
_user_leagues_provider_col_checked = False


def _user_leagues_provider_column() -> Optional[str]:
    """The user_leagues provider/platform column name, or None.

    Probed once per process from information_schema so the cohort's
    provider split only builds when the column really exists; a failed
    probe (no DB, missing table) returns None and the split is skipped.
    """
    global _user_leagues_provider_col, _user_leagues_provider_col_checked
    if _user_leagues_provider_col_checked:
        return _user_leagues_provider_col
    _user_leagues_provider_col_checked = True
    try:
        rows = _fetchall(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'user_leagues'
            """
        )
        names = {str(r["column_name"]) for r in rows}
        for cand in _PROVIDER_COLUMN_CANDIDATES:
            if cand in names:
                _user_leagues_provider_col = cand
                break
    except Exception:
        logger.debug("[analytics] user_leagues provider probe failed", exc_info=True)
    return _user_leagues_provider_col


def revenue_summary(days: int = 30) -> Dict[str, Any]:
    """Checkout / subscription numbers for the last `days` plus current PRO.

    Event-based counts come from analytics_events; current/new PRO counts
    come from user_league_subscriptions. No MRR: subscription rows do not
    carry a reliably priced amount (see subscriptions schema), so no
    revenue figure is estimated. Table-based counts degrade to 0 when
    the subscriptions table is unavailable, mirroring the funnel.
    """
    days = _window_days(days)

    def _one(sql: str, args: Sequence[Any] = ()) -> int:
        try:
            rows = _fetchall(sql, args)
            return int(rows[0]["count"]) if rows else 0
        except Exception:
            logger.debug("[analytics] revenue query failed", exc_info=True)
            return 0

    window = "created_at >= now() - make_interval(days => %s)"
    excl = _exclusion_clause()
    checkout_identities = _one(
        f"SELECT COUNT(DISTINCT {_IDENT}) AS count FROM analytics_events "
        f"WHERE event = 'checkout_started' AND {window} {excl}",
        (days,),
    )
    subscribed_events = _one(
        f"SELECT COUNT(*) AS count FROM analytics_events "
        f"WHERE event = 'pro_subscribed' AND {window} {excl}",
        (days,),
    )
    subscribed_identities = _one(
        f"SELECT COUNT(DISTINCT {_IDENT}) AS count FROM analytics_events "
        f"WHERE event = 'pro_subscribed' AND {window} {excl}",
        (days,),
    )
    cancelled_events = _one(
        f"SELECT COUNT(*) AS count FROM analytics_events "
        f"WHERE event = 'pro_cancelled' AND {window} {excl}",
        (days,),
    )
    # user_league_subscriptions.user_id is a provider-side TEXT id, not an
    # accounts.id, so the exclusion list does not apply to these two.
    # "Currently active" is the entitlement predicate used across the app:
    # status 'active' and not past expires_at.
    active_pro = _one(
        "SELECT COUNT(*) AS count FROM user_league_subscriptions "
        "WHERE subscription_status = 'active' AND expires_at > now()"
    )
    new_pro = _one(
        "SELECT COUNT(*) AS count FROM user_league_subscriptions "
        f"WHERE {window}",
        (days,),
    )
    return {
        "days": days,
        "checkout_identities": checkout_identities,
        "subscribed_events": subscribed_events,
        "subscribed_identities": subscribed_identities,
        "cancelled_events": cancelled_events,
        "checkout_conversion_pct": (
            round(100.0 * subscribed_identities / checkout_identities, 1)
            if checkout_identities else 0.0
        ),
        "active_pro_subscriptions": active_pro,
        "new_pro_subscriptions": new_pro,
    }


def account_retention() -> List[Dict[str, Any]]:
    """Week-over-week return for signed-in accounts only.

    Same shape and New York Monday weeks as week_over_week_return(), but
    the weekly active set is account ids only (no anonymous sessions),
    so this tracks the signed-in core rather than all traffic.
    """
    rows = _fetchall(
        f"""
        WITH weekly AS (
            SELECT date_trunc('week', created_at AT TIME ZONE 'America/New_York')::date AS w,
                   account_id
            FROM analytics_events
            WHERE event = 'pageview' AND account_id IS NOT NULL
              AND created_at >= now() - interval '9 weeks'
            {_exclusion_clause()}
            GROUP BY 1, 2
        ),
        flagged AS (
            SELECT w, account_id,
                   (LAG(w) OVER (PARTITION BY account_id ORDER BY w)
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


def cohort_heatmap(weeks: int = 8, active_weeks: int = 6) -> List[Dict[str, Any]]:
    """Signup-week cohort retention heatmap.

    For each signup week (last `weeks` Monday-anchored New York weeks): the
    share of that week's signup cohort (signed-in accounts only, exclusion
    list applied) with >= 1 pageview in week 0..`active_weeks`-1 after
    their signup week. Always returns `weeks` rows, oldest cohort first;
    weeks with no signups have size 0 and all-None pcts, and offsets whose
    target week has not happened yet for that cohort read as None.
    """
    try:
        weeks = max(1, int(weeks))
    except (TypeError, ValueError):
        weeks = 8
    try:
        active_weeks = max(1, min(12, int(active_weeks)))
    except (TypeError, ValueError):
        active_weeks = 6

    # One LEFT JOIN per offset week: explicit and test-friendly. Offsets are
    # ints validated above, so interpolating them is safe.
    joins = " ".join(
        "LEFT JOIN activity a%(o)d ON a%(o)d.account_id = c.id"
        " AND a%(o)d.active_w = c.cohort_w + %(d)d" % {"o": o, "d": o * 7}
        for o in range(active_weeks)
    )
    counts = ", ".join(
        "COUNT(a%d.account_id) AS w%d" % (o, o) for o in range(active_weeks)
    )
    rows = _fetchall(
        f"""
        WITH cohorts AS (
            SELECT id,
                   date_trunc('week', created_at AT TIME ZONE 'America/New_York')::date AS cohort_w
            FROM accounts
            WHERE (created_at AT TIME ZONE 'America/New_York')::date
                  >= date_trunc('week', now() AT TIME ZONE 'America/New_York')::date
                     - make_interval(weeks => %s)
            {_exclusion_clause("id")}
        ),
        activity AS (
            SELECT account_id,
                   date_trunc('week', created_at AT TIME ZONE 'America/New_York')::date AS active_w
            FROM analytics_events
            WHERE event = 'pageview' AND account_id IS NOT NULL
            {_exclusion_clause()}
            GROUP BY 1, 2
        )
        SELECT c.cohort_w AS cohort_week, COUNT(*) AS size, {counts}
        FROM cohorts c
        {joins}
        GROUP BY 1 ORDER BY 1
        """,
        (weeks - 1,),
    )
    by_week = {str(r["cohort_week"]): r for r in rows}
    today = ny_today()
    monday = today - _dt.timedelta(days=today.weekday())
    out: List[Dict[str, Any]] = []
    for i in range(weeks - 1, -1, -1):
        cw = monday - _dt.timedelta(weeks=i)
        iso = cw.isoformat()
        r = by_week.get(iso)
        size = int(r["size"]) if r else 0
        entry: Dict[str, Any] = {"cohort_week": iso, "size": size}
        for o in range(active_weeks):
            key = "w%d" % o
            if cw + _dt.timedelta(weeks=o) > monday:
                entry[key] = None  # that week has not happened yet
            elif r and size:
                entry[key] = round(100.0 * int(r[key]) / size, 1)
            else:
                entry[key] = None
        out.append(entry)
    return out


def feature_usage_ranking(days: int = 30) -> List[Dict[str, Any]]:
    """Non-pageview events ranked by use over the last `days`.

    Per event: total uses and distinct identities (_IDENT: account when
    signed in, else session), sorted by uses desc. This is the direct
    "which features are used most" view; the by-week table shows trend.
    """
    days = _window_days(days)
    rows = _fetchall(
        f"""
        SELECT event, COUNT(*) AS uses, COUNT(DISTINCT {_IDENT}) AS users
        FROM analytics_events
        WHERE event <> 'pageview'
          AND created_at >= now() - make_interval(days => %s)
        {_exclusion_clause()}
        GROUP BY event
        ORDER BY uses DESC, event
        """,
        (days,),
    )
    return [
        {"event": str(r["event"]), "uses": int(r["uses"]), "users": int(r["users"])}
        for r in rows
    ]


# ── Gap filling (continuous axes for the charts) ────────────────────────────

def _utc_today() -> _dt.date:
    return _dt.datetime.now(_dt.timezone.utc).date()


def ny_today() -> _dt.date:
    """Today's date in America/New_York (the DAU/WAU bucket timezone)."""
    try:
        from zoneinfo import ZoneInfo

        return _dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return _utc_today()


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
