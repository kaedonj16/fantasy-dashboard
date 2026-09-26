"""Churn reduction: dunning, cancellation save flow, trial reminders, win-back.

Owns the ``churn_dunning`` (open payment-failure episodes) and ``churn_events``
(append-only log of churn touches) tables, plus the daily scan that escalates
dunning, sends trial-expiry reminders, and sends the one-time win-back offer.

Trial integration point: the free-trial workstream has not landed yet, so
``find_trials_due()`` / ``trial_days_left_for_account()`` query a ``pro_trials``
table *if it exists* and return empty otherwise. When the trial workstream
lands, it should write rows with (account_id, trial_ends_at, plan); everything
below starts working without further changes.

Never import stripe at module level: only the scan paths that need it call
``_stripe()`` so unit tests and pages stay Stripe-free.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SCHEMA_READY = False

# Dunning policy: immediate touch on invoice.payment_failed, one follow-up
# ~3 days later while still past due, then quiet (banner keeps nudging).
DUNNING_TOUCHES = 2
DUNNING_ESCALATION_DAYS = 3

# Trial reminders go out 2 days and 1 day before trial expiry.
TRIAL_REMINDER_DAYS = (2, 1)

# Win-back: one email per account per year, only when the sub ended >30d ago
# and nothing active remains.
WINBACK_MIN_LAPSE_DAYS = 30
WINBACK_RESEND_DAYS = 365

PAUSE_MONTHS = 2  # "pause PRO for 2 months instead" save offer


def _now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_schema(conn=None) -> None:
    """Create churn tables. Safe to call repeatedly."""
    global _SCHEMA_READY
    if _SCHEMA_READY and conn is None:
        return

    def _run(c):
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS churn_dunning (
                stripe_subscription_id TEXT PRIMARY KEY,
                account_id INTEGER,
                email TEXT NOT NULL DEFAULT '',
                plan TEXT NOT NULL DEFAULT '',
                first_failed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                touch_count INTEGER NOT NULL DEFAULT 0,
                last_touch_at TIMESTAMPTZ,
                resolved_at TIMESTAMPTZ
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS churn_events (
                id SERIAL PRIMARY KEY,
                account_id INTEGER,
                kind TEXT NOT NULL,
                detail JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        c.execute(
            """CREATE INDEX IF NOT EXISTS churn_events_kind_idx
               ON churn_events (kind, created_at)"""
        )
        try:
            c.commit()
        except Exception:
            pass

    try:
        if conn is not None:
            _run(conn)
        else:
            from dashboard_services.db import get_conn

            with get_conn() as c:
                _run(c)
        _SCHEMA_READY = True
    except Exception:
        logger.debug("[churn] ensure_schema failed", exc_info=True)
        raise


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return {}


def record_event(account_id: Optional[int], kind: str, detail: Optional[dict] = None) -> bool:
    """Append one row to the churn event log. Never raises."""
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            ensure_schema(conn)
            conn.execute(
                "INSERT INTO churn_events (account_id, kind, detail) VALUES (%s, %s, %s)",
                (
                    int(account_id) if account_id else None,
                    str(kind),
                    json.dumps(detail or {}),
                ),
            )
            conn.commit()
        return True
    except Exception:
        logger.debug("[churn] record_event failed kind=%s", kind, exc_info=True)
        return False


def has_event(kind: str, account_id: Optional[int] = None, since_days: Optional[int] = None,
              detail_key: str = "", detail_value: str = "") -> bool:
    """True when a matching churn event already exists (idempotency guard)."""
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            ensure_schema(conn)
            sql = "SELECT 1 FROM churn_events WHERE kind = %s"
            params: list = [kind]
            if account_id:
                sql += " AND account_id = %s"
                params.append(int(account_id))
            if since_days is not None:
                sql += " AND created_at > now() - (%s || ' days')::interval"
                params.append(int(since_days))
            if detail_key:
                sql += " AND detail ->> %s = %s"
                params += [detail_key, detail_value]
            sql += " LIMIT 1"
            row = conn.execute(sql, tuple(params)).fetchone()
            return row is not None
    except Exception:
        logger.debug("[churn] has_event failed kind=%s", kind, exc_info=True)
        return False


def open_dunning(subscription_id: str, account_id: Optional[int], email: str,
                 plan: str) -> bool:
    """Open (or reuse) a dunning episode. Returns True when touch 1 should send.

    Idempotent: a second webhook for the same failure episode returns False.
    A resolved episode older than the escalation window reopens as a new one.
    """
    sub_id = (subscription_id or "").strip()
    if not sub_id:
        return False
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            ensure_schema(conn)
            row = conn.execute(
                "SELECT touch_count, resolved_at, first_failed_at FROM churn_dunning "
                "WHERE stripe_subscription_id = %s",
                (sub_id,),
            ).fetchone()
            row = _row_to_dict(row)
            if row:
                if row.get("resolved_at") is None:
                    return False  # episode already open; daily scan escalates
                first = row.get("first_failed_at")
                try:
                    old = first < _now() - timedelta(days=DUNNING_ESCALATION_DAYS)
                except Exception:
                    old = True
                if not old:
                    return False
                conn.execute(
                    """UPDATE churn_dunning SET account_id=%s, email=%s, plan=%s,
                       first_failed_at=now(), touch_count=1, last_touch_at=now(),
                       resolved_at=NULL WHERE stripe_subscription_id=%s""",
                    (int(account_id) if account_id else None, email or "",
                     plan or "", sub_id),
                )
                conn.commit()
                return True
            conn.execute(
                """INSERT INTO churn_dunning
                   (stripe_subscription_id, account_id, email, plan,
                    first_failed_at, touch_count, last_touch_at)
                   VALUES (%s, %s, %s, %s, now(), 1, now())""",
                (int(account_id) if account_id else None, email or "", plan or "", sub_id),
            )
            conn.commit()
            return True
    except Exception:
        logger.debug("[churn] open_dunning failed sub=%s", sub_id, exc_info=True)
        return False


def bump_dunning_touch(subscription_id: str) -> bool:
    """Advance an open episode to the next touch. Returns False when capped."""
    sub_id = (subscription_id or "").strip()
    if not sub_id:
        return False
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            ensure_schema(conn)
            row = conn.execute(
                "SELECT touch_count FROM churn_dunning "
                "WHERE stripe_subscription_id = %s AND resolved_at IS NULL",
                (sub_id,),
            ).fetchone()
            row = _row_to_dict(row)
            if not row:
                return False
            if int(row.get("touch_count") or 0) >= DUNNING_TOUCHES:
                return False  # quiet after the cap; banner keeps nudging
            conn.execute(
                "UPDATE churn_dunning SET touch_count = touch_count + 1, "
                "last_touch_at = now() WHERE stripe_subscription_id = %s",
                (sub_id,),
            )
            conn.commit()
            return True
    except Exception:
        logger.debug("[churn] bump_dunning_touch failed sub=%s", sub_id, exc_info=True)
        return False


def resolve_dunning(subscription_id: str) -> None:
    """Close a dunning episode (payment recovered or sub canceled)."""
    sub_id = (subscription_id or "").strip()
    if not sub_id:
        return
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            ensure_schema(conn)
            conn.execute(
                "UPDATE churn_dunning SET resolved_at = now() "
                "WHERE stripe_subscription_id = %s AND resolved_at IS NULL",
                (sub_id,),
            )
            conn.commit()
    except Exception:
        logger.debug("[churn] resolve_dunning failed sub=%s", sub_id, exc_info=True)


def dunning_due_for_escalation() -> list[dict]:
    """Open episodes whose follow-up touch is due (touch 1 sent >N days ago)."""
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            ensure_schema(conn)
            rows = conn.execute(
                """SELECT stripe_subscription_id, account_id, email, plan, touch_count
                   FROM churn_dunning
                   WHERE resolved_at IS NULL
                     AND touch_count = 1
                     AND last_touch_at < now() - (%s || ' days')::interval""",
                (DUNNING_ESCALATION_DAYS,),
            ).fetchall()
            return [_row_to_dict(r) for r in (rows or [])]
    except Exception:
        logger.debug("[churn] dunning_due_for_escalation failed", exc_info=True)
        return []


def has_open_dunning(account_id: Optional[int]) -> bool:
    """True when the account has an unresolved payment-failure episode."""
    if not account_id:
        return False
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            ensure_schema(conn)
            row = conn.execute(
                "SELECT 1 FROM churn_dunning WHERE account_id = %s "
                "AND resolved_at IS NULL LIMIT 1",
                (int(account_id),),
            ).fetchone()
            return row is not None
    except Exception:
        logger.debug("[churn] has_open_dunning failed", exc_info=True)
        return False


def _stripe():
    import stripe as _s

    _s.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    return _s


def stripe_sub_is_past_due(subscription_id: str) -> Optional[bool]:
    """Check a Stripe subscription's live status. None when unknowable."""
    try:
        sub = _stripe().Subscription.retrieve(subscription_id)
        if isinstance(sub, dict):
            status = sub.get("status")
        else:
            status = getattr(sub, "status", None)
        if status in ("past_due", "unpaid", "incomplete"):
            return True
        if status in ("active", "trialing"):
            return False
        return None
    except Exception:
        logger.debug("[churn] stripe_sub_is_past_due failed", exc_info=True)
        return None


# ── Trial integration point (free-trial workstream has not landed) ────────────

def find_trials_due() -> list[dict]:
    """Trials expiring within the reminder window.

    Reads the ``pro_trials`` table (expected columns: account_id,
    trial_ends_at). Trials grant full PRO (Hall of Fame-equivalent), so the
    reminder labels them that way. Returns [] until the table exists.
    """
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'pro_trials'"
            ).fetchone()
            if not exists:
                return []
            rows = conn.execute(
                """SELECT account_id, trial_ends_at
                   FROM pro_trials
                   WHERE trial_ends_at > now()
                     AND trial_ends_at <= now() + (%s || ' days')::interval""",
                (max(TRIAL_REMINDER_DAYS),),
            ).fetchall()
            out = []
            for r in rows or []:
                d = _row_to_dict(r)
                ends = d.get("trial_ends_at")
                try:
                    if ends and ends.tzinfo is None:
                        ends = ends.replace(tzinfo=timezone.utc)
                    days_left = (ends - _now()).total_seconds() / 86400
                except Exception:
                    continue
                d["plan"] = "hall_of_fame"
                d["days_left"] = days_left
                out.append(d)
            return out
    except Exception:
        logger.debug("[churn] find_trials_due failed", exc_info=True)
        return []


def trial_days_left_for_account(account_id: Optional[int]) -> Optional[int]:
    """Whole days until this account's trial ends, or None when unknown/none."""
    if not account_id:
        return None
    for trial in find_trials_due():
        try:
            if int(trial.get("account_id") or 0) == int(account_id):
                return max(0, int(trial.get("days_left") or 0))
        except (TypeError, ValueError):
            continue
    return None


# ── Win-back ──────────────────────────────────────────────────────────────────

def account_has_active_sub(account_id: Optional[int]) -> bool:
    """True when any subscription table shows an active, unexpired row."""
    if not account_id:
        return False
    keys = (f"acct:{int(account_id)}", str(int(account_id)))
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            now = _now()
            for table, user_col in (
                ("user_subscriptions", "user_id"),
                ("league_subscriptions", "subscriber_user_id"),
                ("user_league_subscriptions", "user_id"),
            ):
                row = conn.execute(
                    f"SELECT 1 FROM {table} WHERE {user_col} = ANY(%s) "
                    "AND subscription_status = 'active' AND expires_at > %s LIMIT 1",
                    (list(keys), now),
                ).fetchone()
                if row:
                    return True
            return False
    except Exception:
        logger.debug("[churn] account_has_active_sub failed", exc_info=True)
        return False


def find_winback_candidates(limit: int = 200) -> list[dict]:
    """Lapsed PRO accounts: canceled >30d ago, nothing active, no recent win-back.

    Returns account rows (id, email, first_name) ready for the offer email.
    """
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            ensure_schema(conn)
            rows = conn.execute(
                """SELECT DISTINCT
                       NULLIF(regexp_replace(user_id, '^acct:', ''), '')::int AS account_id
                   FROM user_subscriptions
                   WHERE subscription_status = 'canceled'
                     AND user_id LIKE 'acct:%%'
                     AND regexp_replace(user_id, '^acct:', '') ~ '^[0-9]+$'
                     AND updated_at < now() - (%s || ' days')::interval
                   UNION
                   SELECT DISTINCT
                       NULLIF(regexp_replace(subscriber_user_id, '^acct:', ''), '')::int
                   FROM league_subscriptions
                   WHERE subscription_status = 'canceled'
                     AND subscriber_user_id LIKE 'acct:%%'
                     AND regexp_replace(subscriber_user_id, '^acct:', '') ~ '^[0-9]+$'
                     AND updated_at < now() - (%s || ' days')::interval
                   LIMIT %s""",
                (WINBACK_MIN_LAPSE_DAYS, WINBACK_MIN_LAPSE_DAYS, limit),
            ).fetchall()
            out = []
            seen = set()
            for r in rows or []:
                d = _row_to_dict(r)
                aid = d.get("account_id")
                try:
                    aid = int(aid) if aid is not None else None
                except (TypeError, ValueError):
                    aid = None
                if not aid or aid in seen:
                    continue
                seen.add(aid)
                if account_has_active_sub(aid):
                    continue
                if has_event("winback_sent", account_id=aid, since_days=WINBACK_RESEND_DAYS):
                    continue
                if has_open_dunning(aid):
                    continue  # past-due is dunning's job, not win-back's
                acct = conn.execute(
                    "SELECT id, email, first_name FROM accounts WHERE id = %s", (aid,)
                ).fetchone()
                acct = _row_to_dict(acct)
                if not acct.get("email"):
                    continue
                out.append(acct)
            return out
    except Exception:
        logger.debug("[churn] find_winback_candidates failed", exc_info=True)
        return []


# ── Cancellation save flow ────────────────────────────────────────────────────

CANCEL_REASONS = (
    "too_expensive",
    "not_using",
    "missing_feature",
    "switched_tool",
    "seasonal",
    "other",
)


def active_subscriptions_for_user(user_id: str) -> list[dict]:
    """Active subscription rows owned by a checkout identity.

    Returns dicts with table, stripe_subscription_id, plan, expires_at,
    league_id, platform. Used by the pricing manage card and the cancel endpoints.
    """
    uid = (user_id or "").strip()
    if not uid:
        return []
    out: list[dict] = []
    try:
        from dashboard_services.db import get_conn

        now = _now()
        with get_conn() as conn:
            cur = conn.cursor() if hasattr(conn, "cursor") else conn
            specs = (
                ("user_subscriptions", "user_id", "plan_key", "user_id"),
                ("league_subscriptions", "subscriber_user_id", None, "league_id"),
                ("user_league_subscriptions", "user_id", None, "league_id"),
            )
            # plan_key may not exist on databases created before migration 039.
            _have_plan_key = True
            for table, user_col, key_col, league_col in specs:
                try:
                    sel = ", plan_key" if key_col else ""
                    rows = cur.execute(
                        f"""SELECT stripe_subscription_id, expires_at, {league_col}{sel}, platform
                            FROM {table}
                            WHERE {user_col} = %s
                              AND subscription_status = 'active'
                              AND expires_at > %s""",
                        (uid, now),
                    ).fetchall()
                except Exception:
                    if key_col and _have_plan_key:
                        # plan_key column missing (pre-039 DB): retry without it.
                        _have_plan_key = False
                        try:
                            rows = cur.execute(
                                f"""SELECT stripe_subscription_id, expires_at, {league_col}, platform
                                    FROM {table}
                                    WHERE {user_col} = %s
                                      AND subscription_status = 'active'
                                      AND expires_at > %s""",
                                (uid, now),
                            ).fetchall()
                        except Exception:
                            continue
                    else:
                        continue
                for r in rows or []:
                    d = _row_to_dict(r)
                    if not d.get("stripe_subscription_id"):
                        continue
                    if key_col == "plan_key":
                        plan = (d.get("plan_key") or "").strip().lower() or "user"
                    else:
                        plan = {"league_subscriptions": "league",
                                "user_league_subscriptions": "single_league"}[table]
                    out.append({
                        "table": table,
                        "stripe_subscription_id": d["stripe_subscription_id"],
                        "plan": plan,
                        "expires_at": d.get("expires_at"),
                        "league_id": d.get("league_id") or "",
                        "platform": (d.get("platform") or "sleeper").strip().lower() or "sleeper",
                    })
    except Exception:
        logger.debug("[churn] active_subscriptions_for_user failed", exc_info=True)
    return out


def pause_subscription(subscription_id: str, months: int = PAUSE_MONTHS) -> dict:
    """Pause collection on a Stripe subscription; auto-resumes after ``months``.

    Uses Stripe's ``pause_collection`` (behavior mark_uncollectible) with
    ``resumes_at`` so billing restarts on its own. This was chosen over a
    stay-discount coupon: one API call, no invoice math, the subscription stays
    active so entitlement rows keep working, and nothing needs a follow-up.
    """
    sub_id = (subscription_id or "").strip()
    if not sub_id:
        return {"ok": False, "error": "Missing subscription id."}
    resumes_at = int((_now() + timedelta(days=30 * months)).timestamp())
    try:
        _stripe().Subscription.modify(
            sub_id,
            pause_collection={"behavior": "mark_uncollectible", "resumes_at": resumes_at},
        )
        return {"ok": True, "resumes_at": resumes_at}
    except Exception as e:
        logger.warning("[churn] pause_subscription failed sub=%s: %s", sub_id, e)
        return {"ok": False, "error": "Stripe could not pause this subscription."}


def cancel_at_period_end(subscription_id: str) -> dict:
    """Schedule a Stripe subscription to cancel at the current period end."""
    sub_id = (subscription_id or "").strip()
    if not sub_id:
        return {"ok": False, "error": "Missing subscription id."}
    try:
        _stripe().Subscription.modify(sub_id, cancel_at_period_end=True)
        return {"ok": True}
    except Exception as e:
        logger.warning("[churn] cancel_at_period_end failed sub=%s: %s", sub_id, e)
        return {"ok": False, "error": "Stripe could not cancel this subscription."}


# ── Win-back offer token ──────────────────────────────────────────────────────

def _winback_secret() -> str:
    return (
        os.environ.get("WINBACK_TOKEN_SECRET", "").strip()
        or os.environ.get("FLASK_SECRET_KEY", "").strip()
        or os.environ.get("CRON_SECRET", "").strip()
    )


def make_winback_token(account_id: int, ttl_days: int = 30) -> str:
    """Signed token for the /pro/winback checkout link (no login needed)."""
    secret = _winback_secret()
    exp = int((_now() + timedelta(days=ttl_days)).timestamp())
    body = f"{int(account_id)}:{exp}"
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{body}:{sig}"


def verify_winback_token(token: str) -> Optional[int]:
    """Return the account id for a valid token, else None."""
    try:
        account_s, exp_s, sig = (token or "").split(":")
        body = f"{account_s}:{exp_s}"
        want = hmac.new(
            _winback_secret().encode(), body.encode(), hashlib.sha256
        ).hexdigest()[:32]
        if not hmac.compare_digest(want, sig):
            return None
        if int(exp_s) < int(_now().timestamp()):
            return None
        return int(account_s)
    except Exception:
        return None


def winback_offer_label() -> str:
    return (os.environ.get("WINBACK_OFFER_LABEL") or "").strip() or "20% off your first year back"


def winback_coupon_id() -> str:
    return (os.environ.get("WINBACK_COUPON_ID") or "").strip()


# ── Daily scan (cron entry point) ─────────────────────────────────────────────

def run_daily_scan() -> dict:
    """One pass of churn touches. Returns counts; never raises."""
    from utils import churn_email

    summary = {"dunning_touch_2": 0, "dunning_resolved": 0,
               "trial_reminders": 0, "winback": 0}
    # 1. Dunning escalation: touch 2 while still past due, resolve if recovered.
    for row in dunning_due_for_escalation():
        sub_id = str(row.get("stripe_subscription_id") or "")
        status = stripe_sub_is_past_due(sub_id)
        if status is False:
            resolve_dunning(sub_id)
            summary["dunning_resolved"] += 1
            continue
        if status is None:
            continue  # Stripe unreachable; try again tomorrow, stay quiet
        if bump_dunning_touch(sub_id):
            acct = _account_row(row.get("account_id"), row.get("email"))
            if acct and churn_email.send_dunning_touch(
                account_id=acct.get("id"),
                email=acct.get("email") or row.get("email") or "",
                first_name=acct.get("first_name"),
                plan=str(row.get("plan") or ""),
                touch=2,
            ):
                record_event(acct.get("id"), "dunning_touch_2",
                             {"sub_id": sub_id, "plan": row.get("plan")})
                summary["dunning_touch_2"] += 1
    # 2. Trial reminders (no-op until the free-trial workstream lands).
    for trial in find_trials_due():
        days = float(trial.get("days_left") or 99)
        want = None
        for d in TRIAL_REMINDER_DAYS:
            if d - 0.5 <= days <= d + 0.5:
                want = d
                break
        if want is None:
            continue
        aid = trial.get("account_id")
        if has_event(f"trial_reminder_{want}d", account_id=aid):
            continue
        acct = _account_row(aid, "")
        if acct and churn_email.send_trial_reminder(
            account_id=acct.get("id"),
            email=acct.get("email") or "",
            first_name=acct.get("first_name"),
            days_left=want,
            plan=str(trial.get("plan") or ""),
        ):
            record_event(aid, f"trial_reminder_{want}d", {"plan": trial.get("plan")})
            summary["trial_reminders"] += 1
    # 3. Win-back: one email per lapsed account.
    if winback_coupon_id():
        for acct in find_winback_candidates():
            aid = acct.get("id")
            token = make_winback_token(int(aid))
            if churn_email.send_winback(
                account_id=aid,
                email=acct.get("email") or "",
                first_name=acct.get("first_name"),
                token=token,
            ):
                record_event(aid, "winback_sent", {})
                summary["winback"] += 1
    else:
        logger.info("[churn] WINBACK_COUPON_ID not set; win-back emails skipped")
    return summary


def _account_row(account_id: Any, fallback_email: str = "") -> Optional[dict]:
    try:
        aid = int(account_id) if account_id else 0
    except (TypeError, ValueError):
        aid = 0
    if aid:
        try:
            from dashboard_services.db import get_conn

            with get_conn() as conn:
                row = conn.execute(
                    "SELECT id, email, first_name FROM accounts WHERE id = %s", (aid,)
                ).fetchone()
                row = _row_to_dict(row)
                if row.get("email"):
                    return row
        except Exception:
            logger.debug("[churn] _account_row lookup failed", exc_info=True)
    if fallback_email and "@" in fallback_email:
        return {"id": aid or None, "email": fallback_email, "first_name": None}
    return None
