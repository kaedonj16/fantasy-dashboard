#!/usr/bin/env python3
"""
Admin script to manually grant premium access.

Usage:
  python scripts/grant_premium.py league <league_id> <subscriber_user_id> [--days 365] [--platform sleeper]
  python scripts/grant_premium.py user <user_id> [--days 365] [--platform sleeper] [--plan hall_of_fame]
  python scripts/grant_premium.py user-email <email> [--days 365] [--plan hall_of_fame]
  python scripts/grant_premium.py single_league <user_id> <league_id> [--days 365] [--platform sleeper]

Plans: hall_of_fame (default; PRO everywhere, no league-slot cap),
       starter (1 league slot), all_pro (5 league slots).
"""
import argparse
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")
from dashboard_services.db import get_conn
from dashboard_services.subscriptions import (
    create_league_subscription,
    create_user_league_subscription,
    create_user_subscription,
)


def resolve_account_id(email):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM accounts WHERE email = %s", (email.strip().lower(),)
        )
        row = cur.fetchone()
    if not row:
        return None
    return row[0] if not isinstance(row, dict) else row["id"]


def add_common(p):
    p.add_argument("--days", type=int, default=365, help="Days until expiry (default: 365)")
    p.add_argument("--platform", default="sleeper", help="Platform (default: sleeper)")
    p.add_argument("--plan", default="hall_of_fame",
                   help="Plan key: hall_of_fame (default, PRO everywhere), starter, all_pro")


def main():
    parser = argparse.ArgumentParser(description="Grant premium access to a league or user")
    sub = parser.add_subparsers(dest="type", required=True)

    lp = sub.add_parser("league", help="Grant a shared league subscription")
    lp.add_argument("league_id", help="League ID")
    lp.add_argument("subscriber_user_id", help="Sleeper username of the subscriber")
    add_common(lp)

    up = sub.add_parser("user", help="Grant a personal subscription (Sleeper user_id/username)")
    up.add_argument("user_id", help="Sleeper user id or username")
    add_common(up)

    ue = sub.add_parser("user-email", help="Grant a personal subscription to a Google-login user by email")
    ue.add_argument("email", help="Google account email")
    add_common(ue)

    sl = sub.add_parser("single_league", help="Grant a personal single-league subscription")
    sl.add_argument("user_id", help="User id / username / acct:N key")
    sl.add_argument("league_id", help="League ID")
    add_common(sl)

    args = parser.parse_args()
    plan_key = (args.plan or "hall_of_fame").strip().lower()
    expires_at = datetime.now(timezone.utc) + timedelta(days=args.days)

    if args.type == "league":
        ok = create_league_subscription(
            args.league_id, args.subscriber_user_id, expires_at, args.platform
        )
        if ok:
            print(f"[OK] League premium granted: {args.league_id} via {args.subscriber_user_id}, expires {expires_at.date()}")
        else:
            print("[ERROR] Failed to grant league premium", file=sys.stderr)
            sys.exit(1)
    elif args.type == "single_league":
        ok = create_user_league_subscription(
            args.user_id, args.league_id, expires_at, args.platform
        )
        if ok:
            print(f"[OK] Single-league premium granted: {args.user_id} @ {args.league_id}, expires {expires_at.date()}")
        else:
            print("[ERROR] Failed to grant single-league premium", file=sys.stderr)
            sys.exit(1)
    else:
        if args.type == "user-email":
            acct_id = resolve_account_id(args.email)
            if not acct_id:
                print(f"[ERROR] No Google account for {args.email}: never logged in", file=sys.stderr)
                sys.exit(1)
            user_id = f"acct:{acct_id}"
        else:
            user_id = args.user_id
        ok = create_user_subscription(user_id, expires_at, args.platform, plan_key=plan_key)
        if ok:
            print(f"[OK] User premium granted: {user_id} (plan={plan_key}), expires {expires_at.date()}")
        else:
            print("[ERROR] Failed to grant user premium", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
