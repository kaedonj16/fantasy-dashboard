#!/usr/bin/env python3
"""
Admin script to manually grant premium access.

Usage:
  python scripts/grant_premium.py league <league_id> <subscriber_user_id> [--days 365] [--platform sleeper]
  python scripts/grant_premium.py user <user_id> [--days 365] [--platform sleeper]
"""
import argparse
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")
from dashboard_services.subscriptions import create_league_subscription, create_user_subscription


def main():
    parser = argparse.ArgumentParser(description="Grant premium access to a league or user")
    sub = parser.add_subparsers(dest="type", required=True)

    lp = sub.add_parser("league", help="Grant a league subscription")
    lp.add_argument("league_id", help="Sleeper league ID")
    lp.add_argument("subscriber_user_id", help="Sleeper username of the subscriber")
    lp.add_argument("--days", type=int, default=365, help="Days until expiry (default: 365)")
    lp.add_argument("--platform", default="sleeper", help="Platform (default: sleeper)")

    up = sub.add_parser("user", help="Grant a user subscription")
    up.add_argument("user_id", help="Sleeper username")
    up.add_argument("--days", type=int, default=365, help="Days until expiry (default: 365)")
    up.add_argument("--platform", default="sleeper", help="Platform (default: sleeper)")

    args = parser.parse_args()
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
    else:
        ok = create_user_subscription(args.user_id, expires_at, args.platform)
        if ok:
            print(f"[OK] User premium granted: {args.user_id}, expires {expires_at.date()}")
        else:
            print("[ERROR] Failed to grant user premium", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
