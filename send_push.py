#!/usr/bin/env python3
"""
Send a push notification to all subscribers (or a specific user).

Usage:
  python send_push.py "Title" "Body text"
  python send_push.py "Title" "Body text" --url /trade
  python send_push.py "Title" "Body text" --username hoodiekj1
  python send_push.py "Title" "Body text" --league 123456789

Requires VAPID_PUBLIC_KEY, VAPID_PRIVATE_KEY, and DATABASE_URL env vars
(same ones used by the app — already set on Render).
"""
import argparse
import json
import os
import sys

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Send a push notification")
parser.add_argument("title", help="Notification title")
parser.add_argument("body",  help="Notification body text")
parser.add_argument("--url",      default="/",     help="URL to open on tap (default: /)")
parser.add_argument("--tag",      default="admin",  help="Notification tag (dedupes on device)")
parser.add_argument("--username", default=None,     help="Send only to this Sleeper username (owner_id)")
parser.add_argument("--league",   default=None,     help="Send only to subscribers in this league_id")
parser.add_argument("--dry-run",  action="store_true", help="Print subscriber count without sending")
args = parser.parse_args()

# ── DB connection ─────────────────────────────────────────────────────────────

def get_db():
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        sys.exit("ERROR: DATABASE_URL env var not set.")
    try:
        import psycopg
        return psycopg.connect(db_url, row_factory=psycopg.rows.dict_row)
    except Exception as e:
        sys.exit(f"ERROR: Could not connect to DB: {e}")

# ── VAPID keys ─────────────────────────────────────────────────────────────────

def get_vapid_keys():
    pub  = os.environ.get("VAPID_PUBLIC_KEY",  "").strip()
    priv = os.environ.get("VAPID_PRIVATE_KEY", "").replace("\\n", "\n").strip()
    if not pub or not priv:
        sys.exit("ERROR: VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY env vars must be set.")
    return pub, priv

# ── Fetch subscribers ─────────────────────────────────────────────────────────

def get_subscribers(conn, username=None, league=None):
    if username:
        rows = conn.execute(
            "SELECT endpoint, p256dh, auth FROM push_subscriptions WHERE owner_id = %s",
            (username,)
        ).fetchall()
    elif league:
        rows = conn.execute(
            "SELECT endpoint, p256dh, auth FROM push_subscriptions WHERE league_id = %s",
            (str(league),)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT endpoint, p256dh, auth FROM push_subscriptions"
        ).fetchall()
    return rows

# ── Send ──────────────────────────────────────────────────────────────────────

def send(rows, title, body, url, tag, priv_key):
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        sys.exit("ERROR: pywebpush not installed. Run: pip install pywebpush")

    payload = json.dumps({
        "title": title,
        "body":  body,
        "url":   url,
        "tag":   tag,
        "actions": [{"action": "view", "title": "View"}],
    })

    sent = failed = stale = 0
    for row in rows:
        ep, p256dh, auth = row["endpoint"], row["p256dh"], row["auth"]
        try:
            webpush(
                subscription_info={"endpoint": ep, "keys": {"p256dh": p256dh, "auth": auth}},
                data=payload,
                vapid_private_key=priv_key,
                vapid_claims={"sub": "mailto:admin@brfantasy.com"},
            )
            sent += 1
        except WebPushException as e:
            if e.response and e.response.status_code in (404, 410):
                stale += 1
            else:
                print(f"  WARN: {ep[:60]}... → {e}")
                failed += 1
        except Exception as e:
            print(f"  WARN: {ep[:60]}... → {e}")
            failed += 1

    return sent, failed, stale

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _, priv_key = get_vapid_keys()

    conn = get_db()
    rows = get_subscribers(conn, username=args.username, league=args.league)

    target = f"@{args.username}" if args.username else (f"league {args.league}" if args.league else "all subscribers")
    print(f"\nTarget : {target}")
    print(f"Count  : {len(rows)} subscriber(s)")
    print(f"Title  : {args.title}")
    print(f"Body   : {args.body}")
    print(f"URL    : {args.url}")
    print(f"Tag    : {args.tag}")

    if not rows:
        print("\nNo subscribers matched — nothing sent.")
        return

    if args.dry_run:
        print("\n[dry-run] No notifications sent.")
        return

    print("\nSending...")
    sent, failed, stale = send(rows, args.title, args.body, args.url, args.tag, priv_key)
    print(f"  Sent   : {sent}")
    if failed: print(f"  Failed : {failed}")
    if stale:  print(f"  Stale  : {stale} (expired subscriptions, safe to ignore)")
    print("Done.")

if __name__ == "__main__":
    main()
