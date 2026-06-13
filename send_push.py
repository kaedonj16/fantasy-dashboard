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
parser.add_argument("title", nargs="?", help="Notification title")
parser.add_argument("body",  nargs="?", help="Notification body text")
parser.add_argument("--url",           default="/",    help="URL to open on tap (default: /)")
parser.add_argument("--tag",           default="admin", help="Notification tag (dedupes on device)")
parser.add_argument("--username",      default=None,   help="Send only to this owner_id/username")
parser.add_argument("--league",        default=None,   help="Send only to subscribers in this league_id")
parser.add_argument("--dry-run",       action="store_true", help="Print subscriber count without sending")
parser.add_argument("--generate-keys", action="store_true", help="Generate fresh VAPID keys and print them")
args = parser.parse_args()

# ── Generate keys mode ────────────────────────────────────────────────────────

if args.generate_keys:
    from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, generate_private_key
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
    import base64
    priv_key = generate_private_key(SECP256R1())
    pub_raw  = priv_key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    pub_b64  = base64.urlsafe_b64encode(pub_raw).rstrip(b"=").decode()
    priv_pem = priv_key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()).decode()
    print("\nNew VAPID keys — set these in your Render environment variables:\n")
    print(f"VAPID_PUBLIC_KEY={pub_b64}")
    print(f"VAPID_PRIVATE_KEY={priv_pem.replace(chr(10), r'\\n')}")
    print("\nIMPORTANT: After updating Render env vars, redeploy the app so the")
    print("service worker picks up the new public key from /api/push/vapid-public-key.")
    sys.exit(0)

if not args.title or not args.body:
    parser.error("title and body are required (unless using --generate-keys)")

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
    # Normalize the private key to PEM — pywebpush 2.x needs proper PEM format.
    # The env var may be a raw URL-safe base64 key or a PEM key.
    priv = _normalize_vapid_private_key(priv)
    return pub, priv


def _normalize_vapid_private_key(priv):
    """Accept raw base64url or any PEM variant; always return TraditionalOpenSSL EC PEM."""
    from cryptography.hazmat.primitives.serialization import (
        load_pem_private_key, Encoding, PrivateFormat, NoEncryption,
    )
    from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, derive_private_key
    import base64

    # If it looks like PEM, load it and re-serialize to force TraditionalOpenSSL
    # (-----BEGIN EC PRIVATE KEY-----). This handles PKCS#8 (BEGIN PRIVATE KEY)
    # and explicit-parameter keys that py_vapid can't parse directly.
    if "BEGIN" in priv:
        try:
            loaded = load_pem_private_key(priv.encode(), password=None)
            pem = loaded.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()).decode()
            print(f"  [key] Loaded PEM and re-serialized to TraditionalOpenSSL EC PEM")
            return pem
        except Exception as e:
            print(f"  [key] PEM load failed: {e}")

    # Try as raw URL-safe base64 private key scalar (32 bytes)
    try:
        raw = base64.urlsafe_b64decode(priv + "==")
        if len(raw) == 32:
            key = derive_private_key(int.from_bytes(raw, "big"), SECP256R1())
            pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()).decode()
            print("  [key] Converted raw base64url key to EC PEM")
            return pem
        else:
            print(f"  [key] base64url decoded to {len(raw)} bytes (expected 32)")
    except Exception as e:
        print(f"  [key] base64url parse failed: {e}")

    print(f"  [key] WARNING: could not normalize key (len={len(priv)}, starts={priv[:30]!r})")
    print("  [key] HINT: run with --generate-keys to create fresh VAPID keys")
    return priv

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

def _make_vapid(pem):
    """Build Vapid object directly from PEM, bypassing from_string()->from_der() bug."""
    from cryptography.hazmat.primitives.serialization import (
        load_pem_private_key, Encoding, PrivateFormat, NoEncryption,
    )
    from py_vapid import Vapid
    loaded = load_pem_private_key(pem.encode(), password=None)
    der = loaded.private_bytes(Encoding.DER, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    return Vapid.from_der(der)


def send(rows, title, body, url, tag, priv_key):
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        sys.exit("ERROR: pywebpush not installed. Run: pip install pywebpush")

    try:
        vapid_obj = _make_vapid(priv_key)
        print(f"  [vapid] Key loaded OK via from_der()")
    except Exception as e:
        sys.exit(f"ERROR: Could not build Vapid object: {e}")

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
                vapid_private_key=vapid_obj,
                vapid_claims={"sub": "mailto:admin@brfantasy.com"},
            )
            sent += 1
        except WebPushException as e:
            resp = e.response
            status = resp.status_code if resp else "?"
            body = resp.text[:200] if resp else ""
            if resp and resp.status_code in (404, 410):
                stale += 1
            else:
                print(f"  FAIL [{status}]: {ep[:70]}...")
                if body:
                    print(f"         {body}")
                failed += 1
        except Exception as e:
            print(f"  FAIL [exc]: {ep[:70]}...")
            print(f"         {type(e).__name__}: {e}")
            failed += 1

    return sent, failed, stale

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pub_key, priv_key = get_vapid_keys()
    print(f"\nVAPID pub  : {pub_key[:30]}...")
    print(f"VAPID priv : {priv_key[:40].strip()!r}...")

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
