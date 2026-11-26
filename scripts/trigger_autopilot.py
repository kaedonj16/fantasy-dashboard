#!/usr/bin/env python3
"""Trigger one Autopilot scheduler pass on the web service.

Mirrors scripts/trigger_notifications.py: the Render cron service
itself is just a clock; the work runs on the web service via the
secret-gated /api/cron/autopilot hook, which builds plans for every
usable league with Autopilot on, executes Auto leagues, and queues
Approve leagues with a "moves waiting" notification.

The render.yaml cron services fire at every candidate trigger time
the slate-derived windows can need (the legacy Thu-evening and
Sunday passes plus weekday/weekend slate specials); the web endpoint
due-gates each firing against the week's real schedule, so a firing
with no window due is a cheap no-op.

Env: APP_URL (web service base URL), CRON_SECRET (shared with the
web service). Stdlib only; no project imports.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

# Render's egress is sometimes IPv6-first and urllib can pick a dead
# IPv6 route; prefer IPv4 like trigger_notifications.py does.
import socket

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
    results = _orig_getaddrinfo(host, port, family, type, proto, flags)
    v4 = [r for r in results if r[0] == socket.AF_INET]
    return v4 or results


socket.getaddrinfo = _ipv4_first


def main() -> int:
    app_url = (os.environ.get("APP_URL") or "").rstrip("/")
    secret = os.environ.get("CRON_SECRET") or ""
    if not app_url or not secret:
        print("[trigger_autopilot] APP_URL / CRON_SECRET not set; exiting",
              file=sys.stderr)
        return 1
    url = f"{app_url}/api/cron/autopilot"
    body = json.dumps({"secret": secret}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Cron-Secret": secret,
            # Cloudflare rejects the default Python-urllib UA (error
            # 1010); present as a browser like trigger_notifications.py.
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read().decode("utf-8", "replace")
            print(f"[trigger_autopilot] HTTP {resp.status}: {payload[:300]}")
            return 0 if resp.status < 300 else 1
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        print(f"[trigger_autopilot] HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface anything
        print(f"[trigger_autopilot] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
