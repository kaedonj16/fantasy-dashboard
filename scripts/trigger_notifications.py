#!/usr/bin/env python3
"""Trigger the web app's notification cron hook.

Render cron services should not import the Flask app. This script POSTs
``/api/cron/notifications`` with ``CRON_SECRET`` (same credential the daily job
uses to flush caches).

Usage:
    python scripts/trigger_notifications.py hourly
    python scripts/trigger_notifications.py weekly
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

ALLOWED = {"hourly", "daily", "weekly"}
DEFAULT_TIMEOUT = {"hourly": 120, "daily": 180, "weekly": 900}


def trigger(kind: str, app_url: str | None = None, secret: str | None = None,
            timeout: int | None = None) -> int:
    kind = (kind or "").strip().lower()
    if kind not in ALLOWED:
        print(f"[notify-cron] unknown type {kind!r}; expected {sorted(ALLOWED)}")
        return 2

    app_url = (app_url if app_url is not None else os.environ.get("APP_URL", "")).rstrip("/")
    secret = secret if secret is not None else os.environ.get("CRON_SECRET", "")
    if not app_url or not secret:
        print("[notify-cron] skipped — APP_URL or CRON_SECRET not set")
        return 1

    body = json.dumps({"type": kind, "secret": secret}).encode()
    req = urllib.request.Request(
        f"{app_url}/api/cron/notifications",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Cron-Secret": secret,
        },
        method="POST",
    )
    wait = timeout if timeout is not None else DEFAULT_TIMEOUT[kind]
    try:
        with urllib.request.urlopen(req, timeout=wait) as resp:
            payload = resp.read()[:800]
            print(f"[notify-cron] {kind}: HTTP {resp.status} {payload!r}")
            return 0 if 200 <= getattr(resp, "status", 0) < 300 else 1
    except urllib.error.HTTPError as exc:
        print(f"[notify-cron] {kind} failed: HTTP {exc.code} {exc.read()[:800]!r}")
        return 1
    except Exception as exc:
        print(f"[notify-cron] {kind} failed: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    kind = args[0] if args else "hourly"
    return trigger(kind)


if __name__ == "__main__":
    raise SystemExit(main())
