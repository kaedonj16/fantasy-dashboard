#!/usr/bin/env python3
"""Diagnose WHY the OneID call fails — surfaces the real exception the broker hides.

    export ESPN_ONEID_API_KEY='<full authorization header value>'
    python3 diagnose_oneid.py you@your-espn-email.com

Prints python/httpx versions, any proxy env in effect, and the exact error (or the
HTTP status + a snippet of the body) for the first two OneID calls. No secrets are
printed — not the key, not your email's response contents beyond a short status.
"""
import os
import sys
import uuid

BASE = "https://registerdisney.go.com/jgc/v8/client/ESPN-ONESITE.WEB-PROD"


def main(email: str) -> int:
    try:
        import httpx
    except ModuleNotFoundError:
        print("httpx is not installed in this interpreter. Run: python3 -m pip install httpx")
        return 2

    print(f"python {sys.version.split()[0]}   httpx {httpx.__version__}")
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "NO_PROXY"):
        if os.getenv(var):
            print(f"  proxy env: {var}={os.getenv(var)}")
    key = os.getenv("ESPN_ONEID_API_KEY", "").strip()
    if not key:
        print("ESPN_ONEID_API_KEY is not set.")
        return 2
    print(f"  api key: present, scheme={key.split(' ', 1)[0]!r}, length={len(key)}")

    conv = str(uuid.uuid4())
    headers = {"Content-Type": "application/json", "conversation-id": conv,
               "correlation-id": str(uuid.uuid4())}

    def call(label, path, body, auth=False):
        h = dict(headers)
        if auth:
            h["authorization"] = key
        print(f"\n→ {label}  POST {path}")
        try:
            with httpx.Client(base_url=BASE, timeout=15.0) as c:
                r = c.post(path, json=body, headers=h)
            print(f"   HTTP {r.status_code}")
            snippet = r.text[:300].replace("\n", " ")
            print(f"   body: {snippet}")
            return r
        except Exception as exc:  # noqa: BLE001 — we want the raw type here
            print(f"   FAILED: {type(exc).__module__}.{type(exc).__name__}: {exc}")
            cause = exc.__cause__ or exc.__context__
            if cause:
                print(f"   caused by: {type(cause).__module__}.{type(cause).__name__}: {cause}")
            return None

    call("guest-flow", "/guest-flow", {"email": email})
    call("recovery-methods", "/guest/recovery-methods", {"loginValue": email}, auth=True)
    print("\nSend me everything above (it contains no secrets).")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 diagnose_oneid.py you@your-espn-email.com")
    raise SystemExit(main(sys.argv[1]))
