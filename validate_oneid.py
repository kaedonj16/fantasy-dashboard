#!/usr/bin/env python3
"""One-off: prove the live OneID OTP driver works, end to end, before flipping the flag.

Runs the REAL OneIdOtpBroker against Disney/ESPN with your email. The api-key and the
returned cookies stay on this machine — the script prints only lengths, never values.

    export ESPN_ONEID_API_KEY='<the full authorization header value, e.g. APIKEY abc...>'
    python3 validate_oneid.py you@your-espn-email.com

It sends a code to your inbox, you paste it back here, and it reports whether ESPN
returned a usable SWID + espn_s2. Run it from the repo root so the import resolves.
"""
import os
import sys

os.environ["ESPN_OTP_BROKER"] = "oneid"  # force the real driver regardless of env


def main(email: str) -> int:
    if not os.getenv("ESPN_ONEID_API_KEY", "").strip():
        print("Set ESPN_ONEID_API_KEY first (the full authorization header value).")
        return 2
    from dashboard_services.providers import espn_login as L

    broker = L.OneIdOtpBroker()
    try:
        login_id = broker.start(email)
    except L.EspnLoginError as exc:
        print(f"start failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"→ code sent to {email} (login id {login_id[:8]}…). Check your email.")

    code = input("Enter the 6-digit code: ").strip()
    try:
        creds = broker.verify(login_id, code)
    except L.EspnLoginError as exc:
        print(f"verify failed: {type(exc).__name__}: {exc}")
        return 1

    swid, s2 = creds.get("swid", ""), creds.get("espn_s2", "")
    ok = swid.startswith("{") and swid.endswith("}") and len(s2) > 100
    print(f"  SWID: {'present' if swid else 'MISSING'} (len {len(swid)})")
    print(f"  espn_s2: {'present' if s2 else 'MISSING'} (len {len(s2)})")
    print("\n" + ("PASS — the driver is live-validated; safe to enable the flag."
                  if ok else "FAIL — cookies look wrong; do not enable the flag."))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 validate_oneid.py you@your-espn-email.com")
    raise SystemExit(main(sys.argv[1]))
