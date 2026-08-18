"""ESPN email + one-time-code (OTP) login broker.

Feature-flagged via ``ESPN_OTP_LOGIN_ENABLED`` (OFF by default). When on, this
obtains ``espn_s2`` + ``SWID`` from an email one-time passcode and hands them to
the normal ``connect_league`` pipeline — so a member can sign in with an email
and a code instead of copying cookies. Cookie-paste and the browser extension
remain the fallback for every failure, and the flag is a kill-switch.

Because automating ESPN/Disney login carries Terms-of-Service and reliability
risk, the real headless driver (``PlaywrightEspnLoginBroker``) is deliberately
stubbed until the flow is validated on a networked host (see the OTP spike). The
broker abstraction, the short-lived session store, rate limiting, and the whole
request/verify contract are complete and covered by tests against the mock.
"""
from __future__ import annotations

import os
import secrets
import threading
import time
import uuid
from typing import Optional


# ── errors (mapped to user-facing messages + HTTP status in the route) ────────
class EspnLoginError(Exception):
    """Base for login-broker failures."""


class EspnLoginUnavailable(EspnLoginError):
    """The broker can't run (flag off, not configured, or not yet implemented)."""


class EspnLoginInvalidCode(EspnLoginError):
    """The submitted code was wrong."""


class EspnLoginExpired(EspnLoginError):
    """The login session (login_id) expired or is unknown."""


class EspnLoginCaptchaRequired(EspnLoginError):
    """ESPN presented a human challenge we can't clear headlessly."""


class EspnLoginTooManyAttempts(EspnLoginError):
    """Too many wrong codes for one login session."""


class EspnLoginRateLimited(EspnLoginError):
    """Too many code requests for one email in the window."""


def otp_login_enabled() -> bool:
    """Kill-switch. OFF unless the env var is explicitly truthy."""
    return os.getenv("ESPN_OTP_LOGIN_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


# ── tunables ──────────────────────────────────────────────────────────────────
_SESSION_TTL = 300          # seconds a login_id stays valid (inbox round-trip)
_MAX_VERIFY_ATTEMPTS = 5    # wrong-code attempts before a session is burned
_START_LIMIT = 5           # OTP requests allowed per email...
_START_WINDOW = 900        # ...within this many seconds


class _Store:
    """In-memory, TTL'd login sessions + per-email request throttle.

    NOTE: process-local. Fine for the mock and single-worker runs; a multi-worker
    deployment needs a shared store (e.g. Redis) before the real broker ships.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}
        self._starts: dict[str, list[float]] = {}

    def rate_ok(self, email: str, now: float) -> bool:
        with self._lock:
            recent = [t for t in self._starts.get(email, []) if now - t < _START_WINDOW]
            if len(recent) >= _START_LIMIT:
                self._starts[email] = recent
                return False
            recent.append(now)
            self._starts[email] = recent
            return True

    def create(self, email: str, now: float, extra: Optional[dict] = None) -> str:
        with self._lock:
            for lid in [k for k, v in self._sessions.items() if now - v["created_at"] > _SESSION_TTL]:
                self._sessions.pop(lid, None)
            login_id = secrets.token_urlsafe(24)
            self._sessions[login_id] = {"email": email, "created_at": now, "attempts": 0, **(extra or {})}
            return login_id

    def get(self, login_id: str, now: float) -> Optional[dict]:
        with self._lock:
            s = self._sessions.get(login_id)
            if not s:
                return None
            if now - s["created_at"] > _SESSION_TTL:
                self._sessions.pop(login_id, None)
                return None
            return s

    def drop(self, login_id: str) -> None:
        with self._lock:
            self._sessions.pop(login_id, None)


class EspnLoginBroker:
    """Broker contract: start → (email arrives) → verify. Subclasses implement the
    provider-specific hooks; this base owns session lifetime, throttling, and the
    attempt cap so every implementation behaves the same."""

    def __init__(self) -> None:
        self._store = _Store()

    def start(self, email: str) -> str:
        email = (email or "").strip()
        if not email or "@" not in email:
            raise EspnLoginError("A valid email is required.")
        now = time.time()
        if not self._store.rate_ok(email, now):
            raise EspnLoginRateLimited("Too many code requests. Wait a few minutes and try again.")
        extra = self._begin(email)
        return self._store.create(email, now, extra)

    def verify(self, login_id: str, code: str) -> dict:
        now = time.time()
        s = self._store.get(login_id, now)
        if not s:
            raise EspnLoginExpired("This sign-in expired. Request a new code.")
        code = (code or "").strip()
        if not code:
            raise EspnLoginInvalidCode("Enter the code from your email.")
        s["attempts"] += 1
        if s["attempts"] > _MAX_VERIFY_ATTEMPTS:
            self._store.drop(login_id)
            raise EspnLoginTooManyAttempts("Too many tries. Start the sign-in again.")
        creds = self._submit(s, code)  # returns {"swid","espn_s2"} or raises
        self._store.drop(login_id)
        return creds

    def resend(self, login_id: str) -> None:
        s = self._store.get(login_id, time.time())
        if not s:
            raise EspnLoginExpired("This sign-in expired. Request a new code.")
        self._resend(s)

    # subclass hooks ----------------------------------------------------------
    def _begin(self, email: str) -> dict:
        raise NotImplementedError

    def _submit(self, session: dict, code: str) -> dict:
        raise NotImplementedError

    def _resend(self, session: dict) -> None:
        raise NotImplementedError


class MockEspnLoginBroker(EspnLoginBroker):
    """Deterministic broker for tests/dev (``ESPN_OTP_BROKER=mock``). The code
    ``123456`` succeeds and yields placeholder cookies."""

    CODE = "123456"

    def _begin(self, email: str) -> dict:
        return {"mock": True}

    def _submit(self, session: dict, code: str) -> dict:
        if code != self.CODE:
            raise EspnLoginInvalidCode("That code isn't right. Check your email and try again.")
        return {"swid": "{MOCK-SWID-0000-0000}", "espn_s2": "MOCK_ESPN_S2_VALUE"}

    def _resend(self, session: dict) -> None:
        return None


class PlaywrightEspnLoginBroker(EspnLoginBroker):
    """Real headless Disney OneID driver.

    NOT finalized: the OTP spike confirmed the flow is drivable and captcha-free,
    but the production driver (whether a live headless context per login or a
    replay of the OTP API — decided from the spike's network capture) still needs
    a networked validation run. Until then it refuses cleanly, so with the flag on
    users fall back to cookie paste instead of hitting a half-built browser.
    """

    def _begin(self, email: str) -> dict:
        raise EspnLoginUnavailable("ESPN email sign-in isn't available yet.")

    def _submit(self, session: dict, code: str) -> dict:
        raise EspnLoginUnavailable("ESPN email sign-in isn't available yet.")

    def _resend(self, session: dict) -> None:
        raise EspnLoginUnavailable("ESPN email sign-in isn't available yet.")


# ── real driver: the Disney OneID email-OTP API (Option B — no browser) ───────
# The spike captured this exact five-call flow, and no reCAPTCHA token rides on
# any request. Shapes are validated; the flow still needs a networked run before
# the flag is enabled. httpx is imported lazily so this module stays importable
# (and the broker/mock tests run) without it.
_ONEID_BASE = "https://registerdisney.go.com/jgc/v8/client/ESPN-ONESITE.WEB-PROD"
_ONEID_TIMEOUT = 15.0


class OneIdOtpBroker(EspnLoginBroker):
    """Drives Disney OneID passwordless (email OTP) directly over HTTPS.

    Config: ``ESPN_ONEID_API_KEY`` is the full ``authorization`` header value for
    the ESPN-ONESITE.WEB-PROD client (a public client key from OneID.js). Without
    it the broker is unavailable, so the flow degrades to cookie paste.
    """

    def _api_key(self) -> str:
        key = os.getenv("ESPN_ONEID_API_KEY", "").strip()
        if not key:
            raise EspnLoginUnavailable("ESPN email sign-in isn't configured.")
        return key

    def _headers(self, conversation_id: str, authorization: Optional[str] = None) -> dict:
        headers = {
            "Content-Type": "application/json",
            "conversation-id": conversation_id,
            "correlation-id": str(uuid.uuid4()),
        }
        if authorization:
            headers["authorization"] = authorization
        return headers

    @staticmethod
    def _data(resp) -> dict:
        try:
            body = resp.json()
        except Exception:
            raise EspnLoginError("Unexpected response from ESPN sign-in.")
        if isinstance(body, dict) and body.get("error"):
            raise EspnLoginError("ESPN rejected the sign-in step.")
        return (body or {}).get("data") or {}

    def _begin(self, email: str) -> dict:
        api_key = self._api_key()  # raises Unavailable before importing httpx
        import httpx
        conversation_id = str(uuid.uuid4())
        try:
            with httpx.Client(base_url=_ONEID_BASE, timeout=_ONEID_TIMEOUT) as client:
                client.post("/guest-flow", json={"email": email}, headers=self._headers(conversation_id))
                client.post("/guest/recovery-methods", json={"loginValue": email},
                            headers=self._headers(conversation_id, api_key))
                data = self._data(client.post("/notification/otp/recovery", json={"lookupValue": email},
                                              headers=self._headers(conversation_id, api_key)))
        except httpx.HTTPError:
            raise EspnLoginError("Couldn't reach ESPN sign-in. Try again.")
        session_id = data.get("sessionId")
        if not session_id:
            raise EspnLoginError("ESPN didn't start the sign-in. Try again.")
        return {"email": email, "conversation_id": conversation_id, "api_key": api_key, "session_id": session_id}

    def _submit(self, session: dict, code: str) -> dict:
        import httpx
        conversation_id, api_key, session_id = session["conversation_id"], session["api_key"], session["session_id"]
        try:
            with httpx.Client(base_url=_ONEID_BASE, timeout=_ONEID_TIMEOUT) as client:
                redeem = client.post("/otp/redeem", json={"passcode": code, "sessionIds": [session_id]},
                                     headers=self._headers(conversation_id, api_key))
                if redeem.status_code in (400, 401, 403):
                    raise EspnLoginInvalidCode("That code isn't right. Check your email and try again.")
                redeemed = self._data(redeem)
                swid = redeemed.get("swid")
                recovery_token = (redeemed.get("recoveryToken") or {}).get("access_token")
                if not swid or not recovery_token:
                    raise EspnLoginInvalidCode("That code isn't right. Check your email and try again.")
                login = self._data(client.post("/guest/login/recoveryToken",
                                               json={"swid": swid, "recoveryToken": recovery_token},
                                               headers=self._headers(conversation_id, api_key)))
        except httpx.HTTPError:
            raise EspnLoginError("Couldn't reach ESPN sign-in. Try again.")
        espn_s2 = login.get("s2")
        final_swid = (login.get("token") or {}).get("swid") or swid
        if not espn_s2 or not final_swid:
            raise EspnLoginError("ESPN sign-in completed but didn't return a session.")
        return {"swid": final_swid, "espn_s2": espn_s2}

    def _resend(self, session: dict) -> None:
        import httpx
        try:
            with httpx.Client(base_url=_ONEID_BASE, timeout=_ONEID_TIMEOUT) as client:
                data = self._data(client.post("/notification/otp/recovery", json={"lookupValue": session["email"]},
                                              headers=self._headers(session["conversation_id"], session["api_key"])))
        except httpx.HTTPError:
            raise EspnLoginError("Couldn't resend the code. Try again.")
        if data.get("sessionId"):
            session["session_id"] = data["sessionId"]


_BROKER: Optional[EspnLoginBroker] = None


def get_broker() -> EspnLoginBroker:
    """Singleton broker (holds the session store, so start/verify share state).

    ``ESPN_OTP_BROKER``: ``mock`` (tests/dev) → deterministic; ``oneid`` → the
    real OneID API driver; anything else → the unavailable default, so the flag
    can be on for staging without a working driver.
    """
    global _BROKER
    if _BROKER is None:
        kind = os.getenv("ESPN_OTP_BROKER", "").strip().lower()
        if kind == "mock":
            _BROKER = MockEspnLoginBroker()
        elif kind == "oneid":
            _BROKER = OneIdOtpBroker()
        else:
            _BROKER = PlaywrightEspnLoginBroker()
    return _BROKER


def _reset_broker() -> None:
    """Test hook: forget the cached broker so env changes take effect."""
    global _BROKER
    _BROKER = None
