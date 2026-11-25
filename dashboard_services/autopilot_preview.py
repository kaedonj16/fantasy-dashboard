"""Owner-preview gate for the Autopilot page mock.

Autopilot (spec: ~/workspace/autopilot-spec.md) is an unbuilt, PRO-gated
feature. This module gates the in-site UI mock so only the site owner can see
it while it is evaluated: the Autopilot tab, the page itself, and the
portfolio / dashboard status indicators all consult the same check and fail
closed.

Visibility is granted when ANY of these holds for the current request:

1. The session is admin-authenticated (``dashboard_services.admin_auth``,
   the same mechanism the /admin surfaces use).
2. The signed-in account id is listed in ``AUTOPILOT_PREVIEW_ACCOUNT_IDS``
   (comma-separated).
3. The signed-in account email is listed in ``AUTOPILOT_PREVIEW_EMAILS``
   (comma-separated, case-insensitive; the web service sets this to the
   owner's email in render.yaml).

On top of identity, the mock is scoped to the one proven pilot league
(ESPN 887776065, where the 2026-11-24 probe proved lineup writes). Every
other league and platform renders exactly as before.
"""
from __future__ import annotations

import os

#: The pilot league the mock is scoped to (the probe league).
PILOT_PLATFORM = "espn"
PILOT_LEAGUE_ID = "887776065"

#: Sample mode the indicators report for the pilot league. The page mock
#: opens in Approve; the badge/chip name that mode.
SAMPLE_MODE_LABEL = "Approve"


def _env_set(name: str) -> frozenset:
    raw = os.environ.get(name, "") or ""
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def autopilot_preview_allowed() -> bool:
    """Whether the current viewer may see the Autopilot mock surfaces."""
    try:
        from dashboard_services.admin_auth import is_admin

        if is_admin():
            return True
    except Exception:
        pass
    try:
        from flask import session

        account_id = session.get("account_id")
        email = (session.get("account_email") or "").strip().lower()
    except Exception:
        return False
    if account_id is not None and str(account_id) in _env_set(
        "AUTOPILOT_PREVIEW_ACCOUNT_IDS"
    ):
        return True
    if email and email in _env_set("AUTOPILOT_PREVIEW_EMAILS"):
        return True
    return False


def autopilot_page_visible(platform, league_id) -> bool:
    """Gate for the Autopilot tab and page: pilot league + preview viewer."""
    if str(platform or "").lower() != PILOT_PLATFORM:
        return False
    if str(league_id or "") != PILOT_LEAGUE_ID:
        return False
    return autopilot_preview_allowed()


def autopilot_indicator_mode(platform, league_id):
    """Sample Autopilot mode label for status indicators, or None.

    Used by the Portfolio card badge, the league dashboard header chip, and
    the /api/my-leagues payload. None means: render no indicator.
    """
    if autopilot_page_visible(platform, league_id):
        return SAMPLE_MODE_LABEL
    return None


# ---------------------------------------------------------------------------
# Real Phase 1 access seam (backend; the UI mock above still uses the
# preview-only checks until the UI PR rewires it).
#
# autopilot_access(account_id, platform, league_id, season) is THE gate
# every Phase 1 backend path consults. It is the conjunction of:
#
# 1. Pilot rollout: the account is the site admin session, or its id is in
#    AUTOPILOT_PREVIEW_ACCOUNT_IDS, or its email is in
#    AUTOPILOT_PREVIEW_EMAILS (render.yaml sets the owner's email).
# 2. PRO: subscriptions.has_premium_access for the account + league
#    (Kaedon's decision 2026-11-24: Autopilot is PRO-gated at every mode).
# 3. Usable league, data-driven: the platform is ESPN, a user_leagues row
#    links THIS account to the league, and ESPN credentials resolve for
#    this account (league-attached or same-account fallback). Sleeper and
#    Yahoo leagues are never usable in Phase 1 and resolve False, so they
#    render no Autopilot surface at all.
# ---------------------------------------------------------------------------

def _account_email(account_id):
    """The account's email from the accounts table, or None."""
    if not account_id:
        return None
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            row = conn.execute(
                "SELECT email FROM accounts WHERE id = %s", (int(account_id),)
            ).fetchone()
        if row:
            return str(row["email"] or "").strip().lower() or None
    except Exception:
        pass
    return None


def pilot_account_allowed(account_id) -> bool:
    """Pilot-rollout check for an explicit account id (no session needed)."""
    try:
        from dashboard_services.admin_auth import is_admin

        if is_admin():
            return True
    except Exception:
        pass
    if account_id is None:
        return False
    if str(account_id) in _env_set("AUTOPILOT_PREVIEW_ACCOUNT_IDS"):
        return True
    email = None
    try:
        from flask import has_request_context, session

        if has_request_context() and session.get("account_id") is not None \
                and str(session.get("account_id")) == str(account_id):
            email = (session.get("account_email") or "").strip().lower() or None
    except Exception:
        email = None
    if email is None:
        email = _account_email(account_id)
    return bool(email and email in _env_set("AUTOPILOT_PREVIEW_EMAILS"))


def league_is_autopilot_usable(account_id, platform, league_id, season) -> bool:
    """Data-driven usability: ESPN + linked to this account + credentials
    resolve for this account. Any doubt resolves False (fail closed)."""
    if not account_id or str(platform or "").lower() != "espn":
        return False
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            row = conn.execute(
                """SELECT id FROM user_leagues
                   WHERE account_id = %s AND platform = 'espn'
                     AND league_id = %s AND season = %s""",
                (int(account_id), str(league_id), int(season)),
            ).fetchone()
        if not row:
            return False
    except Exception:
        return False
    try:
        from dashboard_services.accounts import (
            get_any_espn_account_credentials,
            get_espn_league_credentials,
        )

        creds = get_espn_league_credentials(
            int(account_id), str(league_id), int(season)
        ) or {}
        if not (creds.get("espn_s2") and creds.get("swid")):
            creds = get_any_espn_account_credentials(int(account_id)) or {}
        return bool(creds.get("espn_s2") and creds.get("swid"))
    except Exception:
        return False


def autopilot_access(account_id, platform, league_id, season) -> bool:
    """The single Phase 1 gate: pilot AND PRO AND usable league."""
    if not pilot_account_allowed(account_id):
        return False
    if not league_is_autopilot_usable(account_id, platform, league_id, season):
        return False
    try:
        from dashboard_services.subscriptions import has_premium_access

        return bool(
            has_premium_access(
                None, str(league_id), str(platform or "espn"),
                account_id=int(account_id),
            )
        )
    except Exception:
        return False
