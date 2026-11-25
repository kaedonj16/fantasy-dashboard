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
