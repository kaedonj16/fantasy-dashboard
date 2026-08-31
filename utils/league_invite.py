"""League-shared PRO invite helpers (roadmap R11).

Invite links land on ``/invite/<platform>/<season>/<league_id>`` so teammates
can identify into that league and inherit the league plan entitlement.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import quote


_PLATFORMS = frozenset({"sleeper", "espn", "yahoo", "mfl", "fleaflicker"})


def normalize_invite_platform(platform: Optional[str]) -> str:
    p = (platform or "sleeper").strip().lower()
    return p if p in _PLATFORMS else "sleeper"


def league_invite_path(platform: str, season: int, league_id: str) -> str:
    """Relative path for a shareable league-PRO invite."""
    plat = normalize_invite_platform(platform)
    lid = quote(str(league_id or "").strip(), safe="")
    return f"/invite/{plat}/{int(season)}/{lid}"


def league_invite_url(base: str, platform: str, season: int, league_id: str) -> str:
    return f"{(base or '').rstrip('/')}{league_invite_path(platform, season, league_id)}"


def dashboard_after_invite(platform: str, season: int, league_id: str) -> str:
    plat = normalize_invite_platform(platform)
    lid = quote(str(league_id or "").strip(), safe="")
    return f"/{plat}/{int(season)}/{lid}/dashboard?league_pro=1"


def is_league_plan_buyer(viewer_ids: set[str], subscriber_user_id: Optional[str]) -> bool:
    """True when the current viewer matches the league plan's Stripe buyer id."""
    buyer = str(subscriber_user_id or "").strip()
    if not buyer:
        return False
    return buyer in {str(v).strip() for v in viewer_ids if v}
